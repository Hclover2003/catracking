# import packages
import os
import cv2
import numpy as np;
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import albumentations as album
import joblib

from scipy import ndimage
from typing import Tuple, List
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

## Define UNet Architecture

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)
   
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of the UNet model"""
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

def get_training_augmentation():
    """ Add augmentation to the training data. Crop it to 256, 256 and flip it horizontally, vertically or rotate it by 90 degrees.
    
    Returns:
        album.Compose: Composed augmentation functions
    """
    train_transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True), # crop it to 256, 256
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():   
    """ Add augmentation to the validation data. Add padding to make it 1536, 1536. 
    
    Returns:
        album.Compose: Composed augmentation functions
    """
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0), # crop it to 1536, 1536
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    """ Convert image to tensor 
    
    Returns: 
        numpy.ndarray: Converted image
    """
    return x.transpose(2, 0, 1).astype('float32') # convert to tensor

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

def visualize(**images):
    """
    Plot images in one row
    images: List of images in the form (width, height, channels)
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    # for each item in images, it has an index, name, and image
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values: # for each "label" (here, it would be black/white or background/neuron)
        equality = np.equal(label, colour) # check if the given label is equal to this particular value in the labels list
        class_map = np.all(equality, axis = -1) # check if all values in the array are True (if the label is equal to this particular value in the labels list)
        semantic_map.append(class_map) # append the result to the semantic map
    semantic_map = np.stack(semantic_map, axis=-1) # join sequence of arrays along a new axis

    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

class CaImagesDataset(torch.utils.data.Dataset):

    """Calcium imaging images dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks # they have 3 values (BGR) --> read as RGB
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

def find_centroids(segmented_img):
  centroids = []
  cont, hierarchy = cv2.findContours(segmented_img, 
                          cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
  for c in cont:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroids.append((cX, cY))
  
  return centroids

# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[1500,1500,3]):
   
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]
    
# Background is black, calcium is white
class_rgb_values = [(0,0,0), # black
 (255, 255, 255) # white
 ]
class_names=['background', 'calcium']

x_train_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/train"
y_train_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/train"

x_valid_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/valid"
y_valid_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/valid"

x_test_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/test"
y_test_dir="/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/test"


# Get train and val dataset instances
train_dataset = CaImagesDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=class_rgb_values,
)
valid_dataset = CaImagesDataset(
    x_valid_dir, y_valid_dir, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn=None),
    class_rgb_values=class_rgb_values,
)

# Get train and val data loaders (load large amounts of data more efficiently)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


LOAD = False # True if loading a model, False if creating a new model
TRAINING = True # True if training, False if testing
load_num = 1
save_num = 1

model_folder = "/Users/huayinluo/Desktop/code/catracking-1/models/unet"

if LOAD:
    model = joblib.load(f'{model_folder}/model{load_num}.pkl')
else:
    model = UNet()
    
if TRAINING:
    EPOCHS = 12
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00008),
    ])

    # define data loaders for training and validation sets
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    best_iou_score = 0.0 # current best iou score
    train_logs_list, valid_logs_list = [], [] # train and valid logs

    for i in range(0, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        
    joblib.dump(model, f'{model_folder}/model{save_num}.pkl')
    
## Using Model to Perform Segmentation on Test Data
# create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
test_dataset = CaImagesDataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn=None),   
    class_rgb_values=class_rgb_values,
)
test_dataloader = DataLoader(test_dataset)

# test dataset for visualization (without preprocessing transformations)
test_dataset_vis = CaImagesDataset(
    x_test_dir, y_test_dir, 
    augmentation=get_validation_augmentation(),
    class_rgb_values=class_rgb_values,
)

# get a random test image/mask index
random_idx = random.randint(0, len(test_dataset_vis)-1)
image, mask = test_dataset_vis[random_idx]
visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)

sample_preds_folder = '/Users/huayinluo/Desktop/code/catracking-1/results/unet'


# for idx in range(len(test_dataset)):
random_idx = random.randint(0, len(test_dataset)-1)
image, gt_mask = test_dataset[random_idx] # image and ground truth from test dataset
image_vis = crop_image(np.transpose(test_dataset[random_idx][0].astype('uint8'), (1, 2, 0)))
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

# Predict test image
pred_mask = model(x_tensor)
pred_mask = pred_mask.detach().squeeze().cpu().numpy()
# Convert pred_mask from `CHW` format to `HWC` format
pred_mask = np.transpose(pred_mask,(1,2,0))
# Get prediction channel corresponding to calcium
pred_calcium_heatmap = pred_mask[:,:,class_names.index('calcium')]
pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), class_rgb_values))
# Convert gt_mask from `CHW` format to `HWC` format
gt_mask = np.transpose(gt_mask,(1,2,0))
gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), class_rgb_values))

cv2.imwrite(
    os.path.join(sample_preds_folder, f"sample_pred_{i}.png"), 
    np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1]
    )  
visualize(
    original_image = image_vis,
    ground_truth_mask = gt_mask,
    predicted_mask = pred_mask,
    predicted_building_heatmap = pred_calcium_heatmap
)

### Evaluating Test Metrics

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

    
    