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
import torch.nn.functional as F
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
import torchvision.transforms as transforms

import segmentation_models_pytorch.utils.metrics

class CaImagesDataset(torch.utils.data.Dataset):

    """Calcium imaging images dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512)
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
    
    def __getitem__(self, i):
        
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
        
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        img_size = 512
        width, height = self.image_dim
        max_dim = max(img_size, width, height)
        pad_left = (max_dim-width)//2
        pad_right = max_dim-width-pad_left
        pad_top = (max_dim-height)//2
        pad_bottom = max_dim-height-pad_top
        _transform.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
                                        padding_mode='edge'))
        _transform.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))  
        mask = transforms.Compose(_transform)(mask)

        ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()

        _transform_img = _transform.copy()
        _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)
            
        return image, ont_hot_mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
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
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
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
        self.down_conv1 = DownBlock(1, 64) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1) 

    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
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


def get_preprocessing(preprocessing_fn=None, image_dim=(512, 512)):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    _transform.append(transforms.ToTensor())

    img_size = 512
    width, height = image_dim
    max_dim = max(img_size, width, height)
    pad_left = (max_dim-width)//2
    pad_right = max_dim-width-pad_left
    pad_top = (max_dim-height)//2
    pad_bottom = max_dim-height-pad_top
    _transform.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
                                    padding_mode='edge'))
    _transform.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))       
    if preprocessing_fn:
        _transform.append(preprocessing_fn)

    return transforms.Compose(_transform)

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
def crop_image(image, target_image_dims):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def get_subset(dataset, subset_size):
    """Get a subset of the dataset
    
    Args:
        dataset (CaImagesDataset): dataset to get subset from
        subset_size (int): size of the subset
        
    Returns:
        subset (CaImagesDataset): subset of the dataset
    """
    # get a random subset of the dataset
    subset = torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), subset_size))
    return subset