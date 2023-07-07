import os
import cv2
import numpy as np;
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import albumentations as album
import math

from sklearn import model_selection
from scipy import ndimage
from typing import Tuple, List
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import joblib
import time
import wandb
import random
import scipy.io
import re

class CaImagesDataset(Dataset):
    """CA Images dataset."""
    # load the dataset
    def __init__(self, x, y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)

    # number of samples in the dataset
    def __len__(self):
        return len(self.x)
    
    # get a sample from the dataset
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]
    
class NeuralNetwork(nn.Module):
    """Neural network with LSTM layer and fully connected layer"""
    def __init__(self, hidden_nodes = 2, num_layers = 1):
        super(NeuralNetwork,self).__init__()
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=hidden_nodes,
                            bidirectional=False,
                            num_layers=num_layers,
                            batch_first=True
                            )

    def forward(self,x):
        output,_status = self.lstm(x)
        return output

def split_lst(lst: List[Tuple], n: int) -> List[List[Tuple]]:
  """
  Split a list into sequences of length n
  
  Parameters
  ----------
  lst: original sequence of x,y coordinates
  n: length of each shortened sequence
  
  Returns
  -------
  List of sequences of length n
  """
  length = len(lst)
  shortened_sequences = []
  for i in range((length+n-1)//n):
    short_seq = lst[i*n: (i+1)*n]
    if len(short_seq) == n:
      shortened_sequences.append(short_seq)
  return shortened_sequences

def slice_sequence(sequence: np.array, frame_rate: int) -> np.array:
  """
  Slices sequence into smaller sequences, skipping frame_rate number of frames
  
  Parameters
  ----------
      sequence (np.array): Sequence of x,y coordinates
      frame_rate (int): Number of frames per second
  
  Returns
  -------
      List of sequences of length frame_rate
  """
  indices = [i for i in range(0, len(sequence), frame_rate)]
  return sequence[indices]

def find_centroids(segmented_img: np.ndarray) -> Tuple[List, List]:
  """
  Finds centroids and contours of segmented image
  
  Parameters
  ----------
  segmented_img: segmented image (binary)
  
  Returns
  -------
  centroids: list of centroids in image
  contours: list of contours in image
  """
  # initialize centroids list
  centroids = []
  # get contours
  contours, hierarchy = cv2.findContours(segmented_img, 
                          cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
  
  # compute the centroid of each contour
  for c in contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroids.append((cX, cY))
  
  return centroids, contours

def crop_img(img: np.array, coordinate: Tuple, crop_width: int = 20, crop_height: int = 20) -> np.array:
  """
  Crop image around given coordinate.
  
  Parameters
  ----------
  img: image to crop
  position: point to crop around
  crop_width: width of crop
  crop_height: height of crop
  
  Returns
  -------
  cropped image of shape (crop_width, crop_height)
  """
  x, y = coordinate
  return img[max(0, y-(crop_height//2)): min(img.shape[0], y+(crop_height//2)),
              max(0, x-(crop_width//2)): min(img.shape[1], x+(crop_width//2))]

def get_dist_score(point1:Tuple, point2:Tuple) -> float:
  """Returns distance score between two points
  
  Parameters
  ----------
  point1: coordinates of point 1
  point2: coordinates of point 2
  
  Returns
  -------
  Euclidean distance between point1 and point2
  """
  x, y = point1
  x2, y2 = point2
  
  return math.sqrt(((x2-x)**2)+((y2-y)**2))

def get_color_score(img1: np.array, img2: np.array, compare_type: int = cv2.HISTCMP_CORREL) -> float:
    """Returns color score between two images
    
    Parameters
    ----------
    img1: First image
    img2: Second image
    compare_type: Type of opencv histogram comparison to use
    
    Returns
    -------
    Histogram similarity between the two images
    
    Related
    -------
    See https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    """
    hst1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hst2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    score = cv2.compareHist(hst1, hst2, cv2.HISTCMP_CORREL)
    return score

def get_shape_score(cont1 , cont2) -> float:
    """ Get difference in area and perimeter between two contours

    Parameters
    ----------
        cont1 (contour): First contour
        cont2 (contour): Second contour

    Returns
    -------
        Tuple : the absolute value of difference in area and perimeter
        
    Related
    -------
    See https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html 
    """
    area1 = cv2.contourArea(cont1)
    area2 = cv2.contourArea(cont2)
    perimeter1 = cv2.arcLength(cont1, True)
    perimeter2 = cv2.arcLength(cont2, True)
    return abs(area1 - area2), abs(perimeter1-perimeter2)

def get_closest_cent(centroids:List, point:Tuple):
  """ Returns the closest centroid to the predicted coordinates
  centroids: list of centroids
  pred: predicted coordinates"""
  min_score = 10**1000
  closest_centroid = (0,0) # Closest to predicted coords

  for centroid in centroids:
    score = get_dist_score(point, centroid)
    if score <= min_score:
      min_score = score
      closest_centroid = centroid
  return closest_centroid

def get_most_similar_centroid(img, prev_img, pred_pt:Tuple, prev_pt:Tuple, centroids: List[Tuple], contours: List, dist_weight: int = 1, color_weight: int = 10, area_weight:int = 0, log=False):
    """ 
    Returns the closest centroid to the predicted coordinates
    
    Parameters
    ----------
    img: image for this frame
    prev_img: image for previous frame
    pred_pt: predicted coordinate
    prev_pt: previous coordinate
    centroids: list of centroids
    contours: list of contours
    dist_weight: weighting of distance score (Default is 1)
    color_weight: weighting of color score (Default is 10)
    area_weight: weighting of area score (Default is 0)
    
    Returns
    -------
    Tuple: closest centroid (closest in terms of weighted distance, color, and area)
    """
    cropped_prev_img = crop_img(prev_img, prev_pt)
    prev_centroids, prev_contours = find_centroids(cropped_prev_img)
    prev_contour = prev_contours[0]
    
    min_score = 10**1000
    closest_centroid = (0,0)

    # Loop through potential centroids
    for i in range(len(centroids)):
        # Get centroid, contour, and cropped centroid image for this centroid
        centroid = centroids[i]
        contour = contours[i]
        cropped_centroid_img = crop_img(img, centroid)
        
        # Get scores for this centroid
        dist_score = get_dist_score(pred_pt, centroid) # Distance between predicted point and centroid
        color_score = get_color_score(cropped_centroid_img, cropped_prev_img) # Color similarity between cropped centroid and cropped previous point
        area_score, perimeter_score = get_shape_score(contour, prev_contour) # Area and perimeter difference between this contour and previous contour
        
        # Weighted score
        score = (dist_weight*dist_score) + (color_weight*color_score) + (area_weight*area_score)
        
        # Print scores for each centroid
        if log:
            print(f"Centroid: {centroid} | Score: {score:.2f} | Dist: {dist_score:.2f}, Color: {color_score:.3f}, Area: {area_score:.2f}")
        
        # Check if this centroid is closer than the previous closest
        if score <= min_score:
            min_score = score
            closest_centroid = centroid
            
    return closest_centroid

def train_epoch(model: NeuralNetwork, 
                train_sequences: List[List[Tuple]], 
                valid_sequences: List[List[Tuple]] , 
                batch_size: int, num_predict_coords: int, sequence_length: int, frame_rate: int,
                criterion, optimizer):
  """ 
  Trains model for one epoch. 

  Parameters
  ----------
      model (NeuralNetwork): A neural network model with one LSTM layer
      train_sequences (List[List[Tuple]]): List of short sequences of x,y coordinates for training
      valid_sequences (List[List[Tuple]]): List of short sequences of x,y coordinates for validation
      batch_size (int): Batch size (number of sequences to train on at once)
      num_predict_coords (int): Number of coordinates to predict
      sequence_length (int): Length of each sequence
      frame_rate (int): Frame rate of video (sequence interval)
      criterion: Loss function
      optimizer: Optimizer
      
  Returns
  -------
      final train loss, final valid loss, and average train loss of model for epoch
      
  """
  # Preprocess train sequences
  # Slice sequences to get desired frame rate
  train_sequences = [slice_sequence(seq, frame_rate) for seq in train_sequences]
  # Split sequences to get desired sequence length
  shortened_train_sequences = []
  for sequence in train_sequences:
    shortened_train_sequences.extend(split_lst(sequence, sequence_length))
  shortened_train_sequences = np.stack(shortened_train_sequences)
  # Shuffle order of sequences
  shortened_train_sequences = np.random.permutation(shortened_train_sequences) 
  
  # Preprocess valid sequences
  shortened_valid_sequences = []
  for sequence in valid_sequences:
    shortened_valid_sequences.extend(split_lst(sequence, 100))
  shortened_valid_sequences = np.stack(shortened_valid_sequences)
  
  # Number of batches
  num_batches = len(shortened_train_sequences)//batch_size
  # Total loss for epoch
  total_loss = 0
  
  # Train model
  for i in range(num_batches):
    # Get batch of sequences and convert to tensors
    input_sequences = torch.tensor(shortened_train_sequences[i*batch_size:(i+1)*batch_size][:-num_predict_coords], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    # Get actual sequences and convert to tensors
    actual_sequences = torch.tensor(shortened_train_sequences[i*batch_size:(i+1)*batch_size][num_predict_coords:], dtype=torch.float32)
    # Predict next coordinates
    pred_sequences = model(input_sequences)

    # Calculate loss
    loss = criterion(pred_sequences, actual_sequences)
    total_loss += loss
    loss.backward()
    optimizer.step()
    
    # Calculate validation loss
    valid_input = torch.tensor(shortened_valid_sequences[:][:-1], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    valid_actual = torch.tensor(shortened_valid_sequences[:][1:], dtype=torch.float32)
    valid_pred = model(valid_input)
    valid_loss = criterion(valid_pred, valid_actual)
    
  # Log losses to wandb
  wandb.log({"loss": loss, "valid_loss": valid_loss})
  
  return loss, valid_loss, (total_loss/num_batches)

def train(train_sequences: List[List[Tuple]], valid_sequences: List[List[Tuple]], model_dir:str, model_name:str, create_new_model: bool, config: dict=None):
  """
  Trains model for multiple epochs and logs to wandb
  
  Parameters
  ----------
      train_sequences (List[List[Tuple]]): List of short sequences of x,y coordinates for training
      valid_sequences (List[List[Tuple]]): List of short sequences of x,y coordinates for validation
      config: Hyperparameters (set by wandb sweep)
  
  """
  matplotlib.use('Agg')
  with wandb.init(config=config):
    
    start = time.time()
    config = wandb.config
    
    # Initialize model, loss function, and optimizer
    if create_new_model:
      model = NeuralNetwork(hidden_nodes=config.hidden_nodes, num_layers=config.num_layers)
      print("New Model Created")
    else:
      model = joblib.load(os.path.join(model_dir, "lstm_model5e2.pkl"))
      print("Model Loaded")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_num = 0
    table = wandb.Table(columns=["Epoch", "Image"])
    # Create directory for model
    if not os.path.exists(os.path.join(model_dir, model_name)):
      os.mkdir(os.path.join(model_dir, model_name))
    # Train model for multiple epochs
    for epoch in range(config.epochs):
      train_loss, valid_loss, avg_train_loss = train_epoch(model=model, 
                                train_sequences=train_sequences,
                                valid_sequences=valid_sequences, 
                                batch_size=config.batch_size, 
                                num_predict_coords=config.num_predict_coords, 
                                sequence_length=config.sequence_length, 
                                frame_rate=config.frame_rate,
                                criterion=criterion, optimizer=optimizer)
      wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})    
        
      if epoch % 100 == 0:
        joblib.dump(model, os.path.join(model_dir, model_name, str(checkpoint_num)))
        checkpoint_num += 1
        valid_input = torch.tensor(valid_sequences[0][:250], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
        valid_actual = torch.tensor(valid_sequences[0][1:251], dtype=torch.float32)
        valid_pred = model(valid_input)
        colours = np.arange(valid_pred.detach().shape[0])
        predx = valid_pred.detach()[:,0]
        predy = valid_pred.detach()[:,1]
        actx = valid_actual.detach()[:,0]
        acty = valid_actual.detach()[:,1]
        fig = plt.figure()
        plt.scatter(valid_input.detach()[0,0], valid_input.detach()[0,1], label="Input")
        plt.scatter(predx, predy, c = colours, cmap="Greens", label="Predicted") #1st sequence in batch
        plt.colorbar()
        plt.scatter(actx, acty, c = colours, cmap = "Oranges", label="Actual") # 1st sequence in batch
        plt.colorbar()
        plt.legend()
        table.add_data(epoch, wandb.Image(fig))
        plt.close()
    wandb.log({"predictions": table})
    print(f"Time: {time.time()-start}")
    wandb.finish()

# Set seed for reproducibility
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print("Seed set")

# Login to wandb
WANDB_API_KEY = "9623d3970461071fa95cf35f8c34d09b2f3fa223"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Change to your own directory
project_dir = "/Users/huayinluo/Desktop/code/ca-track/ca-track"
data_dir = os.path.join(project_dir, "data")
video_dir = os.path.join(data_dir, "imgs")
position_dir = os.path.join(data_dir, "positions")
model_dir = os.path.join(project_dir, "models", "lstm-predict")
img_dir = os.path.join(project_dir, "images")
results_dir = os.path.join(project_dir, "results")
save_dir = os.path.join(results_dir, f"july4-1")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# project_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm"
# data_dir = "/Users/huayinluo/Desktop/code/catracking-1/data"
# video_dir = os.path.join(data_dir, "imgs")
# position_dir = os.path.join(data_dir, "positions")
# model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm"
# img_dir = "/Users/huayinluo/Desktop/code/catracking-1/images"
# results_dir = "/Users/huayinluo/Desktop/code/catracking-1/results"

# Loop through videos and get positions
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415', "11310", "11311", "11313", "11315", "11316", "11317_1", "11317_2", "11318", "11320_1", "11320_2", "11323", "11324", "11325_1", "11325_2", "11327_1", "11327_2", "11328_1", "11332", "11334", "11350-a_crop", "11352-a_crop", "11363-a_crop", "11364_a_crop", "11365-a", "11403-a_crop", "11405-a_crop", "11407_crop_1", "11549_crop", "11550_crop", "11551_crop", "11552_crop", "11553_crop", "11554_crop_1", "11554_crop_2", "11555_crop", "11556_crop", "11558_crop", "11560_crop", "11563_crop", "11565_crop", "11569_crop_1", "11569_crop_2", "11570_crop", "11595_crop", "11596_crop", "11597_crop", "11598_crop"]
positions_dct={} 
for video in videos:
  # Load AVA and AVB positions
    try:
        ava = os.path.join(position_dir, f"AVA_{video}.mat")
        avb = os.path.join(position_dir, f"AVB_{video}.mat")
        mat_ava = scipy.io.loadmat(ava)
        mat_avb = scipy.io.loadmat(avb)
        positions_ava = np.concatenate(mat_ava['dual_position_data'])
        positions_ava = np.array([list(coord.squeeze(0)) for coord in positions_ava])
        positions_avb = np.concatenate(mat_avb['dual_position_data'])
        positions_avb = np.array([list(coord.squeeze(0)) for coord in positions_avb])
    except:
        try:
            positions_ava = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
            positions_avb = np.load(os.path.join(position_dir, f"AVB_{video}.mat.npy"))
        except:
            videos.remove(video)
            print(f"Error loading {video}")
            continue
    all_neurons_positions = np.stack((positions_ava, positions_avb))
    positions_dct[video] = all_neurons_positions
    width, height = positions_ava[:, 0].max(), positions_ava[:, 1].max()
    print(f"Video {video} | {width}, {height} | {positions_ava.shape}")
print(f"Finished loading images and positions: {len(positions_dct)} positions")

# Original data test/train split (# Add 80% of each video to training set, 20% to testing set)
train_sequences = []
test_sequences = []
max_video_width = 550 
max_video_height = 570
for video in videos:
  # Normalize positions
  try:
    all_positions = positions_dct[video]
  except:
    videos.remove(video)
    print(f"Error getting {video}")
    continue
  norm_positions = np.multiply(all_positions, [1/max_video_width, 1/max_video_height]) # Norm positions shape: [Num neurons = 2, Sequence length = 2571, Num coordinates = 2]
  split = math.floor(norm_positions.shape[1]*0.8)
  ava, avb = norm_positions
  train_sequences.append(ava[:split])
  train_sequences.append(avb[:split])
  test_sequences.append(ava[split:])
  test_sequences.append(avb[split:])
print("Test/Train split complete")

# Load model
model_name = "724"
model = joblib.load(os.path.join(model_dir, model_name))

# track number of correct and incorrect predictions
num_correct = 0
num_incorrect = 0
RESET = False
num_reset = 0
max_reset = 100

# choose set of frames to predict on
video = '11409'
start_frame = 0
end_frame = 200 # frames 0-100 of video 11408
start_index = 10 # give first 10 frames as input to model

# actual positions and chosen positions
norm_width, norm_height = 550, 570
actual_positions = np.array(positions_dct[video][0][start_frame:end_frame+1])
actual_positions_norm = np.multiply(actual_positions, [1/norm_width, 1/norm_height])
predicted_positions_norm = actual_positions_norm.copy()[:10] # running log of chosen coordinates (use as feature x)
chosen_positions_norm = actual_positions_norm.copy()[:10] # running log of chosen coordinates (use as feature x)
# predicted_positions = actual_positions[start_frame:start_frame+start_index, :] # running log of predictions
# pred = model(torch.tensor(actual_positions_norm[:end_frame], dtype=torch.float32))
# act = actual_positions_norm[1:]
  
# loop through each frame
for frame in range(start_frame+start_index, end_frame+1):
    pred = model(torch.tensor(predicted_positions_norm[:frame], dtype=torch.float32))
    pred = pred.detach().numpy()[-1].reshape(1, 2)
    predicted_positions_norm = np.concatenate((predicted_positions_norm, pred))
    mask_image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, "ground_truth", "11408", f"{frame}.png")), cv2.COLOR_BGR2GRAY)
    centroids, contours = find_centroids(mask_image)
    closest_centroid = get_closest_cent(centroids, pred)
    chosen_positions_norm = np.concatenate((chosen_positions_norm, np.multiply(closest_centroid, [1/norm_width, 1/norm_height])))
    
    fig = plt.figure()
    colours = np.arange(predicted_positions_norm.shape[0])
    plt.scatter(predicted_positions_norm[:, 0], predicted_positions_norm[:, 1], c = colours, cmap="Greens", label="Predicted") #1st sequence in batch
    plt.colorbar()
    plt.scatter(actual_positions_norm[:len(colours), 0], actual_positions_norm[:len(colours), 1], c = colours, cmap = "Oranges", label="Actual") # 1st sequence in batch
    plt.colorbar()
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{frame}.png"))
    plt.close(fig)

print("Finished predicting")