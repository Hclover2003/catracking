# import packages
import os
import cv2
import numpy as np;
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
import itertools
import sys
from matplotlib.animation import FuncAnimation
import scipy

class CaPositionsDataset(Dataset):
    """Calcium tracking positions dataset."""
    # load the dataset
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # number of samples in the dataset
    def __len__(self):
        return len(self.x)
    
    # get a sample from the dataset
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]
    
class NeuralNetworkClassifier(nn.Module):
    """Neural network with LSTM layer and fully connected layer"""
    def __init__(self):
        super(NeuralNetworkClassifier,self).__init__()
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=2,
                            bidirectional=False,
                            num_layers=1,
                            batch_first=True
                            )
        self.fc1 = nn.Linear(2,1)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = self.fc1(output)
        return output.squeeze(1)

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
  Slices sequence into smaller sequences of length frame_rate
  
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

def save_centroids(videos):
  max_width, max_height = 450, 550
  for video in videos:
    # Load AVA and AVB positions
    video_centroids = []
    positions_ava = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
    positions_avb = np.load(os.path.join(position_dir, f"AVB_{video}.mat.npy"))
    num_frames = len(positions_ava)
    for frame in range(num_frames):
      frame_mask = cv2.cvtColor(cv2.imread(os.path.join(img_dir, "ground_truth", video, f"{frame}.png")), cv2.COLOR_BGR2GRAY)
      centroids, contours = find_centroids(frame_mask)
      video_centroids.append(centroids)
      print(f"Video {video} | Frame {frame} | Centroids: {centroids}")
    save_dir = os.path.join(data_dir, "centroids", video)
    padded_video_centroids = zip(*itertools.zip_longest(*video_centroids, fillvalue=(0,0)))
    padded_video_centroids = np.array(list(padded_video_centroids))
    padded_video_centroids = padded_video_centroids[:, :5, :]
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "centroids.npy"), padded_video_centroids)
  # Set seed (for reproducibility)
  np.random.seed(0)
  random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  os.environ["PYTHONHASHSEED"] = str(0)
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Seed set")


# Set seed for reproducibility
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Seed set")

# Login to wandb
WANDB_API_KEY = "9623d3970461071fa95cf35f8c34d09b2f3fa223"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Preprocessing
data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data"
position_dir = rf"{data_dir}\positions"
model_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\lstm"
img_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images"
results_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results"

# data_dir = "/Users/huayinluo/Desktop/code/catracking-1/data"
# position_dir = os.path.join(data_dir, "positions")
# model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm_classify"
# img_dir = "/Users/huayinluo/Desktop/code/catracking-1/images"
# results_dir = "/Users/huayinluo/Desktop/code/catracking-1/results"

# Save all video positions in dictionary
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']
# videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415', "11310", "11311", "11315", "11317_1", "11317_2", "11318", "11320_1", "11323", "11324", "11325_1", "11325_2", "11327_1", "11327_2", "11328_1", "11332","11350-a_crop", "11352-a_crop", "11363-a_crop", "11364_a_crop", "11365-a", "11403-a_crop", "11405-a_crop", "11407_crop_1", "11549_crop", "11551_crop", "11552_crop", "11553_crop", "11554_crop_1", "11554_crop_2", "11555_crop", "11556_crop", "11558_crop", "11560_crop", "11565_crop", "11569_crop_1", "11569_crop_2", "11570_crop", "11595_crop", "11596_crop", "11597_crop", "11598_crop"]

 
# Train/Test split
sequence_length = 50
all_sequences = []
all_labels = []

norm_width, norm_height = 450, 500

# Get training set
for video in videos:
  positions_ava = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
  positions_avb = np.load(os.path.join(position_dir, f"AVB_{video}.mat.npy"))
  norm_positions_ava = np.multiply(positions_ava, [1/norm_width, 1/norm_height])
  norm_positions_avb = np.multiply(positions_avb, [1/norm_width, 1/norm_height])
  all_centroids = np.load(os.path.join(data_dir, "centroids", video, "centroids.npy"))
  split = math.floor(len(norm_positions_ava)*0.8)
  
  frame = sequence_length
  while frame < len(norm_positions_ava):
    centroids = np.unique(all_centroids[frame], axis=0)
    closest_centroid_ava = get_closest_cent(centroids, positions_ava[frame])
    closest_centroid_avb = get_closest_cent(centroids, positions_avb[frame])
    
    short_ava = norm_positions_ava[frame-sequence_length:frame]
    short_avb = norm_positions_avb[frame-sequence_length:frame]

    for centroid in centroids:
      centroid_norm = np.multiply(centroid, [1/norm_width, 1/norm_height])
      if ((centroid == closest_centroid_ava).all()):
        all_labels.append(1)
        all_sequences.append([*short_ava, centroid_norm])
      elif ((centroid== closest_centroid_avb).all()):
        all_labels.append(1)
        all_sequences.append([*short_avb, centroid_norm])
      else:
        all_labels.append(0)
        all_sequences.append([*short_ava, centroid_norm])
        all_labels.append(0)
        all_sequences.append([*short_avb, centroid_norm])
    frame += sequence_length

split = (len(all_sequences)//10)*8
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)
train_sequences = all_sequences[:split]
train_labels = all_labels[:split]
test_sequences = all_sequences[split:]
test_labels = all_labels[split:]
print("Test/Train split complete")

train_dataset = CaPositionsDataset(train_sequences, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = CaPositionsDataset(test_sequences, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

TRAINING = True # Set to True if training model
TESTING = False # Set to True if testing model

if TRAINING:
  # Initialise model and parameters
  # model = joblib.load(os.path.join(model_dir, "lstm_classifier_5.pkl"))
  # model_name = "lstm_classifier_5.pkl"
  model = NeuralNetworkClassifier()
  model_name = "lstm_classifier_july7-morning"

  epochs = 1000000
  learning_rate = 0.0000001
  batch_size = 16
  criterion = nn.BCEWithLogitsLoss()
  alpha = 0.25
  gamma = 2
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  print(f"Training model {model_name}")
  print(f"Learning rate: {learning_rate} | Epochs: {epochs} | Batch size: {batch_size}")

  # Initialise wandb
  wandb.init(
      project="lstm-classify",
      config={
      "existing_model": "none",
      "model_name": model_name,
      "hidden_nodes": 2,
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size,
      "number of training sequences": len(train_sequences)
      }
  )
  
  # Train model
  start_time = time.time()
  best_loss = 10000
  for epoch in range(epochs):
    # Go through batches
    for i, (input_sequences, input_labels) in enumerate(train_loader):
      input_sequences = torch.tensor(input_sequences, dtype=torch.float32)
      input_labels = torch.tensor(input_labels, dtype=torch.float32)
      pred_labels = model(input_sequences)[:, -1, :]
      pred_labels = pred_labels.squeeze(1) # Remove dimension of size 1 [Batch size, Sequence length, 1] -> [Batch size, Seqeuence length]
      loss = criterion(pred_labels, input_labels)
      # p_t = torch.exp(-bce_loss)
      # focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
      # loss = focal_loss.mean()
      loss.backward()
      optimizer.step()
    # Log losses to wandb
    if epoch % 10 == 0:
      wandb.log({"loss": loss})
      test_input_sequences = torch.tensor(test_sequences[:batch_size], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
      test_input_labels = torch.tensor(test_labels[:batch_size], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
      test_pred = model(test_input_sequences)[:, -1, :]
      test_pred = test_pred.squeeze(1)
      test_loss = criterion(test_pred, test_input_labels)
      if (test_loss < best_loss):
        best_loss = test_loss
        joblib.dump(model, os.path.join(model_dir, model_name))
        print("Saved Model")
      # p_t = torch.exp(-bce_test_loss)
      # focal_loss = alpha* (1 - p_t) ** gamma * bce_test_loss
      # test_loss = focal_loss.mean()
      wandb.log({"valid_loss": test_loss})
      print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {test_loss}")
  print(time.time() - start_time)
  wandb.finish()
  
if TESTING:
  model_name = "lstm_classifier_2"
  model = joblib.load(os.path.join(model_dir, model_name))
  
  criterion = nn.BCEWithLogitsLoss()
  for i, (input_sequence, input_label) in enumerate(test_loader):
    test_pred = model(input_sequence)
    loss = criterion(test_pred.squeeze(2), input_label)

  