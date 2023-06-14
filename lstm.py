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
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.lstm = nn.LSTM(input_size=2, 
                            hidden_size=2,
                            bidirectional=False,
                            num_layers=1,
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
  return [lst[i*n: (i+1)*n] for i in range((length+n-1)//n)]

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
  centroids = []
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

def get_dist_score(x, y, x2, y2):
  """Returns distance score between two points
  x, y: coordinates of point 1
  x2, y2: coordinates of point 2"""
  return math.sqrt(((x2-x)**2)+((y2-y)**2))

def get_closest_cent(centroids:List, pred:Tuple, log=False):
  """ Returns the closest centroid to the predicted coordinates
  centroids: list of centroids
  pred: predicted coordinates"""
  max_score = 10**1000
  predx, predy = pred
  coords = (0,0) # Closest to predicted coords

  for (pot_x, pot_y) in centroids:
    score = get_dist_score(predx, predy, pot_x, pot_y)
    if log:
      print(f"Centroid: {pot_x}, {pot_y} | Score: {round(score)}")
    if score <= max_score:
      max_score = score
      coords = (pot_x, pot_y)
  return coords

def crop_img(img, x, y, w, h):
  """Crop image to x, y, w, h"""
  return img[max(0, y-(h//2)): min(img.shape[0], y+(h//2)),
              max(0, x-(w//2)): min(img.shape[1], x+(w//2))]

def get_color_score(img1, img2):
  hst1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
  hst2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
  score = cv2.compareHist(hst1, hst2, cv2.HISTCMP_CORREL)
  return score

def get_shape_score(cont1, cont2):
  area1 = cv2.contourArea(cont1)
  area2 = cv2.contourArea(cont2)
  return abs(area1 - area2)

def train_epoch(model: NeuralNetwork, train_lst: List[List[Tuple]], valid_lst: List[List[Tuple]] , batch_size, criterion, optimizer):
  """ 
  Trains model for one epoch

  Parameters
  ----------
      model (NeuralNetwork): A neural network model with one LSTM layer
      train_lst (List[List[Tuple]]): List of short sequences of x,y coordinates for training
      valid_lst (List[List[Tuple]]): List of short sequences of x,y coordinates for validation
      criterion: Loss function
      optimizer: Optimizer
      
  Returns
  -------
      avg loss of model for epoch
  """
  total_loss = 0 # Total loss for epoch
  num_batches = len(train_lst)//batch_size # Number of batches for training
  num_batches_valid = len(valid_lst)//batch_size # Number of batches for validation
  
  # Train model
  for i in range(num_batches):
    train_input = torch.tensor(train_lst[i*batch_size:(i+1)*batch_size][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    pred = model(train_input)
    actual = torch.tensor(train_lst[i*batch_size:(i+1)*batch_size][:, 1:, :], dtype=torch.float32)
    loss = criterion(pred, actual)
    total_loss += loss
    loss.backward()
    optimizer.step()
    
  # Validate model
  for j in range(num_batches_valid):
    valid_input = torch.tensor(valid_lst[j*batch_size:(j+1)*batch_size][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    valid_pred = model(valid_input)
    valid_actual = torch.tensor(valid_lst[j*batch_size:(j+1)*batch_size][:, 1:, :], dtype=torch.float32)
    valid_loss = criterion(valid_pred, valid_actual)
  
  # Log loss to wandb
  wandb.log({"batch_loss": loss, "batch_valid_loss": valid_loss})
  
  return total_loss/num_batches

def train(train_lst: List[List[Tuple]], valid_lst: List[List[Tuple]], config: dict=None):
  """
  Trains model for multiple epochs and logs to wandb
  
  Parameters
  ----------
      train_lst (List[List[Tuple]]): List of short sequences of x,y coordinates for training
      valid_lst (List[List[Tuple]]): List of short sequences of x,y coordinates for validation
      config: Hyperparameters (set by wandb sweep)
  
  """
  with wandb.init(config=config):
    start = time.time()
    model = NeuralNetwork() # New model
    
    config = wandb.config
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        avg_loss = train_epoch(model=model, train_lst=train_lst, valid_lst=valid_lst, batch_size=config.batch_size, criterion=criterion, optimizer=optimizer)
        wandb.log({"avg_loss": avg_loss, "epoch": epoch})    

    joblib.dump(model, os.path.join(model_dir, "lstm_model_1.pkl"))
    print(f"Time: {time.time()-start}")
    wandb.finish()

# SET CONSTANTS
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
os.environ["PYTHONHASHSEED"] = str(0)
print("Seed set")

# data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data"
# model_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\lstm"
# video_dir = rf"{data_dir}\imgs"
# position_dir = rf"{data_dir}\positions"

data_dir = "/Users/huayinluo/Desktop/code/CaTracking/data"
model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm"
img_dir = "/Users/huayinluo/Desktop/code/CaTracking/data/imgs"
video_dir = os.path.join(data_dir, "imgs")
position_dir = os.path.join(data_dir, "positions")

# Save all video arrays and positions in dictionary
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']
imgs_dct = {}
positions_dct={}
for video in videos:
  imgs_dct[video] = np.load(os.path.join(video_dir, f"{video}_crop.nd2.npy"))
  positions_dct[video] = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
  print(f"Loading {video}...")
print(f"Finished loading images and positions: {len(imgs_dct)} videos, {len(positions_dct)} positions")

# Test/train split (# Add 80% of each video to training set, 20% to testing set)
df_train_lst = []
df_test_lst = []
for video in videos:
  positions = positions_dct[video]
  h, w = imgs_dct[video].shape[2:]
  norm_positions = [(x/w, y/h) for (x,y) in positions]
  split = math.floor(len(norm_positions)*0.8)
  df_train_lst.append(norm_positions[:split])
  df_test_lst.append(norm_positions[split:])
print("Test/Train split complete")


df_train_lst_shortened = np.concatenate([split_lst(lst, 100)[:-1] for lst in df_train_lst])
df_test_lst_shortened = np.concatenate([split_lst(lst, 100)[:-1] for lst in df_test_lst])

# Sweep config
parameters_dct = {
"seq_len": {"values": [10, 25, 50, 100, 200, 250]},
"batch_size": {"values": [8, 16, 32, 64]},
"learning_rate": {"max": 0.001, "min": 0.00001},
"epochs": {"values": [50, 100, 200, 500, 1000, 2000]}
}

parameters_dct.update({
"epochs": {"value": 1000},
"seq_len": {"value": 100},
"learning_rate": {"value": 0.001}
}) # Set constant parameters

sweep_config = {
  "method": "random",
  "name": "sweep",
  "metric": {
    "goal": "minimize",
    "name": "loss"
  },
  "parameters": parameters_dct,
}

sweep_id = wandb.sweep(sweep_config, project="lstm-positions")

wandb.agent(sweep_id, function=lambda: train(train_lst=df_train_lst_shortened, valid_lst=df_test_lst_shortened), count=3)