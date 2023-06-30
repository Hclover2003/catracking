# import packages
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
        self.linear = nn.Linear(hidden_nodes, 2)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = self.linear(output)
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
      model = joblib.load(os.path.join(model_dir, "lstm_model5e6.pkl"))
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
project_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm"
data_dir = os.path.join(project_dir, "data")
video_dir = os.path.join(data_dir, "imgs")
position_dir = os.path.join(data_dir, "positions")
model_dir = os.path.join(project_dir, "models", "lstm")
img_dir = os.path.join(project_dir, "images")
results_dir = os.path.join(project_dir, "results")
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
    print(f"Loading {video}...")
print(f"Finished loading images and positions: {len(positions_dct)} positions")

# # Visualize position difference between AVA and AVB for video
# for video in videos:
#     try:
#         ava, avb = positions_dct[video]
#     except:
#        print(f"Error saving {video}")
#        continue
#     colours = np.arange(ava.shape[0])
#     plt.scatter(ava[:, 0], ava[:, 1], c=colours, cmap="Greens", label="AVA")
#     plt.scatter(avb[:, 0], avb[:, 1], c=colours, cmap="Oranges", label="AVB")
#     plt.legend()
#     plt.savefig(os.path.join(project_dir, "cases", f"{video}.png"))
#     print(f"Saved {video}")
#     plt.close()

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
    print(f"Error loading {video}")
    continue
  norm_positions = np.multiply(all_positions, [1/max_video_width, 1/max_video_height]) # Norm positions shape: [Num neurons = 2, Sequence length = 2571, Num coordinates = 2]
  split = math.floor(norm_positions.shape[1]*0.8)
  ava, avb = norm_positions
  train_sequences.append(ava[:split])
  train_sequences.append(avb[:split])
  test_sequences.append(ava[split:])
  test_sequences.append(avb[split:])

print("Test/Train split complete")

print("Starting sweep...")
# Sweep config
parameters_dct = {
"sequence_length": {"values": [10, 25, 50, 100, 200, 250]},
"frame_rate":{"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]},
"num_predict_coords": {"values": [1, 2, 3, 4, 5, 10]},
"batch_size": {"values": [8, 16, 32, 64]},
"learning_rate": {"values": [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]},
"epochs": {"values": [50, 100, 200, 500, 1000, 2000]},
"hidden_nodes": {"values": [2, 3, 4, 5, 6, 7, 8]},
"num_layers": {"values": [1, 2, 3]},
}

# Set constant parameters
parameters_dct.update({
"sequence_length": {"value": 250},
"frame_rate":{"value": 1},
"num_predict_coords": {"value": 1},
"batch_size": {"value": 16},
"learning_rate": {"value": 0.001},
"epochs": {"value": 10000},
"hidden_nodes": {"value": 3},
"num_layers": {"value": 1},
})

sweep_config = {
  "method": "grid",
  "name": "hidden_node_sweep_existing",
  "metric": {
    "goal": "minimize",
    "name": "loss"
  },
  "parameters": parameters_dct,
}

sweep_id = wandb.sweep(sweep_config, project="lstm-predict")

# Run sweep to find hyperparameters
wandb.agent(sweep_id, function=lambda: train(train_sequences=train_sequences, 
                                              valid_sequences=test_sequences,
                                              model_dir=model_dir,
                                              model_name="lstm_predict_june28-2",
                                              create_new_model=True,
                                              ))