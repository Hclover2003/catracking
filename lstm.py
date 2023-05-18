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
import math

from scipy import ndimage
from typing import Tuple, List
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
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

class neural_network(nn.Module):
    """Neural network with LSTM layer and fully connected layer"""
    def __init__(self):
        super(neural_network,self).__init__()
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=5,
                            num_layers=1,
                            batch_first=True
                            )
        self.fc1 = nn.Linear(in_features=5,
                             out_features=1)

    def forward(self,x):
        output,_status = self.lstm(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output

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

def scale_data_scaler(data, opt):
  """Scales data with MinMaxScaler or StandardScaler"""
  if (opt == "minmax"):
    print("minmax")
    scaler = MinMaxScaler(feature_range=(0,1))
    norm_data = [ lst[0] for lst in scaler.fit_transform(np.array(data).reshape((-1, 1)))]
  elif (opt == "stand"):
    print("stand")
    scaler = StandardScaler()
    norm_data = [ lst[0] for lst in scaler.fit_transform(np.array(data).reshape((-1, 1)))]
  return scaler, norm_data

def scale_data(data, full:int):
  """Scales data with image dimensions"""
  norm_data = [i/full for i in data]
  return norm_data

def get_features_and_outcome(num_prev, neuron_positions, isX:bool):
  """Returns dataframe with features and outcome variables"""
  i = 0
  norm_features = []
  
  # scale data (x or y position)
  if isX:
    norm_neuron_positions = scale_data([x for (x, y) in neuron_positions], full=width)
  else:
    norm_neuron_positions = scale_data([y for (x, y) in neuron_positions], full=width)
    
  # since we need 10 previous frames as features, make sure we stop in time
  while i <= len(neuron_positions) - num_prev -1:
    # each loop = feature for one "sample" (num_prev previous points)
    norm_features.append(norm_neuron_positions[i:i+num_prev])
    i+=1

  # make dataframe with features and outcome variables
  dict = {'prev_n_position': norm_features, 'curr_position': norm_neuron_positions[num_prev:]} 
  df = pd.DataFrame(dict)
  return df


# Set seed (for reproducibility)
num = 0
torch.manual_seed(num)
random.seed(num)
np.random.seed(num)

# constants
videos = ['11409', "11410", '11411', '11413', '11414', '11415']
imgs_dct = {}
positions_dct={}

# Get max height and width between all videos (for scaling)
height, width = 0, 0 
for video in videos:
    # Save imgs and positions in dictionary
    imgs_dct[video] = np.load(f"./data/imgs/{video}_crop.nd2.npy")
    positions_dct[video] = np.load(f"./data/positions/AVA_{video}.mat.npy")
    
    h, w = imgs_dct[video].shape[2:]
    if h > height:
        height = h
    if w > width:
        width = w

# Concatenate all videos into one dataframe
df_lst = []
for ava in positions_dct.values():
    df_lst.append(get_features_and_outcome(10, ava)) # Get features and outcome variables for each video
df = pd.concat(df_lst)

# Separate features X and outcome Y variables
X = np.array(df.prev_n_position.tolist())
Y = np.array(df.curr_position.tolist())

# Train-test split
split = math.floor(len(df)*0.8)
x_train = X[:split]
x_test = X[split:]
y_train = Y[:split]
y_test = Y[split:]

# Create dataset and dataloader
train_set = CaImagesDataset(x_train,y_train)
train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=256 # Each batch has 256 samples. (e.g. if dataset has 2048 samples total, there are 8 training batches)
                          )
torch.manual_seed(0)

# Create model
model = neural_network()

# Optimizer and loss function 
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

# Training loop
epochs = 500
existing_epochs=0
train_times = {} #key= epoch, value=loss for that epoch
for i in range(epochs+1):
    for j,data in enumerate(train_loader):
        y_pred = model(data[:][0].view(-1,10,1)).reshape(-1)
        loss = criterion(y_pred,data[:][1])
        loss.backward()
        optimizer.step()
    existing_epochs+=1
    train_times[existing_epochs] = loss
    if i%50 == 0:
        print(existing_epochs,"th iteration : ",loss)

losses = [tsr.detach().numpy().flat[0] for tsr in train_times.values()]
plt.plot(train_times.keys(), losses)
plt.title("Train Loss Curve")
plt.imshow()