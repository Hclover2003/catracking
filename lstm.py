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
import joblib
import time

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

def unscale_data(data, full:int):
  """"""
  try:
      norm_data = [i*full for i in data]
  except:
      norm_data = data*full
  return np.array(norm_data)

def get_features_and_outcome(num_prev, neuron_positions):
  """Returns dataframe with features and outcome variables"""
  i = 0
  features_x = []
  features_y = []
  
  # scale data (x or y position)
  norm_neuron_positions_x = scale_data([x for (x, y) in neuron_positions], full=width)
  norm_neuron_positions_y = scale_data([y for (x, y) in neuron_positions], full=width)
    
  # since we need 10 previous frames as features, make sure we stop in time
  while i <= len(neuron_positions) - num_prev -1:
    # each loop = feature for one "sample" (num_prev previous points)
    features_x.append(norm_neuron_positions_x[i:i+num_prev])
    features_y.append(norm_neuron_positions_y[i:i+num_prev])
    i+=1

  # make dataframe with features and outcome variables
  dict = {'prev_n_x': features_x, 'curr_x': norm_neuron_positions_x[num_prev:], 
          'prev_n_y': features_y, 'curr_y': norm_neuron_positions_y[num_prev:], 'curr_frame': [i for i in range(num_prev, len(neuron_positions))]
          } 
  df = pd.DataFrame(dict)
  return df

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

def get_dist_score(x, y, x2, y2):
  return ((x2-x)**2)+((y2-y)**2)

def get_closest_cent(centroids:List, pred:Tuple):
    """ Returns the closest centroid to the predicted coordinates"""
    max_score = 10**1000
    predx, predy = pred
    coords = (0,0) # Closest to predicted coords

    for (pot_x, pot_y) in centroids:
        score = get_dist_score(predx, predy, pot_x, pot_y)
        print(f"Centroid: {pot_x}, {pot_y} | Score: {round(score)}")
        if score <= max_score:
            max_score = score
        coords = (pot_x, pot_y)
    return coords

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

if False:
  # Concatenate all videos into one dataframe
  df_lst = []
  for ava in positions_dct.values():
      df_lst.append(get_features_and_outcome(10, ava)) # Get features and outcome variables for each video
  df = pd.concat(df_lst)

  # Separate features X and outcome Y variables
  X = np.array(df.prev_n_x.tolist())
  Y = np.array(df.curr_x.tolist())


  # Train-test split
  split = math.floor(len(df)*0.8)
  x_train = X[:split]
  x_test = X[split:]
  y_train = Y[:split]
  y_test = Y[split:]


df_train_lst = []
df_test_lst=[]
for ava in positions_dct.values():
    video_df= get_features_and_outcome(10, ava)
    split = math.floor(len(video_df)*0.8)
    df_train_lst.append(video_df[:split]) # Get features and outcome variables for each video
    df_test_lst.append(video_df[split:]) # Get features and outcome variables for each video
df_train = pd.concat(df_train_lst)
df_test = pd.concat(df_test_lst)

x_train = np.array(df_train.prev_n_x.tolist())
x_test = np.array(df_test.prev_n_x.tolist())
y_train = np.array(df_train.curr_x.tolist())
y_test = np.array(df_test.curr_x.tolist())

x_train2 = np.array(df_train.prev_n_y.tolist())
x_test2 = np.array(df_test.prev_n_y.tolist())
y_train2 = np.array(df_train.curr_y.tolist())
y_test2 = np.array(df_test.curr_y.tolist())

# Create dataset and dataloader
train_set = CaImagesDataset(x_train,y_train)
train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=256 # Each batch has 256 samples. (e.g. if dataset has 2048 samples total, there are 8 training batches)
                          )

train_set2 = CaImagesDataset(x_train2,y_train2)
train_loader2 = DataLoader(train_set2,
                          shuffle=True,
                          batch_size=256 # Each batch has 256 samples. (e.g. if dataset has 2048 samples total, there are 8 training batches)
                          )
test_set = CaImagesDataset(x_test,y_test)
test_set2 = CaImagesDataset(x_test2,y_test2)


train = False # True if training x model, False if loading model
train2 = False # True if training y model, False if loading model

try_num = 1 # for saving model x
try_num2 = 0 # for saving model y

if train:
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
  joblib.dump(model, f'model{try_num}.pkl')

else:
  model = joblib.load(f'model{try_num}.pkl')


if train2:
  # Create model
  model2 = neural_network()

  # Optimizer and loss function 
  criterion2 = torch.nn.MSELoss()
  optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.0001)

  # Training loop
  epochs2 = 500
  existing_epochs2=0
  train_times2 = {} #key= epoch, value=loss for that epoch
  for i in range(epochs2+1):
      for j,data in enumerate(train_loader2):
          y_pred = model2(data[:][0].view(-1,10,1)).reshape(-1)
          loss2 = criterion2(y_pred,data[:][1])
          loss2.backward()
          optimizer2.step()
      existing_epochs2+=1
      train_times2[existing_epochs2] = loss2
      if i%50 == 0:
          print(existing_epochs2,"th iteration : ",loss2)
  joblib.dump(model2, f'ymodel{try_num2}.pkl')

else:
  model2 = joblib.load(f'ymodel{try_num2}.pkl')

# losses = [tsr.detach().numpy().flat[0] for tsr in train_times.values()]
# plt.plot(train_times.keys(), losses)
# plt.title("Train Loss Curve")

#training set actual vs predicted
fig, ax = plt.subplots(1,2)
train_pred = model(train_set[:][0].view(-1,10,1)).view(-1)
test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)

ax[0].plot(train_pred.detach().numpy(),label='predicted')
ax[0].plot(train_set[:][1].view(-1),label='original')
ax[0].title.set_text("Training Set Actual vs Predicted (X Coordinates)")
ax[0].legend()

ax[1].plot(test_pred.detach().numpy(),label='predicted')
ax[1].plot(test_set[:][1].view(-1),label='original')
ax[1].title.set_text("Test Set Actual vs Predicted (X Coordinates)")
ax[1].legend()
plt.show()

fig2, ax2 = plt.subplots(1,2)
train_pred2 = model(train_set2[:][0].view(-1,10,1)).view(-1)
test_pred2 = model(test_set2[:][0].view(-1,10,1)).view(-1)

ax2[0].plot(train_pred2.detach().numpy(),label='predicted')
ax2[0].plot(train_set2[:][1].view(-1),label='original')
ax2[0].title.set_text("Training Set Actual vs Predicted (X Coordinates)")
ax2[0].legend()

ax2[1].plot(test_pred2.detach().numpy(),label='predicted')
ax2[1].plot(test_set2[:][1].view(-1),label='original')
ax2[1].title.set_text("Test Set Actual vs Predicted (X Coordinates)")
ax2[1].legend()
plt.show()


inputx = np.array(scale_data([x for (x, y) in ava][:10], width)).reshape((-1, 1))
inputy = np.array(scale_data([y for (x, y) in ava][:10], height)).reshape((-1, 1))

start_time = time.time() 

n_input=10 # number of previous frames to use as features
num_correct = 0 # number of correct predictions
num_wrong = 0 # number of wrong predictions
frame_reset_pts = {} # dictionary of frames to reset at (key: frame, value: predicted (x, y) coordinates)
streak_dct = {} # dictionary of streaks (key: frame, value: streak count)
streak_count = 0

# Dictionaries for plotting
centroid_dct = {} # key: frame, value: list of centroids
pred_dct = {} # key: frame, value: (predx, predy)
chosen_path = {} # key: frame, value: selected (x, y) coordinates

for i in range(0, 201):
  print(f"frame {i+n_input}")

  # features
  x_init = torch.from_numpy(np.float32(np.expand_dims(inputx[i:i+n_input].reshape(-1, 1), 0))) # these are normalized values
  y_init = torch.from_numpy(np.float32(np.expand_dims(inputy[i:i+n_input].reshape(-1, 1), 0))) # these are standardized values
  print(x_init, y_init)
  
  # predicted coordinates
  predx = unscale_data(model(x_init).detach().numpy()[0][0], full=width)
  predy = unscale_data(model(y_init).detach().numpy()[0][0], full=height)
  pred_dct[i]=(predx, predy)

  # actual coordinates
  actx, acty = ava[i+n_input] 

  # Get list of centroids
  ground_truth_dir=r'C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\ground_truth\11408'
  mask = cv2.imread(f"{ground_truth_dir}/{i+n_input}.png", cv2.IMREAD_GRAYSCALE)
  centroids = find_centroids(mask)  # list of potential centroids
  centroid_dct[i] = centroids

  # Find closest centroid
  max_score = 10**1000
  max_act_score = 10**10000
  coords = (0,0) # Closest to predicted coords
  coords_act = (0,0) # Closest to actual coords

  for (pot_x, pot_y) in centroids:
    score = get_dist_score(predx, predy, pot_x, pot_y)
    act_score = get_dist_score(actx, acty, pot_x, pot_y)
    print(f"Centroid: {pot_x}, {pot_y} | Score: {round(score)}")
    if score <= max_score:
      max_score = score
      coords = (pot_x, pot_y)
    if act_score <= max_act_score:
      max_act_score = score
      coords_act = (pot_x, pot_y)

  print(f"Original Prediction: {'%.2f'%(predx)}, {'%.2f'%(predy)}")
  print(f"Actual Coords: {'%.2f'%(actx)}, {'%.2f'%(acty)}")
  print(f"Closest to Pred Coords: {coords[0]}, {coords[1]}") # last element = most recently appended "curr"
  print(f"Closest to Actual Coords: {coords_act[0]}, {coords_act[1]}") # last element = most recently appended "curr"


  # prediction is correct if closest centroid to predicted coords is the same as closest centroid to actual coords
  if (coords[0] == coords_act[0]):
    print("Correct")
    num_correct +=1
    inputx = np.append(inputx, unscale_data(np.array(coords[0]).reshape(-1, 1)[0][0], width)) 
    inputy = np.append(inputy,  unscale_data(np.array(coords[1]).reshape(-1, 1)[0][0], height))
    streak_count+=1
  else:
    print("False")
    num_wrong +=1
    frame_reset_pts[i+n_input] = (predx, predy)
    actual_norm_x = scale_data(ava[i+n_input][0].reshape(-1, 1), width)
    actual_norm_y = scale_data(ava[i+n_input][1].reshape(-1, 1), height)
    inputx = np.append(inputx, actual_norm_x) 
    inputy = np.append(inputy, actual_norm_y)
    streak_dct[i]=streak_count
    streak_count=0
  chosen_path[i] = (coords[0], coords[1])
  print(f"{num_correct}, {num_wrong}")
  print("\n")
print(f"Time: {time.time() -start_time}")