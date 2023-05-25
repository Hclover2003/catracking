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

# Set seed (for reproducibility)
# num = 0
# torch.manual_seed(num)
# random.seed(num)
# np.random.seed(num)

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

def get_blobs_adaptive(img, bound_size, min_brightness_const, min_area):
    im_gauss = cv2.GaussianBlur(img, (5, 5), 0) # "smoothing" the image with Gaussian Blur
    thresh = cv2.adaptiveThreshold(im_gauss,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bound_size,(min_brightness_const))
    # Find contours
    cont, hierarchy = cv2.findContours(thresh, 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE # only stores necessary points to define contour (avoids redundancy, saves memory)
                            )
    cont_filtered = []
    for con in cont: 
        area = cv2.contourArea(con) # calculate area, filter for contours above certain size
        if area>min_area: # chosen by trial/error
            cont_filtered.append(con)    
    
    # Draw + fill contours
    new_img = np.full_like(img, 0) # image has black background
    for c in cont_filtered:
        cv2.drawContours(new_img, # image to draw on
                        [c], # contours to draw
                        -1, # contouridx: since negative, all contours are drawn
                        255, # colour of contours: white
                        -1 # thickness: since negative, fill in the shape
                        )
    return new_img, cont_filtered
  
def scale_data_scaler(data, opt):
  """Scales data with MinMaxScaler or StandardScaler
  data: list of data to scale
  opt: "minmax" or "stand" 
  
  Returns scaler and scaled data"""
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
  """Scales data with image dimensions
  data: list of data to scale
  full: image dimension (height or width)
  
  Returns scaled data"""
  norm_data = [i/full for i in data]
  return norm_data

def unscale_data(data, full:int):
  """Unscales data with image dimensions
  data: list of data to scale
  full: image dimension (height or width)
  
  Returns unscaled data"""
  try:
      norm_data = [i*full for i in data]
  except:
      norm_data = data*full
  return np.array(norm_data)

def get_features_and_outcome(num_prev, neuron_positions):
  """Returns dataframe with features and outcome variables
  num_prev: number of previous frames to use as features
  neuron_positions: list of neuron positions (x, y)
  
  Returns dataframe with features and outcome variables"""
  i = 0
  features_x = []
  features_y = []
  
  # scale data (x or y position)
  norm_neuron_positions_x = scale_data([x for (x, y) in neuron_positions], full=width)
  norm_neuron_positions_y = scale_data([y for (x, y) in neuron_positions], full=height)
    
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

def get_norm_df(df, lst_cols:List, non_lst_cols: List):
  """Normalize dataframe values, even when some columns have elements that are lists

  df: dataframe to normalize
  lst_cols: list of columns where element dtype is list, not number
  non_lst_cols: list of columns where element dtype is number
  
  Returns dataframe with normalized columns"""
  df_expanded = df.copy()
  df_final = df.copy()
  for col in lst_cols:
      n = len(df_expanded[col][0])
      print(col, n)
      labels = [f"{col}_{i}" for i in range(n)]
      df_expanded[labels] = pd.DataFrame(df_expanded[col].tolist())
      df_expanded = df_expanded.drop(col, axis=1)
  
  scaler = MinMaxScaler()
  df_norm = pd.DataFrame(scaler.fit_transform(df_expanded),
                    columns=df_expanded.columns)
  
  for col_name in lst_cols:
      sub_cols = [col for col in df_norm.columns if col_name in col]
      df_final[col_name] = df_norm[sub_cols].values.tolist()
  df_final[non_lst_cols] = df_norm[non_lst_cols]

  return df_final

def get_features_and_outcome_w_visual(num_prev, max_height, max_width, neuron_positions, img_dir):
  """Returns dataframe with features and outcome variables
  num_prev: number of previous frames to use as features
  neuron_positions: list of neuron positions (x, y)
  
  Returns dataframe with features and outcome variables"""
  i = 0 # index of current frame
  features_x = []
  features_y = []
  features_x_colorhist = []
  features_x_mean= []
  features_x_meanstd = []
  PURPLE = (69, 6, 90)
  
  # scale data (x or y position)
  neuron_positions_x = [x for (x, y) in neuron_positions]
  neuron_positions_y = [y for (x, y) in neuron_positions]
    
  # since we need 10 previous frames as features, make sure we stop in time
  while i <= len(neuron_positions) - num_prev -1:
    frame = i+num_prev
    
    # Get features from image
    actx, acty = neuron_positions[frame]
    actx, acty = round(actx), round(acty)
    img_path = f"{img_dir}\{frame}.png"
    print(img_path)
    img= cv2.imread(img_path)
    img = cv2.copyMakeBorder(img, 0, max_height-img.shape[0], 0, max_width-img.shape[1], borderType=cv2.BORDER_CONSTANT, value=PURPLE) # Add padding
    crop_size=12

    
    cropped_img =  img[(acty)-crop_size:acty+crop_size, actx-crop_size:actx+crop_size, :]
    
    # Get visual features from cropped image (neuron)
    features_x_meanstd.append(np.concatenate(cv2.meanStdDev(cropped_img)).flatten()) # mean and std of each channel
    features_x_mean.append(cv2.mean(cropped_img)[:3]) # mean of each channel
    features_x_colorhist.append(cv2.calcHist([cropped_img],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]).flatten()) # color histogram
    
    # Get features from neuron positions
    features_x.append(neuron_positions_x[i:frame])
    features_y.append(neuron_positions_y[i:frame])
    
    i+=1

  # Make dataframe with features and outcome variables
  dict = {'prev_n_x': features_x, 'curr_x': neuron_positions_x[num_prev:], 
          'prev_n_y': features_y, 'curr_y': neuron_positions_y[num_prev:], 
          'channel_means': features_x_mean, 'channel_std': features_x_meanstd
          } 

  # Normalize features
  scaler = MinMaxScaler()
  df = pd.DataFrame(dict)
  df_final = get_norm_df(df, ['prev_n_x', 'prev_n_y', 'channel_means', 'channel_std'], ['curr_x', 'curr_y'])
  
  df_final['curr_frame']= [j for j in range(num_prev, len(neuron_positions))]
  
  return df_final, scaler

def find_centroids(segmented_img):
  """Returns list of centroids of segmented image
  segmented_img: segmented image (binary)
  """
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
  """Returns distance score between two points
  x, y: coordinates of point 1
  x2, y2: coordinates of point 2"""
  return ((x2-x)**2)+((y2-y)**2)

def get_closest_cent(centroids:List, pred:Tuple):
    """ Returns the closest centroid to the predicted coordinates
    centroids: list of centroids
    pred: predicted coordinates"""
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

def get_norm_width_height(video_dir, position_dir, videos, imgs_dct, positions_dct):
  width, height = 0, 0 
  for video in videos:
      # Save imgs and positions in dictionary
      imgs_dct[video] = np.load(rf"{video_dir}\{video}_crop.nd2.npy")
      positions_dct[video] = np.load(rf"{position_dir}\AVA_{video}.mat.npy")
      print(f"Loaded {video}")
      h, w = imgs_dct[video].shape[2:]
      if h > height:
          height = h
      if w > width:
          width = w
  return width, height

# SET CONSTANTS
data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data"

videos = ['11409', "11410", '11411', '11413', '11414', '11415']
imgs_dct = {}
positions_dct={}
width, height = get_norm_width_height( rf"{data_dir}\imgs", rf"{data_dir}\positions", videos, imgs_dct, positions_dct) # Get max height and width between all videos (for scaling)
print(f"Max width: {width} | Max height: {height}")
print(f"Finished loading images and positions: {len(imgs_dct)} images, {len(positions_dct)} positions")

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

# PREPARE DATA FOR LSTM
# Train-test split
df_train_lst = []
df_test_lst=[]
for video in videos:
    print(f"video {video} ...")
    positions = positions_dct[video]
    img_dir =rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}"
    video_df= get_features_and_outcome_w_visual(10, max_height=height, max_width=width, img_dir=img_dir, neuron_positions=positions) # Get features and outcome variables for each video
    split = math.floor(len(video_df)*0.8)
    df_train_lst.append(video_df[:split]) # Get features and outcome variables for each video
    df_test_lst.append(video_df[split:]) # Get features and outcome variables for each video
print("Loaded data")
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
print("Finished test-train split")
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
print("Finished creating dataset and dataloader")


# TRAIN MODEL
train = False # True if training x model, False if loading model
load=False
load_num = 2 # for saving/loading model x
save_num = 1

train2 = True # True if training y model, False if loading model
load2=False
load_num2 = 3 # for saving/loading model y
save_num2=4
# Get X model
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
  joblib.dump(model, f'model{save_num}.pkl')
  print(f"saved x model {save_num}")

    # Plot loss curve
  losses = [tsr.detach().numpy().flat[0] for tsr in train_times.values()]
  plt.figure()
  plt.plot(train_times.keys(), losses)
  plt.title("Train Loss Curve")
  plt.show()
else:
  model = joblib.load(f'model{load_num}.pkl')
  print("loaded x model")

# Get Y model
if train2:
  # Create model
  if load2:
     model2=joblib.load(f'ymodel{load_num2}.pkl')
  else:
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
          print(existing_epochs2,"th iteration : ", loss2)
  joblib.dump(model2, f'ymodel{save_num2}.pkl')
  print(f"saved y model {save_num2}")

  # Plot loss curve
  losses2 = [tsr.detach().numpy().flat[0] for tsr in train_times2.values()]
  plt.figure()
  plt.plot(train_times2.keys(), losses2)
  plt.title("Train Loss Curve")
  plt.show()
else:
  model2 = joblib.load(f'ymodel{load_num2}.pkl')
  print("loaded y model")

# VISUALIZE PREDICTIONS
# Plot x coordinates actual vs predicted
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

ax[0].set_xlabel("Frame")
ax[0].set_ylabel("X Coordinates")
ax[1].set_xlabel("Frame")
ax[1].legend()
plt.show()

# Plot y coordinates actual vs predicted
fig2, ax2 = plt.subplots(1,2)
train_pred2 = model(train_set2[:][0].view(-1,10,1)).view(-1)
test_pred2 = model(test_set2[:][0].view(-1,10,1)).view(-1)

ax2[0].plot(train_pred2.detach().numpy(),label='predicted')
ax2[0].plot(train_set2[:][1].view(-1),label='original')
ax2[0].title.set_text("Training Set Actual vs Predicted (Y Coordinates)")
ax2[0].legend()

ax2[1].plot(test_pred2.detach().numpy(),label='predicted')
ax2[1].plot(test_set2[:][1].view(-1),label='original')
ax2[1].title.set_text("Test Set Actual vs Predicted (Y Coordinates)")
ax2[0].set_xlabel("Frame")
ax2[0].set_ylabel("Y Coordinates")
ax2[1].set_xlabel("Frame")
ax2[1].legend()
plt.show()

# Plot x and y coordinates
fig3, ax3 = plt.subplots(1,2)
ax3[0].plot(train_pred.detach().numpy(), train_pred2.detach().numpy(),label='predicted')
ax3[0].plot(train_set[:][1].view(-1), train_set2[:][1].view(-1),label='original')
ax3[1].plot(test_pred.detach().numpy(), test_pred2.detach().numpy(),label='predicted')
ax3[1].plot(test_set[:][1].view(-1), test_set2[:][1].view(-1),label='original')
ax3[0].title.set_text("Training Set Path of Neuron (X and Y Coordinates)")
ax3[1].title.set_text("Test Set Path of Neuron (X and Y Coordinates)")
ax3[0].set_xlabel("X Coordinates")
ax3[0].set_ylabel("Y Coordinates")
ax3[1].set_xlabel("X Coordinates")
ax3[0].legend()
ax3[1].legend()
plt.show()

# EVALUATE MODEL
# RMSE
print("X coordinates")
print(f"Train RMSE: {np.sqrt(mean_squared_error(train_pred.view(-1).detach().numpy(), train_set[:][1].view(-1).detach().numpy()))}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(test_pred.view(-1).detach().numpy(), test_set[:][1].view(-1).detach().numpy()))}")
# print(f"Train R^2: { model_selection.cross_val_score(model, train_set[:][0].view(-1,10,1), train_set[:][1].view(-1), cv=5).mean() }")
# print(f"Test R^2: { model_selection.cross_val_score(model, test_set[:][0].view(-1,10,1), test_set[:][1].view(-1), cv=5).mean() }")

print("Y coordinates")
print(f"Train RMSE: {np.sqrt(mean_squared_error(train_pred2.view(-1).detach().numpy(), train_set2[:][1].view(-1).detach().numpy()))}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(test_pred2.view(-1).detach().numpy(), test_set2[:][1].view(-1).detach().numpy()))}")

