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
  # Train model
  for i in range(num_batches):
    train_input = torch.tensor(train_lst[i*batch_size:(i+1)*batch_size][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    pred = model(train_input)
    actual = torch.tensor(train_lst[i*batch_size:(i+1)*batch_size][:, 1:, :], dtype=torch.float32)
    loss = criterion(pred, actual)
    total_loss += loss
    loss.backward()
    optimizer.step()
    
    valid_input = torch.tensor(valid_lst[i:(i+1)][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    valid_pred = model(valid_input)
    valid_actual = torch.tensor(valid_lst[i:(i+1)][:, 1:, :], dtype=torch.float32)
    valid_loss = criterion(valid_pred, valid_actual)
  # Log loss to wandb
  wandb.log({"loss": loss, "valid_loss": valid_loss})
  
  return loss, valid_loss, (total_loss/num_batches)

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
    config = wandb.config
    
    
    train_lst_shortened = np.concatenate([split_lst(lst, config.seq_len)[:-1] for lst in train_lst])
    valid_lst_shortened = np.concatenate([split_lst(lst, config.seq_len)[:-1] for lst in valid_lst])
    model = NeuralNetwork() # New model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        avg_loss = train_epoch(model=model, train_lst=train_lst_shortened, valid_lst=valid_lst_shortened, batch_size=config.batch_size, criterion=criterion, optimizer=optimizer)
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

data_dir = "/Users/huayinluo/Desktop/code/catracking-1/data"
video_dir = os.path.join(data_dir, "imgs")
position_dir = os.path.join(data_dir, "positions")
model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm"
img_dir = "/Users/huayinluo/Desktop/code/catracking-1/images"
results_dir = "/Users/huayinluo/Desktop/code/catracking-1/results"


# Save all video arrays and positions in dictionary
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415', '11433', '11434']
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

TRAIN = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TRAIN:
  model = joblib.load(os.path.join(model_dir, "lstm_model5A.pkl"))
  model_name = "lstm_model5B.pkl"
  epochs = 300000
  seq_len = 250
  lr = 0.000001
  batch_size = 8
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  df_train_lst_shortened = np.concatenate([split_lst(lst, seq_len)[:-1] for lst in df_train_lst])
  df_test_lst_shortened = np.concatenate([split_lst(lst, seq_len)[:-1] for lst in df_test_lst])
  num_batches = len(df_train_lst_shortened)//batch_size
  print(f"Number of Sequences = {len(df_train_lst_shortened)}, Batch size = {batch_size}, Num batches: {num_batches}")

  wandb.init(
      project="lstm",
      config={
        "existing_model": "lstm_model5.pkl",
        "model_name": model_name,
        "seq_len": seq_len,
      "learning_rate": lr,
      "epochs": epochs,
      "batch_size": batch_size,
      "architecture": "lstm",
      "dataset": "CIFAR-100",
      "epochs": epochs,
      }
  )

  table = wandb.Table(columns=["Epoch", "Image"])
  
  start_time = time.time()
  for epoch in range(epochs):
    # Train model
    for j in range(num_batches):
      train_input = torch.tensor(df_train_lst_shortened[j*batch_size:(j+1)*batch_size][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
      pred = model(train_input)
      actual = torch.tensor(df_train_lst_shortened[j*batch_size:(j+1)*batch_size][:, 1:, :], dtype=torch.float32)
      loss = criterion(pred, actual)
      loss.backward()
      optimizer.step()
    # Log train and valid loss every 10 epochs
    if epoch % 10 == 0:
      wandb.log({"loss": loss})
      test_input = torch.tensor(df_test_lst_shortened[0:batch_size][:, :-1, :], dtype=torch.float32)
      test_actual = torch.tensor(df_test_lst_shortened[0:batch_size][:, 1:, :], dtype=torch.float32)
      test_pred = model(test_input)
      valid_loss = criterion(test_pred, test_actual)
      wandb.log({"valid_loss": valid_loss})
      print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
    # Plot valid prediction every 100 epochs
    if epoch % 100 == 0:
      colours = np.arange(test_pred.detach()[1].numpy().shape[0])
      predx = test_pred.detach()[1].numpy()[:,0]
      predy = test_pred.detach()[1].numpy()[:,1]
      actx = test_actual.detach()[1][:,0]
      acty = test_actual.detach()[1][:,1]
      fig = plt.figure()
      plt.scatter(test_input.detach()[1].numpy()[0,0], test_input.detach()[1].numpy()[0,1], label="Input")
      plt.scatter(predx, predy, c = colours, cmap="Greens", label="Predicted") #1st sequence in batch
      plt.colorbar()
      plt.scatter(actx, acty, c = colours, cmap = "Oranges", label="Actual") # 1st sequence in batch
      plt.colorbar()
      plt.legend()
      table.add_data(epoch, wandb.Image(fig))
      plt.close()
  joblib.dump(model, os.path.join(model_dir, model_name))
  print(time.time() - start_time)
  wandb.log({"predictions": table})
  wandb.finish()
  
PREDICT = False
if PREDICT:
  # load model
  model_name = "lstm_model5e2.pkl"
  model = joblib.load(os.path.join(model_dir, model_name))
  
  # track number of correct and incorrect predictions
  num_correct = 0
  num_incorrect = 0
  RESET = False
  num_reset = 0
  max_reset = 100
  
  # choose set of frames to predict on
  video = '11408'
  start_frame = 0
  end_frame = 200 # frames 0-100 of video 11408
  start_index = 1 # give first 10 frames as input to model
  height, width = imgs_dct[video].shape[2:] # dimensions to normalize positions with
  crop_size = 20
  
  dst_weight = 10 # Distance score is from [0, infinity]. Closer to 0 is better
  hst_weight = 100 # Color Histogram score is from [0, 1]. Closer to 1 is better.
  area_weight = 0
  save_dir = os.path.join(results_dir, f"model{model_name}_{video}_{start_index}start_{end_frame}frames_RESET{RESET}{max_reset}_{dst_weight}dst{hst_weight}color{area_weight}area")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
  # actual positions and chosen positions
  actual_positions = np.array(positions_dct[video][start_frame:end_frame+1]) # full actual sequence of video 11408 (use as label y)
  actual_positions_norm = np.multiply(actual_positions, [1/width, 1/height])
  chosen_positions_norm = actual_positions_norm[start_frame:start_frame+start_index, :] # running log of chosen coordinates (use as feature x)
  predicted_positions = actual_positions[start_frame:start_frame+start_index, :] # running log of predictions
  
  # loop through each frame
  for frame in range(start_frame+start_index, end_frame+1):
    # get image for this frame and previous frame
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame-1}.png")), cv2.COLOR_BGR2GRAY)
    prev_position = [int(position) for position in list(np.multiply(chosen_positions_norm[-1], [width, height]))]
    cropped_prev_img = crop_img(prev_img, *prev_position, crop_size, crop_size)
    prev_contours, hierarchy = cv2.findContours(cropped_prev_img, 
                          cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
    prev_contour = prev_contours[0]

    # Get actual and predicted points
    pred_positions_norm = model(torch.tensor(chosen_positions_norm, dtype=torch.float32))
    pred_position = np.multiply(pred_positions_norm[-1].detach(), [width, height]).detach().numpy() # Predicted coord for this frame
    predicted_positions = np.concatenate((predicted_positions, np.array(pred_position).reshape(-1,2)))
    actual_position = actual_positions[frame]# Actual coord for this frame

    # Segment image and get centroids (possible positions)
    mask = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"ground_truth/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    centroids, contours = find_centroids(mask)
    num_centroids = len(centroids)
    max_pred_score = 10**1000
    max_act_score = 10*1000
    centroid_scores_dict = {} # key: centroid value, value: list of scores
    
    # Find "closest" centroid: closest positionally to predicted point, closest visually to previous point
    for i in range(num_centroids):
      centroid = centroids[i]
      centroid_contour = contours[i]
      cropped_centroid_img = crop_img(img, *centroid, crop_size, crop_size)
      

      dst_score = get_dist_score(*centroid, *pred_position) # distance between predicted point and centroid
      hst_score = get_color_score(cropped_prev_img, cropped_centroid_img) # color similarity between previous frame and centroid
      area_score = get_shape_score(prev_contour, centroid_contour) # shape similarity between previous frame and centroid
      score = (dst_weight*dst_score) + (hst_weight*hst_score) + (area_weight*area_score)
      centroid_scores_dict[centroid] = [dst_score, hst_score, area_score]
      
      actual_score = get_dist_score(*centroid, *actual_position) # distance between actual point and centroid
      
      # Choose centroid with closest score
      if score <= max_pred_score:
        max_pred_score = score
        closest = centroid
      if actual_score <= max_act_score:
        max_act_score = actual_score
        closest_act = centroid

    # Chosen position normalized
    closest_position_norm = [closest[0]/width, closest[1]/height]
          
    if closest_act == closest:
      num_correct += 1
    else:
      num_incorrect += 1
      
      # If incorrect, reset to actual position
      if RESET and num_reset < max_reset:
        closest_position_norm = [closest_act[0]/width, closest_act[1]/height]
        num_reset += 1
        print(f"RESET: {num_reset} resets left")
        
    # Add chosen position to list
    chosen_positions_norm = np.concatenate((chosen_positions_norm, np.array(closest_position_norm).reshape(-1,2)))

    # Un-normalized actual positions
    chosen_positions = np.multiply(chosen_positions_norm, [width, height])
    
    plt.imshow(img, cmap='gray')
    # plt.plot(act_pt[0], act_pt[1], 'ro', markersize=3, label= "Actual")
    # plt.plot(pred_pt[0], pred_pt[1], 'bo', markersize=3, label = "Predicted")
    # plt.plot(closest[0], closest[1], 'mo', markersize=3, label = "Closest Predicted")
    # plt.plot(closest_act[0], closest_act[1], 'go', markersize=3, label = "Closest Actual")
    plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], c=np.arange(len(predicted_positions)), cmap='Blues', s = 3, label="Predicted")
    plt.scatter(actual_positions[:frame+1, 0], actual_positions[:frame+1, 1], c=np.arange(len(actual_positions[:frame+1])), cmap='Greens', s = 3, label="Input")
    plt.colorbar()
    plt.scatter(chosen_positions[:, 0], chosen_positions[:, 1], c= np.arange(len(chosen_positions)), cmap='Oranges', s = 3, label="Input")
    plt.colorbar()
    plt.legend()
    plt.title(f"Frame {frame-1} | Correct: {num_correct} | Wrong: {num_incorrect} | Accuracy: {num_correct/(num_correct+num_incorrect)}")
    plt.savefig(os.path.join(save_dir, f"{frame-1}.png"))
    plt.close()
    print(f"Saved image for frame {frame} (Progress: {frame}/{end_frame} {round(frame/end_frame)} %)")

    print(f"Correct: {num_correct} | Wrong: {num_incorrect} | Accuracy: {num_correct/(num_correct+num_incorrect)}")
    
  
exit()
  
TESTING = False
RESET = False
max_reset = 2
reset_num = 0
crop_size = 20

if TESTING:
  model = joblib.load(os.path.join(model_dir, "lstm_model0.pkl"))
  num_correct = 0
  num_incorrect = 0
  start_index = 10 # Starting index

  video = '11408' # video to test on
  actual_positions = np.array(df_test_lst[0]) # full actual sequence of video 11408 (normalized)  (use as label y)
  chosen_positions = actual_positions[:start_index, :] # running log of predictions (use as feature x)

  height, width = imgs_dct[video].shape[2:]
  split_frame = math.floor(len(positions_dct[video])*0.8) # split that we did for the test set (ie. the first "frame/position" in act_seq is actually frame split (2131) in the original video )
  total_frames = len(actual_positions)-start_index
  
  for i in range(start_index, len(actual_positions)):
    frame = split_frame+i # Actual frame number
    print(f"Predicting frame: {frame} of video {video}. Finished {i-start_index}/{total_frames} Frames | Progress: {round((i-start_index)/(total_frames)*100, 2)}%")

    # Get previous frame (to compare color & shape)
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame-1}.png")), cv2.COLOR_BGR2GRAY)
    prev_pt = [int(x) for x in list(np.multiply(chosen_positions[i-1], [w, h]))]  # point at previous index in chosen positions, un-normalize it
    cropped_prev_img = crop_img(prev_img, *prev_pt, crop_size, crop_size)
    prev_cont, hierarchy = cv2.findContours(cropped_prev_img, 
                          cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
    prev_cont = prev_cont[0]

    # Get actual and predicted points
    act_pt = np.multiply(actual_positions[i], [width, height]) # Actual coord for this frame (un-normalize it)
    pred = model(torch.tensor(chosen_positions, dtype=torch.float32))
    pred_pt = np.multiply(pred[-1].detach(), [width, height]).detach().numpy() # Predicted coord for this frame
    
    # Get centroids and contours from segmented image
    # TODO: in practice, this is where UNet would segment the image
    mask = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"ground_truth/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    cnts, contours = find_centroids(mask)
    
    max_pred_score = 10**1000
    max_act_score = 10**1000
    cropped_imgs = []
    dst_scores = []
    color_scores = []
    shape_scores = []
    num_centroids = len(cnts)

    # For each centroid, get distance score, color score, and shape score
    # Choose closest centroid based on this
    for i in range(num_centroids):
      # Crop image around centroid
      cnt = cnts[i]
      cont = contours[i]
      cropped_img = crop_img(img, *cnt, crop_size, crop_size)
      cropped_imgs.append(cropped_img)

      # Get scores
      dst_weight = 1
      hst_weight = 1
      area_score = 0
      dst_score = get_dist_score(*cnt, *pred_pt) # distance between predicted point and centroid
      hst_score = get_color_score(cropped_prev_img, cropped_img) # color similarity between previous frame and centroid
      area_score = get_shape_score(prev_cont, cont) # shape similarity between previous frame and centroid
      score = dst_score + hst_score
      
      act_score = get_dist_score(*cnt, *act_pt) # distance between actual point and centroid
      
      # Choose centroid with closest score
      if score <= max_pred_score:
        max_pred_score = score
        closest = cnt
      if act_score <= max_act_score:
        max_act_score = act_score
        closest_act = cnt
        
      dst_scores.append(dst_score)
      color_scores.append(hst_score)
      shape_scores.append(area_score)

    # Visualize each cropped centroid and their respective scores
    fig, ax = plt.subplots(1, num_centroids+5)
    ax[0].imshow(prev_img)
    ax[0].plot(prev_pt[0], prev_pt[1], 'ro', markersize=3, label= "Previous")
    ax[0].set_title(f"Previous Img: Frame {frame-1}")
    ax[1].imshow(cropped_prev_img) # Plot previous cropped
    ax[1].set_title("Previous Cropped")
    ax[2].imshow(img)
    ax[2].set_title(f"Current Img: Frame {frame}")
    ax[2].plot(act_pt[0], act_pt[1], 'ro', markersize=3, label= "Actual")
    ax[2].plot(pred_pt[0], pred_pt[1], 'bo', markersize=3, label = "Predicted")
    ax[2].plot(closest[0], closest[1], 'mo', markersize=3, label = "Closest Predicted")
    ax[2].plot(closest_act[0], closest_act[1], 'go', markersize=3, label = "Closest Actual")
    ax[3].imshow(mask)
    ax[3].set_title("Mask")
    for i in range(num_centroids):
      ax[3].plot(cnts[i][0], cnts[i][1], 'ro', markersize=1, label= f"Centroid {i}")
      ax[i+4].imshow(cropped_imgs[i]) # Plot centroid cropped
      title = f"Cnt {i}"
      if cnts[i] == closest:
        title += " | Pred"
      if cnts[i] == closest_act:
        title += " | Act"
      ax[i+4].set_title(title)
      ax[i+4].set_xlabel(f"Dst Score: {round(dst_scores[i])} \n HST Score: {round(color_scores[i], 2)} \n Area Score: {round(shape_scores[i])}")
      print(title + f" | Dst Score: {round(dst_scores[i])} | HST Score: {round(color_scores[i], 2)} | Area Score: {round(shape_scores[i])}")
    ax[3].legend()
    ax[-1].imshow(crop_img(img, act_pt[0], act_pt[1], crop_size, crop_size))
    

    accuracy = num_correct/(num_correct+num_incorrect) if (num_correct+num_incorrect) != 0 else 0
    print(f"Correct: {num_correct} | Wrong: {num_incorrect} | Accuracy: {accuracy}")
    # plt.show()
    plt.close()

    if closest_act == closest:
      num_correct += 1
    else:
      num_incorrect += 1
    
    # Add predicted point or add actual point
    if RESET and reset_num < max_reset:
      closest_norm = [closest_act[0]/w, closest_act[1]/h]
      reset_num += 1
    else:
      closest_norm = [closest[0]/w, closest[1]/h]
    
    # Add chosen position to list
    chosen_positions= np.concatenate((chosen_positions, np.array(closest_norm).reshape(-1,2)))

    # Un-normalized actual positions
    act_seq_coords = np.multiply(actual_positions[:i], [width, height])
    act_seqx, act_seqy = act_seq_coords[:i, 0], act_seq_coords[:i][:, 1]
    pred_coords = np.multiply(chosen_positions[:i], [width, height])
    predx, predy = pred_coords[:i, 0], pred_coords[:i, 1]

    plt.imshow(img, cmap='gray')
    plt.plot(act_pt[0], act_pt[1], 'ro', markersize=3, label= "Actual")
    plt.plot(pred_pt[0], pred_pt[1], 'bo', markersize=3, label = "Predicted")
    plt.plot(closest[0], closest[1], 'mo', markersize=3, label = "Closest Predicted")
    plt.plot(closest_act[0], closest_act[1], 'go', markersize=3, label = "Closest Actual")
    plt.scatter(act_seqx, act_seqy, c=np.arange(len(act_seqx)), cmap='Greens', s = 3, label="Input")
    plt.colorbar()
    plt.scatter(predx, predy, c= np.arange(len(predx)), cmap='Oranges', s = 3, label="Input")
    plt.colorbar()
    plt.legend()
    plt.savefig(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\lstm_pred\{video}\{frame}.png")
    plt.close()
    print(f"Saved image for frame {frame}")

  print(f"Correct: {num_correct} | Wrong: {num_incorrect} | Accuracy: {num_correct/(num_correct+num_incorrect)}")
  
SWEEP = False
if SWEEP:
  # Sweep config
  parameters_dct = {
  "seq_len": {"values": [10, 25, 50, 100, 200, 250]},
  "batch_size": {"values": [8, 16, 32, 64]},
  "learning_rate": {"max": 0.001, "min": 0.00001},
  "epochs": {"values": [50, 100, 200, 500, 1000, 2000]}
  }

  parameters_dct.update({
  "seq_len": {"value": 100},
  "learning_rate": {"value": 0.001},
  "batch_size": {"value": 16}
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

  # Run sweep to find hyperparameters
  wandb.agent(sweep_id, function=lambda: train(train_lst=df_train_lst, valid_lst=df_test_lst), count=3)