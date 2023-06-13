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

def split_lst(lst, n):
  """
  Split a list into sequences of length n
  """
  length = len(lst)
  return [lst[i*n: (i+1)*n] for i in range((length+n-1)//n)]

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
  
  return centroids, cont

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
    
# SET CONSTANTS
data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data"
model_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\lstm"
results_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results\lstm"

video_dir = rf"{data_dir}\imgs"
position_dir = rf"{data_dir}\positions"

# Save all video arrays and positions in dictionary
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']
imgs_dct = {}
positions_dct={}
for video in videos:
  imgs_dct[video] = np.load(rf"{video_dir}\{video}_crop.nd2.npy")
  positions_dct[video] = np.load(rf"{position_dir}\AVA_{video}.mat.npy")
  print(f"Loading {video}...")
print(f"Finished loading images and positions: {len(imgs_dct)} videos, {len(positions_dct)} positions")

# Split into train and test sets
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

# Split into sequences of length 100
seq_len=200
df_train_lst_shortened = [split_lst(lst, seq_len)[:-1] for lst in df_train_lst]
df_train_lst_shortened = np.concatenate(df_train_lst_shortened)
df_test_lst_shortened = [split_lst(lst, seq_len)[:-1] for lst in df_test_lst]
df_test_lst_shortened = np.concatenate(df_test_lst_shortened)
print(f"Input shape: {np.array(df_train_lst_shortened[0]).shape}")

TRAINING = False
NEW = False
if NEW:
  model = NeuralNetwork()
  losses = []
else:
  model = joblib.load(rf"{model_dir}\lstm_model9.pkl")
  losses = joblib.load(rf"{model_dir}\lstm_losses9.pkl")
print("Model loaded")

if TRAINING:
  lr = 0.00001
  epochs = 1000
  wandb.init(
      project="lstm",
      config={
      "learning_rate": lr,
      "architecture": "lstm",
      "dataset": "CIFAR-100",
      "epochs": epochs,
      }
  )
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  batch_size = 16
  num_batches = len(df_train_lst_shortened)//batch_size
  print(f"Batch size = {batch_size}, Num batches: {num_batches}")
  table = wandb.Table(columns=["Epoch", "Image"])
  for epoch in range(epochs):
    for j in range(num_batches):
      train_input = torch.tensor(df_train_lst_shortened[j*batch_size:(j+1)*batch_size][:, :-1, :], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
      pred = model(train_input)
      actual = torch.tensor(df_train_lst_shortened[j*batch_size:(j+1)*batch_size][:, 1:, :], dtype=torch.float32)
      loss = criterion(pred, train_input)
      loss.backward()
      optimizer.step()
    if epoch % 10 == 0:
      losses.append(loss)
      wandb.log({"loss": loss})
      test_input = torch.tensor(df_test_lst_shortened[0:batch_size][:, :-1, :], dtype=torch.float32)
      test_actual = torch.tensor(df_test_lst_shortened[0:batch_size][:, 1:, :], dtype=torch.float32)
      test_pred = model(test_input)
      valid_loss = criterion(test_pred, test_actual)
      wandb.log({"valid_loss": valid_loss})
      print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
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
  wandb.log({"predictions": table})
  joblib.dump(model, rf"{model_dir}\lstm_model10.pkl")
  joblib.dump(losses, rf"{model_dir}\lstm_losses10.pkl")
  wandb.finish()

video = '11409'
h, w = imgs_dct[video].shape[2:]

head = 10 # Starting index
split = math.floor(len(positions_dct[video])*0.8) # split that we did for the test set (ie. the first "frame/position" in act_seq is actually frame split (2131) in the original video )

act_seq = np.array(df_test_lst[1]) # full actual sequence of video 11408 (normalized)  (use as label y)
input = act_seq[:head, :] # running log of predictions (use as feature x)

correct = 0
wrong = 0

TESTING = True
RESET = False
max_reset = 2
reset_num = 0

crop_size = 20

if TESTING:
  total_frames = len(act_seq)-head
  for i in range(head, len(act_seq)):
    frame = split+i # Actual frame number
    print(f"Given first {head-1} frames, predict frame: {frame} of video {video}. Finished {i-head}/{total_frames} Frames | Progress: {round((i-head)/(total_frames)*100, 2)}%")

    # Get previous frame (to compare color & shape)
    img = cv2.cvtColor(cv2.imread(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}\{frame}.png"), cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(cv2.imread(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}\{frame-1}.png"), cv2.COLOR_BGR2GRAY)
    prev_pt = np.multiply(input[i-1], [w, h])
    prev_pt = [int(x) for x in list(prev_pt)]
    cropped_prev_img = crop_img(prev_img, *prev_pt, crop_size, crop_size)
    prev_cont, hierarchy = cv2.findContours(cropped_prev_img, 
                          cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
    prev_cont = prev_cont[0]

    # Get actual and predicted points
    act_pt = np.multiply(act_seq[i], [w, h]) # Actual coord for this frame
    pred = model(torch.tensor(input, dtype=torch.float32))
    pred_pt = np.multiply(pred[-1].detach(), [w, h]).detach().numpy() # Predicted coord for this frame
    
    # Get closest centroid to predicted point
    mask = cv2.cvtColor(cv2.imread(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\ground_truth\{video}\{frame}.png"), cv2.COLOR_BGR2GRAY)
    cnts, contours = find_centroids(mask)
    
    max_pred_score = 10**1000
    max_act_score = 10**1000
    cropped_imgs = []
    dst_scores = []
    color_scores = []
    shape_scores = []
    n = len(cnts)

    # For each centroid, get distance score, color score, and shape score
    for i in range(n):
      cnt = cnts[i]
      cont = contours[i]
      cropped_img = crop_img(img, *cnt, crop_size, crop_size)
      cropped_imgs.append(cropped_img)

      dst_score = get_dist_score(*cnt, *pred_pt) # distance between predicted point and centroid
      act_dst_score = get_dist_score(*cnt, *act_pt) # distance between actual point and centroid
      hst_score = get_color_score(cropped_prev_img, cropped_img) # color similarity between previous frame and centroid
      area_score = get_shape_score(prev_cont, cont) # shape similarity between previous frame and centroid

      if dst_score <= max_pred_score:
        max_pred_score = dst_score
        closest = cnt
      if act_dst_score <= max_act_score:
        max_act_score = act_dst_score
        closest_act = cnt
      dst_scores.append(dst_score)
      color_scores.append(hst_score)
      shape_scores.append(area_score)

    # Visualize color/shape scores
    fig, ax = plt.subplots(1, n+4)
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
    for i in range(n):
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

    accuracy = correct/(correct+wrong) if (correct+wrong) != 0 else 0
    print(f"Correct: {correct} | Wrong: {wrong} | Accuracy: {accuracy}")
    # plt.show()
    plt.close()

    if closest_act == closest:
      correct += 1
    else:
      wrong += 1
    
    # Add predicted point or add actual point
    if RESET and reset_num < max_reset:
      closest_norm = [closest_act[0]/w, closest_act[1]/h]
      reset_num += 1
    else:
      closest_norm = [closest[0]/w, closest[1]/h]
    
    input = np.concatenate((input, np.array(closest_norm).reshape(-1,2)))

    act_seq_coords = np.multiply(act_seq[:i], [w, h])
    act_seqx, act_seqy = act_seq_coords[:i, 0], act_seq_coords[:i][:, 1]
    pred_coords = np.multiply(input[:i], [w, h])
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

  print(f"Correct: {correct} | Wrong: {wrong} | Accuracy: {correct/(correct+wrong)}")