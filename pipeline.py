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
import natsort
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

def show_crop_images(img, mask, prev_img, act_pt, pred_pt, prev_pt, cnts, contours):
    cropped_prev_img = crop_img(prev_img, prev_pt)
    num_centroids = len(centroids)
    cropped_imgs = []
    for i in range(num_centroids):
      # Crop image around centroid
      cnt = cnts[i]
      cont = contours[i]
      cropped_img = crop_img(img, cnt)
      cropped_imgs.append(cropped_img)
        # Visualize each cropped centroid and their respective scores
    
    closest = get_most_similar_centroid(img, prev_img, pred_pt, prev_pt, cnts, contours)
    closest_act = get_closest_cent(cnts, act_pt)
    
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
      title = f"Cnt {cnts[i]}"
      if cnts[i] == closest:
        title += " | Pred"
      if cnts[i] == closest_act:
        title += " | Act"
      ax[i+4].set_title(title)
    # ax[3].legend()
    ax[-1].imshow(crop_img(img, act_pt.astype(int), crop_size, crop_size))
    ax[-1].set_title("Actual Cropped")
    plt.show()


# List of data directories: change to your directory
data_dir = "/Users/huayinluo/Desktop/code/catracking-1/data"
raw_video_dir = os.path.join(data_dir, "imgs")
position_dir = os.path.join(data_dir, "positions")

img_dir = "/Users/huayinluo/Desktop/code/catracking-1/images"
ground_truth_dir = os.path.join(img_dir, "ground_truth")
original_dir = os.path.join(img_dir, "original")

model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm"
results_dir = "/Users/huayinluo/Desktop/code/catracking-1/results"
save_dir = os.path.join(results_dir, "pipeline")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Video number we are tracking
video = "11408"
start_frame = 0
end_frame = 200 
video_original_dir = os.path.join(original_dir, video)

# Get video height and width for normalization
video_height, video_width = np.load(os.path.join(raw_video_dir, f"{video}_crop.nd2.npy")).shape[2:]

# Number of frames to give as input to model
num_start_frames = 10 

# Get actual positions (to compare results against)
actual_positions = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
actual_positions_norm = np.multiply(actual_positions, [1/video_width, 1/video_height])

# Running log of predictions
chosen_positions_norm = actual_positions_norm[start_frame:start_frame+num_start_frames, :] # First 10 frames are given as input
lstm_predicted_positions = actual_positions[start_frame:start_frame+num_start_frames, :]

# Load model
model_name = "lstm_model5e2.pkl"
model = joblib.load(os.path.join(model_dir, model_name))

# Track number of correct and incorrect predictions
num_correct = 0
num_incorrect = 0

# Choose whether to reset model at incorrect predictions
RESET = False
num_reset = 0
max_reset = 100 # maximum number of times to reset model

# Parameters for scoring
crop_size = 20 # Size of cropped image
dst_weight = 10 # Distance score is from [0, infinity]. Closer to 0 is better
hst_weight = 100 # Color Histogram score is from [0, 1]. Closer to 1 is better.
area_weight = 0 # Area score is from [0, 1]. Closer to 0 is better.

# Loop through each frame
for frame in range(start_frame+num_start_frames, end_frame+1):
    # Get current and previous images
    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/{video}/{frame-1}.png")), cv2.COLOR_BGR2GRAY)
    segmented_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"ground_truth/{video}/{frame}.png")), cv2.COLOR_BGR2GRAY)
    segmented_prev_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"ground_truth/{video}/{frame-1}.png")), cv2.COLOR_BGR2GRAY)
    
    centroids, contours = find_centroids(segmented_img)
    prev_position = (np.multiply(chosen_positions_norm[-1], [video_width, video_height])).astype(int)
    actual_position = actual_positions[frame]
    
    # Predict position with LSTM model
    pred_positions_norm = model(torch.tensor(chosen_positions_norm, dtype=torch.float32))
    pred_position_norm = pred_positions_norm[-1].detach()
    pred_position = (np.multiply(pred_position_norm, [video_width, video_height]).numpy()).astype(int)
    
    # Get closest centroid
    closest_centroid = get_most_similar_centroid(img=img, prev_img=prev_img, pred_pt=pred_position, prev_pt=prev_position, centroids=centroids, contours=contours, log=True)
    
    actual_closest_centroid = get_closest_cent(centroids, actual_position)
    
    # Normalize chosen position
    closest_position_norm = [closest_centroid[0]/video_width, closest_centroid[1]/video_height]
    
    # Check if prediction is correct
    if closest_centroid == actual_closest_centroid:
        num_correct += 1
    else:
        num_incorrect += 1
        
        # If incorrect, reset to actual position
        if RESET and num_reset < max_reset:
            closest_position_norm = [actual_closest_centroid[0]/video_width, actual_closest_centroid[1]/video_height]
        num_reset += 1
        print(f"RESET: {num_reset} resets left")
        
    # Add chosen position to list
    chosen_positions_norm = np.concatenate((chosen_positions_norm, np.array(closest_position_norm).reshape(-1,2)))

    # VISUALIZE results
    # Un-normalizeactual positions
    chosen_positions = np.multiply(chosen_positions_norm, [video_width, video_height])

    show_crop_images(img, segmented_img, prev_img, actual_position, pred_position, prev_position, centroids, contours)
    
    plt.imshow(img, cmap='gray')
    # plt.plot(act_pt[0], act_pt[1], 'ro', markersize=3, label= "Actual")
    # plt.plot(pred_pt[0], pred_pt[1], 'bo', markersize=3, label = "Predicted")
    # plt.plot(closest[0], closest[1], 'mo', markersize=3, label = "Closest Predicted")
    # plt.plot(closest_act[0], closest_act[1], 'go', markersize=3, label = "Closest Actual")
    plt.scatter(lstm_predicted_positions[:, 0], lstm_predicted_positions[:, 1], c=np.arange(len(lstm_predicted_positions)), cmap='Blues', s = 3, label="Predicted")
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


