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
import scipy.io

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
  shortened_sequences = []
  for i in range((length+n-1)//n):
    short_seq = lst[i*n: (i+1)*n]
    if len(short_seq) == n:
      shortened_sequences.append(short_seq)
  return shortened_sequences

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
    valid_input = torch.tensor(shortened_valid_sequences[0][:-1], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
    valid_actual = torch.tensor(shortened_valid_sequences[0][1:], dtype=torch.float32)
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
  with wandb.init(config=config):
    
    start = time.time()
    config = wandb.config
    
    # Initialize model, loss function, and optimizer
    if create_new_model:
      model = NeuralNetwork()
    else:
      model = joblib.load(os.path.join(model_dir, "lstm_model5e6.pkl"))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_num = 0
    
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
        # Save model every 10 epochs
        joblib.dump(model, os.path.join(model_dir, model_name, str(checkpoint_num)))
        checkpoint_num += 1
    print(f"Time: {time.time()-start}")
    wandb.finish()

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

# SET CONSTANTS
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
os.environ["PYTHONHASHSEED"] = str(0)
print("Seed set")

data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data"
video_dir = rf"{data_dir}\imgs"
position_dir = rf"{data_dir}\positions"
model_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\lstm"
img_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images"
results_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results"

# data_dir = "/Users/huayinluo/Desktop/code/catracking-1/data"
# video_dir = os.path.join(data_dir, "imgs")
# position_dir = os.path.join(data_dir, "positions")
# model_dir = "/Users/huayinluo/Desktop/code/catracking-1/models/lstm"
# img_dir = "/Users/huayinluo/Desktop/code/catracking-1/images"
# results_dir = "/Users/huayinluo/Desktop/code/catracking-1/results"

# Loop through videos and get positions
old_videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']
new_videos = ['11310', '11311', '11313', '11315', '11316', '11317_1', '11317_2', '11318', '11320_1', '11323', '11325_1', '11325_2', '11327_1', '11327_2', '11328_1']
videos = old_videos + new_videos
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
    positions_ava = np.load(os.path.join(position_dir, f"AVA_{video}.mat.npy"))
    positions_avb = np.load(os.path.join(position_dir, f"AVB_{video}.mat.npy"))
  all_neurons_positions = np.stack((positions_ava, positions_avb))
  positions_dct[video] = all_neurons_positions
  print(f"Loading {video}...")
print(f"Finished loading images and positions: {len(positions_dct)} positions")

# Original data test/train split (# Add 80% of each video to training set, 20% to testing set)
train_sequences = []
test_sequences = []
max_video_width = 500
max_video_height = 570
for video in videos:
  # Normalize positions
  all_positions = positions_dct[video]
  norm_positions = np.multiply(all_positions, [1/max_video_width, 1/max_video_height]) # Norm positions shape: [Num neurons = 2, Sequence length = 2571, Num coordinates = 2]
  split = math.floor(norm_positions.shape[1]*0.8)
  ava, avb = norm_positions
  train_sequences.append(ava[:split])
  train_sequences.append(avb[:split])
  test_sequences.append(ava[split:])
  test_sequences.append(avb[split:])

print("Test/Train split complete")


TRAIN = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONT_EXISTING_MODEL = True
if TRAIN:
  # Continue training existing model or create new model
  if CONT_EXISTING_MODEL:
    existing_model = "lstm_model_june23_1.pkl"
    model = joblib.load(os.path.join(model_dir, existing_model))
  else:
    existing_model = "N/A"
    model = NeuralNetwork()
    
  # Set parameters
  model_name = "lstm_model_june23_1.pkl"
  epochs = 1000
  lr = 0.000001
  sequence_length = 250
  frame_rate = 3
  batch_size = 8
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)


  # Slice sequence by frame rate
  train_sequences = [slice_sequence(seq, frame_rate) for seq in train_sequences]

  sequence_length = 250
  # Split sequences into desired sequence length
  shortened_train_sequences = []
  for sequence in train_sequences:
    shortened_train_sequences.extend(split_lst(sequence, sequence_length))
  shortened_train_sequences = np.concatenate(shortened_train_sequences)

  shortened_test_sequences = []
  for sequence in test_sequences:
    shortened_test_sequences.extend(split_lst(sequence, sequence_length))
  shortened_test_sequences = np.concatenate(shortened_test_sequences)
  num_batches = len(shortened_train_sequences)//batch_size
  print(f"Number of Sequences = {len(shortened_train_sequences)}, Batch size = {batch_size}, Num batches: {num_batches}")

  # Record wandb hyperparameters
  wandb.init(
      project="lstm",
      config={
      "existing_model": "lstm_model5.pkl",
      "model_name": model_name,
      "sequence_length": sequence_length,
      "frame_rate": frame_rate,
      "batch_size": batch_size,
      "epochs": epochs,
      "learning_rate": lr,
      "architecture": "lstm",
      "dataset": ",".join(videos),
      }
  )

  table = wandb.Table(columns=["Epoch", "Image"])
  
  start_time = time.time()
  for epoch in range(epochs):
    # Train model
    for j in range(num_batches):
      train_input = torch.tensor(shortened_train_sequences[j*batch_size:(j+1)*batch_size][:-1], dtype=torch.float32) # Shape: [N: batch size (8), L: sequence length (100), H: input dimension (2))]
      pred = model(train_input)
      actual = torch.tensor(shortened_train_sequences[j*batch_size:(j+1)*batch_size][1:], dtype=torch.float32)
      loss = criterion(pred, actual)
      loss.backward()
      optimizer.step()
    # Log train and valid loss every 10 epochs
    if epoch % 10 == 0:
      wandb.log({"loss": loss})
      test_input = torch.tensor(shortened_test_sequences[0:batch_size][:-1], dtype=torch.float32)
      test_actual = torch.tensor(shortened_test_sequences[0:batch_size][1:], dtype=torch.float32)
      test_pred = model(test_input)
      valid_loss = criterion(test_pred, test_actual)
      wandb.log({"valid_loss": valid_loss})
      print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
    # Plot valid prediction and save model every 100 epochs
    if epoch % 100 == 0:
      colours = np.arange(test_pred.detach().shape[0])
      predx = test_pred.detach()[:,0]
      predy = test_pred.detach()[:,1]
      actx = test_actual.detach()[:,0]
      acty = test_actual.detach()[:,1]
      fig = plt.figure()
      plt.scatter(test_input.detach()[0,0], test_input.detach()[0,1], label="Input")
      plt.scatter(predx, predy, c = colours, cmap="Greens", label="Predicted") #1st sequence in batch
      plt.colorbar()
      plt.scatter(actx, acty, c = colours, cmap = "Oranges", label="Actual") # 1st sequence in batch
      plt.colorbar()
      plt.legend()
      table.add_data(epoch, wandb.Image(fig))
      plt.close()
      joblib.dump(model, os.path.join(model_dir, model_name))
      print("Saved model")
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
  # height, width = imgs_dct[video].shape[2:] # dimensions to normalize positions with
  crop_size = 20
  
  dst_weight = 10 # Distance score is from [0, infinity]. Closer to 0 is better
  hst_weight = 100 # Color Histogram score is from [0, 1]. Closer to 1 is better.
  area_weight = 0
  save_dir = os.path.join(results_dir, f"model{model_name}_{video}_{start_index}start_{end_frame}frames_RESET{RESET}{max_reset}_{dst_weight}dst{hst_weight}color{area_weight}area")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
  # actual positions and chosen positions
  actual_positions = np.array(positions_dct[video][start_frame:end_frame+1]) # full actual sequence of video 11408 (use as label y)
  actual_positions_norm = np.multiply(actual_positions, [1/max_video_width, 1/max_video_height])
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
  
  
SWEEP = True
if SWEEP:
  print("Starting sweep...")
  # Sweep config
  parameters_dct = {
  "sequence_length": {"values": [10, 25, 50, 100, 200, 250]},
  "frame_rate":{"values": [1, 3, 5, 10]},
  "num_predict_coords": {"values": [1, 2, 3, 4, 5, 10]},
  "batch_size": {"values": [8, 16, 32, 64]},
  "learning_rate": {"max": 0.001, "min": 0.00001},
  "epochs": {"values": [50, 100, 200, 500, 1000, 2000]}
  }

  # Set constant parameters
  parameters_dct.update({
  "sequence_length": {"value": 100},
  # "frame_rate":{"value": 1},
  "num_predict_coords": {"value": 1},
  "batch_size": {"value": 16},
  "learning_rate": {"value": 0.0001},
  "epochs": {"value": 1000}
  })

  sweep_config = {
    "method": "grid",
    "name": "sweep",
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
                                               model_name="lstm_predict-A",
                                               create_new_model=False,
                                               ))