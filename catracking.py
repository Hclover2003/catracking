import os
import tkinter as tk
from PIL import Image, ImageTk
import re
import numpy as np
import joblib
import torch
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

    def forward(self,x):
        output,_status = self.lstm(x)
        return output

# Directory containing the images
image_dir = r"images/original/dummy_data"
mask_dir = r"images/ground_truth/dummy_data"
total_frames = 50
thumbnail_size = (100, 100)
model_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\lstm"
img_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images"

model_name = "lstm_model0.pkl"
model = joblib.load(os.path.join(model_dir, model_name))
video_img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, f"original/11408/1.png")), cv2.COLOR_BGR2GRAY)
video_height, video_width = video_img.shape
print(f"Video size: {video_width} x {video_height}")


class ImageGallery:
    """
    A simple image gallery that displays a list of images in a directory
    """
    def __init__(self, root):
        """
        Create the image gallery
        """
        self.root = root
        self.gallery_frame = None
        self.thumbnails_frame = None
        self.thumbnails_canvas = None

        self.stage = None
        self.enlarged_image_canvas = None
        self.enlarged_mask_canvas = None
        self.thumbnail_images = []
        self.mask_images = []
        self.tracking = False

        self.information_frame = None
        self.frame_label = None
        self.current_frame = 0
        self.selected_neuron = "AVA"
        self.selected_neuron_label = None
        self.neuron_positions_dct = {"AVA": np.ones((total_frames,2)), "AVB": np.ones((total_frames,2))}
        self.neuron_colours_dct = {"AVA": "red", "AVB": "blue"}
        self.neuron_positions_frame = None

        # Create the main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Information
        self.information_frame = tk.Frame(main_frame, padx=20)
        self.information_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Information - title Label
        title = tk.Label(self.information_frame, text="Ca Tracking", font=("Arial", 60))
        title.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - current frame Label
        self.frame_label = tk.Label(self.information_frame, text=f"Frame {self.current_frame}", font=("Arial", 20))
        self.frame_label.pack(side=tk.TOP, fill = tk.BOTH)

        # Information - selected neuron Label
        self.selected_neuron_label = tk.Label(self.information_frame, text=f"Selected Neuron: {self.selected_neuron}", font=("Arial", 20))
        self.selected_neuron_label.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - neuron positions Label for multiple selected neurons
        self.neuron_positions_frame = tk.Frame(self.information_frame)
        self.neuron_positions_frame.pack(side=tk.TOP, fill=tk.BOTH)
        for neuron in self.neuron_positions_dct.keys():
            neuron_positions = self.neuron_positions_dct[neuron]
            neuron_current_position = neuron_positions[self.current_frame]
            neuron_button = tk.Button(self.neuron_positions_frame, text=f"{neuron}", font=("Arial", 20), width=10, bg=self.neuron_colours_dct[neuron], 
                                      command=lambda neuron=neuron: self.set_selected_neuron(neuron))
            neuron_button.pack(side=tk.TOP, fill=tk.BOTH)
            neuron_label = tk.Label(self.neuron_positions_frame, text=f"{neuron}: {neuron_current_position}", font=("Arial", 20))
            neuron_label.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - start tracking Button
        start_tracking_button = tk.Button(self.information_frame, text="Start Tracking", font=("Arial", 20))
        start_tracking_button.pack(side=tk.TOP, fill=tk.BOTH)
        start_tracking_button.bind("<Button-1>", lambda event: self.track_neuron(self.selected_neuron))

        # Information - stop tracking Button
        stop_tracking_button = tk.Button(self.information_frame, text="Stop Tracking", font=("Arial", 20))
        stop_tracking_button.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - save positions Button
        save_positions_button = tk.Button(self.information_frame, text="Save Positions", font=("Arial", 20))
        save_positions_button.pack(side=tk.TOP, fill=tk.BOTH)
        save_positions_button.bind("<Button-1>", lambda event: self.stop_tracking() )

        # Gallery
        self.gallery_frame = tk.Frame(main_frame)
        self.gallery_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Gallery - enlarged image Frame
        self.stage = tk.Frame(self.gallery_frame, bg="black")
        self.stage.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Enlarged Image Frame - enlarged image Canvas
        self.enlarged_image_canvas = tk.Canvas(self.stage, bg="red")
        self.enlarged_image_canvas.pack(side=tk.LEFT, fill=tk.NONE, expand=False)
        # self.enlarged_image_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.enlarged_mask_canvas= tk.Canvas(self.stage, bg="blue")
        self.enlarged_mask_canvas.pack(side=tk.RIGHT, fill=tk.NONE, expand=False)
        # self.enlarged_mask_canvas.place(relx=0.9, rely=0.9, anchor=tk.CENTER)

        # Enlarged Image Canvas - plot neuron positions and bind mouse click event      
        self.enlarged_image_canvas.bind('<Button 1>', lambda event: self.set_neuron_position_for_frame(self.selected_neuron, np.array((event.x, event.y)), self.current_frame))
        self.plot_neuron_positions_for_frame(self.current_frame)

        # Gallery - thumbnails Canvas
        self.thumbnails_canvas = tk.Canvas(self.gallery_frame, height=100, width=1000)
        self.thumbnails_canvas.pack(side=tk.BOTTOM, fill=tk.X)

        # Thumbnails Canvas - scrollbar
        scrollbar = tk.Scrollbar(self.gallery_frame, orient=tk.HORIZONTAL, command=self.thumbnails_canvas.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.thumbnails_canvas.configure(xscrollcommand=scrollbar.set)
        self.thumbnails_canvas.bind('<Configure>', lambda event: self.thumbnails_canvas.configure(scrollregion=self.thumbnails_canvas.bbox('all')))

        # Thumbnails Canvas - arrow keys scroll
        root.bind('<Right>', lambda event: self.next_frame())
        root.bind('<Left>', lambda event: self.prev_frame())

        # Thumbnails Canvas - thumbnails frames
        self.thumbnails_frame = tk.Frame(self.thumbnails_canvas)
        self.thumbnails_canvas.create_window((0, 0), window=self.thumbnails_frame, anchor='nw')


        # Load and display the images
        self.load_thumbnail_images()
        print("Hello")

    def stop_tracking(self):
        """
        Stop tracking the neuron
        """
        self.tracking = False
        print("Tracking stopped")
        
    def start_tracking(self):
        """
        Start tracking the neuron
        """
        self.tracking = True
        print("Tracking started")
        self.track_neuron(self.selected_neuron)

    def track_neuron(self, neuron):
        """
        Track the neuron
        """
        print(f"Tracking neuron {neuron}")
        for frame in range(self.current_frame, total_frames):
            if not self.tracking:
                print("Tracking stopped")
                break
            input_positions = self.neuron_positions_dct["AVA"][:frame]
            input_positions_norm = np.multiply(input_positions, [1/video_width, 1/video_height])
            input_positions_norm = torch.tensor(input_positions_norm, dtype=torch.float32)
            pred_positions_norm = model(input_positions_norm)
            pred_positions = np.multiply(pred_positions_norm.detach(), [video_width, video_height])
            pred_position_for_frame = pred_positions[-1]
            print(f"Input positions: {input_positions}")
            print(f"Predicted point: {pred_position_for_frame}")
            self.set_neuron_position_for_frame(neuron, pred_position_for_frame, frame)
            print(f"Done predicting frame {frame}: {pred_position_for_frame}")
        print("Done tracking")


    def next_frame(self):
        """
        Display the next frame
        """
        if (self.current_frame + 1 < total_frames):
            print(f"Switching from frame {self.current_frame} to frame {self.current_frame + 1}")
            self.current_frame += 1
            self.set_current_frame(self.current_frame)
            self.update_neuron_position_labels(self.current_frame)
            self.plot_neuron_positions_for_frame(self.current_frame)
            self.display_enlarged_image(self.thumbnail_images[self.current_frame][0], self.mask_images[self.current_frame][0])
            # Scroll right to next thumbnail
            thumbnail_x_coord = (thumbnail_size[0])*self.current_frame
            print(f"Thumbnail width: {thumbnail_x_coord}, canvas width: { self.thumbnails_canvas.winfo_width()}, frame width: { self.thumbnails_frame.winfo_width()}")
            if thumbnail_x_coord > self.thumbnails_canvas.winfo_width():
                self.thumbnails_canvas.xview_moveto((thumbnail_x_coord/self.thumbnails_frame.winfo_width())-10)
            
    def prev_frame(self):
        """
        Display the next frame
        """
        if (self.current_frame > 0):
            print(f"Switching from frame {self.current_frame} to frame {self.current_frame - 1}")
            self.current_frame += -1
            self.set_current_frame(self.current_frame)
            self.update_neuron_position_labels(self.current_frame)
            self.plot_neuron_positions_for_frame(self.current_frame)
            self.display_enlarged_image(self.thumbnail_images[self.current_frame][0], self.mask_images[self.current_frame][0])
            # Scroll right to next thumbnail
            thumbnail_x_coord = (thumbnail_size[0])*self.current_frame
            print(f"Thumbnail width: {thumbnail_x_coord}, canvas width: { self.thumbnails_canvas.winfo_width()}, frame width: { self.thumbnails_frame.winfo_width()}")
            if thumbnail_x_coord > self.thumbnails_canvas.winfo_width():
                self.thumbnails_canvas.xview_moveto(thumbnail_x_coord/self.thumbnails_frame.winfo_width())

    def set_selected_neuron(self, neuron):
        self.selected_neuron = neuron
        self.selected_neuron_label.config(text=f"Selected Neuron: {self.selected_neuron}")
        print(f"Selected neuron: {self.selected_neuron}")

    def plot_neuron_positions_for_frame(self, frame):
        """
        Plot the neuron positions for the current frame
        """
        # Clear the canvas
        self.enlarged_image_canvas.delete("neuron_position")

        # Plot the neuron positions
        colours = ["red", "blue"]
        for neuron in self.neuron_positions_dct.keys():
            x, y = self.neuron_positions_dct[neuron][frame]
            colour = self.neuron_colours_dct[neuron]
            print(f"Plotting {neuron} at ({x}, {y}) with colour {colour}")
            self.enlarged_image_canvas.create_oval(x,y,x+1, y+1,fill=colour, outline=colour, width=5, tags="neuron_position")

    def clear_frame(self, frame):
        """
        Clear the frame label
        """
        for widgets in frame.winfo_children():
            widgets.destroy()

    def set_current_frame(self, frame):
        """
        Update the frame label
        """
        self.current_frame = frame
        self.frame_label.config(text=f"Frame {frame}")

    def update_neuron_position_labels(self, frame):
        # Clear the frame
        self.clear_frame(self.neuron_positions_frame)

        for neuron in self.neuron_positions_dct.keys():
            neuron_positions = self.neuron_positions_dct[neuron]
            neuron_current_position = neuron_positions[self.current_frame]
            neuron_button = tk.Button(self.neuron_positions_frame, text=f"{neuron}", font=("Arial", 20), width=10, bg=self.neuron_colours_dct[neuron], activebackground="white", 
                                      command=lambda neuron=neuron: self.set_selected_neuron(neuron))
            neuron_button.pack(side=tk.TOP, fill=tk.BOTH)
            neuron_label = tk.Label(self.neuron_positions_frame, text=f"{neuron}: {neuron_current_position}", font=("Arial", 20))
            neuron_label.pack(side=tk.TOP, fill=tk.BOTH)

    def set_neuron_position_for_frame(self, neuron, neuron_position, frame):
        """
        Update the neuron position for neuron at current frame
        """
        print(f"Neuron {neuron} at frame {frame} has position {neuron_position}")

        # Update dictionary
        self.neuron_positions_dct[neuron][frame] = neuron_position
        # Show the updated neuron positions
        self.update_neuron_position_labels(frame)
        self.plot_neuron_positions_for_frame(frame)

    def load_thumbnail_images(self):
        """
        Load the images from the directory and display them in the thumbnails frame
        """
        # Get the list of image files in the directory
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        image_files.sort(key=lambda f: int(re.sub('\D', '', f)))

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)
            if os.path.isfile(image_path):
                # Load the image and create a thumbnail
                frame_num = image_file.split('.')[0]
                mask_image = Image.open(mask_path)
                image = Image.open(image_path)
                thumbnail = image.copy()
                thumbnail.thumbnail(thumbnail_size)
                thumbnail_tk = ImageTk.PhotoImage(thumbnail)

                # Create a label with the thumbnail image
                thumbnail_label = tk.Label(self.thumbnails_frame, image=thumbnail_tk)
                thumbnail_label.image = thumbnail_tk  # Store a reference to prevent garbage collection
                thumbnail_label.pack(side=tk.LEFT, padx=5)

                # Bind the label to display the enlarged image
                thumbnail_label.bind("<Button-1>", lambda event, img=image, mask=mask_image: self.display_enlarged_image(img))

                # Add the image and its label to the list
                self.thumbnail_images.append((image, thumbnail_label))
                self.mask_images.append((mask_image, thumbnail_label))

    def display_enlarged_image(self, image, mask_image = None):
        """
        Display the image in the enlarged image label while maintaining aspect ratio and fitting window height
        """
        # Get the size of the window
        window_width = self.enlarged_image_canvas.winfo_width()
        window_height = self.enlarged_image_canvas.winfo_height()

        # Get the size of the image
        image_width, image_height = image.size
        mask_width, mask_height = mask_image.size
        print(f"Mask size: {mask_width} x {mask_height}")
        image_tk = ImageTk.PhotoImage(image)
        mask_image_tk = ImageTk.PhotoImage(mask_image)
        

        # # Calculate the aspect ratio of the image
        # aspect_ratio = image_width / image_height

        # # Calculate the new width and height to fit the window height while maintaining aspect ratio
        # new_height = window_height
        # new_width = int(new_height * aspect_ratio)

        # # Resize the image with the calculated size
        # resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # # Convert the resized image to Tkinter-compatible format
        # resized_image_tk = ImageTk.PhotoImage(resized_image)
        print(f"Image size: {image_width} x {image_height}")
        print(f"Original canvas size: {self.enlarged_image_canvas.winfo_width()} x {self.enlarged_image_canvas.winfo_height()}")
        self.enlarged_image_canvas.config(width=image_width, height=image_height)
        self.enlarged_image_canvas.place()
        print(f"Updated canvas size: {self.enlarged_image_canvas.winfo_width()} x {self.enlarged_image_canvas.winfo_height()}")
        # Update the enlarged image label
        self.enlarged_image_canvas.create_image(window_width/2, window_height/2, image=image_tk, anchor=tk.CENTER)
        self.enlarged_image_canvas.image = image_tk

        self.enlarged_mask_canvas.create_image(window_width/2, window_height/2, image=mask_image_tk, anchor=tk.CENTER)
        self.enlarged_mask_canvas.image = mask_image_tk

        image_frame = int(os.path.basename(image.filename).split('.')[0])
        self.set_current_frame(image_frame)
        self.update_neuron_position_labels(self.current_frame)
        self.plot_neuron_positions_for_frame(self.current_frame)


# Create the main window
root = tk.Tk()
root.title("Image Gallery")
root.geometry("1000x800")

# Create the image gallery
gallery = ImageGallery(root)

# Run the GUI main loop
root.mainloop()
