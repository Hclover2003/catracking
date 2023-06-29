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
import cv2
import os
import numpy as np;
import matplotlib.pyplot as plt
import shutil


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
    return thresh, new_img


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
        self.neuron_positions_dct = {"AVA": np.ones((total_frames,2)), "AVB": np.ones((total_frames,2))}
        self.neuron_colours_dct = {"AVA": "red", "AVB": "blue"}
        self.neuron_positions_frame = None

        self.bound_size=11
        self.min_brightness=-5
        self.min_area=10

        self.rect = None
        self.start_x = None
        self.start_y = None


        # Create the main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Information
        self.information_frame = tk.Frame(main_frame, padx=40, pady=40)
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
            neuron_button = tk.Button(self.neuron_positions_frame, text=f"{neuron}: {neuron_current_position}", font=("Arial", 20), width=10, bg=self.neuron_colours_dct[neuron], 
                                      command=lambda neuron=neuron: self.set_selected_neuron(neuron))
            neuron_button.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - bound size Scale
        bound_size_slider = tk.Scale(self.information_frame, from_=5, to=50, orient=tk.HORIZONTAL, label="Bound Size", length=200, command=lambda value: self.set_bound_size(value))
        bound_size_slider.set(self.bound_size)
        bound_size_slider.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - min brightness Scale
        min_brightness_slider = tk.Scale(self.information_frame, from_=-30, to=10, orient=tk.HORIZONTAL, label="Min Brightness", length=200, command=lambda value: self.set_min_brightness(value))
        min_brightness_slider.set(self.min_brightness)
        min_brightness_slider.pack(side=tk.TOP, fill=tk.BOTH)

        # Information - min area Scale
        min_area_slider = tk.Scale(self.information_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Min Area", length=200, command=lambda value: self.set_min_area(value))
        min_area_slider.set(self.min_area)
        min_area_slider.pack(side=tk.TOP, fill=tk.BOTH)

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
        self.enlarged_image_canvas = tk.Canvas(self.stage, bg="black")
        self.enlarged_image_canvas.pack(side=tk.LEFT, fill=tk.NONE, expand=False)
        # self.enlarged_image_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.enlarged_mask_canvas= tk.Canvas(self.stage, bg="black")
        self.enlarged_mask_canvas.pack(side=tk.RIGHT, fill=tk.NONE, expand=False)
        self.enlarged_mask_canvas.bind("<ButtonPress-1>", self.on_mask_button_press)
        self.enlarged_mask_canvas.bind("<B1-Motion>", self.on_mask_move_press)
        self.enlarged_mask_canvas.bind("<ButtonRelease-1>", self.on_mask_button_release)

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

    def on_mask_button_press(self, event):
        self.start_x = self.enlarged_mask_canvas.canvasx(event.x)
        self.start_y = self.enlarged_mask_canvas.canvasy(event.y)
        self.rect = self.enlarged_mask_canvas.create_rectangle(self.x, self.y, 1, 1, outline="red")

    
    def on_mask_move_press(self, event):
        curX = self.enlarged_mask_canvas.canvasx(event.x)
        curY = self.enlarged_mask_canvas.canvasy(event.y)
        self.enlarged_mask_canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
    
    def on_mask_button_release(self, event):
        pass

    def set_min_area(self, value):
        """
        Set the min area
        """
        self.min_area = int(value)
        print(f"Min area: {self.min_area}")
        image_path = self.thumbnail_images[self.current_frame][0].filename
        self.display_enlarged_mask(image_path)

    def set_min_brightness(self, value):
        """
        Set the min brightness
        """
        self.min_brightness = int(value)
        print(f"Min brightness: {self.min_brightness}")
        image_path = self.thumbnail_images[self.current_frame][0].filename
        self.display_enlarged_mask(image_path)

    def set_bound_size(self, value):
        """
        Set the bound size
        """
        self.bound_size = int(value)
        print(f"Bound size: {self.bound_size}")
        image_path = self.thumbnail_images[self.current_frame][0].filename
        self.display_enlarged_mask(image_path)

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
            self.display_enlarged_mask(self.thumbnail_images[self.current_frame][0].filename)
            # Scroll right to next thumbnail
            thumbnail_x_coord = (thumbnail_size[0])*self.current_frame
            print(f"Thumbnail width: {thumbnail_x_coord}, enlarged_mask_canvas width: { self.thumbnails_canvas.winfo_width()}, frame width: { self.thumbnails_frame.winfo_width()}")
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
            self.display_enlarged_mask(self.thumbnail_images[self.current_frame][0].filename)
            # Scroll right to next thumbnail
            thumbnail_x_coord = (thumbnail_size[0])*self.current_frame
            print(f"Thumbnail width: {thumbnail_x_coord}, enlarged_mask_canvas width: { self.thumbnails_canvas.winfo_width()}, frame width: { self.thumbnails_frame.winfo_width()}")
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
        # Clear the enlarged_mask_canvas
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
            neuron_button = tk.Button(self.neuron_positions_frame, text=f"{neuron}: {neuron_current_position}", font=("Arial", 20), width=10, bg=self.neuron_colours_dct[neuron], activebackground="white", 
                                      command=lambda neuron=neuron: self.set_selected_neuron(neuron))
            neuron_button.pack(side=tk.TOP, fill=tk.BOTH)

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
                # Get mask image
                image = Image.open(image_path)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                thresh, mask_image = get_blobs_adaptive(img, bound_size=self.bound_size, min_brightness_const=self.min_brightness, min_area=self.min_area)
                mask_image = Image.fromarray(mask_image)

                # Get thumbnail image
                thumbnail = image.copy()
                thumbnail.thumbnail(thumbnail_size)
                thumbnail_tk = ImageTk.PhotoImage(thumbnail)

                # Create a label with the thumbnail image
                thumbnail_label = tk.Label(self.thumbnails_frame, image=thumbnail_tk)
                thumbnail_label.image = thumbnail_tk  # Store a reference to prevent garbage collection
                thumbnail_label.pack(side=tk.LEFT, padx=5)

                # Bind the label to display the enlarged image
                thumbnail_label.bind("<Button-1>", lambda event, img=image, img_path=image_path: self.on_thumbnail_click(img, img_path))

                # Add the image and its label to the list
                self.thumbnail_images.append((image, thumbnail_label))
                self.mask_images.append((mask_image, thumbnail_label))

    def on_thumbnail_click(self, image, mask_image):
        """
        Display the enlarged image when the thumbnail is clicked
        """
        # Display the enlarged image
        self.display_enlarged_image(image)
        # Display the mask image
        self.display_enlarged_mask(mask_image)

    def display_enlarged_image(self, image, mask_image = None):
        """
        Display the image in the enlarged image label while maintaining aspect ratio and fitting window height
        """
        image_width, image_height = image.size
        image_tk = ImageTk.PhotoImage(image)
        self.enlarged_image_canvas.config(width=image_width, height=image_height)
        self.enlarged_image_canvas.place()
        self.enlarged_image_canvas.create_image(image_width/2, image_height/2, image=image_tk, anchor=tk.CENTER)
        self.enlarged_image_canvas.image = image_tk


        image_frame = int(os.path.basename(image.filename).split('.')[0])
        self.set_current_frame(image_frame)
        self.update_neuron_position_labels(self.current_frame)
        self.plot_neuron_positions_for_frame(self.current_frame)
        print(f"Displaying image {image.filename}")
    
    def display_enlarged_mask(self, image_path):
        """
        Display the mask image
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        thresh, mask_image = get_blobs_adaptive(img, bound_size=self.bound_size, min_brightness_const=self.min_brightness, min_area=self.min_area)
        mask_image = Image.fromarray(mask_image)
        mask_width, mask_height = mask_image.size
        mask_image_tk = ImageTk.PhotoImage(mask_image)
        self.enlarged_mask_canvas.config(width=mask_width, height=mask_height)
        self.enlarged_mask_canvas.create_image(mask_width/2, mask_height/2, image=mask_image_tk, anchor=tk.CENTER)
        self.enlarged_mask_canvas.image = mask_image_tk



# Create the main window
root = tk.Tk()
root.title("Image Gallery")
root.geometry("1000x800")

# Create the image gallery
gallery = ImageGallery(root)

# Run the GUI main loop
root.mainloop()
