# import packages
import cv2
import os
import numpy as np;
import matplotlib.pyplot as plt
import shutil
# import natsort

def save_imgs(imgs, save_dir):
    channel=0
    for frame in range(0, imgs.shape[0]):
        plt.imsave(f"{save_dir}/{frame}.png", imgs[frame, channel, :, :])
        print(f"{'{0:.2f}'.format(frame/imgs.shape[0])} Progress: Saved {frame}/{imgs.shape[0]} images")
    print("done")

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

def get_blobs(img, min_brightness=45, min_area=20):
    """ Returns image with identified blobs

    Args:
        img (array of shape (height, width)): grayscale image
        min_brightness: minimum color to count as blob (0 = black, 255 = white)
        min_area: minimum area to count as blob

    Returns:
        array of shape (height, width): black/white image of background/blob
    """
    # Remove noise
    im_gauss = cv2.GaussianBlur(img, (5, 5), 0) # "smoothing" the image with Gaussian Blur
    
    # Threshold image
    threshold = min_brightness # chosen by trial/error
    ret, thresh = cv2.threshold(im_gauss, threshold, 255, 0) # get threshold image
    
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
    return new_img

def save_og_ground(save_dir, frame, label, save_x_dir, save_y_dir):
    """ Saves original image and ground truth image """
    img = cv2.imread(f"{save_dir}/{frame}", cv2.IMREAD_GRAYSCALE) # read as grayscale
    thres, blob_img = get_blobs_adaptive(img, bound_size=11, min_brightness_const=-5, min_area=10) # get segmented image (set parameters)
    plt.imsave(f"{save_x_dir}/{label}.png", img) # save original image (feature)
    # plt.imsave(f"{save_y_dir}/{label}.png", blob_img, cmap="gray") # save segmented image (label)

skip = 27
save_dir = r'C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\ground_truth\valid'
load_dir = r'C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data\ground_truth\valid'
files = [f for f in os.listdir(load_dir) if os.path.isfile(os.path.join(load_dir, f))]
selected_files = [files[i] for i in range(0, len(files), skip)]
print(len(selected_files))
for file_name in selected_files:
    shutil.copy(f'{load_dir}\{file_name}', f'{save_dir}\{file_name}')
    print(f"Saved {file_name} to {save_dir} from {load_dir}")

exit()
# file_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lst" # change to your directory
video_num = "11409" # change to video num 
videos = ['11409', "11410", '11411', '11413', '11414', '11415', '11433', '11434']
channel = 0 # for now, only look at first channel

blob=False # Set to True to create ground truth
save_og=False # Set to True to save original colour images (from numpy array)
make_dir=False # Set to True to create directories for saving images
rename=False # Set to True to rename files
traintestvalid = True # Set to True to create train/test/valid directories

if rename:
    last_num = -2571 #len(imgs) +1 # change to last number of files in previous directory
    # Rename files
    video = "11409" # change to video you want to rename
    files = natsort.natsorted([file for file in os.listdir(f"./images/ground_truth/{video}")], reverse=False)
    for file in files:
        os.rename(f"./images/ground_truth/{video}/{file}", f"./images/ground_truth/{video}/{int(file.split('.')[0])+last_num}.png")
        print(f"Renamed {file} to {int(file.split('.')[0])+last_num}.png")
    print("Done Renaming!")

# create directory if it doesn't exist
if make_dir:
    os.mkdir(f"images/thres_adaptive/{video_num}")
    os.mkdir(f"images/ground_truth/{video_num}")
    os.mkdir(f"images/original/{video_num}")
       
if save_og:
    imgs = np.load(f"./data/imgs/{video_num}_crop.nd2.npy")
    save_imgs(imgs, video_num) # replace with video you want to save
    
# Creating ground truth
if blob:
    imgs = np.load(f"./data/imgs/{video_num}_crop.nd2.npy")
    ava = np.load(f"./data/positions/AVA_{video_num}.mat.npy")
    avb = np.load(f"./data/positions/AVB_{video_num}.mat.npy")

    start_frame = 370
    end_frame = 500 #imgs.shape[0] # set to imgs.shape[0] to run through entire video
    channel = 0 # for now, only look at first channel
    min_brightness=-5
    min_area=10
    bound_size=11 # Must be odd number
    for frame in range(start_frame, end_frame):
        img = cv2.imread(f"./images/original/{video_num}/{frame}.png", cv2.IMREAD_GRAYSCALE) # read as grayscale
        thresh, blob_img = get_blobs_adaptive(img, bound_size=bound_size, min_brightness_const=min_brightness, min_area=min_area) # get segmented image (set parameters)
        

        plt.imsave(f"images/thres_adaptive/{video_num}/{frame}.png", thresh, cmap="gray") # save as black/white
        plt.imsave(f"images/ground_truth/{video_num}/{frame}.png", blob_img, cmap="gray") # save as black/white
        print(f"{'{0:.2f}'.format((frame-start_frame)/(end_frame-start_frame))} Progress: Saved {frame-start_frame}/{end_frame-start_frame} images")
    print(f"Done Video {video_num}! min brightness: {min_brightness}, min area: {min_area}, bound size: {bound_size}")

if traintestvalid:
    data_dir = f"/Users/huayinluo/Desktop/code/CaTracking"
    
    x_train_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/train" # original images
    y_train_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/train" # segmented images
    
    x_valid_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/valid"
    y_valid_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/valid"
    
    x_test_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/original/test"
    y_test_dir=f"/Users/huayinluo/Desktop/code/catracking-1/unet_data/ground_truth/test"
    
    train_split = 0.8
    valid_split = 0.9
    
    videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415', '11433', '11434']
    
    train_i = 0
    valid_i = 0
    test_i = 0
    for video in videos:
        imgs = np.load(f"{data_dir}/data/imgs/{video}_crop.nd2.npy")
        vid_originals = [file for file in os.listdir(f"{data_dir}/original/{video}") if file.endswith(".png")] # get all original images for video
        for i in range(imgs.shape[0]):
            if i < train_split*imgs.shape[0]:
                plt.imsave(f"{x_train_dir}/{train_i}.png", imgs[i, 0, :, :])
                img = cv2.imread(f"{save_dir}/{frame}", cv2.IMREAD_GRAYSCALE) # read as grayscale
                thres, blob_img = get_blobs_adaptive(img, bound_size=11, min_brightness_const=-5, min_area=10) # get segmented image (set parameters)
                plt.imsave(f"{save_x_dir}/{label}.png", img) # save original image (feature)
                # plt.imsave(f"{save_y_dir}/{label}.png", blob_img, cmap="gray") # save segmented image (label)
                train_i+=1
            elif i < valid_split*imgs.shape[0]:
                plt.imsave(f"{x_train_dir}/{valid_i}.png", imgs[i, 0, :, :])
                valid_i+=1
            else:
                plt.imsave(f"{x_train_dir}/{test_i}.png", imgs[i, 0, :, :])
                test_i+=1
            # save_og_ground(f"{data_dir}/original/{video}", img, label, x_save_dir, y_save_dir)
            print(f"Saved image {i} for video {video}")
        print(f"Done video {video}!")
    print("Done all videos!")
                

    