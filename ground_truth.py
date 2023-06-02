# import packages
import cv2
import os
import numpy as np;
import matplotlib.pyplot as plt
import natsort
    # import packages
import cv2
import os
import numpy as np;
import matplotlib.pyplot as plt
import shutil

def save_imgs(imgs, save_dir):
    channel=0
    for frame in range(0, imgs.shape[0]):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(rf"{save_dir}\{frame}.png", imgs[frame, channel, :, :])
        print(f"{'{0:.2f}'.format(frame/imgs.shape[0])} Progress: Saved {frame}/{imgs.shape[0]} images")

def get_blobs_adaptive(img, bound_size, min_brightness_const, min_area):
    new_img = np.full_like(img, 0)
    im_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(im_gauss,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bound_size,(min_brightness_const))
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_filtered = []
    
    for con in cont: 
        area = cv2.contourArea(con)
        if area>min_area:
            cont_filtered.append(con)    
    
    for c in cont_filtered:
        cv2.drawContours(new_img, [c], -1, 255,-1)

    return new_img, cont_filtered
def get_blob_img(video, frame):
    img = cv2.imread(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}\{frame}.png", cv2.IMREAD_GRAYSCALE)
    thres, blob_img = get_blobs_adaptive(img, bound_size=11, min_brightness_const=-5, min_area=10)
    return blob_img

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


SAVEOG = True
SAVEUNET = False
SAVEUNETMOVE = False
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']

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
                




if SAVEOG:
    for video in videos[3:]:
        imgs = np.load(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\data\imgs\{video}_crop.nd2.npy")
        # save_imgs(imgs, rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}")
        # print(f"Done Video {video}! Saved {imgs.shape[0]} images. ")
        print("Video {video} has {imgs.shape[0]} frames. Saving ground truth images...")
        for frame in range(imgs.shape[0]):
            img = cv2.imread(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\original\{video}\{frame}.png", cv2.IMREAD_GRAYSCALE)
            blob_img, conts = get_blobs_adaptive(img, bound_size=11, min_brightness_const=-5, min_area=10)
            try:
                plt.imsave(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\ground_truth\{video}\{frame}.png", blob_img, cmap="gray")
            except:
                os.makedirs(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\ground_truth\{video}")
                plt.imsave(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\images\ground_truth\{video}\{frame}.png", blob_img, cmap="gray")
            print("Saved frame {frame}")
        print(f"Done video {video}")

if SAVEUNETMOVE:
    data_dir = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm"
    train_split = 0.8
    valid_split = 0.9
    train_label=0
    valid_label=0
    test_label=0
    for video in videos:
        vid_originals = [file for file in os.listdir(rf"{data_dir}\images\original\{video}") if file.endswith(".png")] # get all original images for video
        vid_truth = [file for file in os.listdir(rf"{data_dir}\images\ground_truth\{video}") if file.endswith(".png")] # get all original images for video
        vid_originals.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # sort by number
        vid_truth.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # sort by number
        i=0
        for frame in range(len(vid_originals)):
            if i < train_split*len(vid_originals):
                save_dir = rf"{data_dir}\unet_data\original\train"
                save_dir_gt = rf"{data_dir}\unet_data\ground_truth\train"
                shutil.copyfile(rf"{data_dir}\images\original\{video}\{vid_originals[frame]}", rf"{save_dir}\{train_label}.png")
                shutil.copyfile(rf"{data_dir}\images\ground_truth\{video}\{vid_truth[frame]}", rf"{save_dir_gt}\{train_label}.png")
                train_label+=1
                print(f"Saved train image {train_label}. Progress = {'{0:.2f}'.format(i/(train_split*len(vid_originals)))}")
            elif i < valid_split*len(vid_originals):
                save_dir = rf"{data_dir}\unet_data\original\valid"
                save_dir_gt = rf"{data_dir}\unet_data\ground_truth\valid"
                shutil.copyfile(rf"{data_dir}\images\original\{video}\{vid_originals[frame]}", rf"{save_dir}\{valid_label}.png")
                shutil.copyfile(rf"{data_dir}\images\ground_truth\{video}\{vid_truth[frame]}", rf"{save_dir_gt}\{valid_label}.png")
                valid_label+=1
                print(f"Saved valid image {valid_label}. Progress = {'{0:.2f}'.format((i-train_split*len(vid_originals))/((valid_split-train_split)*len(vid_originals)))}")
            else:
                save_dir = rf"{data_dir}\unet_data\original\test"
                save_dir_gt = rf"{data_dir}\unet_data\ground_truth\test"
                shutil.copyfile(rf"{data_dir}\images\original\{video}\{vid_originals[frame]}", rf"{save_dir}\{test_label}.png")
                shutil.copyfile(rf"{data_dir}\images\ground_truth\{video}\{vid_truth[frame]}", rf"{save_dir_gt}\{test_label}.png")
                test_label+=1
                print(f"Saved test image {test_label}. Progress = {'{0:.2f}'.format((i-valid_split*len(vid_originals))/((1-valid_split)*len(vid_originals)))}")
            i+=1
            print(f"Done frame {frame}!.")
