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

SAVEOG = True
SAVEUNET = False
SAVEUNETMOVE = False
videos = ['11408', '11409', "11410", '11411', '11413', '11414', '11415']


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
