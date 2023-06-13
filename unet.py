if __name__ == '__main__':    
    
# import packages
    import os
    import cv2
    import numpy as np;
    import pandas as pd
    import random, tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import segmentation_models_pytorch as smp
    import albumentations as album
    import joblib
    import torchvision.transforms as transforms

    from scipy import ndimage
    from typing import Tuple, List
    from scipy import stats
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from torch.utils.data import DataLoader
    from caimages import *
    import segmentation_models_pytorch.utils.metrics
    import wandb
    import random
    import torch
    import time

    ## DEFINE UNET MODEL

    x_train_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\original\train"
    y_train_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\ground_truth\train"

    x_valid_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\original\valid"
    y_valid_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\ground_truth\valid"

    x_test_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\original\test"
    y_test_dir=r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\unet_data_excerpt\ground_truth\test"

    height, width = cv2.imread(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape[:2]

    # Get train and val dataset instances
    train_dataset = CaImagesDataset(
        x_train_dir, y_train_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )
    valid_dataset = CaImagesDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("Data loaders created.")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAINING = True
    NEW = True
    TESTING = True

    if TRAINING:
        lr = 0.001
        epochs = 2
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="unet-segmentation",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": epochs,
            }
        )
        if NEW:
            model = UNet()
        else:
            model = joblib.load(r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\unet\model8_full.pkl") # load model
        class_labels = {
        0: "background",
        1: "calcium",
        }

        print("Starting training...")
        start = time.time()
        alpha = 0.25
        gamma = 3
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        loss_list = [] # train and valid logs
        table = wandb.Table(columns=['ID', 'Image'])

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                print("Progress: {:.2%}".format(i/len(train_loader)))
                inputs, labels = data
                pred = model(inputs)
                bce_loss = criterion(pred, labels) # calculate loss (binary cross entropy)
                p_t = torch.exp(-bce_loss)
                focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
                loss = focal_loss.mean()
                loss.backward() # calculate gradients (backpropagation)
                optimizer.step() # update model weights (values for kernels)
                print(f"Step: {i}, Loss: {loss}")
                loss_list.append(loss)
                wandb.log({"loss": loss})

            for i, data in enumerate(valid_loader):
                valid_inputs, valid_labels = data
                valid_pred = model(valid_inputs)
                bce_loss = criterion(valid_pred, valid_labels) # calculate loss (binary cross entropy)
                p_t = torch.exp(-bce_loss)
                focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
                valid_loss = focal_loss.mean()
                mask_img = wandb.Image(valid_inputs[0].squeeze(0).numpy(), 
                                       masks = {
                                           "predictions" : {
                                "mask_data" : np.argmax(valid_pred[0].detach(), 0).numpy(),
                                "class_labels" : class_labels
                            },
                            "ground_truth" : {
                                "mask_data" : valid_labels[0][1].numpy(),
                                "class_labels" : class_labels
                            }}
                )
                table.add_data(f"Epoch {epoch} Step {i}", mask_img)
                wandb.log({"valid_loss": valid_loss})

            print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
            print(f"Time elapsed: {time.time() - start} seconds")
        print(f"Total time: {time.time() - start} seconds")
        wandb.log({"Table" : table})
        joblib.dump(model, r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\unet\model_1_full.pkl")
        wandb.finish()
        try:
            joblib.dump(loss_list, r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\unet\loss_list_1_full.pkl")
        except:
            print("Failed to save loss list")

    if TESTING:
        model = joblib.load(r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\unet\model5_full.pkl")
        image, gt_mask = train_dataset[2] # image and ground truth from test dataset
        print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        print(image)
        suffix = "_1"
        plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        plt.savefig(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results\unet\test{suffix}.png")
        plt.show()
        plt.imshow(gt_mask[1].numpy(), cmap="gray")
        plt.savefig(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results\unet\test_gt{suffix}.png")
        plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        print(pred_mask.shape)
        pred_mask_binary = np.argmax(pred_mask.squeeze(0).detach(), 0)
        plt.imshow(pred_mask_binary.numpy(), cmap="gray")
        plt.savefig(rf"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results\unet\test_pred{suffix}.png")
        plt.show()
    exit()

    LOAD = False # True if loading a model, False if creating a new model
    TRAINING = True # True if training, False if testing
    load_num = 3
    save_num = 3
    target_width = 454 # change to max width of images in dataset (make sure dividable by 2)
    target_height = 546 # change to max height of images in dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_folder = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\models\unet"

    img, mask = train_dataset[0]

    print(img.shape, mask.shape)
    # exit()
    #TRAIN MODEL
    if TRAINING:
        if LOAD:
            model = joblib.load(f'{model_folder}/model{load_num}.pkl')
        else:
            model = UNet()
        # loss = smp.utils.losses.DiceLoss()
        # metrics = [
        #     smp.utils.metrics.IoU(threshold=0.5),
        # ]
        # optimizer = torch.optim.Adam([ 
        #     dict(params=model.parameters(), lr=0.00008),
        # ])

        # # define data loaders for training and validation sets
        # train_epoch = smp.utils.train.TrainEpoch(
        #     model, 
        #     loss=loss, 
        #     metrics=metrics, 
        #     optimizer=optimizer,
        #     device=DEVICE,
        #     verbose=True,
        # )

        # valid_epoch = smp.utils.train.ValidEpoch(
        #     model, 
        #     loss=loss, 
        #     metrics=metrics, 
        #     device=DEVICE,
        #     verbose=True,
        # )
        
        print("Starting training...")
        EPOCHS = 1
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        train_logs_list, valid_logs_list = [], [] # train and valid logs

        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            j=0
            for img, mask in train_loader:
                print("Progress: {:.2%}".format(j/len(train_loader)))
                pred = model(img)
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()
                j+=1
            
            if i % 1 == 0:
                print(f"Epoch: {i}, Loss: {loss}")
            # train_logs = train_epoch.run(train_loader)
            # valid_logs = valid_epoch.run(valid_loader)
            # train_logs_list.append(train_logs)
            # valid_logs_list.append(valid_logs)
        
        joblib.dump(model, f'{model_folder}/model{save_num}.pkl')
    else:
        model = joblib.load(f'{model_folder}/model{load_num}.pkl')

    ## TEST MODEL
    test_dataset = CaImagesDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn=None),   
    )
    
    test_dataloader = DataLoader(test_dataset)

    random_idx = random.randint(0, len(test_dataset)-1)
    image, mask = test_dataset[random_idx]
    plt.imshow(image.squeeze().to_numpy(), cmap='gray')
    plt.show()
    plt.imshow(mask.squeeze().to_numpy(), cmap='gray')
    plt.show()


    sample_preds_folder = r"C:\Users\hozhang\Desktop\CaTracking\huayin_unet_lstm\results\unet"
    for idx in range(2):
        image, gt_mask = test_dataset[random_idx] # image and ground truth from test dataset
        print(gt_mask.shape)
        print(gt_mask) #CHW (2, H, W)
        image_vis = crop_image(np.transpose(test_dataset[random_idx][0].astype('uint8'), (1, 2, 0)), (target_height, target_width, 3)) # image for visualization
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        # Predict test image
        pred_mask = model(x_tensor)
        print(pred_mask.shape)
        print(pred_mask)


        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        print(pred_mask.shape)
        # Get prediction channel corresponding to calcium
        pred_calcium_heatmap = pred_mask[:,:,class_names.index('calcium')]
        print(pred_calcium_heatmap.shape)
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), class_rgb_values), (target_height, target_width, 1))
    
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask,(1,2,0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), class_rgb_values), (target_height, target_width, 3))

        print(pred_mask)
        print(pred_mask.shape)
        plt.imshow(pred_mask)
        plt.show()

        # cv2.imwrite(
        #     os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), 
        #     np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1]
        #     )  
        # visualize(
        #     original_image = image_vis,
        #     ground_truth_mask = gt_mask,
        #     predicted_mask = pred_mask,
        #     predicted_building_heatmap = pred_calcium_heatmap
        # )

    # EVALUATE MODEL
    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

        
        