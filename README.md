
# File Descriptions

Files
- `README.md`: This file.
- `ground_truth.py`: This file contains code for UNet ground truth generation
- `caimages.py` : This file contains function & class definitions for UNet model
- `unet.py`: This file contains code for UNet training and testing
- `lstm.py`: This file contains code for LSTM training and testing

Folders
- `data/`: This folder contains the raw data, including numpy array of video and labelled position of neurons AVA and AVB (imgs/video, positions/video)
- `images/`: This folder contains the images (ground_truth/video and original/video)
- `unet_data/`: This folder contains the data used for training and testing UNet, which is a selection of images/ in train test split
- `models/`: This folder contains the trained models (lstm and unet)
- `results/`: This folder contains prediction results and visualizations

# How to train
LSTM.py
* Run `python lstm.py` to train the LSTM model. The model will be saved in `models/` folder.
* Hyperparameters:
  * `--seq_len`: sequence length of input sequences
  * `--epochs`: number of epochs to train
  * `--batch_size`: batch size
  * `--lr`: learning rate
