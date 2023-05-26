
# File Structure

.
├── data (numpy video and position)/
│   ├── imgs
│   └── positions
├── images (original microscope images)/
│   ├── original
│   └── ground_truth
├── models (trained ML models)/
│   ├── lstm
│   └── unet
├── visualizations(visualizations of predictions + performance)/
│   ├── lstm
│   └── unet
├── results (saved prediction data)/
│   ├── lstm/
│   │   ├── 11408 (each try)
│   │   └── 11409
│   └── unet
└── unet_data (images split into train/valid/test)/
    ├── original/
    │   ├── train
    │   ├── valid
    │   └── test
    └── ground_truth/
        ├── train
        ├── valid
        └── test