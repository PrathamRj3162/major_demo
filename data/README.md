# Dataset Info

## Chest X-Ray Pneumonia Dataset (Kaggle)

We used this dataset for training: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Original paper: Kermany et al., Cell, 2018

### Structure
```
chest_xray/
├── train/
│   ├── NORMAL/       (~1341 images)
│   └── PNEUMONIA/    (~3875 images)
├── val/
│   ├── NORMAL/       (8 images)
│   └── PNEUMONIA/    (8 images)
└── test/
    ├── NORMAL/       (234 images)
    └── PNEUMONIA/    (390 images)
```

Total around 5856 images. Pretty imbalanced towards pneumonia.

### How to use it

1. Download from Kaggle (either through the website or `kaggle datasets download`)
2. Extract to `data/chest_xray/`
3. Use the training script to fine-tune the model
4. Save weights to `backend/models/best_model.pth`

### Without training

The app still works fine without trained weights — it uses ImageNet pretrained DenseNet121 by default. Obviously the predictions won't be accurate without fine-tuning on chest xrays, but all the features (Grad-CAM, federated learning, UI) work properly for demonstration.

### Testing with random images

You can just search "chest xray" on Google, download any image, and upload it through the UI. The pipeline handles resizing and conversion automatically.

### Supported formats
PNG, JPEG, BMP, TIFF — any size, grayscale or RGB (gets converted to RGB internally)
