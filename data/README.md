# 📊 Sample Dataset Usage Guide

## Chest X-Ray Images (Pneumonia) Dataset

### Source
- **Kaggle**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Original Paper**: Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning (Kermany et al., Cell, 2018)

### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/       (1,341 images)
│   └── PNEUMONIA/    (3,875 images)
├── val/
│   ├── NORMAL/       (8 images)
│   └── PNEUMONIA/    (8 images)
└── test/
    ├── NORMAL/       (234 images)
    └── PNEUMONIA/    (390 images)
```

### Total: 5,856 images
- **Normal**: 1,583 images
- **Pneumonia**: 4,273 images (includes both bacterial and viral)

### Image Details
- Format: JPEG
- Size: Variable (typically 1000-2000px)
- Modality: Anterior-Posterior chest X-rays
- Source: Guangzhou Women and Children's Medical Center

---

## How to Use

### 1. Download Dataset
```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# Option B: Manual download from Kaggle website
# Go to the link above, download, and extract to data/
```

### 2. Train the Model
Create a training script (`train.py`) or use a Jupyter notebook:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from backend.models.densenet_model import PneumoniaNet

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('data/chest_xray/train', transform=transform)
test_dataset = datasets.ImageFolder('data/chest_xray/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model
model = PneumoniaNet(num_classes=2, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.densenet.classifier.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), 'backend/models/best_model.pth')
```

### 3. Using Without Training
The system works out-of-the-box with ImageNet pretrained weights. The prediction accuracy won't be medically relevant without fine-tuning, but all features (Grad-CAM, federated learning, UI) work perfectly for demonstration purposes.

### 4. Quick Test
You can use any chest X-ray image from the internet for testing:
- Search "normal chest xray" or "pneumonia chest xray" on Google Images
- Download a sample and upload it through the UI
- The system will process it through the full pipeline

---

## Image Requirements
- **Format**: PNG, JPEG, BMP, TIFF
- **Content**: Chest X-ray (anterior-posterior view recommended)
- **Size**: Any size (automatically resized to 224×224)
- **Mode**: Grayscale or RGB (automatically converted to RGB)
