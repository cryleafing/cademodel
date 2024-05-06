#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch
from torch import nn, optim
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# get the needed imports for preprocessing


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")


# In[1]:


# custom isic dataset class !!
class ISICDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.labels = self.annotations.columns[1:]
        self.annotations['target'] = self.annotations[self.labels].idxmax(axis=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.annotations.iloc[idx, -1])

        if self.transform:
            image = self.transform(image)

        return image, label


# In[ ]:


# image transformations, should resize images to the same 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[ ]:


# file paths
root_dir = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/SIC2018_Task3_Training_Input/ISIC2018_Task3_Training_Input"
csv_file = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"


# In[ ]:


# load pre-trained ResNet50 model and modify the final layer
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Change to 7 classes

# move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model


# In[ ]:


# paths to training, validation, and testing datasets, along with the ground truth csv
train_root = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_Input"
train_csv = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"

val_root = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_Input"
val_csv = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"

test_root = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Test_Input/ISIC2018_Task3_Test_Input"
test_csv = "OneDrive/Documents/Y3 CS/Artificial Intelligence/CADe Archive/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv"

# create dataset instances
train_dataset = ISICDataset(csv_file=train_csv, root_dir=train_root, transform=train_transforms)
val_dataset = ISICDataset(csv_file=val_csv, root_dir=val_root, transform=test_transforms)
test_dataset = ISICDataset(csv_file=test_csv, root_dir=test_root, transform=test_transforms)

# data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# dataloaders dictionary
dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}


# In[ ]:


# Train the model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)


# In[ ]:




