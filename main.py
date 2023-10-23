# %%
DATA_PATH = "C:\\Users\\USER\\Downloads\\dog-and-cat\\dog-and-cat"
# %%
import os
from os import makedirs, listdir
from shutil import copyfile
from random import seed, random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# %%
# check data
train_folder = f"{DATA_PATH}\\training_set"
test_folder = f"{DATA_PATH}\\testing_set"
train_dataset = ImageFolder(train_folder)
test_dataset = ImageFolder(test_folder)
# %%
# check data
train_file_names = []
train_labels = []
test_file_names = []
test_labels = []

for label in os.listdir(train_folder):
    label_folder = os.path.join(train_folder, label)
    if os.path.isdir(label_folder):
        for file in os.listdir(label_folder):
            if file.endswith('.jpg'):
                train_file_names.append(file)
                train_labels.append(file[0:3])

for label in os.listdir(test_folder):
    label_folder = os.path.join(test_folder, label)
    if os.path.isdir(label_folder):
        for file in os.listdir(label_folder):
            if file.endswith('.jpg'):
                test_file_names.append(file)
                test_labels.append(file[0:3])

training_data = {'id': train_file_names, 'label': train_labels, 'split': "train"}
testing_data = {'id': test_file_names, 'label': test_labels, 'split': "test"}

training_df = pd.DataFrame(training_data)
testing_df = pd.DataFrame(testing_data)

df = pd.concat([training_df, testing_df], ignore_index=True)
# split val and train 1:9
train_df, val_df = train_test_split(training_df, test_size=0.2, random_state=2023)
train_df = train_df.iloc[:, :2]
val_df = val_df.iloc[:, :2]
test_df = testing_df.iloc[:, :2]

print('The shape of train data',train_df.shape)
print('The shape of test data',test_df.shape)
print('The shape of val data',val_df.shape)
print('The shape of all data', df.shape)
# %%
# build torch dataset
class DogCatLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(
            dataset
        )  # some images are CMYK, Grayscale, check only RGB
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        classCategory = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, classCategory

    # 非 RGB 圖片會被忽略
    def checkChannel(self, dataset):
        datasetRGB = []
        for index in range(len(dataset)):
            if Image.open(dataset[index][0]).getbands() == (
                "R",
                "G",
                "B",
            ):  # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB

# %%
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) params from Imagenet
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) params from Imagenet
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# %%
# 將訓練資料再切分為 Train、Validation
training_set = ImageFolder(train_folder)

train_data, valid_data, train_label, valid_label = train_test_split(
    training_set.imgs, training_set.targets, test_size=0.1, random_state=42
)

train_dataset = DogCatLoader(train_data, train_transform)
valid_dataset = DogCatLoader(valid_data, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

# %%
image_size = 224
image_channel = 3 # RGO
bat_size = 64 # 64
# %%
# building Network

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size: 3*224*224
        self.conv1 = nn.Sequential(
            nn.Conv2d(image_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            ) # 32*112*112
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            ) # 64*56*56
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2), 
            ) # 128*28*28
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
            ) # 128*28*28

        

# %%

# %%
