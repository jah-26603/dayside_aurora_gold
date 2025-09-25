# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:18:17 2025

@author: JDawg
"""

from torch.utils.data import DataLoader, random_split
import torch
from dataloader import SegmentationDataset, JointRandomAffine, JointCompose, JointRandomChannelShuffle, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointNormalize, JointPad
from training_evals_functions import run_training, visualize_predictions, load_model
from model import  UNetSimpleDropout
import os
import netCDF4 as nc
import glob 
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

#%%
here = os.path.dirname(os.path.abspath(__file__))
#get all northern files from classical directory
fp = yaml.safe_load(open(os.path.join(here, "..", "config.yaml")))['classical_product_output']
files = glob.glob(os.path.join(fp, str(2020), '*.nc')) #training was only done on 2020 outputs

img = []
label = []
for m in tqdm(files):
    
    ds = nc.Dataset(m, 'r')
    label.append(ds.variables['final_auroral_mask'][:])
    img.append(ds.variables['north_scan'][:])
    lat_mask = ds.variables['lat'][:]
    ds.close()

lat_mask[np.isnan(lat_mask)] = 0
lat_mask[lat_mask != 0] = 1

label_tensor = torch.tensor(np.concatenate(label, axis = 0).data)
img_tensor = torch.tensor(np.concatenate(img, axis = 1).data * lat_mask)

means = img_tensor.mean(dim=(1, 2, 3))
stds = img_tensor.std(dim=(1, 2, 3))
class TensorSegmentationDataset(Dataset):
    def __init__(self, images, labels,lat_mask, transform=None):
        images = torch.permute(images, (1,0,2,3))
        images = torch.nan_to_num(images, nan=0.0)  # if needed
        lat_mask = torch.tensor(lat_mask.data)
        
        self.images = images
        self.labels = labels
        self.lat_mask = lat_mask
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]       # shape: [C, H, W]
        label = self.labels[idx]     # shape: [H, W] or [C, H, W]
        lat_mask = self.lat_mask
        # Unsqueeze label if needed to add channel dimension
        if label.ndim == 2:
            label = torch.stack([label, lat_mask])
        if self.transform:
            im, lb = self.transform(img, label)
        return im, lb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetSimpleDropout(n_channels=3, n_classes=1).to(device)

#Dataloading stuff
joint_transform = JointCompose([
    JointRandomHorizontalFlip(),
    JointRandomAffine(degrees=(-5,5), translate=(0.1, 0.1), scale=(1, 1.15), shear=(-5,5)),
    JointNormalize(mean = means.tolist(), 
                   std = stds.tolist()),
    ])



dataset = TensorSegmentationDataset(
    images=img_tensor, 
    labels=label_tensor, 
    lat_mask = lat_mask,
    transform=joint_transform
)

train_size = int(0.6 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size =len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

##Training

# # # # #For finetuning
# model = load_model(model, r"C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\deep_learning\best_seg_model.pth", device='cuda')
# model.to(device)

# run_training(model, train_loader=train_loader, val_loader= val_loader, patience = 7,
#              device=device, epochs=50, check_val = True, lr = 1e-3, weight_decay = 1e-2)
visualize_predictions(model, test_loader, device)



