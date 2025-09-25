# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 21:23:48 2025

@author: JDawg
"""

from deep_learning.model import UNet, UNetSimpleDropout
from deep_learning.training_evals_functions import load_model
import torch
import torchvision.transforms.v2.functional as TF
import numpy as np
import matplotlib.pyplot as plt


def dayglow_gen(image, north, nan_mask):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=8, n_classes=3).to(device)
    model = load_model(model, r"deep_learning\best_reg_model.pth", device=device)
    model.eval()

    raw_x = torch.tensor(np.stack([np.cos(image[:,0] *np.pi /180),#sza
                                   np.sin(image[:,0] *np.pi /180), 
                                   np.cos(image[:,1] * 2*np.pi/366),#doy
                                   np.sin(image[:,1]* 2*np.pi/366),
                                   np.cos(image[:,2] * np.pi/180), #ema
                                   np.sin(image[:,2]* np.pi/180),
                                   image[:,3],image[:,4]], 
                                  axis=0), dtype=torch.float32)
    #these magic numbers can be found from the deep learning training files -> unet dayglow_reg
    raw_x[-2:] = TF.normalize(raw_x[-2:], mean=1.16006117, std=1.2479701)
    # add batch dimension
    raw_x = raw_x.permute(1, 0, 2, 3)# this will be fine i think when i have more than 1                                 
       
    normalized_images = []
    with torch.no_grad():
        x = raw_x.to(device)
        output = model(x).permute(1,0,2,3)
        pred = output.cpu().squeeze().numpy()
        
    full_south = np.zeros_like(pred)
    full_diff = np.zeros_like(pred)
    a = 20
    b = 52
    for c in range(pred.shape[0]):
        for btc in range(pred.shape[1]):
            
            south = pred[c,btc] - np.nanmin(pred[c,btc])
            south = south * np.flipud(nan_mask)
            norte = np.flipud(north[c,btc] * nan_mask)
            
            na = norte[a:b]
            sa = south[a:b]
            mask = na > 100
            na = na[mask]
            sa = sa[mask]
        
            # Remove any NaNs in the filtered values
            valid = ~np.isnan(na) & ~np.isnan(sa)
            na = na[valid]
            sa = sa[valid]
            beta = np.dot(sa, na) / np.dot(sa, sa)
            full_south[c,btc] = south*beta
            full_diff[c,btc] = np.clip(norte - south*beta, 0, np.inf)
    return full_south, full_diff
   
    
   
def mask_gen(north):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetSimpleDropout(n_channels=3, n_classes=1).to(device)
    model = load_model(model, r"deep_learning\best_seg_model.pth", device=device)
    north = np.transpose(north, (1,0,2,3))
    
    north[np.isnan(north)] = 0
    standardized = np.zeros_like(north)
    for b in range(north.shape[0]):
        for c in range(north.shape[1]):
            dn = north[b,c] / (np.nanmax(north[b,c]) + 1e-6)
            dn = (dn - dn.mean()) / (dn.std() + 1e-6)
            standardized[b,c] = dn
    
    x = torch.tensor(standardized, dtype=torch.float32).to(device)
    
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        preds = preds.squeeze().detach().cpu().numpy()

    return preds