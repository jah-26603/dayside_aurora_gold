# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 23:55:43 2025

@author: JDawg
"""

import numpy as np
import glob 
import os
from tqdm import tqdm
import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import os
import torch 
from torch.utils.data import DataLoader, random_split
from dataloader import SegmentationDataset, JointRandomAffine, JointCompose, JointResize, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointNormalize, JointPad
from training_evals_functions import run_training, visualize_predictions, load_model
from model import UNet, UNetSimpleDropout
import pandas as pd
import spaceweather as sw
import datetime
import yaml

sys.path.append(os.path.abspath('..'))
import functions

here = os.path.dirname(os.path.abspath(__file__))
configs_fp = os.path.join(here, "..", "config.yaml")
configs = yaml.safe_load(open(configs_fp))['dl_configs']


#this will create the feature to image pairs necessary for training
def get_training_data(configs_fp, configs):
    species_info_dict = yaml.safe_load(open(configs_fp))['species']
    base = yaml.safe_load(open(configs_fp))['north_day_base']
    #kp indice data from potsdam, gathering kp values within a certain range and all of 2020.
    kp_df = sw.gfz_3h()["2018-11":"2022"]
    kp_df['date'], kp_df['datetime'] = kp_df.index.date,  kp_df.index
    high_kp_dates = kp_df[(kp_df.Kp <= 6) & (kp_df.Kp >= 4)]['date'].unique()
    kp_df = kp_df[kp_df['date'].isin(high_kp_dates)]
    
    ogs = sw.gfz_3h()['2020-01':'2020-12']
    ogs['date'], ogs['datetime'] = ogs.index.date, ogs.index
    kp_df = pd.merge(kp_df, ogs, how = 'outer')
    
    #unique date times
    dfps = kp_df.date.unique() 
    days = np.array([[d.timetuple().tm_yday, d.timetuple().tm_year] for d in dfps])
    
    #files for retrieval
    sub_dirs = [os.path.join(base, str(year), str(doy).zfill(3)) for doy,year in days]
    files = [file for sub_dir in sub_dirs for file in glob.glob(os.path.join(sub_dir, '*.nc'))]
    kp_df = sw.gfz_3h()["2018-11":"2024"]
    kp_df['date'], kp_df['datetime'] = kp_df.index.date,  kp_df.index
    
    
    #save and store required data for climatology model.
    for i, file_idx in tqdm(enumerate(range(0, len(files))), total=len(files) - 0):

        file = files[file_idx]
        try:
            ds = nc.Dataset(file, 'r')
        except OSError:
            continue
        
        
        sza = ds.variables['SOLAR_ZENITH_ANGLE'][:].data[:52]
        if np.isnan(sza[0,:]).all():
            continue
        
        else:
            ema = ds.variables['EMISSION_ANGLE'][:]
            ema = ema.data[:52]
            radiance = ds.variables['RADIANCE'][:]
            radiance = np.clip(radiance, 0, np.inf)
            wavelength = ds.variables['WAVELENGTH'][:]

            label = []
            for specie in species_info_dict.keys():
                filled_indices, one_pixel = functions.filled_indices(wavelength)
                brightnesses = functions.get_data_info(radiance, one_pixel, **species_info_dict[specie]).data[:52]
                label.append(brightnesses)
            label = np.array(label)
            
            doy = int(file[52:55])
            year = int(file[47:51])
            utc_hr, utc_min = file[56:61].split('_')
            gold_utc = int(utc_hr) + int(utc_min)/60
            
            
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(days= doy - 1)
            gold_datetime = datetime.datetime.combine(date.date(), datetime.time()) + datetime.timedelta(hours=gold_utc)
            
            filtered_kp = kp_df[((gold_datetime - kp_df.datetime) < datetime.timedelta(days = .25 - 1/24))
                                & ((kp_df.datetime - gold_datetime) <= datetime.timedelta(days = 1/24))].copy()
            
            filtered_kp['time_diff'] = np.abs((filtered_kp['datetime'] - gold_datetime).dt.total_seconds())
            filtered_kp = filtered_kp.sort_values('time_diff')
            
            
            if len(filtered_kp.Kp) != 2:
                continue
                
            kp_arr = [np.full(sza.shape, int(kp)) for kp in filtered_kp.Kp]
            doy_arr = np.full(sza.shape, int(doy))
            kp_stack = np.stack(kp_arr, axis=-1)
            
            
            img = np.dstack([sza, doy_arr, ema] + [kp_stack[..., i] for i in range(kp_stack.shape[-1])])
            
            
            np.save(os.path.join(configs['south_input_out'], f'sza_{i}.npy'), img)
            np.save(os.path.join(configs['south_emissions_out'], f'south_{i}.npy'), label)
    
        
#IF RUNNING FOR FIRST TIME UNCOMMENT THIS TO GET TRAINING DATA
get_training_data(configs_fp, configs)

#%%
root_dir = configs['root_dir']

#Computes mean and std for dataset. Necessary for regression standardization
def compute_mean_std(label_dir, image_dir = None):
    if image_dir is None:
        sum_ = [0, 0 , 0]
        sum_sq = [0, 0 , 0]
        count = [0, 0 , 0]
        
        for fname in tqdm(os.listdir(label_dir), desc= 'gathering standardization info...'):
            if fname.endswith('.npy'):
                data = np.load(os.path.join(label_dir, fname)).astype(np.float32)
                
                for c in range(data.shape[0]):
                    pixels = data[c].flatten()
                    sum_[c] += pixels.sum()
                    sum_sq[c] += (pixels ** 2).sum()
                    count[c] += pixels.size
                    
        mean = np.array(sum_) / np.array(count)
        std = np.sqrt(np.array(sum_sq) / count - mean ** 2)
        return mean, std
    else: 
        sum_ = [0,0, 0, 0]
        sum_sq = [0,0, 0 , 0]
        count = [0,0, 0 , 0]
        
        for fname,pname in tqdm(zip(os.listdir(label_dir),os.listdir(image_dir)), desc= 'gathering standardization info...'):
            if fname.endswith('.npy'):
                data = np.load(os.path.join(label_dir, fname)).astype(np.float32)
                kp_data = np.load(os.path.join(image_dir, pname)).astype(np.float32)
                for c in range(data.shape[0]):
                    pixels = data[c].flatten()
                    sum_[c] += pixels.sum()
                    sum_sq[c] += (pixels ** 2).sum()
                    count[c] += pixels.size
                    
                sum_[-1] += kp_data[0,0,-2:].sum()
                sum_sq[-1] += (kp_data[0,0,-2:]**2).sum()
                count[-1] += kp_data[0,0,-2:].size
        mean = np.array(sum_) / np.array(count)
        std = np.sqrt(np.array(sum_sq) / count - mean ** 2)
        return mean, std

#already calculated standardization information
mean = [424.5253015,   94.00546396, 478.16959464,   1.21870199]
std = [610.35117947, 137.95994337, 702.63847917,   1.2907782 ]
# # Usage
# label_dir = r"E:\south\south"
# image_dir = r"E:\south\sza"
# mean, std = compute_mean_std(label_dir, image_dir)
# print(mean)
# print(std)


joint_transform = JointCompose([
    JointNormalize(mean= mean, std=std, aurora = False)])


dataset = SegmentationDataset(root_dir= root_dir, transform = joint_transform, aurora = False)
train_size = int(0.6 * len(dataset)) 
val_size = int(0.2 * len(dataset))
test_size =len(dataset) - val_size - train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size,test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=8, n_classes=3).to(device)
# # For regression task (aurora=False):
    
    
# run_training(model, train_loader=train_loader, val_loader= val_loader, aurora = False,
#              device=device, epochs=100, check_val = True, lr = 1e-3, weight_decay =1e-2,
#              patience = 10)



model.eval()
visualize_predictions(model, test_loader, device, aurora=False)  # For regression






