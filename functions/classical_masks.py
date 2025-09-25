# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:35:37 2024

@author: JDawg
"""
import numpy as np
import functions
import cv2
import matplotlib.pyplot as plt
from deep_learning.model import UNet
from deep_learning.training_evals_functions import load_model
import torch
import torchvision.transforms.v2.functional as TF

def classical_masks(time_of_scan, difference, border_image , latitude,  brightnesses):
                
        dp = np.flipud(difference.astype(np.float32))
        gk1 = functions.gabor_fil(time_of_scan)
        filtered_image, kernel = functions.LoG_filter_opencv(dp, sigma_x = .65, sigma_y =.35, size_x = 7, size_y = 5)
        filtered_image = cv2.convertScaleAbs(filtered_image)

        filtered_image = np.abs(cv2.filter2D(filtered_image, -1, gk1).astype(float))
        filtered_image[filtered_image == 0] = np.nan
        filtered_image[filtered_image < np.nanmedian(filtered_image)] = np.nan
        filtered_image[~np.isnan(filtered_image)] = 1
        
        mask = np.zeros_like(filtered_image)
        try:
            results = functions.clustering_routine(dp, filtered_image, np.flipud(difference), latitude, lat_threshold = 50)
            results *= border_image
            results[results<50] = 0 
            mask = np.copy(results)
            mask[np.isnan(mask)] = 0
            mask[mask != 0] = 1 
            
        except ValueError:
            print('no points meet criteria in this scan')
            mask = np.zeros((52,92))
            results = np.zeros((52,92))
        
        return mask, results

