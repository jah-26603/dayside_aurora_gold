# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:11:10 2024

@author: dogbl
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt as medfilt



def limb_data (arr, bi):
    k0 = (arr[52:]*bi).T
    k1 = k0.flatten()
    k2 = k1[~np.isnan(k1)]
    return k2


def limb_difference(north, south, latitude):
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    lat = latitude[52:]
    lat[np.isnan(lat)] = 0
    lat[lat != 0 ]= 1
    nb = cv2.filter2D(lat, -1, kernel)
    limb = np.where((nb <= 6) & (lat == 1), 1, np.nan) #creates mask for limb pixels
    limb_diff_arr = np.zeros((south.shape[0], south.shape[1], 124))
    
    #This for loop: flattens limb pixels of the earth into 1d signals, smooths them
    #normalizes them, and finds the scaling factor, then subtracts, this is the final result
    for c in range(south.shape[0]):
        for btc in range(south.shape[1]):
            sur = np.flipud(south[c,btc])
            norte = north[c,btc]

        
        
            kk1 = ((norte*limb).T).flatten()
            kk2 = kk1[~np.isnan(kk1)]
            ss1 = (sur*limb).T.flatten()
            ss2 = np.copy(ss1)[~np.isnan(ss1)]        
            ss3 = gaussian_filter1d(ss2, sigma = 7)
        
            
            kk3 = kk2/np.max(np.concatenate((kk2[:50], kk2[100:]))) #Finds the correct scale. not based on the information near the peaks
            du = (kk3 / np.max(kk3))
            du1 = np.concatenate(((kk3 / np.max(kk3))[:50], (kk3 / np.max(kk3))[100:]))
            su =  ss3 / np.max(ss3)
            su1 = np.concatenate((su[:50], su[100:]))
            suu = np.clip(su - np.mean(su1) + 1*np.mean(du1), 0 ,1) +.01
            diff = np.clip((du/np.max(du)  - suu/np.max(suu)), 0,1)
            diff/= np.max(diff)
            
            sig = 5
            du = medfilt(du, kernel_size = sig)
            result = np.clip(du - suu, 0, 1 )/ np.max(np.clip(du - suu, 0, 1 ))
            limb_diff_arr[c,btc] = result #the result smoothed
            

    return limb_diff_arr


