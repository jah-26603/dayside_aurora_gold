# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:57:26 2024

@author: JDawg
"""
import numpy as np
import scipy
import scipy.signal
import cv2
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
def limb_edge(difference, diff, latitude):
    
        ''' alot of these numbers in this function are magic numbers. In essence,
        this function finds the smooth second derivative of the differenced limb signal (convolved with an LoG filter). 
        We find the zero crossings of this result where the slope of the 2nd derivative is decreasing for the left boundary,
        and increasing for the right boundary within a certain range. This captures most of the boundary behaviors. '''
        
        
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        dummy = np.where(np.isnan(latitude), 0, 1)
        nb = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
        limb = np.where((nb < 7) & (dummy == 1), 1, np.nan)[52:]
        neighbor_check = cv2.filter2D(dummy.astype(np.float32), -1, kernel)
        col, row = np.where(~np.isnan((np.flipud(difference[0,0]) * limb).T))
        sig = 5
        
        border_image_arr = np.zeros_like(difference)
        col_list = np.zeros((diff.shape[0], diff.shape[1],2))
        for c in range(diff.shape[0]):
            for btc in range(diff.shape[1]):
                
                
                #magic numbers for spatial acceleration regions
                min_v = 27
                max_v = 95
                
                # Compute second derivative and detect zero crossings across limb
                second_der = gaussian_laplace(diff[c,btc], sigma=sig)
                second_der /= np.max(np.abs(second_der))
                zrs = np.where(np.diff(np.sign(second_der)) != 0)[0]
                
                peaks = scipy.signal.find_peaks(second_der, prominence = 0.08, distance = 5 )
                o_peaks = scipy.signal.find_peaks(-second_der, prominence = 0.08, distance = 5 )[0]
                peaks = np.asarray(peaks[0])
                try:
                    left_peak = peaks[np.min(np.where(peaks > min_v))] 
                    right_peak = peaks[np.max(np.where(peaks < max_v))]
                except ValueError:
                    left_peak = np.copy(min_v)
                    right_peak = np.copy(max_v)
                o_peaks = o_peaks[np.logical_and(o_peaks > left_peak, o_peaks < right_peak)]
        
                valid_range = zrs[np.where((zrs > left_peak) & (zrs < right_peak))]
        
                grad_array = np.diff(second_der)
                try:
                    min_v = np.min(valid_range[np.where(grad_array[valid_range] <0)]) 
        
                except ValueError or UnboundLocalError:
                    min_v = np.min(zrs[zrs>30])
                try:
                    max_v = np.max(valid_range[np.where(grad_array[valid_range] >0)]) + 1
                except ValueError or UnboundLocalError:
                    if len(zrs == 0):
                        max_v = 85
                    else:
                        max_v = np.max(zrs[zrs<85])
        
                border_image = np.where((neighbor_check < 8) & (dummy == 1), 1, 0)      
                border_image = border_image[52:]
                border_image = np.abs(border_image - 1)
                    
                border_image[:, col[min_v] :col[max_v]] = 1
                
                border_image_arr[c,btc] = border_image
                col_list[c,btc] = np.array([col[min_v], col[max_v]])
                
                
        return border_image_arr, col_list