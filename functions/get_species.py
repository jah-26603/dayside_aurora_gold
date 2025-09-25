# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:03:23 2024

@author: JDawg
"""
import numpy as np
import matplotlib.pyplot as plt

def get_data_info(radiance, one_pixel, region):
    
    radiances_of_interest = []
    
    for lb,ub in region:
        lb1 = np.abs(one_pixel.data - lb).argmin()
        ub1 = np.abs(one_pixel.data - ub).argmin()
        radiances_of_interest.append(radiance[:,:,lb1: ub1 + 1].data)
        
        
    brightnesses = 0.04*np.nansum(np.concatenate(radiances_of_interest, axis = -1), axis = -1)
    condition = np.sum(radiance == 0, axis=2) < 5  # Checks for stars along emission profile of each pixel, might need a different check
    brightnesses[condition] = 0  # Set brightnesses to 0 where condition is True
    
    return brightnesses
                




            