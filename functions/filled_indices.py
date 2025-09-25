# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:57:40 2024

@author: JDawg
"""
import numpy as np

import numpy as np

def filled_indices(wavelength):
    # Create a boolean mask for finite values
    mask = np.isfinite(wavelength)
    
    # Get indices where the mask is True
    filled_indices = np.argwhere(mask.any(axis=2))  # Find non-empty pixel indices
    
    if filled_indices.size == 0:
        # Handle case when there are no filled indices
        return filled_indices, None

    # Extract one pixel's wavelength values
    one_pixel = wavelength[filled_indices[0, 0], filled_indices[0, 1], :]
    
    return filled_indices, one_pixel

# def filled_indices(wavelength):
#     #Find i,j indices of wavelengths that are filled with data
#     filled_indices = []
#     for i in range(wavelength.shape[0]):
#         for j in range(wavelength.shape[1]):
#             if np.isfinite(wavelength[i][j]).any():
#                 filled_indices.append((i, j))
#     filled_indices = np.array(filled_indices)
#     one_pixel = np.array(wavelength[filled_indices[0][0],filled_indices[0][1],:]) #array of 800 wavelength values for one pixel
    
    
#     return filled_indices, one_pixel