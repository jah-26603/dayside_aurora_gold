# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:05:12 2024

@author: JDawg
"""

import numpy as np
import cv2

def LoG_filter_opencv(image, sigma_x = 1, sigma_y = 1, size_x=None, size_y=None):
    # Set filter size based on sigma if not provided
    if size_x is None:
        size_x = int(6 * sigma_x + 1) if sigma_x >= 1 else 7
    if size_y is None:
        size_y = int(6 * sigma_y + 1) if sigma_y >= 1 else 7
    
    # Ensure sizes are odd
    if size_x % 2 == 0:
        size_x += 1
    if size_y % 2 == 0:
        size_y += 1
    
    # Generate 2D grid for kernel computation
    x, y = np.meshgrid(np.arange(-size_x//2+1, size_x//2+1), 
                       np.arange(-size_y//2+1, size_y//2+1))
    
    # Compute anisotropic LoG kernel
    kernel = -(1/(np.pi * sigma_x**2 * sigma_y**2)) * (1 - ((x**2 / sigma_x**2) + (y**2 / sigma_y**2)) / 2) * \
              np.exp(-((x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))
    
    # Normalize kernel
    kernel = kernel / np.sum(np.abs(kernel))
    # Perform convolution using OpenCV filter2D
    result = cv2.filter2D(image, -1, kernel)
    return result, kernel
