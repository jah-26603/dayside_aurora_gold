# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:00:37 2024

@author: JDawg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gabor_fil(hour, ksize = 7, sigma = 20, lambd = 5, gamma = 0, psi = np.deg2rad(0), theta = None, storm = False): 

    # Parameters for Gabor filter
      # Size of the Gabor kernel
      # Standard deviation of the Gaussian function
      # Wavelength of the sinusoidal factor
      # Aspect ratio of the Gaussian function
     # Phase offset
    if storm:
        theta = np.deg2rad(theta)
        gk1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
    else:
        a = 13
        b = 17
        if hour <= a:
            theta = np.deg2rad(105)  # Orientation of the Gabor kernel (135 degrees for southwest)
        elif (hour > a and hour < b):
            theta = np.deg2rad(90)  # Orientation of the Gabor kernel (135 degrees for southwest)
        elif (hour >= b):
            theta = np.deg2rad(75)
    
        # Create Gabor kernel
        gk1 = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)

    
    return gk1


if __name__ == '__main__':
    
    hour = 12
    psi = np.deg2rad(0)
    sigma = 1
    lambd = 5
    gamma = 0
    
    gk1 = gabor_fil(hour = hour)
    pp = gk1/ np.sum(gk1)
    plt.figure()
    plt.imshow(gk1)
    plt.show()
    
    print(pp)