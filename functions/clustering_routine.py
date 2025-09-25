# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:03:23 2024

@author: JDawg
"""
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
def clustering_routine(dp, filtered_image, difference_LBHS, latitude, lat_threshold = 50):
    
    
    
    # qq = dp* filtered_image
    # qq[qq< np.nanmedian(qq)] = np.nan
    qq = np.copy(filtered_image)
    qq[np.isnan(qq)] = 0 
    qq[qq!=0] = 1


    # #CLUSTERING
    i, j = np.indices(qq.shape)
    vector = np.column_stack((i.ravel(), j.ravel(), qq.ravel()))
    vector= vector[qq.ravel() != 0]
    X = vector[:, :2]  # Use only x and y (i, j) coordinates for spatial clustering

    # Apply DBSCAN
    dbscan = DBSCAN(eps=3, min_samples=10)

    labels = dbscan.fit_predict(X)
    aurora_list = []
    latitude = latitude[52:]
    # First loop: Filter clusters based on latitude condition and store in aurora_list
    for i in range(int(max(labels) + 1)):
        class_vector = X[labels == i]
        row_indices = class_vector[:, 0].astype(int)  # Assuming rows are in the first column
        col_indices = class_vector[:, 1].astype(int)  # Assuming cols are in the second column
    
        # Check if the cluster's median latitude exceeds the threshold
        if np.nanmedian(latitude[row_indices, col_indices]) > lat_threshold:
            aurora_list.append(class_vector)
    
    dummy = np.zeros_like(qq)
    # Second loop: Plot each cluster's points on a copy of the qq array
    for idx, vec in enumerate(aurora_list):
        row_indices = vec[:, 0].astype(int)  # Assuming rows are in the first column
        col_indices = vec[:, 1].astype(int)  # Assuming cols are in the second column
    
        # Mark the points from this cluster in the 'dummy' array
        dummy[row_indices, col_indices] = 1  # Use a distinct value to mark these points


    
    results = dummy*difference_LBHS
    return results

        
        