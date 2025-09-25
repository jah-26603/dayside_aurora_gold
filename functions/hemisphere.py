# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:43:00 2024

@author: JDawg
"""
import numpy as np
def hemisphere(hemisphere_order, sza, skip_s = True, skip_n = False, print_b = False): 
    
    
    first_row_unpopulated = np.isnan(sza[0].data).all()   #Boolean: if True, first row of sza matrix is not populated
    last_row_unpopulated = np.isnan(sza[len(sza) - 1].data).all()   #Boolean: if True, last row of sza matrix is not populated
    
    #We want values in bottom half of data - corresponds to northern hemisphere of Earth (latitude data is flipped)
    if last_row_unpopulated and not first_row_unpopulated:
        
        if print_b : print('This is a Southern Hemisphere Scan')
        hemisphere = 'Southern'
        hemisphere_order.append(1)
        
    if first_row_unpopulated and not last_row_unpopulated :
        if print_b: print('This is a Northern Hemisphere Scan')
        hemisphere = 'Northern'
        hemisphere_order.append(0)
    return hemisphere_order, hemisphere, skip_s, skip_n