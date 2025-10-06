# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 21:20:09 2025

@author: JDawg
"""

import requests
import os
seg_url = r'https://data.lib.vt.edu/ndownloader/files/58265452'
reg_url = r'https://data.lib.vt.edu/ndownloader/files/58265455'
file_Path = 'deep_learning'

response = requests.get(seg_url)
if response.status_code == 200:
    with open(os.path.join(file_Path, 'best_seg_model.pth'), 'wb') as file:
        file.write(response.content)
    print('Segmentation weights downloaded successfully')
else:
    print('Segmentation weights failed to download file')
    
    
response = requests.get(reg_url)
if response.status_code == 200:
    with open(os.path.join(file_Path, 'best_reg_model.pth'), 'wb') as file:
        file.write(response.content)
    print('Regression weights downloaded successfully')
else:

    print('Regression weights failed to download file')
