# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:30:13 2024

@author: JDawg
"""

import os
import glob

def scans_at_time(filepath):
    

    file_list = [name.replace(filepath,"") for name in glob.glob(rf'{filepath}/*.nc')]
    dict_keys = [file[27:32] for file in file_list]
    dict_scans = {key: [] for key in dict_keys}


    return dict_scans


if __name__ == "__main__":
    filepath = r'C:\Users\JDawg\OneDrive\Desktop\England Research\Aurora\jordan_aurora\Aurora_Dates\2020_05_21'
    dict_scans = scans_at_time(filepath)