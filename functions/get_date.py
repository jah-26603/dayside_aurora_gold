# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:59:47 2024

@author: JDawg
"""
import numpy as np
from datetime import datetime

def time_str_to_utc_float(tstr):
    if tstr is None or tstr.strip() == '':
        return np.nan
    try:
        dt = datetime.strptime(tstr, "%H:%M:%S.%fZ")
        return dt.hour + dt.minute / 60 + dt.second / 3600 + dt.microsecond / 3_600_000_000
    except Exception:
        return np.nan

def date_and_time(filled_indices, time):
    # Extract the date from the first valid timestamp
    raw_str = ''.join([c.decode('utf-8') for c in time[filled_indices[0, 0], filled_indices[0, 1]]])
    date = raw_str[:10]  # e.g., '2020-01-01'

    # Initialize time_array with NaNs
    time_array = np.full((104, 92), np.nan)

    # Populate float UTC values
    for i, j in filled_indices:
        raw_str = ''.join([c.decode('utf-8') for c in time[i, j]])
        time_str = raw_str[11:]  # e.g., '23:03:36.747Z'
        time_array[i, j] = time_str_to_utc_float(time_str)

    return date, time_array