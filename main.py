# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:17:32 2024

@author: JDawg
"""

from data_process import process_loop
from tqdm import tqdm
import os 
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import glob
import yaml


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

months = np.arange(1,12 + 1,1)   
species_info_fp = r"config.yaml"
configs = yaml.safe_load(open(species_info_fp))
years = configs['years']
product_outputs = configs['product_output']

if configs['classical_method']:
    years = configs['classical_years']
    product_outputs = configs['classical_product_output']
    
    
for year in years:
    north_day_base = os.path.join(configs['north_day_base'], str(year))
    north_days = np.array([int(d) for d in os.listdir(north_day_base)])
    
    dt_list = [[doy, datetime.strptime(f"{year}-{doy:03d}", "%Y-%j"),
                os.path.join(north_day_base,str(doy).zfill(3),'data')] 
               for doy in north_days]
    
    df = pd.DataFrame(data = dt_list, columns = ['doy', 'datetime', 'n_fp'])
    param = ['LBH','1493', '1356']

    for month in tqdm(months):
        dirs = df[df['datetime'].dt.month == month].copy() 
        if len(dirs) == 0:
            continue
        
        #need to filter out days with VERY low cadence, instrument failure or something ig. This throws up an error for some reason
        valid = [len(glob.glob(os.path.join(fp, '*.nc'))) for fp in np.array(dirs.n_fp)]
        dirs['num_files'] = valid
        dirs = dirs[dirs['num_files'] >= 6]
        
        aurora_product_fp = os.path.join(product_outputs,f'{year}')
        process_loop(param, dirs, aurora_product_fp, configs['species'], classical_method = configs['classical_method'], save_data = configs['save_data'])
            
            

            
