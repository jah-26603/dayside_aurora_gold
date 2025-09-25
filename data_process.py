# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:09:45 2024

@author: JDawg
"""
import functions 
import netCDF4 as nc
import numpy as np
import glob
from tqdm import tqdm
from datetime import timedelta
import copy
import yaml
import os
import spaceweather as sw
import datetime
import apexpy




def process_loop(species, dirs, aurora_product_fp, species_info_dict, classical_method = True, save_data = True):
    # Specie-specific info
    day_data = {
        specie: {
            'north': [],
            'dayglow': [],
            'auroral_pixels': []
        } for specie in species
    }
    day_data['sza'] = []
    day_data['ema'] = []
    day_data['time_arr'] = []
    day_data['ut'] = []
    day_data['final_auroral_mask'] = []
    
    # One deep-copied day_data per doy
    month_data = {str(doy): copy.deepcopy(day_data) for doy in dirs.doy}
    kp_df = sw.gfz_3h()['2018':'2025']
    kp_df['date'], kp_df['datetime'] = kp_df.index.date,  kp_df.index
    
    for doy in dirs.doy:
        
        nfp = dirs.n_fp[dirs.doy == doy].iloc[0]
        file_list = glob.glob(os.path.join(nfp, '*.nc'))
        hemisphere_order = []

        for file in tqdm(range(len(file_list)), desc = f'Day of Year: {doy}'):
            try:
                ds = nc.Dataset(file_list[file], 'r')
            except OSError:
                print('Error in reading netcdf4 file')
                continue
            
            #loads data
            try:
                latitude = ds.variables['GRID_LAT'][:]    
                radiance = ds.variables['RADIANCE'][:]
                wavelength = ds.variables['WAVELENGTH'][:]
                radiance = np.clip(radiance, 0, np.inf)
                sza = ds.variables['SOLAR_ZENITH_ANGLE'][:]
                time_ar = ds.variables['TIME_UTC'][:]
                ema = ds.variables['EMISSION_ANGLE'][:]
                longitude = ds.variables['GRID_LON'][:]
            except RuntimeError:
                continue
            
            
    
            hemisphere_order, hemisphere, *_ = functions.hemisphere(hemisphere_order, sza) #which hemisphere  
            if hemisphere_order[-1] == 1:
                continue
            
            filled_indices, one_pixel = functions.filled_indices(wavelength)  # acceptable indices for analysis
            date, time_array = functions.date_and_time(filled_indices,time_ar)  # gets date and time
            time_of_scan = np.nanmin(time_array)

            for specie in species:
                try:
                    north_scan = functions.get_data_info(radiance, one_pixel, **species_info_dict[specie])
                except Exception as e:
                    print(f"An error occurred: {e}")
                    north_scan = np.zeros(52,92)
                month_data[str(doy)][specie]['north'].append(north_scan[52:]) 

            #You'll see 52 thrown around a lot, this is essentially selecting half of the image, since the other half
            #is empty. See the other 52 rows for example to see the empty, useless data.

    
            month_data[str(doy)]['sza'].append(np.flipud(sza[52:])) 
            month_data[str(doy)]['ema'].append(np.flipud(ema[52:]))
            month_data[str(doy)]['time_arr'].append(time_array[52:])
            month_data[str(doy)]['ut'].append(dirs.datetime[dirs.doy == doy].iloc[0] + timedelta(hours = time_of_scan))
        
        qq = 1
        if doy % qq == 0:
            nnor = np.stack([np.vstack(tuple([month_data[str(doy + 1 - qq + i) ][specie]['north'] for i in range(qq)])) for specie in species])
            nsza = np.vstack(tuple([month_data[str(doy + 1 - qq + i)]['sza'] for i in range(qq)]))
            nema = np.vstack(tuple([month_data[str(doy + 1 - qq + i)]['ema'] for i in range(qq)]))
            ndoy = np.vstack(tuple([np.full(np.array(month_data[str(doy + 1 - qq + i) ]['sza']).shape, doy) for i in range(qq)]))
            
            nsza[np.isnan(nsza)] = 0
            nema[np.isnan(nema)] = 0
            
            
            dtl = [mm  for i in range(qq) for mm in month_data[str(doy + 1 - qq + i)]['ut']]            
            kps = []
            
            for dt in dtl:
                filtered_kp = kp_df[((dt - kp_df.datetime) < datetime.timedelta(days = .25 - 1/24))
                                    & ((kp_df.datetime - dt) <= datetime.timedelta(days = 1/24))].copy()
                filtered_kp['time_diff'] = np.abs((filtered_kp['datetime'] - dt).dt.total_seconds())
                filtered_kp = filtered_kp.sort_values('time_diff')
                kps.append(list(filtered_kp.Kp))
                
            kps = np.array(kps)
            nkps = kps[:,:,None, None]
            nkps = np.broadcast_to(nkps, (nsza.shape[0], 2, 52, 92))  # Now shape matches desired
            
            channels = [nsza, ndoy, nema] + [nkps[:, i] for i in range(nkps.shape[1])]
            img = np.stack(channels, axis=1)  # shape: (B, C, 52, 92)
            nan_mask = np.where(np.isnan(latitude[52:]),0,1)

            #Background Subtraction
            south, difference = functions.dayglow_gen(img, nnor, nan_mask)
            
            #classical method for mask generation (convolution, limb profile, threshold, and candidate voting)
            if classical_method:    
                limb_diff = functions.limb_difference(nnor, south, copy.deepcopy(latitude)) #1d limb profile differences between north & south   
                border_image_arr, col_list = functions.limb_edge(difference, limb_diff, copy.deepcopy(latitude)) #makes limb/auroral borders
                _, best_spec = functions.track_limb_boundary(col_list, low_cadence = False) #best trajectory
                
                masks = np.zeros_like(nnor)
                results = np.zeros_like(nnor)
                
                for ch in range(nnor.shape[0]):
                    for btc in range(nnor.shape[1]):
                        # (time_of_scan, difference, border_image , latitude,  brightnesses, classical_method = True)
                        mask, _ = functions.classical_masks(month_data[str(doy)]['ut'][btc].hour,
                                                             difference[ch,btc], 
                                                             border_image_arr[best_spec, btc],
                                                             latitude,  nnor[ch,btc])
                        masks[ch,btc] = mask
                final_masks = np.nansum(masks, axis = 0)
                final_masks = np.where(final_masks >= 2, 1, 0)
            
            #UNET architecture for mask generation
            else:
                final_masks = functions.mask_gen(nnor) * nan_mask
                
            month_data[str(doy + 1 -qq)]['final_auroral_mask'] = final_masks
            results = difference*final_masks
            
            for n,specie in enumerate(species):
                month_data[str(doy + 1 -qq)][specie]['dayglow'] = south[n]
                month_data[str(doy + 1 -qq)][specie]['auroral_pixels'] = results[n]
                
            #magnetic coordiantes
            apex = apexpy.Apex(date = datetime.datetime.strptime(date, '%Y-%m-%d'))
            mag_lat, mag_lon = apex.convert(latitude[52:], longitude[52:], 'geo', 'qd')
            mag_lt = apex.mlon2mlt(mag_lon, np.array(month_data[str(doy)]['time_arr']))  # Magnetic Local Time
            
            month_data[str(doy)]['magnetic_local_time'] = mag_lt
            
         

        
    month_data['geographic_latitude'] = ds.variables['GRID_LAT'][:][52:]
    month_data['geographic_longitude'] = ds.variables['GRID_LAT'][:][52:]
    month_data['magnetic_latitude'] = mag_lat    
    month_data['magnetic_longitude'] = mag_lon
    
    if save_data:
        functions.create_netcdf(month_data,aurora_product_fp)
        print('Auroral products saved to', aurora_product_fp)
    else:
        print('no auroral products created for this month')
        
