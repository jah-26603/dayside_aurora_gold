# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 21:41:35 2025

@author: JDawg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:39:38 2025

@author: JDawg
"""


from netCDF4 import Dataset
import numpy as np
import netCDF4 as nc
import pickle 
import os

def create_nc(month_data, fp = None):
    
    os.makedirs(fp, exist_ok= True)
    day_keys_str = [k.zfill(3) for k in month_data.keys() if isinstance(k, str) and k.isdigit()]
    day_keys_int = sorted([int(k) for k in day_keys_str])  # Sort numerically
    # day[:] = day_keys_str
    
    try:
        for dd in day_keys_int:
            
            dataset = nc.Dataset(os.path.join(fp, f'{str(dd).zfill(3)}.nc'), 'w', format='NETCDF4')
            dataset.createDimension(f'scans_{str(dd).zfill(3)}',len(month_data[str(dd)]['sza']))
            dataset.createDimension('row', 52)    # rows of image
            dataset.createDimension('col', 92)    # columns of image
            dataset.createDimension('specie', 3)
            
            
            
            #META DATA STUFF
            specie_var = dataset.createVariable('species', str, ('specie',))
            lat = dataset.createVariable('lat', np.float32, ('row','col'))
            lon = dataset.createVariable('lon', np.float32, ('row','col'))
            mag_lat = dataset.createVariable('mag_lat', np.float32, ('row','col'))
            mag_lon = dataset.createVariable('mag_lon', np.float32, ('row','col'))
        

            lat.units = 'degrees_north'
            lat.long_name = 'latitude'
            lon.units = 'degrees_east'
            lon.long_name = 'longitude'
            mag_lat.units = 'degrees_north'
            mag_lat.long_name = 'magnetic_latitude'
            mag_lon.units = 'degrees_east'
            mag_lon.long_name = 'magnetic_longitude'
            
            
            speciees = ['LBH', '1493', '1356']
            specie_var[:] = np.array(speciees)
            lat[:] = month_data['geographic_latitude']
            lon[:] = month_data['geographic_longitude']
            mag_lat[:] = month_data['magnetic_latitude']
            mag_lon[:] = month_data['magnetic_longitude']
    
                
                
                
                
            # Create variables with proper data types
            sza = dataset.createVariable('sza', np.float32, (f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            ema = dataset.createVariable('ema', np.float32, (f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            ut_time = dataset.createVariable('ut_time', np.float32, (f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            mlt = dataset.createVariable('mag_LT', np.float32, (f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            ut_start = dataset.createVariable('scan_start', np.float32, (f'scans_{str(dd).zfill(3)}',))
            fam = dataset.createVariable('final_auroral_mask', np.int32, (f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            north = dataset.createVariable('north_scan', np.float32, ('specie', f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            dayglow = dataset.createVariable('dayglow_model', np.float32, ('specie', f'scans_{str(dd).zfill(3)}', 'row', 'col'))
            
            
            # Set units
            sza.units = 'degrees'
            ema.units = 'degrees'
            ut_time.units = 'hours'
            mlt.units = 'hours'
            ut_start.units = 'hours'
            fam.units = 'binary mask'
            north.units = 'Rayleighs'
            dayglow.units = 'Rayleighs'
            
            # Set long names
            sza.long_name = 'Solar_Zenith_Angle'
            ema.long_name = 'Emission_Angle'
            ut_time.long_name = 'Universal_Time_Array'
            mlt.long_name = 'Magnetic_Local_Time_Array'
            ut_start.long_name = 'UT_Scan_Start'
            fam.long_name = 'Final_Auroral_Mask'
            north.long_name = 'Northern_Hemispheric_Scan'
            dayglow.long_name = 'Dayglow_Model'
            
            
            sza[:] = np.array(month_data[str(dd)]['sza'])
            ema[:] = np.array(month_data[str(dd)]['ema'])
            ut_time[:] = np.array(month_data[str(dd)]['time_arr'])
            mlt[:] = np.array(month_data[str(dd)]['magnetic_local_time'])
            ut_start[:] = np.array([(int(d.hour) + int(d.minute)/60) for d in month_data[str(dd)]['ut']])
            fam[:] = np.array(month_data[str(dd)]['final_auroral_mask'])
            
            north[:] = np.stack(np.array([month_data[str(dd)][spec]['north'] for spec in speciees]))
            dayglow[:] = np.stack(np.array([month_data[str(dd)][spec]['dayglow'] for spec in speciees]))
    
    
    
    
            dataset.variables['species'].description = "Emissions species in this order (LBH bands (N2), 149.3 nm (NI), & 135.6 nm (OI))"
            dataset.variables['lat'].description = "Grid geographic latitude locations (degrees)"
            dataset.variables['lon'].description = "Geographic grid longitude locations (degrees)"
            dataset.variables['mag_lat'].description = "Magnetic latitude locations (Apex Quasi-Dipole)"
            dataset.variables['mag_lon'].description = "Magnetic longitude locations (Apex Quasi-Dipole)"
            dataset.variables['sza'].description = "Solar zenith angle at each grid location (degrees)"
            dataset.variables['ema'].description = "Emission angle (degrees)"
            dataset.variables['ut_time'].description = "Grid Universal Time (UT)"
            dataset.variables['mag_LT'].description = "Magnetic local time (MLT) at each grid location (hours)"
            dataset.variables['scan_start'].description = "Start time of scan in UT (hours)"
            dataset.variables['final_auroral_mask'].description = "Final processed auroral mask (1 = aurora, 0 = background)"
            dataset.variables['north_scan'].description = "Raw emission northern scan from GOLD"
            dataset.variables['dayglow_model'].description = "Dayglow model estimate for background subtraction, scaled to northern scan"
            dataset.close()
        
    
    except Exception as e:
        import traceback

        traceback.print_exc()
        breakpoint()
        dataset.close()
        # os.remove(fp)