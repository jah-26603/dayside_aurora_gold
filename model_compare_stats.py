# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 20:21:00 2025

@author: JDawg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import glob
from tqdm import tqdm
import apexpy
import datetime as dt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import seaborn as sns
import yaml


configs = yaml.safe_load(open(r"config.yaml"))['zhang_paxton']

# In[]  Zhang Paxton model

mlat_list = np.arange(30, 90, 0.5) #final digit controls resolution
mlt_list = np.arange(0, 24, 0.25) 

fp = r"model_comparisons\2019_2024_KP_INDICES.txt"
df = pd.read_csv(fp, delim_whitespace=True)
df.rename(columns={'m': 'month', 'dy': 'day'}, inplace=True)
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['doy'] = [df.date[i].day_of_year for i in range(len(df))]

df = df[['year', 'doy', 'hr', 'kp']]

kp_list = sorted(set(df.kp))
kp_list = [x for x in kp_list if x < 9]

Q_list = []
threshold = .25

#Plotting model outputs at different levels
# for kp in kp_list:

#     Q,_,_ = ZP_oval(mlt_list, kp, mlat_list, threshold)
#     Q_list.append(Q.T)
    
#     plt.figure()
#     plt.title(kp)
#     plt.pcolor(mlt_list,mlat_list,Q)

#     plt.xlabel('mlt')
#     plt.ylabel('mlat')
#     plt.show()


    
# In[]  GOLD  
import netCDF4 as nc
results_files = {}

yr = configs['year']
pfp = os.path.join(configs['products_fp'], str(yr))

data = []
x = []
for file in tqdm(glob.glob(os.path.join(pfp, '*.nc'))):
        
    ds = nc.Dataset(file, 'r')
    species = np.array(ds.variables['species'][:])
    aur_mask = ds.variables['final_auroral_mask'][:]
    ut_hr_start = ds.variables['scan_start'][:]
    north = ds.variables['north_scan'][:]
    aur_mask = ds.variables['final_auroral_mask']
        
        
    doy = int(file [-6:-3])
    lo = df[ (df.doy == doy) & (df.year == yr)].copy()
    kp_hrs = lo.hr.to_numpy()
    kps = lo.kp.to_numpy()
    
    for i,hr in enumerate(ut_hr_start):
        kp = kps[np.argmin(np.abs(kp_hrs - hr))]
        results_files[f'{doy}_{hr}'] = [aur_mask[i], kp]
        x.append(north[:,i])
            
            
            
lat = pd.read_csv(r"model_comparisons\latitude.csv").to_numpy()[52:,1:]
lon = pd.read_csv(r"model_comparisons\longitude.csv").to_numpy()[52:,1:]

# In[]  Resample GOLD into any resolution  


fp = r'model_comparisons\subpixel_geometry'
os.chdir(fp)
nan_mask = np.copy(lat)
nan_mask[~np.isnan(nan_mask)] = 1
ES_lats = []
ES_lons = []

for i in glob.glob(os.path.join(fp,"*.csv")):
    if 'lats' in i:
        ES_lats.append(pd.read_csv(i).to_numpy()[51:]*nan_mask)
    if 'lons' in i:
        ES_lons.append(pd.read_csv(i).to_numpy()[51:]*nan_mask)
    
ES_lats = np.array(ES_lats)
ES_lons = np.array(ES_lons)
ES_lats[np.isnan(ES_lats)] = -30
ES_lons[np.isnan(ES_lons)] = 0

num_bins_list = [3]
e_flux_list = [.25]
def resample_gold_vectorized(mlat, mlt, OP_data, fov=True, aur_data=None):
    """Optimized version using vectorized operations"""

    mlat_list = np.arange(30, 90, 0.5) #final digit controls resolution
    mlt_list = np.arange(0, 24, 0.25) 
    
    if fov:
        gold_fov = np.zeros((25, len(mlt_list), len(mlat_list)))
        
        # Create valid mask
        valid_mask = (mlat >= 30) & (mlt < 23.875) & ~np.isnan(mlat) & ~np.isnan(mlt)
        
        # Get valid coordinates
        valid_pts, valid_rows, valid_cols = np.where(valid_mask)
        valid_mlat = mlat[valid_pts, valid_rows, valid_cols]
        valid_mlt = mlt[valid_pts, valid_rows, valid_cols]
        
        # Vectorized nearest neighbor search
        mlat_indices = np.searchsorted(mlat_list, valid_mlat)
        mlat_indices = np.clip(mlat_indices, 0, len(mlat_list)-1)
        
        # Check if we need to look at the previous index
        prev_indices = np.clip(mlat_indices - 1, 0, len(mlat_list)-1)
        closer_to_prev = (np.abs(valid_mlat - mlat_list[prev_indices]) < 
                         np.abs(valid_mlat - mlat_list[mlat_indices]))
        mlat_indices = np.where(closer_to_prev, prev_indices, mlat_indices)
        
        # Same for MLT
        mlt_indices = np.searchsorted(mlt_list, valid_mlt)
        mlt_indices = np.clip(mlt_indices, 0, len(mlt_list)-1)
        
        prev_indices = np.clip(mlt_indices - 1, 0, len(mlt_list)-1)
        closer_to_prev = (np.abs(valid_mlt - mlt_list[prev_indices]) < 
                         np.abs(valid_mlt - mlt_list[mlt_indices]))
        mlt_indices = np.where(closer_to_prev, prev_indices, mlt_indices)
        
        # Set values
        gold_fov[valid_pts, mlt_indices, mlat_indices] = 1
        return gold_fov
    
    else:
        # For aurora data resampling
        resampled_aur = np.zeros((25, len(mlt_list), len(mlat_list)))
        
        # Get aurora pixels
        aur_rows, aur_cols = np.where(aur_data == 1)
        
        # Create arrays for all valid aurora pixels across all time points
        all_pts = []
        all_mlat_vals = []
        all_mlt_vals = []
        all_orig_rows = []
        all_orig_cols = []
        
        for pt in range(25):
            valid_indices = ~np.isnan(mlat[pt, aur_rows, aur_cols])
            if np.any(valid_indices):
                valid_aur_rows = aur_rows[valid_indices]
                valid_aur_cols = aur_cols[valid_indices]
                
                all_pts.extend([pt] * len(valid_aur_rows))
                all_mlat_vals.extend(mlat[pt, valid_aur_rows, valid_aur_cols])
                all_mlt_vals.extend(mlt[pt, valid_aur_rows, valid_aur_cols])
                all_orig_rows.extend(valid_aur_rows)
                all_orig_cols.extend(valid_aur_cols)
        
        if len(all_pts) > 0:
            all_pts = np.array(all_pts)
            all_mlat_vals = np.array(all_mlat_vals)
            all_mlt_vals = np.array(all_mlt_vals)
            all_orig_rows = np.array(all_orig_rows)
            all_orig_cols = np.array(all_orig_cols)
            
            # Vectorized nearest neighbor search
            mlat_indices = np.searchsorted(mlat_list, all_mlat_vals)
            mlat_indices = np.clip(mlat_indices, 0, len(mlat_list)-1)
            
            prev_indices = np.clip(mlat_indices - 1, 0, len(mlat_list)-1)
            closer_to_prev = (np.abs(all_mlat_vals - mlat_list[prev_indices]) < 
                             np.abs(all_mlat_vals - mlat_list[mlat_indices]))
            mlat_indices = np.where(closer_to_prev, prev_indices, mlat_indices)
            
            mlt_indices = np.searchsorted(mlt_list, all_mlt_vals)
            mlt_indices = np.clip(mlt_indices, 0, len(mlt_list)-1)
            
            prev_indices = np.clip(mlt_indices - 1, 0, len(mlt_list)-1)
            closer_to_prev = (np.abs(all_mlt_vals - mlt_list[prev_indices]) < 
                             np.abs(all_mlt_vals - mlt_list[mlt_indices]))
            mlt_indices = np.where(closer_to_prev, prev_indices, mlt_indices)
            
            # Set values
            resampled_aur[all_pts, mlt_indices, mlat_indices] = 1
            
            # Create DataFrame more efficiently
            df = pd.DataFrame({
                'OP_mlat_idx': mlat_indices,
                'OP_mlt_idx': mlt_indices,
                'pt': all_pts,
                'Gold row': all_orig_rows,
                'Gold col': all_orig_cols
            })
        else:
            df = pd.DataFrame(columns=["OP_mlat_idx", "OP_mlt_idx", "pt", "Gold row", "Gold col"])
        
        return resampled_aur, df

def optimize_dataframe_operations(df, o_df, bin_num):
    """Optimized version of the DataFrame matching logic"""
    if df.empty:
        return o_df
    
    # Group by Gold row and col to avoid repeated filtering
    grouped = df.groupby(['Gold row', 'Gold col'])
    
    new_rows = []
    for (row, col), group in grouped:
        # Find matching rows using merge (more efficient than nested loops)
        matching_rows = pd.merge(group, o_df, on=["OP_mlt_idx", "OP_mlat_idx"], how="inner")
        
        if len(matching_rows) >= bin_num:
            # Add unique combinations only
            unique_coords = group[["OP_mlt_idx", "OP_mlat_idx"]].drop_duplicates()
            new_rows.append(unique_coords)
    
    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        # Combine and remove duplicates
        combined_df = pd.concat([o_df, new_df], ignore_index=True)
        return combined_df.drop_duplicates(subset=["OP_mlt_idx", "OP_mlat_idx"]).reset_index(drop=True)
    
    return o_df

# Pre-compute commonly used values
mask_template = np.copy(lat)
mask_template[mask_template != 0] = 1
mask_template[mask_template == 0] = np.nan

lat_processed = np.copy(lat)
lat_processed[np.isnan(lat_processed)] = -30
lon_processed = np.copy(lon)
lon_processed[np.isnan(lon_processed)] = 0

num_bins_list = [3]
#COMPARISON - OPTIMIZED
mcc_list = {k: [] for k in num_bins_list}
cm_list = {k: [] for k in num_bins_list}
acc_list = {k: [] for k in num_bins_list}
f1_list = {k: [] for k in num_bins_list}
prec_list = {k: [] for k in num_bins_list}
rec_list = {k: [] for k in num_bins_list}

# In[]



uj = np.linspace(0, 24, num = 12, endpoint = False)
dj = np.linspace(1, 365, num = 30, endpoint = False)
kj = np.linspace(0,9, num = 9, endpoint = False)

mcc_img = [[[[] for _ in kj] for _ in dj] for _ in uj]

for i, key in tqdm(enumerate(results_files), total=len(results_files), desc="Data Loading"):
    
    # Load and process data
    kp = results_files[key][1]
    OP_data = Q_list[kp_list.index(kp)]
    aur_data = results_files[key][0]
    
    doy = int(key.split('_')[0])
    hr, mi = key.split('_')[1].split('.')
    mi = int(mi[:2])
    
    utime = dt.datetime.strptime(f'{yr} {doy}', '%Y %j')
    utime = utime.replace(hour=int(hr), minute=int(int(mi)/60), second=0)
    
    # Use pre-processed masks and coordinates
    mask = np.copy(nan_mask)
    
    apex = apexpy.Apex(date=utime)
    mlat, mlon = apex.convert(ES_lats, ES_lons, 'geo', 'qd', height=150)
    mlt = apex.mlon2mlt(mlon, dtime=utime)

    aur_data[aur_data != 0] = 1
    aur_data = aur_data * mask
    mlat = mlat * mask
    mlt = mlt * mask
    
    mlat[np.isnan(mlat)] = 0
    mlt[np.isnan(mlt)] = 0
    
    # Pre-compute median filter once
    clean_OP = np.copy(OP_data)
    
    # Use optimized resampling function
    resampled_aur_data, df = resample_gold_vectorized(mlat, mlt, OP_data, fov=False, aur_data=aur_data)
    for bin_num in num_bins_list:
        # Pre-compute FOV once per bin_num
        resampled_gold_fov = resample_gold_vectorized(mlat, mlt, OP_data, fov=True)
        gold_fov = np.clip(np.nansum(resampled_gold_fov, axis=0), 0, 1)
        gold_fov[gold_fov != 1] = np.nan
        
        for thresh in e_flux_list:
            # Vectorized threshold operation
            abc = (clean_OP > thresh).astype(int)
            
            ground_truth = np.clip(np.nansum(resampled_gold_fov , axis=0), 0, 1) * abc
            
            # More efficient way to get indices
            orow, ocol = np.where(ground_truth != 0)
            o_df = pd.DataFrame({
                "OP_mlt_idx": orow,
                "OP_mlat_idx": ocol
            })
            
            # Use optimized DataFrame operations
            o_df = optimize_dataframe_operations(df, o_df, bin_num)
            
            # Vectorized final ground truth creation
            ground_truth_final = np.zeros((len(mlt_list), len(mlat_list)))
            if not o_df.empty:
                r = o_df["OP_mlt_idx"].to_numpy(dtype=int)
                c = o_df["OP_mlat_idx"].to_numpy(dtype=int)
                ground_truth_final[r, c] = 1
            ground_truth_final = ground_truth_final * gold_fov
    
            final_aur_data = np.clip(np.nansum(resampled_aur_data, axis=0), 0, 1)
            final_aur_data = final_aur_data * gold_fov
            
            # Compute metrics
            mask = ~np.isnan(ground_truth_final)
            gt_clean = ground_truth_final[mask].astype(int)
            pred_clean = final_aur_data[mask].astype(int)
            
            mcc = matthews_corrcoef(gt_clean, pred_clean)
            cm = confusion_matrix(gt_clean, pred_clean)
            precision = precision_score(gt_clean, pred_clean)
            recall = recall_score(gt_clean, pred_clean)
            f1 = f1_score(gt_clean, pred_clean)
            accuracy = accuracy_score(gt_clean, pred_clean)
            
            cm_list[bin_num].append(cm)
            mcc_list[bin_num].append(mcc)
            acc_list[bin_num].append(accuracy)
            f1_list[bin_num].append(f1)
            prec_list[bin_num].append(precision)
            rec_list[bin_num].append(recall)

            ut = int(hr) + mi*1e-2
            pp = (np.max(uj) - np.min(uj))+ .5* (np.max(uj) - np.min(uj))/(len(uj) -1)
            ll = (np.max(dj) - np.min(dj))+ .5* (np.max(dj) - np.min(dj))/(len(dj) -1)

            uj_idx = np.argmin( np.abs(ut%pp - uj))
            dj_idx = np.argmin( np.abs(doy%ll - dj ))
            kj_idx = np.argmin( np.abs(kp - kj))
            mcc_img[uj_idx][dj_idx][kj_idx].append(mcc)


# In[]
plt.figure()
sns.histplot(mcc_list, bins=10, kde=True)  # Add kde=True if you want a density curve
plt.xlabel("MCC Coefficient")
plt.ylabel("Number Gold Scans")
plt.title("MCC Coefficient distribution 2020")
plt.show()


confus_matrix = [np.sum(np.array(cm_list[key]), axis = 0) for key in cm_list]
confus_matrix = np.squeeze(confus_matrix/np.sum(confus_matrix))

plt.figure(figsize=(6, 5))

# # Use scientific format (e.g., 1.23e+03)
sns.heatmap(confus_matrix, annot=True, cmap='Blues', cbar=True,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


d_cm,d_mcc,d_acc,d_f1,d_prec,d_rec = [],[],[],[],[], []


for key in cm_list:
    dd = np.sum(cm_list[key], axis = 0)
    d_cm.append(dd)
    d_mcc.append(np.mean(mcc_list[key]))
    d_acc.append(np.mean(acc_list[key]))
    d_f1.append(np.mean(f1_list[key]))
    d_prec.append(np.mean(prec_list[key]))
    d_rec.append(np.mean(rec_list[key]))

print('Accuracy: ', d_acc)
print('Recall: ', d_rec)
print('Precision: ', d_prec)
print('MCC: ', d_mcc)
print('F1: ', d_f1)

import numpy as np

final_mcc_image = np.zeros((len(mcc_img), len(mcc_img[0]), len(mcc_img[0][1])))

for i in range(len(mcc_img)):
    for j in range(len(mcc_img[0])):
        for k in range(len(mcc_img[0][1])):
            if mcc_img[i][j][k]:  # not empty
                final_mcc_image[i][j][k] = np.nanmean(mcc_img[i][j][k])
                if np.nanmean(mcc_img[i][j][k]) == 0:
                    final_mcc_image[i][j][k] = np.nan
            else:
                final_mcc_image[i][j][k] = np.nan  # or some other fill value


plt.figure()
plt.pcolor(uj, dj, np.nanmedian(final_mcc_image[:,:,:3], axis = -1).T)
plt.colorbar()
plt.title('MCC vs doy and UT')
plt.ylabel('doy')
plt.xlabel('UT')
plt.show()



uj_len = len(mcc_img)
dj_len = len(mcc_img[0])
kj_len = len(mcc_img[0][0])

# Store median values across kj (10) for each (uj, dj)
median_kj = np.full((uj_len, dj_len, kj_len), np.nan)

for i in range(uj_len):
    for j in range(dj_len):
        for k in range(kj_len):
            values = mcc_img[i][j][k]
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                median_kj[i, j, k] = np.nanmedian(values)
                
jah = [np.nanmedian(median_kj[:,:,k]) for k in range(median_kj.shape[-1])]                

plt.figure()
plt.plot(kj, jah)
plt.xlabel('kp value')
plt.ylabel('median mcc')
plt.title('median mcc vs kp')
plt.show()



