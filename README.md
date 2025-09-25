# Dayside Aurora Dataset (GOLD Mission)
<img src="https://github.com/jah-26603/dayside_aurora_gold/blob/main/36b48e9dc798b6a129637a9bdd91230f%20(1).gif?raw=true" alt="Aurora" width="1000">
This dataset provides reduced measurements of the dayside aurora from the GOLD mission, spanning October 2018 – June 2025.  
The key scientific result is that the dayside aurora can be extracted pixel-wise with background dayglow removed. This dataset includes raw scans, estimated dayglow, and binary auroral masks to enable direct analysis.

(link to dataset here)

---

## Dataset Structure

`Dataset.zip` (folder): This contains all measurements of the dayside aurora.  

Dataset.zip

├── [2018]

├── [2019]

│ ├── 001.nc

│ ├── 002.nc

│ └── ... 365.nc

├── [2020]

│ └── ...

└── [2025]



**Dataset (zip):** Contains GOLD mission measurements from October 2018 - June 2025 used in this dataset.  
**Year Folders:** Each folder corresponds to a calendar year and contains the reduced daily measurement files.  

**Day Files (.nc):** Within these day files are the main contents of dataset. These files are derived from an entire day of GOLD Level 1C DAY (L1C DAY) products. Each of these products are in array format (52 rows x 92 columns). Since the instrument is onboard a geostationary satellite, the geographic and longitude coordinates are constant throughout the mission. The third dimension of this image, number of scans, indicates how many scans were performed for a given day.

- Raw scans from GOLD files of three species: 135.6nm , 1493.nm, & LBH emissions.
- Dayglow estimates - representing light pollution (non auroral emissions) that contaminate the images.
- Binary Masks - Estimated auroral locations in the images.
- Geographic Latitude & Longitude.
- Universal Time. 
- Magnetic Latitude & Local Time (apex quasi-dipole).
- Solar Zenith Angle & Emission Angle.

---

Model Weights

The folder model_weights/ contains pretrained UNet models used to generate:  
best_model_seg: Binary mask generation (segmentation).  
best_model_reg: Dayglow estimation (regression).  

These weights are automatically downloaded when running the code repository. No manual download required.

---

## Usage Example of dataset in Python

```python
ds = nc.Dataset('2019/001.nc')
aurora = ds.variables['north'][:] - ds.variables['dayglow'][:]
mask = ds.variables['final_auroral_mask'][:] 
aurora = np.clip(aurora, 0, np.inf)
aurora = aurora * mask
```

## To generate dataset (validation of this study)
This requires multiple steps:  
1) To download L1C DAY files -> this code to automatically download from the bash is in the 'download_L1C_files' folder. Edit the bash file for a given year and the output path.  
2) To download the model weights -> just run the 'download_model_weights.py' file, and the mdoel weights will be retrieved from the dataset repo.
3) The only thing that needs to be edited, is the config file for pathing, method used, etc. 

## To reproduce model weights
This requires multiple steps:  
1) To download L1C DAY files -> this code to automatically download from the bash is in the 'download_L1C_files' folder. Edit the bash file for a given year and the output path. The entire missions worth ~10 Tb. 
2) The only thing that needs to be edited, is pathing.
3) The two different files refer to the different model tasks trained. Refer to paper for more specifics.

Reference

If you use this dataset, or weights, please cite here:
(add paper/preprint link)
