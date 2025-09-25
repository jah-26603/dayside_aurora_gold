# Dayside Aurora Dataset (GOLD Mission)

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

## Usage Example in Python

```python
ds = nc.Dataset('2019/001.nc')
aurora = ds.variables['north'][:] - ds.variables['dayglow'][:]
mask = ds.variables['final_auroral_mask'][:] 
aurora = np.clip(aurora, 0, np.inf)
aurora = aurora * mask
```

<img src="https://github.com/jah-26603/dayside_aurora_gold/blob/main/36b48e9dc798b6a129637a9bdd91230f(1).gif?raw=true" alt="Aurora GIF" width="2000">




Reference

If you use this dataset, or weights, please cite here:
(add paper/preprint link)
