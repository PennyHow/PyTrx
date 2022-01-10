"""
Created on Fri Mar  5 17:07:47 2021

@author: sethn
"""

#Import packages
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
sys.path.append('../')
import osgeo.ogr as ogr
import osgeo.osr as osr
from scipy import interpolate
from scipy import stats


#from pyproj import Proj
from CamEnv import GCPs, CamEnv, setProjection, projectUV, projectXYZ, optimiseCamera, computeResidualsXYZ
from PyTrx.Images import CamImage
import DEM
import pandas as pd

# =============================================================================
# Define camera environment
# =============================================================================

directory = os.getcwd()
ingleCamEnv = directory + '/cam_env/Inglefield_CAM_camenv.txt'
ingle_calibimg = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.JPG'
ingle_img = CamImage(ingle_calibimg)


#Define data output directory
destination = directory + '/results'
if not os.path.exists(destination):
    os.makedirs(destination)
print(destination)

# Define camera environment
ingleCam = CamEnv(ingleCamEnv)

# Get DEM from camera environment
dem = ingleCam.getDEM()

# Get GCPs
inglefield_gcps = directory + '/cam_env/GCPs_20190712.txt'
gcps = GCPs(dem, inglefield_gcps, ingle_calibimg)

xy = np.arange(-350,350).reshape(2, 350)
xy = np.swapaxes(xy, 0, 1)
xy[:,0].fill(50)

# Report calibration data
ingleCam.reportCalibData()

# Optimise camera environment
ingleCam.optimiseCamEnv('YPR', 'trf', show=False)

#Get inverse projection variables through camera info            
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
                            ingleCam._camCen, ingleCam._refImage, viewshed=False)

#Inverse project image coordinates using function from CamEnv object                       
ingle_xyz = projectUV(xy, invprojvars)

df = pd.DataFrame(ingle_xyz)
df.columns= ['x', 'y', 'z']


# #------------------   Export xyz locations as .csv file   ---------------------


# df . csv #
df.to_csv(directory + '/results/orthorectification_results.csv')

print('\n\nSAVING TEXT FILE')


# =============================================================================
# Orthorectify all images 
# =============================================================================

## 2019 ##
mergedf2019 = pd.read_csv(directory + '/results/manual_detection_results2019.csv', parse_dates=['key_0'], index_col='key_0')

mergedf2019_filtered = mergedf2019.loc[mergedf2019['lineloc'] <= 300]

xyz_df2019 = pd.DataFrame(columns = ['datetime', 'lineloc', 'x', 'y', 'z'])
for index1, row1 in mergedf2019_filtered.iterrows():
    for index2, row2 in df.iterrows():
        if row1['lineloc'] == index2:
            xyz_df2019 = xyz_df2019.append({'datetime': index1, 'lineloc': row1['lineloc'], 'x': row2['x'], 'y': row2['y'], 'z': row2['z'] }, ignore_index = True)

plt.scatter(xyz_df2019['lineloc'], xyz_df2019['z'])
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(xyz_df2019['lineloc'], xyz_df2019['z'])
print (slope19, intercept19, r_value19, p_value19)


projectdf_z2019 = mergedf2019[['lineloc', 'stage_filtered']]
projectdf_z2019['z']= np.nan



for index, row in projectdf_z2019.iterrows():
   # if the lineloc is <= 300, add z value from df['z'] at that index
    if row['lineloc'] <= 300:
        z_row = row['lineloc'].astype(int)
        df_row = df.iloc[z_row]
        row['z'] = df_row['z']
    # else if lineloc > 300, z-value = slope * lineloc + intercept (y= mx + b)
    elif row['lineloc'] > 300:
        row['z'] = slope19 * row['lineloc'] + intercept19
        
fig1, ax1 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax1[0].plot(projectdf_z2019.index, projectdf_z2019['z'])
ax1[0].set_ylabel('Water Level (m)')
ax1[0].set_title('2019 Data')
ax1[0].grid(linestyle='dashed')
ax1[1].plot(projectdf_z2019.index, projectdf_z2019['stage_filtered'])
ax1[1].set_ylabel('Filtered Stage (m)')
ax1[1].grid(linestyle='dashed')

ax1[0].tick_params(axis = 'x', labelrotation = 45)
ax1[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()    

## 2020 ##

mergedf2020 = pd.read_csv(directory + '/results/manual_detection_results2020.csv', parse_dates=['key_0'], index_col='key_0')

mergedf2020_filtered = mergedf2020.loc[mergedf2020['lineloc'] <= 300]

xyz_df2020 = pd.DataFrame(columns = ['datetime', 'lineloc', 'x', 'y', 'z'])
for index1, row1 in mergedf2020_filtered.iterrows():
    for index2, row2 in df.iterrows():
        if row1['lineloc'] == index2:
            xyz_df2020 = xyz_df2020.append({'datetime': index1, 'lineloc': row1['lineloc'], 'x': row2['x'], 'y': row2['y'], 'z': row2['z'] }, ignore_index = True)

# plt.scatter(xyz_df2020['lineloc'], xyz_df2020['z'])
# slope20, intercept20, r_value20, p_value20, std_er20r = stats.linregress(xyz_df2020['lineloc'], xyz_df2020['z'])
# print (slope20, intercept20, r_value20, p_value20)

projectdf_z2020 = mergedf2020[['lineloc', 'stage_filtered']]
projectdf_z2020['z']= np.nan

for index, row in projectdf_z2020.iterrows():
   # if the lineloc is <= 300, add z value from df['z'] at that index
    if row['lineloc'] <= 300:
        z_row = row['lineloc'].astype(int)
        df_row = df.iloc[z_row]
        row['z'] = df_row['z']
    # else if lineloc > 300, z-value = slope * lineloc + intercept (y= mx + b)
    elif row['lineloc'] > 300:
        row['z'] = slope19 * row['lineloc'] + intercept19
        
fig2, ax2 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax2[0].plot(projectdf_z2020.index, projectdf_z2020['z'])
ax2[0].set_ylabel('Water Level (m)')
ax2[0].set_title('2020 Data')
ax2[0].grid(linestyle='dashed')
ax2[1].plot(projectdf_z2020.index, projectdf_z2020['stage_filtered'])
ax2[1].set_ylabel('Filtered Stage (m)')
ax2[1].grid(linestyle='dashed')

ax2[0].tick_params(axis = 'x', labelrotation = 45)
ax2[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()   







