"""
Created on Fri Mar  5 17:07:47 2021

@author: sethn
"""

#Import packages
import numpy as np
import numpy.ma as ma
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
directory = 'C:///Users/sgoldst3/Inglefield/PyTrx/Seth_Examples'


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

# plt.scatter(xyz_df2019['lineloc'], xyz_df2019['z'])
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
        
projectdf_z2019['z_normalized'] = projectdf_z2019['z']-projectdf_z2019['z'].mean()   

projectdf_z2019['bubbler_resampled'] = projectdf_z2019['stage_filtered'].resample('1D').mean()
projectdf_z2019['z_resampled'] = projectdf_z2019['z_normalized'].resample('1D').mean()

projectdf_z2019['nice_datetimes']= projectdf_z2019.index.strftime("%b %d")


fig1, ax1 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax1[0].plot(projectdf_z2019.index, projectdf_z2019['z_normalized'])
ax1[0].set_ylabel('Water Level (m)')
ax1[0].grid(linestyle='dashed')
ax1[0].set_ylim(-4, 4)
ax1[1].plot(projectdf_z2019.index, projectdf_z2019['stage_filtered'])
ax1[1].set_ylabel('Filtered Stage (m)')
ax1[1].grid(linestyle='dashed')
ax1[1].set_ylim(-4, 4)

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
        
projectdf_z2020['z_normalized'] = projectdf_z2020['z']-projectdf_z2020['z'].mean()  

projectdf_z2020['bubbler_resampled'] = projectdf_z2020['stage_filtered'].resample('1D').mean()
projectdf_z2020['z_resampled'] = projectdf_z2020['z_normalized'].resample('1D').mean()  

projectdf_z2020['nice_datetimes']= projectdf_z2020.index.strftime("%b %d")
        

fig2, ax2 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax2[0].plot(projectdf_z2020.index, projectdf_z2020['z_normalized'])
ax2[0].set_ylabel('Water Level (m)')
ax2[0].grid(linestyle='dashed')
ax2[0].set_ylim(-4, 4)
ax2[1].plot(projectdf_z2020.index, projectdf_z2020['stage_filtered'])
ax2[1].set_ylabel('Filtered Stage (m)')
ax2[1].grid(linestyle='dashed')
ax2[1].set_ylim(-4, 4)

ax2[0].tick_params(axis = 'x', labelrotation = 45)
ax2[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()   

# slope19_corr, intercept19_corr, r_value19_corr, p_value19_corr, std_err19_corr = stats.linregress(projectdf_z2019['stage_filtered'], projectdf_z2019['z'])
# print (slope19_corr, intercept19_corr, r_value19_corr, p_value19_corr)

## 2021 ##

mergedf2021 = pd.read_csv(directory + '/results/manual_detection_results2021.csv', parse_dates=['key_0'], index_col='key_0')

mergedf2021_filtered = mergedf2021.loc[mergedf2021['lineloc'] <= 300]

xyz_df2021 = pd.DataFrame(columns = ['datetime', 'lineloc', 'x', 'y', 'z'])
for index1, row1 in mergedf2021_filtered.iterrows():
    for index2, row2 in df.iterrows():
        if row1['lineloc'] == index2:
            xyz_df2021 = xyz_df2021.append({'datetime': index1, 'lineloc': row1['lineloc'], 'x': row2['x'], 'y': row2['y'], 'z': row2['z'] }, ignore_index = True)

# plt.scatter(xyz_df2020['lineloc'], xyz_df2020['z'])
# slope20, intercept20, r_value20, p_value20, std_er20r = stats.linregress(xyz_df2020['lineloc'], xyz_df2020['z'])
# print (slope20, intercept20, r_value20, p_value20)

projectdf_z2021 = mergedf2021[['lineloc', 'stage_filtered']]
projectdf_z2021['z']= np.nan

for index, row in projectdf_z2021.iterrows():
   # if the lineloc is <= 300, add z value from df['z'] at that index
    if row['lineloc'] <= 300:
        z_row = row['lineloc'].astype(int)
        df_row = df.iloc[z_row]
        row['z'] = df_row['z']
    # else if lineloc > 300, z-value = slope * lineloc + intercept (y= mx + b)
    elif row['lineloc'] > 300:
        row['z'] = slope19 * row['lineloc'] + intercept19
        
projectdf_z2021['z_normalized'] = projectdf_z2021['z']-projectdf_z2021['z'].mean()    

projectdf_z2021['bubbler_resampled'] = projectdf_z2021['stage_filtered'].resample('1D').mean()
projectdf_z2021['z_resampled'] = projectdf_z2021['z_normalized'].resample('1D').mean()


projectdf_z2021['nice_datetimes']= projectdf_z2021.index.strftime("%b %d")

# fig4, ax4 = plt.subplots(2, constrained_layout = True, sharex = 'col')

# ax4[0].plot(projectdf_z2021.index, projectdf_z2021['z_normalized'])
# ax4[0].set_ylabel('Water Level (m)')
# ax4[0].grid(linestyle='dashed')
# ax4[0].set_ylim(-4, 5)
# ax4[1].plot(projectdf_z2021.index, projectdf_z2021['stage_filtered'])
# ax4[1].set_ylabel('Filtered Stage (m)')
# ax4[1].grid(linestyle='dashed')
# ax4[1].set_ylim(-4, 5)

# ax4[0].tick_params(axis = 'x', labelrotation = 45)
# ax4[1].tick_params(axis = 'x', labelrotation = 45)

# plt.show()   


# Scatter plots#


fig5, ax5 = plt.subplots(nrows=2, ncols=3, sharex = 'col', sharey= 'row')

# fig5.suptitle("Water Level/Pressure Transducer Correlation")

ax5[0,0].scatter(projectdf_z2019['stage_filtered'], projectdf_z2019['z_normalized'], facecolors='none', edgecolors='black')
ax5[0,0].set_title('2019')
# ax5[0,0].set_xlabel('Filtered Stage (m)')
# ax5[0,0].set_ylabel('Water level (m)')

ax5[0,1].scatter(projectdf_z2020['stage_filtered'], projectdf_z2020['z_normalized'], facecolors='none', edgecolors='black')
ax5[0,1].set_title('2020')
# ax5[0,1].set_xlabel('Filtered Stage (m)')
# ax5[0,1].set_ylabel('Water level (m)')

ax5[0,2].scatter(projectdf_z2021['stage_filtered'], projectdf_z2021['z_normalized'], facecolors='none', edgecolors='black')
ax5[0,2].set_title('2021')
# ax5[0,2].set_xlabel('Filtered Stage (m)')
# ax5[0,2].set_ylabel('Water level (m)')

ax5[1,0].scatter(projectdf_z2019['bubbler_resampled'], projectdf_z2019['z_resampled'], facecolors='none', edgecolors='black')
# ax5[1,0].set_title('2019 correlation')
# ax5[1,0].set_xlabel('Filtered Stage (m)')
# ax5[1,0].set_ylabel('Water level (m)')

ax5[1,1].scatter(projectdf_z2020['bubbler_resampled'], projectdf_z2020['z_resampled'], facecolors='none', edgecolors='black')
# ax5[1,1].set_title('2020 correlation')
# ax5[1,1].set_xlabel('Filtered Stage (m)')
# ax5[0,1].set_ylabel('Water level (m)')

ax5[1,2].scatter(projectdf_z2021['bubbler_resampled'], projectdf_z2021['z_resampled'], facecolors='none', edgecolors='black')
# ax5[1,2].set_title('2021 correlation')
# ax5[1,2].set_xlabel('Filtered Stage (m)')
# ax5[0,2].set_ylabel('Water level (m)')

fig5.text(0.5, 0.02, 'Bubbler Stage (m)', ha='center')
fig5.text(0.04, 0.5, 'Camera Water Level (m)', va='center', rotation='vertical')

# fig5.add_subplot(1, 1, 1, frame_on=False)

# # Hiding the axis ticks and tick labels of the bigger plot
# # ax5.tick_params(labelcolor="none", bottom=False, left=False)

# # Adding the x-axis and y-axis labels for the bigger plot
# fig5.xlabel('Filtered Stage (m)', fontsize=15, fontweight='bold')
# fig5.ylabel('Water Level (m)', fontsize=15, fontweight='bold')


plt.show()

fig6, ax6 = plt.subplots(3)
fig6.tight_layout()
ax6[0].plot(projectdf_z2019.index, projectdf_z2019['z_normalized'], color = 'black', label='Camera Water Levels')
ax6[0].plot(projectdf_z2019.index, projectdf_z2019['stage_filtered'], color = 'red', label='Bubbler Stage')
ax6[0].grid(linestyle='dashed')
ax6[0].set_ylim(-5, 5)
ax6[0].set_ylabel('2019')
ax6[0].set_xticklabels(projectdf_z2019['nice_datetimes'], fontsize=10)
ax6[0].tick_params(axis='both', which='major', direction='out', labelsize=10, width=2)


ax6[1].plot(projectdf_z2020.index, projectdf_z2020['z_normalized'], color = 'black')
ax6[1].plot(projectdf_z2020.index, projectdf_z2020['stage_filtered'], color = 'red')
ax6[1].grid(linestyle='dashed')
ax6[1].set_ylim(-5, 5)
ax6[1].set_ylabel('2020')
ax6[1].set_xticklabels(projectdf_z2020['nice_datetimes'], fontsize=10)
ax6[1].tick_params(axis='both', which='major', direction='out', labelsize=10, width=2)


ax6[2].plot(projectdf_z2021.index, projectdf_z2021['z_normalized'], color = 'black')
ax6[2].plot(projectdf_z2021.index, projectdf_z2021['stage_filtered'], color = 'red')
ax6[2].grid(linestyle='dashed')
ax6[2].set_ylim(-5, 5)
ax6[2].set_ylabel('2021')
ax6[2].set_xticklabels(projectdf_z2021['nice_datetimes'], fontsize=10)
ax6[2].tick_params(axis='both', which='major', direction='out', labelsize=10, width=2)

# ax6.set_xticks(np.arange(times[0], times[-1], 15))
# ax6.set_xticklabels(projectdf_z2021['nice_datetimes'], fontsize=10)
fig6.legend(loc='upper right')
plt.show()

## Linear Regression stats
#2019
projectdf_z2019_2 = projectdf_z2019.dropna()
slope19_hour, intercept19_hour, r_value19_hour, p_value19_hour, std_err19_hour = stats.linregress(projectdf_z2019_2['stage_filtered'],projectdf_z2019_2['z_normalized'])
print('2019 lin regress values (3 hours): ' + 'slope:' + str(slope19_hour) + ' intercept:' + str(intercept19_hour) +' r squared:' + str(r_value19_hour) + ' p value:' + str(p_value19_hour) + ' std error:' + str(std_err19_hour))
# 
slope19_day, intercept19_day, r_value19_day, p_value19_day, std_err19_day = stats.linregress(projectdf_z2019_2['bubbler_resampled'],projectdf_z2019_2['z_resampled'])
print('2019 lin regress values (days): ' + 'slope:' + str(slope19_day) + ' intercept:' + str(intercept19_day) +' r squared:' + str(r_value19_day) + ' p value:' + str(p_value19_day) + ' std error:' + str(std_err19_day))

#2020
projectdf_z2020_2 = projectdf_z2020.dropna()
slope20_hour, intercept20_hour, r_value20_hour, p_value20_hour, std_err20_hour = stats.linregress(projectdf_z2020_2['stage_filtered'],projectdf_z2020_2['z_normalized'])
print('2020 lin regress values (3 hours): ' + 'slope:' + str(slope20_hour) + ' intercept:' + str(intercept20_hour) +' r squared:' + str(r_value20_hour) + ' p value:' + str(p_value20_hour) + ' std error:' + str(std_err20_hour))
# 
slope20_day, intercept20_day, r_value20_day, p_value20_day, std_err20_day = stats.linregress(projectdf_z2020_2['bubbler_resampled'],projectdf_z2020_2['z_resampled'])
print('2020 lin regress values (days): ' + 'slope:' + str(slope20_day) + ' intercept:' + str(intercept20_day) +' r squared:' + str(r_value20_day) + ' p value:' + str(p_value20_day) + ' std error:' + str(std_err20_day))

#2021
projectdf_z2021_2 = projectdf_z2021.dropna()
slope21_hour, intercept21_hour, r_value21_hour, p_value21_hour, std_err21_hour = stats.linregress(projectdf_z2021_2['stage_filtered'],projectdf_z2021_2['z_normalized'])
print('2021 lin regress values (3 hours): ' + 'slope:' + str(slope21_hour) + ' intercept:' + str(intercept21_hour) +' r squared:' + str(r_value21_hour) + ' p value:' + str(p_value21_hour) + ' std error:' + str(std_err21_hour))
# 
slope21_day, intercept21_day, r_value21_day, p_value21_day, std_err21_day = stats.linregress(projectdf_z2021_2['bubbler_resampled'],projectdf_z2021_2['z_resampled'])
print('2021 lin regress values (days): ' + 'slope:' + str(slope21_day) + ' intercept:' + str(intercept21_day) +' r squared:' + str(r_value21_day) + ' p value:' + str(p_value21_day) + ' std error:' + str(std_err21_day))
