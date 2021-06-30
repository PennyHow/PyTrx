# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:07:33 2021

@author: sethn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import stats
from skimage import feature
from skimage.io import imread
import glob
import pandas as pd
import pathlib
from datetime import datetime
import os, sys

sys.path.append('../')


# Load image
directory = os.getcwd()
inglecam_img1 = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.JPG'

image1 = imread(inglecam_img1, as_gray= True)
image1 = ndi.gaussian_filter(image1, 3)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image1)
edges2 = feature.canny(image1, sigma=3)


# Compute row with most edges
rows1, cols1 = edges2.shape
rowsum = np.sum(edges2.astype(int), axis = 1)
edges2line = [rowsum.argmax()]*cols1

# display results
fig1, ax1 = plt.subplots(nrows=3, ncols=3)

ax1[0,0].imshow(image1, cmap='gray')
ax1[0,0].set_ylabel('7/12/2019', fontsize=10)
ax1[0,0].set_title('Greyscale image')

ax1[0,1].plot(rowsum)
ax1[0,1].set_title('# of edge pixels by row')

ax1[0,2].plot(edges2line, color = "red", linewidth=1)
ax1[0,2].imshow(edges2[:, :], cmap='gray')
ax1[0,2].set_title('Canny filter, $\sigma=3$')


# Image 2
inglecam_img2 = directory + '/Inglefield_Data/INGLEFIELD_CAM/2020/INGLEFIELD_CAM_StarDot1_20200726_000000.jpg'
image2 = imread(inglecam_img2, as_gray=True)

image2 = ndi.gaussian_filter(image2, 3)

# Compute the Canny filter for two values of sigma
edges3 = feature.canny(image2)
edges4 = feature.canny(image2, sigma=3)

# Compute row with most edges
rows2, cols2 = edges4.shape
rowsum2 = np.sum(edges4.astype(int), axis = 1)
edges4line = [rowsum2.argmax()]*cols2

ax1[1,0].imshow(image2, cmap='gray')
ax1[1,0].set_ylabel('7/26/2020', fontsize=10)

ax1[1,1].plot(rowsum2)

ax1[1,2].plot(edges4line, color = "red", linewidth=1)
ax1[1,2].imshow(edges4, cmap='gray')


# Image 3
inglecam_img3 = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190810_210000.jpg'
image3 = imread(inglecam_img3, as_gray=True)

image3 = ndi.gaussian_filter(image3, 3)

# Compute the Canny filter for two values of sigma
edges5 = feature.canny(image3)
edges6 = feature.canny(image3, sigma=3)

# Compute row with most edges
rows3, cols3 = edges6.shape
rowsum3 = np.sum(edges6.astype(int), axis = 1)
edges6line = [rowsum3.argmax()]*cols3

ax1[2,0].imshow(image3, cmap='gray')
ax1[2,0].set_ylabel('8/10/2019', fontsize=10)

ax1[2,1].plot(rowsum3)

ax1[2,2].plot(edges6line, color = "red", linewidth=1)
ax1[2,2].imshow(edges6, cmap='gray')

fig1.tight_layout()
plt.show()


# =============================================================================
# Edge Detect Entire directory
# =============================================================================


# 2019
waterlevels2019 = []
dir2019files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/*'
directory2019 = glob.glob(dir2019files)
time_list2019 = []

for i in directory2019:
   
    # parse time stamp
    filename = pathlib.Path(i)
    filename = filename.stem
    filename = filename.split("_")[-2:]
    filename = ["".join(filename)]
    filename = filename[0]
    time = datetime.strptime(filename, '%Y%m%d%H%M%S')
    # time_list2019.append(time)
    
    # Read and filter image
    image = imread(i, as_gray=True)
    image = ndi.gaussian_filter(image, 3)
       
    # Compute the Canny filter for two values of sigma
    edges = feature.canny(image, sigma=3)

    # Compute row with most edges
    rows, cols = edges.shape
    rowsum = np.sum(edges.astype(int), axis = 1)
    waterlevels2019.append(rowsum.argmax())
    time_list2019.append(tuple([time,rowsum.argmax()]))


df_filter1 = pd.DataFrame (waterlevels2019)
df_filter1.columns = ['water level']
z = np.abs(stats.zscore(df_filter1))
Q1_2019 = df_filter1.quantile(0.25)
Q3_2019 = df_filter1.quantile(0.75)
IQR_2019 = Q3_2019 - Q1_2019

df1 = pd.DataFrame(time_list2019)
df1.columns = ['time', 'water level']
df1 = df1.set_index('time')
print(df1.shape)

fig2, ax2 = plt.subplots(1)
ax2.scatter(df1.index, df1['water level'], c = 'blue')
ax2.tick_params(axis = 'x', labelrotation = 45)
ax2.set_title('2019 unfiltered data')
ax2.set_xlabel('Time')
ax2.invert_yaxis()

df1.drop(df1[df1['water level'] < (Q1_2019[0] -1.5 * IQR_2019[0])].index, inplace=True)
df1.drop(df1[df1['water level'] > (Q3_2019[0] +1.5 *IQR_2019[0])].index, inplace=True)
print(df1.shape)
fig3, ax3 = plt.subplots(1)
ax3.plot(df1.index, df1['water level'], c = 'green')
ax3.tick_params(axis = 'x', labelrotation = 45)
ax3.set_title('2019 corrected data')
ax3.set_xlabel('Time')
ax3.invert_yaxis()

# 2020 Data
waterlevels2020 = []
dir2020files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2020/*'
directory2020 = glob.glob(dir2020files)
time_list2020 = []

for i in directory2020:
   
    # parse time stamp
    filename = pathlib.Path(i)
    filename = filename.stem
    filename = filename.split("_")[-2:]
    filename = ["".join(filename)]
    filename = filename[0]
    time = datetime.strptime(filename, '%Y%m%d%H%M%S')
    # time_list2019.append(time)
    
    # Read and filter image
    image = imread(i, as_gray=True)
    image = ndi.gaussian_filter(image, 3)
       
    # Compute the Canny filter for two values of sigma
    edges = feature.canny(image, sigma=3)

    # Compute row with most edges
    rows, cols = edges.shape
    rowsum = np.sum(edges.astype(int), axis = 1)
    waterlevels2020.append(rowsum.argmax())
    time_list2020.append(tuple([time,rowsum.argmax()]))


df_filter2 = pd.DataFrame (waterlevels2020)
df_filter2.columns = ['water level']
z2 = np.abs(stats.zscore(df_filter2))
Q1_2020 = df_filter2.quantile(0.3)
Q3_2020 = df_filter2.quantile(0.7)
IQR_2020 = Q3_2020 - Q1_2020

df2 = pd.DataFrame(time_list2020)
df2.columns = ['time', 'water level']
df2 = df2.set_index('time')
print(df2.shape)

fig4, ax4 = plt.subplots(1)
ax4.scatter(df2.index, df2['water level'], c = 'blue')
ax4.tick_params(axis = 'x', labelrotation = 45)
ax4.set_title('2020 unfiltered data')
ax4.set_xlabel('Time')
ax4.invert_yaxis()

df2.drop(df2[df2['water level'] < (Q1_2020[0] -1.5 * IQR_2020[0])].index, inplace=True)
df2.drop(df2[df2['water level'] > (Q3_2020[0] +1.5 *IQR_2020[0])].index, inplace=True)
print(df2.shape)

fig5, ax5 = plt.subplots(1)
ax5.plot(df2.index, df2['water level'], c = 'green')
ax5.tick_params(axis = 'x', labelrotation = 45)
ax5.set_title('2020 corrected data')
ax5.set_xlabel('Time')
ax5.invert_yaxis()


# =============================================================================
# Bubbler Data
# =============================================================================

bubblerdata2019 = directory + '/Inglefield_Data/modified_2019_excel.xlsx'
bubdf2019 = pd.read_excel(bubblerdata2019, parse_dates={'datetime': ['Dates', 'Times']}, index_col= 'datetime')

bubblerdata2020 = directory + '/Inglefield_Data/modified_2020_excel.xlsx'
bubdf2020 = pd.read_excel(bubblerdata2020, parse_dates={'datetime': ['Dates', 'Times']}, index_col= 'datetime')

bubdf2019 = pd.to_numeric(bubdf2019['ING Stage DCP-raw'], errors="coerce")
bubdf2020 = pd.to_numeric(bubdf2020['ING Stage DCP-raw'], errors="coerce")

resampled2019 = bubdf2019.resample('3H').mean()
resampled2020 = bubdf2020.resample('3H').mean()

fig6, ax6 = plt.subplots(2,2, constrained_layout = True, sharex = 'col')

ax6[0,0].plot(df1.index, df1['water level'])
ax6[0,0].invert_yaxis()
ax6[0,0].set_ylabel('Water Level (row)')
ax6[0,0].set_title('2019 Data')
ax6[0,0].grid(linestyle='dashed')
ax6[0,1].plot(df2.index, df2['water level'])
ax6[0,1].invert_yaxis()
ax6[0,1].set_title('2020 Data')
ax6[0,1].grid(linestyle='dashed')
ax6[1,0].plot(resampled2019.index, resampled2019)
ax6[1,0].set_ylabel('Raw Stage (m)')
ax6[1,0].grid(linestyle='dashed')
ax6[1,1].plot(resampled2020.index, resampled2020)
ax6[1,1].grid(linestyle='dashed')

ax6[0,0].tick_params(axis = 'x', labelrotation = 45)
ax6[0,1].tick_params(axis = 'x', labelrotation = 45)
ax6[1,0].tick_params(axis = 'x', labelrotation = 45)
ax6[1,1].tick_params(axis = 'x', labelrotation = 45)

plt.show()

mergedf2019 = df1.merge(resampled2019, how='right', left_on=df1.index, right_on=resampled2019.index)
mergedf2020 = df2.merge(resampled2020, how='right', left_on=df2.index, right_on=resampled2020.index)

mergedf2019['stage_filtered'] = mergedf2019['ING Stage DCP-raw']
mergedf2019.loc[(mergedf2019['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan
mergedf2020['stage_filtered'] = mergedf2019['ING Stage DCP-raw']
mergedf2020.loc[(mergedf2020['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan


fig7, ax7 = plt.subplots(2, constrained_layout = True)
ax7[0].scatter(mergedf2019['water level'], mergedf2019['stage_filtered'])
ax7[0].set_title('2019')
ax7[1].scatter(mergedf2020['water level'], mergedf2020['stage_filtered'])
ax7[1].set_title('2020')

plt.show()

x2019 = mergedf2019['stage_filtered']
y2019 = mergedf2019['water level']
mask2019 = ~np.isnan(x2019) & ~np.isnan(y2019)
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x2019[mask2019],y2019[mask2019])
print('2019 lin regress values: ' + 'slope:' + str(slope19) + ' intercept:' + str(intercept19) +' r squared:' + str(r_value19) + ' p value:' + str(p_value19) + ' std error:' + str(std_err19))

x2020 = mergedf2020['stage_filtered']
y2020 = mergedf2020['water level']
mask2020 = ~np.isnan(x2020) & ~np.isnan(y2020)
slope, intercept, r_value, p_value, std_err = stats.linregress(x2020[mask2020],y2020[mask2020])
print('2020 lin regress values: ' + 'slope:' + str(slope) + ' intercept:' + str(intercept) +' r squared:' + str(r_value) + ' p value:' + str(p_value) + ' std error:' + str(std_err))

