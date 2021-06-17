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
fig, ax = plt.subplots(nrows=3, ncols=3)

ax[0,0].imshow(image1, cmap='gray')
ax[0,0].set_title('noisy image', fontsize=10)

ax[0,1].plot(rowsum)

ax[0,2].plot(edges2line, color = "red", linewidth=1)
ax[0,2].imshow(edges2[:, :], cmap='gray')
ax[0,2].set_title(r'Canny filter, $\sigma=3$', fontsize=10)

# fig.tight_layout()
# plt.show()

# Image 2
inglecam_img2 = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190731_210000.jpg'
image2 = imread(inglecam_img2, as_gray=True)

image2 = ndi.gaussian_filter(image2, 3)

# Compute the Canny filter for two values of sigma
edges3 = feature.canny(image2)
edges4 = feature.canny(image2, sigma=3)

# Compute row with most edges
rows2, cols2 = edges4.shape
rowsum2 = np.sum(edges4.astype(int), axis = 1)
edges4line = [rowsum2.argmax()]*cols2

# display results
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[1,0].imshow(image2, cmap='gray')
ax[1,0].set_title('noisy image', fontsize=10)

ax[1,1].plot(rowsum2)

ax[1,2].plot(edges4line, color = "red", linewidth=1)
ax[1,2].imshow(edges4, cmap='gray')
ax[1,2].set_title(r'Canny filter, $\sigma=3$', fontsize=10)

# fig.tight_layout()
# plt.show()

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

# display results
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[2,0].imshow(image3, cmap='gray')
ax[2,0].set_title('noisy image', fontsize=10)

ax[2,1].plot(rowsum3)

ax[2,2].plot(edges6line, color = "red", linewidth=1)
ax[2,2].imshow(edges6, cmap='gray')
ax[2,2].set_title(r'Canny filter, $\sigma=3$', fontsize=10)

fig.tight_layout()
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
plt.scatter(df1.index, df1['water level'], c = 'blue')
plt.title('2019 unfiltered data')
plt.gca().invert_yaxis()

df1.drop(df1[df1['water level'] < (Q1_2019[0] -1.5 * IQR_2019[0])].index, inplace=True)
df1.drop(df1[df1['water level'] > (Q3_2019[0] +1.5 *IQR_2019[0])].index, inplace=True)
print(df1.shape)
plt.plot(df1.index, df1['water level'], c = 'green')
plt.title('2019 corrected data')
plt.gca().invert_yaxis()

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
Q1_2020 = df_filter2.quantile(0.25)
Q3_2020 = df_filter2.quantile(0.75)
IQR_2020 = Q3_2020 - Q1_2020

df2 = pd.DataFrame(time_list2020)
df2.columns = ['time', 'water level']
df2 = df2.set_index('time')
print(df2.shape)
plt.scatter(df2.index, df2['water level'], c = 'blue')
plt.title('2020 unfiltered data')
plt.gca().invert_yaxis()

df2.drop(df2[df2['water level'] < (Q1_2020[0] -1.5 * IQR_2020[0])].index, inplace=True)
df2.drop(df2[df2['water level'] > (Q3_2020[0] +1.5 *IQR_2020[0])].index, inplace=True)
print(df2.shape)
plt.plot(df2.index, df2['water level'], c = 'green')
plt.title('2020 corrected data')
plt.gca().invert_yaxis()


# =============================================================================
# Bubbler Data
# =============================================================================

# dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S')

# bubdf2019 = pd.read_csv('C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data/modified_2019.csv', 
#                     parse_dates={'datetime': ['Dates', 'Times']}, date_parser=dateparse, index_col= 'datetime')

# bubdf2020 = pd.read_csv('C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data/modified_2020.csv', 
#                     parse_dates={'datetime': ['Dates', 'Times']}, date_parser=dateparse, index_col= 'datetime')

# # year2019 = bubdf.iloc[0:5050, :]
# # year2020 = bubdf.iloc[5051:, :]
# bubdf2019.resample('3H').mean()
# bubdf2020.resample('3H').mean()

# fig, ax = plt.subplots(2,2)

# ax[0,0].plot(df1.index, df1['water level'])
# ax[0,0].invert_yaxis()
# ax[0,1].plot(df2.index, df2['water level'])
# ax[0,1].invert_yaxis()
# ax[1,0].scatter(bubdf2019.index, bubdf2019['ING Stage DCP-raw'])
# ax[1,1].scatter(bubdf2020.index, bubdf2020['ING Stage DCP-raw'])

# plt.show()



