# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:35:50 2021

@author: sgoldst3
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

dir2019files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/*'
directory2019 = glob.glob(dir2019files)
choices = []
time_list2020 = []

for i in directory2019:
    
    # parse time stamp
    filename = pathlib.Path(i)
    filename = filename.stem
    filename = filename.split("_")[-2:]
    filename = ["".join(filename)]
    filename = filename[0]
    time = datetime.strptime(filename, '%Y%m%d%H%M%S')
    # Read and filter image
    image = imread(i, as_gray=True)
    image = ndi.gaussian_filter(image, 3)
       
    # Compute the Canny filter for two values of sigma
    edges = feature.canny(image, sigma=3)

    # Compute row with most edges
    rows, cols = edges.shape
    rowsum = np.sum(edges.astype(int), axis = 1)
    maxsum = rowsum.argmax()
    edgeline = [maxsum]*cols
    
    plt.imshow(image, cmap='gray')
    plt.plot(edgeline, color = "red", linewidth=1)
    plt.show()
    
    choice = input("is the image good? 0 = bad, 1 = good")
    c = int(choice)
    choices.append(tuple([time, c, i, maxsum]))
        
df = pd.DataFrame(choices, columns= ['time', 'choice', 'path', 'maxsum'])
df = df.set_index('time')


linelocs = []

for i, row in df.iterrows():
    if row['choice'] == 0:
        image = imread(row['path'], as_gray=True)
        plt.imshow(image, cmap='gray')
        plt.show()
        levelguess = input("What 'y' value for the top of the water level?")
        guess = int(levelguess)
        linelocs.append(guess)
    else:
        linelocs.append(row['maxsum'])
        
df['lineloc'] = linelocs

bubblerdata2019 = directory + '/Inglefield_Data/modified_2019_excel.xlsx'
bubdf2019 = pd.read_excel(bubblerdata2019, parse_dates={'datetime': ['Dates', 'Times']}, index_col= 'datetime')

bubdf2019 = pd.to_numeric(bubdf2019['ING Stage DCP-raw'], errors="coerce")

resampled2019 = bubdf2019.resample('3H').mean()

mergedf2019 = df.merge(resampled2019, how='right', left_on=df.index, right_on=resampled2019.index)
mergedf2019['stage_filtered'] = mergedf2019['ING Stage DCP-raw']
mergedf2019.loc[(mergedf2019['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan

mergedf2019.to_csv(directory + '/results/manual_detection_results.csv')
mergedf2019 = pd.read_csv(directory + '/results/manual_detection_results.csv')

fig1, ax1 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax1[0].plot(mergedf2019.index, mergedf2019['lineloc'])
ax1[0].invert_yaxis()
ax1[0].set_ylabel('Water Level (row)')
ax1[0].set_title('2019 Data')
ax1[0].grid(linestyle='dashed')
ax1[1].plot(mergedf2019.index, mergedf2019['ING Stage DCP-raw'])
ax1[1].set_ylabel('Raw Stage (m)')
ax1[1].grid(linestyle='dashed')

ax1[0].tick_params(axis = 'x', labelrotation = 45)
ax1[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()

fig2, ax2 = plt.subplots(constrained_layout = True)
ax2.scatter(mergedf2019['lineloc'], mergedf2019['stage_filtered'])
ax2.set_title('2019')
ax2.set_ylabel('Raw Stage (m)')
ax2.set_xlabel('Water level (row)')

plt.show()

x2019 = mergedf2019['stage_filtered']
y2019 = mergedf2019['lineloc']
mask2019 = ~np.isnan(x2019) & ~np.isnan(y2019)
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x2019[mask2019],y2019[mask2019])
print('2019 lin regress values: ' + 'slope:' + str(slope19) + ' intercept:' + str(intercept19) +' r squared:' + str(r_value19) + ' p value:' + str(p_value19) + ' std error:' + str(std_err19))

