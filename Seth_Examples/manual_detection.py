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


# Load images
directory = os.getcwd()


## 2019 manual detection ##

dir2019files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/*'
directory2019 = glob.glob(dir2019files)
choices = []
time_list2019 = []

print('\nValidate Edge Detection')
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
    
    # show image with rowsum line
    plt.imshow(image, cmap='gray')
    plt.plot(edgeline, color = "red", linewidth=1)
    plt.show()
    
    # image check
    choice = input("Is the line at the top of the river water level? 0 = no, 1 = yes (left), 2 = yes (right)")
    c = int(choice)
    choices.append(tuple([time, c, i, maxsum]))
     
# Send manual detection to Pandas DataFrame
df = pd.DataFrame(choices, columns= ['time', 'choice', 'path', 'maxsum'])
df = df.set_index('time')

# Correct water levels
linelocs = []

print('\nInput correct water levels')

for i, row in df.iterrows():
    if row['choice'] == 0:
        image = imread(row['path'], as_gray=True)
        plt.imshow(image, cmap='gray')
        plt.show()
        levelguess = input("What 'y' value for the top of the water level on the left?")
        guess = int(levelguess)
        linelocs.append(guess)
    elif row['choice'] == 2:
        corrected = row['maxsum'] + 40
        linelocs.append(corrected)
    else:
        linelocs.append(row['maxsum'])
        
df['lineloc'] = linelocs

## Import bubbler validation data
bubblerdata2019 = directory + '/Inglefield_Data/modified_2019_excel.xlsx'
bubdf2019 = pd.read_excel(bubblerdata2019, parse_dates={'datetime': ['Dates', 'Times']}, index_col= 'datetime')
bubdf2019 = pd.to_numeric(bubdf2019['ING Stage DCP-raw'], errors="coerce")

# Resample bubbler data 
resampled2019 = bubdf2019.resample('3H').mean()

# Merge data to single dataframe
mergedf2019 = df.merge(resampled2019, how='right', left_on=df.index, right_on=resampled2019.index)
mergedf2019['stage_filtered'] = mergedf2019['ING Stage DCP-raw']
mergedf2019.loc[(mergedf2019['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan

## Export DataFrame 
mergedf2019.to_csv(directory + '/results/manual_detection_results2019.csv')
print('\nData exported to CSV file... Recommended to double check values')

## Recommended to check CSV file to see if anything needs to be changed ##
# Re-import DataFrame from CSV file
mergedf2019 = pd.read_csv(directory + '/results/manual_detection_results2019.csv', parse_dates=['key_0'], index_col='key_0')

## Calculate linear regression stats

x2019 = mergedf2019['stage_filtered']
y2019 = mergedf2019['lineloc']
mask2019 = ~np.isnan(x2019) & ~np.isnan(y2019)
slope19, intercept19, r_value19, p_value19, std_err19 = stats.linregress(x2019[mask2019],y2019[mask2019])
print('2019 lin regress values: ' + 'slope:' + str(slope19) + ' intercept:' + str(intercept19) +' r squared:' + str(r_value19) + ' p value:' + str(p_value19) + ' std error:' + str(std_err19))

# spearmanr19, spearmanp19 = stats.spearmanr(x2019[mask2019], y2019[mask2019])
# print('Spearman r squared: ' + str(spearmanr19**2))

## Data Plots ##
fig1, ax1 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax1[0].plot(mergedf2019.index, y2019)
ax1[0].invert_yaxis()
ax1[0].set_ylabel('Water Level (row)')
ax1[0].set_title('2019 Data')
ax1[0].grid(linestyle='dashed')
ax1[1].plot(mergedf2019.index, x2019)
ax1[1].set_ylabel('Filtered Stage (m)')
ax1[1].grid(linestyle='dashed')

ax1[0].tick_params(axis = 'x', labelrotation = 45)
ax1[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()

fig2, ax2 = plt.subplots(constrained_layout = True)
ax2.scatter(x2019, y2019, c = mergedf2019['choice'])
ax2.invert_yaxis()
ax2.set_title('2019')
ax2.set_xlabel('Filtered Stage (m)')
ax2.set_ylabel('Water level (row)')

plt.show()

counts19 = mergedf2019['choice'].value_counts()
print('Percentage automatically detected: ' + str((counts19[1]+counts19[2])/len(mergedf2019)*100))

## 2020 manual detection ##

dir2020files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2020/*'
directory2020 = glob.glob(dir2020files)
choices2 = []
time_list20 = []

print('\nValidate Edge Detection')
for i in directory2020:
    
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
    
    # show image with rowsum line
    plt.imshow(image, cmap='gray')
    plt.plot(edgeline, color = "red", linewidth=1)
    plt.show()
    
    # image check
    choice = input("Is the line at the top of the river water level? 0 = no, 1 = yes(left), 2 = yes (right)")
    c = int(choice)
    choices2.append(tuple([time, c, i, maxsum]))
     
# Send manual detection to Pandas DataFrame
df2 = pd.DataFrame(choices2, columns= ['time', 'choice', 'path', 'maxsum'])
df2 = df2.set_index('time')

# Correct water levels
linelocs2 = []

print('\nInput correct water levels')

for i, row in df2.iterrows():
    if row['choice'] == 0:
        image = imread(row['path'], as_gray=True)
        plt.imshow(image, cmap='gray')
        plt.show()
        levelguess = input("What 'y' value for the top of the water level on the left?")
        guess = int(levelguess)
        linelocs2.append(guess)
    elif row['choice'] == 2:
        corrected = row['maxsum'] + 40
        linelocs2.append(corrected)
    else:
        linelocs2.append(row['maxsum'])
        
df2['lineloc'] = linelocs2

lineadjust = []
for i, row in df2.iterrows():
    if row['choice'] == 0:
        adjustment = row['lineloc'] - 20
        lineadjust.append(adjustment)
    elif row['choice'] == 2:
        adjustment = row['lineloc'] - 10
        lineadjust.append(adjustment)
    else:
        lineadjust.append(row['lineloc'])
df2['lineadjust'] = lineadjust

## Import bubbler validation data
bubblerdata2020 = directory + '/Inglefield_Data/modified_2020_excel.xlsx'
bubdf2020 = pd.read_excel(bubblerdata2020, parse_dates={'datetime': ['Dates', 'Times']}, index_col= 'datetime')
bubdf2020 = pd.to_numeric(bubdf2020['ING Stage DCP-raw'], errors="coerce")

# Resample bubbler data 
resampled2020 = bubdf2020.resample('3H').mean()

# Merge data to single dataframe
mergedf2020 = df2.merge(resampled2020, how='right', left_on=df2.index, right_on=resampled2020.index)
mergedf2020['stage_filtered'] = mergedf2020['ING Stage DCP-raw']
mergedf2020.loc[(mergedf2020['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan

## Export DataFrame 
mergedf2020.to_csv(directory + '/results/manual_detection_results2020.csv')
print('\nData exported to CSV file... Recommended to double check values')

## Recommended to check CSV file to see if anything needs to be changed ##
# Re-import DataFrame from CSV file
mergedf2020 = pd.read_csv(directory + '/results/manual_detection_results2020.csv', parse_dates=['key_0'], index_col='key_0')


## Calculate linear regression stats

x2020 = mergedf2020['stage_filtered']
y2020 = mergedf2020['lineadjust']
mask2020 = ~np.isnan(x2020) & ~np.isnan(y2020)
slope20, intercept20, r_value20, p_value20, std_err20 = stats.linregress(x2020[mask2020],y2020[mask2020])
print('2020 lin regress values: ' + 'slope:' + str(slope20) + ' intercept:' + str(intercept20) +' r squared:' + str(r_value20) + ' p value:' + str(p_value20) + ' std error:' + str(std_err20))

# spearmanr20, spearmanp20 = stats.spearmanr(x2020[mask2020], y2020[mask2020])
# print('Spearman r squared: ' + str(spearmanr20**2))

## Data Plots ##
fig3, ax3 = plt.subplots(2, constrained_layout = True, sharex = 'col')

ax3[0].plot(mergedf2020.index, y2020)
ax3[0].invert_yaxis()
ax3[0].set_ylabel('Water Level (row)')
ax3[0].set_title('2020 Data')
ax3[0].grid(linestyle='dashed')
ax3[1].plot(mergedf2020.index, x2020)
ax3[1].set_ylabel('Filtered Stage (m)')
ax3[1].grid(linestyle='dashed')

ax3[0].tick_params(axis = 'x', labelrotation = 45)
ax3[1].tick_params(axis = 'x', labelrotation = 45)

plt.show()

fig4, ax4 = plt.subplots(constrained_layout = True)
ax4.scatter(x2020, y2020, c= mergedf2020['choice'])
ax4.invert_yaxis()
ax4.set_title('2020')
ax4.set_xlabel('Filtered Stage (m)')
ax4.set_ylabel('Water level (row)')

plt.show()

counts20 = mergedf2020['choice'].value_counts()
print('Percentage automatically detected: ' + str((counts20[1]+counts20[2])/len(mergedf2020)*100))
