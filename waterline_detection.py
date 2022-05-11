"""
Script for delineating water stages from time-lapse images.

@author: Seth Goldstein
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import stats
from skimage import feature
from skimage.io import imread
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

# =============================================================================
# Detection & validation funcs
# =============================================================================

def validateEdge(imgs):
    '''User validate Canny Edge detection from a sequence of images'''
    choices = []
    for i in imgs:
        
        # Parse time stamp
        filename = str(Path(i).stem).split("_")[-2:]
        filename = ["".join(filename)][0]
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
        
        # Show image with rowsum line
        plt.imshow(image, cmap='gray')
        plt.plot(edgeline, color = "red", linewidth=1)
        plt.show()
        
        # Check image with user input
        choice = input("Is the line at the top of the river water level? "+
                       "0 = no, 1 = yes (left), 2 = yes (right) ")
        c = int(choice)
        choices.append(tuple([time, c, i, maxsum]))
         
    # Send manual detection to DataFrame
    df = pd.DataFrame(choices, columns= ['time', 'choice', 'path', 'maxsum'])
    df = df.set_index('time')
    return df


def correctLevels(df):
    '''Correct water levels if Canny Edge line is imprecise'''
    linelocs = []
    print('\nInput correct water levels')
    for i, row in df.iterrows():
        
        # Identify if water level not defined
        if row['choice'] == 0:
            image = imread(row['path'], as_gray=True)
            plt.imshow(image, cmap='gray')
            plt.show()
            levelguess = input("What 'y' value for the top of the water level"+
                               " on the left? ")
            guess = int(levelguess)
            linelocs.append(guess)
            
        # Pass water level forward if defined
        elif row['choice'] == 2:
            corrected = row['maxsum'] + 40
            linelocs.append(corrected)
        else:
            linelocs.append(row['maxsum'])  
            
    # Append line locations to DataFrame
    df['lineloc'] = linelocs
    return df
  

def getBubbler(filename):
    '''Import Bubbler data to DataFrame'''
    bubdf = pd.read_excel(filename, 
                          parse_dates={'datetime': ['Dates', 'Times']}, 
                          index_col= 'datetime')
    bubdf = pd.to_numeric(bubdf['ING Stage DCP-raw'], errors="coerce")
    return bubdf
    

def mergeDF(df, bubdf):
    '''Merge water level DataFrame with bubbler data DataFrame'''
    # Resample bubbler data 
    resampled = bubdf.resample('3H').mean()
    
    # Merge data to single dataframe
    mergedf = df.merge(resampled, how='right', left_on=df.index, 
                       right_on=resampled.index)
    mergedf['stage_filtered'] = mergedf['ING Stage DCP-raw']
    mergedf.loc[(mergedf['stage_filtered'] < 0.07), 'stage_filtered'] = np.nan
    return mergedf


def showLevels(df, year):
    '''Show statistics and detection results'''
    # Print dataset completeness
    counts = df['choice'].value_counts()
    print(counts)
    print('Percentage automatically detected: '+
          f'{(counts[1]+counts[2])/len(df)*100}%')
     
    # Calculate linear regression statistics
    x = df['stage_filtered']
    y = df['lineloc']
    mask = ~np.isnan(x) & ~np.isnan(y)
    slop, inte, rval, pval, stderr = stats.linregress(x[mask], y[mask])
    print('2019 lin regress values')
    print(f'slope: {slop} \nintercept: {inte} \nr squared: {rval}' +
          f'\np value: {pval} \nstd error: {stderr}')

    # Plot water stage-level time-series
    fig1, ax1 = plt.subplots(2, constrained_layout=True, sharex='col')
    ax1[0].plot(df.index, y)
    ax1[0].invert_yaxis()
    ax1[0].set_ylabel('Water Level (row)')
    ax1[0].set_title(year +' Data')
    ax1[0].grid(linestyle='dashed')
    ax1[1].plot(df.index, x)
    ax1[1].set_ylabel('Filtered Stage (m)')
    ax1[1].grid(linestyle='dashed')
    ax1[0].tick_params(axis = 'x', labelrotation = 45)
    ax1[1].tick_params(axis = 'x', labelrotation = 45)
    plt.show()
    
    # Plot water stage vs. level scatter
    fig2, ax2 = plt.subplots(constrained_layout = True)
    ax2.scatter(x, y, c = mergedf2019['choice'])
    ax2.invert_yaxis()
    ax2.set_title(year)
    ax2.set_xlabel('Filtered Stage (m)')
    ax2.set_ylabel('Water level (pixel)')
    plt.show()

   
# =============================================================================
# 2019 data
# =============================================================================

# Define input images and variables
directory2019 = glob.glob('cam_data/images/2019/*')
choices = []

# Validate and correct water level
df2019 = validateEdge(directory2019)
df2019 = correctLevels(df2019)

# Import and resample bubbler validation data
bubblerdata2019 = 'river_stage_data/modified_2019_excel.xlsx'
bubdf2019 = getBubbler(bubblerdata2019)

# Merge water level and water stage DataFrames
mergedf2019 = mergeDF(df2019, bubdf2019)

# Calculate statistics and plot results
showLevels(mergedf2019, '2019')

# Export DataFrame to csv
# ## Recommended to check CSV file to see if anything needs to be changed ##
mergedf2019.to_csv('cam_results/manual_detection_results2019.csv')
print('\nData exported to CSV file... Recommended to double check values')


# =============================================================================
# 2020 data
# =============================================================================

# Define input images and variables
directory2020 = glob.glob('cam_data/images/2020/*')
choices = []

# Validate and correct water level
df2020 = validateEdge(directory2020)
df2020 = correctLevels(df2020)

# Import and resample bubbler validation data
bubblerdata2020 = 'river_stage_data/modified_2020_excel.xlsx'
bubdf2020 = getBubbler(bubblerdata2020)

# Merge water level and water stage DataFrames
mergedf2020 = mergeDF(df2020, bubdf2020)

# Calculate statistics and plot results
showLevels(mergedf2020, '2020')

# Export DataFrame to csv
# ## Recommended to check CSV file to see if anything needs to be changed ##
mergedf2020.to_csv('cam_results/manual_detection_results2020.csv')
print('\nData exported to CSV file... Recommended to double check values')


# =============================================================================
# 2021 data
# =============================================================================

# Define input images and variables
directory2021 = glob.glob('cam_data/images/2021/*')
choices = []

# Validate and correct water level
df2021 = validateEdge(directory2021)
df2021 = correctLevels(df2021)

# Import and resample bubbler validation data
bubblerdata2021 = 'river_stage_data/modified_2021_excel.xlsx'
bubdf2021 = getBubbler(bubblerdata2021)

# Merge water level and water stage DataFrames
mergedf2021 = mergeDF(df2021, bubdf2021)

# Calculate statistics and plot results
showLevels(mergedf2021, '2021')

# Export DataFrame to csv
# ## Recommended to check CSV file to see if anything needs to be changed ##
mergedf2021.to_csv('cam_results/manual_detection_results2021.csv')
print('\nData exported to CSV file... Recommended to double check values')


# =============================================================================
print('Finished')