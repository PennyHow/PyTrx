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

dir2019files = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019_sample/*'
directory2019 = glob.glob(dir2019files)
choices = []

for i in directory2019:
    
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
    choices.append(tuple([c, i, maxsum]))
        
df = pd.DataFrame(choices, columns= ['choice', 'path', 'maxsum'])
df['lineloc'] = ""

for i, row in df.iterrows():
    if row['choice'] == 0:
        image = imread(row['path'], as_gray=True)
        plt.imshow(image, cmap='gray')
        plt.show()
        levelguess = input("What 'y' value for the top of the water level?")
        guess = int(levelguess)
        row['lineloc'] = guess
    else:
        row['lineloc'] = row['maxsum']