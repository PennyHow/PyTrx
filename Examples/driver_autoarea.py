'''
PYTRX EXAMPLE AUTOMATED AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates supraglacial lake surface area at Kronebreen, Svalbard, 
for a small subset of the 2014 melt season using modules in PyTrx. Specifically 
this script performs automated detection of supraglacial lakes through 
sequential images of the glacier to derive surface areas which have been 
corrected for image distortion.

Previously defined areas can also be imported from file (this can be changed 
by commenting and uncommenting commands in the "Calculate areas" section of 
this script).


@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)
'''

#Import packages
import sys
import os

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from FileHandler import writeAreaFile, writeSHPFile
from Utilities import plotPX, plotXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR5_2014.txt'
cammask = '../Examples/camenv_data/masks/KR5_2014_amask.JPG'
camimgs = '../Examples/images/KR5_2014_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/autoarea/'
if not os.path.exists(destination):
    os.makedirs(destination)


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)


#Define Area class initialisation variables
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'R'                #Desired image band
quiet = 2                   #Level of commentary
loadall = False             #Load all images?
time = 'EXIF'              #Method to derive image times


#Set up Area object, from which areal extent will be measured
lakes = Area(camimgs, cameraenvironment, calibFlag, cammask, maxim, imband, 
             quiet, loadall, time)
             

#---------------------   Set area detection parameters   ----------------------             

#Set image enhancement parameters. If these are undefined then they will be 
#set to a default enhancement of ('light', 50, 20)
lakes.setEnhance('light', 50, 20)


#Set colour range, from which extents will be distinguished. If colour range 
#is not specified, it will be manually defined 
lakes.setColourrange(8, 1)                                                                                                                                                      


#Set px plotting extent for easier colourrange definition and area verification
lakes.setPXExt(0,1200,2000,1500)


#Set polygon threshold (i.e. number of polygons kept)
lakes.setThreshold(5)

#Set automated area detection input arguments
px=None                 #Pre-defined pixel extents?
colour=False            #Define colour range for every image?
verify=False            #Manually verify detected areas?


#-------------------------   Calculate areas   --------------------------------

#Calculate real areas
rpolys, rareas = lakes.calcAutoAreas(px, colour, verify)

##Calculate pixel extents. Use this function if pixel extents are only needed, 
##or a DEM is not available. Pixel extents are automatically calculated if 
##calcAreas is called.
#polys, areas = lakes.calcExtents()

##Import data from file
#rpolys, rareas, pxpolys, pxareas = lakes.importData(destination)


#----------------------------   Export data   ---------------------------------

#Write results to file
writeAreaFile(lakes, destination)


#Create shapefiles
target1 = destination + 'shpfiles/'
if not os.path.exists(target1):
    os.makedirs(target1)
    
proj = 32633
writeSHPFile(lakes, target1, proj) 


#Write all image extents and dems 
target2 = destination + 'outputimgs/'
if not os.path.exists(target2):
    os.makedirs(target2)
    
for i in range(len(rpolys)):
    plotPX(lakes, i, target2, crop=False, show=True)
    plotXYZ(lakes, i, target2, dem=False, show=True)


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'