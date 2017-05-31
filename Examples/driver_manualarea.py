'''
PYTRX EXAMPLE MANUAL AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates meltwater plume surface extent at Kronebreen, Svalbard, 
for a small subset of the 2014 melt season using modules in PyTrx. Specifically 
this script performs manual detection of supraglacial lakes through 
sequential images of the glacier to derive surface areas which have been 
corrected for image distortion.

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)

'''

#Import packages
import sys
import time

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from FileHandler import writeAreaFile, writeAreaSHP
from Utilities import plotPX, plotXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_cam5_2014.txt'
cammask = '../Examples/camenv_data/masks/c5_2014_amask.JPG'
camimgs = '../Examples/images/KR5_2014_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/KR5_autoarea/'


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)


#Define Area class initialisation variables
method = 'auto'             #Method for detection ('auto' or 'manual')
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'L'                #Desired image band
loadall = False             #Load all images?
timem = 'EXIF'              #Method to derive image times
quiet = 2                   #Level of commentary

#Set up Area object, from which areal extent will be measured
lakes = Area(camimgs, cameraenvironment, method, calibFlag, cammask, maxim, 
             imband, quiet, loadall, time)


#---------------------   Set area detection parameters   ----------------------             

#Set image enhancement parameters. If these are undefined then they will be 
#set to a default enhancement of ('light', 50, 20)
lakes.setEnhance('light', 50, 20)


#Set colour range, from which extents will be distinguished. If colour range 
#is not specified, it will be manually defined 
lakes.setColourrange(8, 1)                                                                                                                                                      


#Set plotting extent for easier colour range definition and area verification
lakes.setPXExt(0,1200,2000,1500)


#Set polygon threshold (i.e. number of polygons kept)
lakes.setThreshold(5)


#-------------------------   Calculate areas   --------------------------------

#Calculate real areas
rpolys, rareas = lakes.calcAreas(color=False, verify=False)

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
proj = 32633
writeAreaSHP(lakes, target1, proj) 


#Write all image extents and dems 
target2 = destination + 'outputimgs/'
for i in range(len(rpolys)):
    plotPX(lakes, i, destination, crop=False)
    plotXYZ(lakes, i, destination, dem=False, show=True)


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print 'Finished'