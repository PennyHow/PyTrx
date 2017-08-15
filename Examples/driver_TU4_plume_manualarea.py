'''
PYTRX EXAMPLE MANUAL AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates meltwater plume surface extent at Tunabreen, Svalbard, 
for a small subset of the 2015 melt season using modules in PyTrx. Specifically 
this script performs manual detection of meltwater plumes through 
sequential images of the glacier to derive surface areas which have been 
corrected for image distortion.

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)

'''

#Import packages
import sys
import time
import os

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from FileHandler import writeAreaFile, writeSHPFile, importMeasureData
from Utilities import plotPX, plotXYZ


#-----------------------------   Map data files   -----------------------------

#Define plume detection specifications
camera = 'TU2'
date = '150819'
plume = 'plume1'


#Define data inputs
camdata = ('../Examples/camenv_data/camenvs/CameraEnvironmentData_' + camera +
          '_2015.txt')

#camimgs = ('F:/imagery/tunabreen/pytrx/' + camera + 
#           '_hourly_2015/' + date + '/*.JPG')

camimgs = ('C:/Users/s0824923/Desktop/' + camera + 
           '_hourly_2015/' + date + '/*.JPG')

#Define data output directory
destination = ('../Examples/results/tuna_plume/' + camera + '/' + 
               date + '/' + plume + '/')


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)

#Show GCPs
extent=[550000,565000,8706000,8716000]
cameraenvironment.showGCPs(extent)

#Show Principal Point
cameraenvironment.showPrincipalPoint()


#Define Area class initialisation variables
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'L'                #Desired image band
loadall = False             #Load all images?
timem = 'EXIF'              #Method to derive image times
quiet = 2                   #Level of commentary

#Set up Area object, from which areal extent will be measured
plumes = Area(camimgs, cameraenvironment, calibFlag, None, maxim, imband, 
              quiet, loadall, time)


#-------------------------   Calculate areas   --------------------------------

##Import data from file
#rpolys, rareas, pxpolys, pxareas = importMeasureData(plumes, destination)

#Calculate real areas
rpolys, rareas = plumes.calcManualAreas()

#plumes.setColourrange(8, 1)
#plumes.setEnhance('light', 50, 20)
#pxpolys, pxareas = plumes.calcAutoExtents(True, True)
#rpolys, rareas = plumes.calcAutoAreas()


##Calculate pixel extents. Use this function if pixel extents are only needed, 
##or a DEM is not available. Pixel extents are automatically calculated if 
##calcAreas is called.
#polys, areas = plumes.calcExtents()


#----------------------------   Export data   ---------------------------------

#Write results to file
writeAreaFile(plumes, destination)


#Create shapefiles
target1 = destination + 'shpfiles/'
if not os.path.exists(target1):
    os.makedirs(target1)
proj = 32633
writeSHPFile(plumes, target1, proj) 


#Write all image extents and dems 
target2 = destination + 'outputimgs/'
if not os.path.exists(target2):
    os.makedirs(target2)
for i in range(len(rpolys)):
    plotPX(plumes, i, target2, show=False, crop=False)
    plotXYZ(plumes, i, target2, dem=True, show=False)


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'