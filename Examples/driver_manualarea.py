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
from FileHandler import writeAreaFile, writeSHPFile, importAreaData
from Utilities import plotPX, plotXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR1_2014.txt'
camimgs = '../Examples/images/KR1_2014_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/KR1_manualarea/'


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)


#Define Area class initialisation variables
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'R'                #Desired image band
loadall = False             #Load all images?
timem = 'EXIF'              #Method to derive image times
quiet = 2                   #Level of commentary

#Set up Area object, from which areal extent will be measured
plumes = Area(camimgs, cameraenvironment, calibFlag, None, maxim, imband, 
              quiet, loadall, time)


#-------------------------   Calculate areas   --------------------------------

##Calculate real areas
#rpolys, rareas = plumes.calcManualAreas()

##Calculate pixel extents. Use this function if pixel extents are only needed, 
##or a DEM is not available. Pixel extents are automatically calculated if 
##calcAreas is called.
#polys, areas = plumes.calcExtents()

#Import data from file
rpolys, rareas, pxpolys, pxareas = importAreaData(plumes, destination)


#----------------------------   Export data   ---------------------------------

#Write results to file
writeAreaFile(plumes, destination)


#Create shapefiles
target1 = destination + 'shpfiles/'
proj = 32633
writeSHPFile(plumes, target1, proj) 


#Write all image extents and dems 
target2 = destination + 'outputimgs/'
for i in range(len(rpolys)):
    plotPX(plumes, i, destination, crop=False)
    plotXYZ(plumes, i, destination, dem=True, show=True)


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'