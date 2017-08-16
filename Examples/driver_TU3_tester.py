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

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from Utilities import plotPX, plotXYZ


#-----------------------------   Map data files   -----------------------------

#Define plume detection specifications
camera = 'TU3'


#Define data inputs
camdata = ('../Examples/camenv_data/camenvs/CameraEnvironmentData_' + camera +
          '_2015.txt')

camimgs = ('../Examples/images/TU3_2015_subset/*.JPG')


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)

##Show GCPs
#extent=[550000,565000,8706000,8716000]
#cameraenvironment.showGCPs(extent)
#
##Show Principal Point
#cameraenvironment.showPrincipalPoint()


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

#Show all image extents and dems 
for i in range(len(rpolys)):
    plotPX(plumes, i, None, show=True, crop=False)
    plotXYZ(plumes, i, None, dem=True, show=True)


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'