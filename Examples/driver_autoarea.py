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

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)

'''

#Import packages
import sys
import time
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import interpolate

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from Images import TimeLapse
from FileHandler import writeAreaFile, writeAreaSHP
from Utilities import plotPXarea, plotXYZarea


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_cam5_2014.txt'
cammask = '../Examples/camenv_data/masks/c5_2014_amask.JPG'
camimgs = '../Examples/images/KR5_2014_subset/*.JPG'

#Define data output directory
destination = '../results/KR5_2014_lakes/'


#-------------------------   Create camera object   ---------------------------
#Define camera environment
cameraenvironment = CamEnv(camdata)


##Get PP and approximate YPR
#ypr_data = fileDirectory + 'Data/GCPdata/gcps/c5_2014_ypr.txt'
#camera = cam5.getFullCamera()
#print 'Full camera info: ' + str(camera)
#cam5.showPrincipalPoint()
#cam5.approxYPR(ypr_data)
#print 'Camera pose: ' 
#print cam5.getCameraPose()


##Show viewshed
#demobj = DEM_FromMat('C:/Users/s0824923/Local Documents/python_workspace/pytrx/Data/GCPdata/dem/dem.mat')
#
#X = demobj.getData(0)
#Y = demobj.getData(1)
#Z = demobj.getZ()
#cellsize = X[0][1]-X[0][0]
#
#ext = demobj.getExtent()
#
#camxyz = [451632.669, 8754648.786, 624.699]
#
#vis = voxelviewshed(X,Y,Z,camxyz)
#vis = np.flipud(vis)
#print vis.shape
#
#plt.imshow(vis, cmap='gray', origin ='upper', extent=ext)
#plt.scatter(camxyz[0], camxyz[1], color = 'r')
#plt.show()


#------------------------   Calculate lake areas   ----------------------------

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
             imband, loadall, time, quiet)
                 
#Set image enhancement parameters. If these are undefined then they will be 
#set to a default enhancement of ('light', 50, 20)
lakes.setEnhance('light', 50, 20)

##Show example of image enhancement
#im=lakes.getMaxImgData()
#im=lakes.maskImg(im)  
#im=lakes.enhanceImg(im)
#plt.imshow(im)
#plt.show()

#Set colour range, from which extents will be distinguished. If colour range 
#is not specified, it will be manually defined 
lakes.setColourrange(8, 1)                                                                                                                                                      

#Set plotting extent for easier colour range definition and area verification
lakes.setPXExt(0,1200,2000,1500)

#Set polygon threshold (i.e. number of polygons kept)
lakes.setThreshold(5)

##Calculate pixel extents. Use this function if pixel extents are only needed, 
##or a DEM is not available. Pixel extents are automatically calculated if 
##calcAreas is called.
#polys, areas = lakes.calcExtents()

#Calculate real areas
rpolys, rareas = lakes.calcAreas(method='auto', color=False, verify=False)

##Import data from file
#rpolys, rareas, pxpolys, pxareas = lakes.importData(destination)


#----------------------------   Export data   ---------------------------------
#Write results to file
writeAreaFile(lakes, destination)

#Create shapefiles
target = destination + 'shpfiles/'
proj = 32633
writeAreaSHP(target, proj) 


#Write all image extents and dems to destination file
target=destination + 'extentimgs/'
length=len(rpolys)
for i in range(length):
    plotPXarea(i, None)
    plotXYZarea(i, None, dem=True, show=True)


#Show image path list for checking
lakes.printImageList() 


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      
print 'Finished'

