'''
PYTRX EXAMPLE MANUAL AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates meltwater plume surface extent at Kronebreen (camera 
site 1, KR1) for a small subset of the 2014 melt season using modules in PyTrx. 
Specifically this script performs manual detection of supraglacial lakes 
through sequential images of the glacier to derive surface areas which have 
been corrected for image distortion. Previously defined pixel areas can also 
be imported from file (this can be changed by commenting and uncommenting 
commands in the "Calculate areas" section of this script).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)
'''

#Import packages
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob

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
destination = '../Examples/results/manualarea/'
if not os.path.exists(destination):
    os.makedirs(destination)


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)


#Define Area class initialisation variables
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'R'                #Desired image band
loadall = False             #Load all images?
time = 'EXIF'               #Method to derive image times
quiet = 2                   #Level of commentary

#Set up Area object, from which areal extent will be measured
plumes = Area(camimgs, cameraenvironment, calibFlag, None, maxim, imband, 
              quiet, loadall, time)


#-------------------------   Calculate areas   --------------------------------

##Calculate pixel extents. Use this function if pixel extents are only needed, 
##or a DEM is not available
#polys, areas = plumes.calcManualExtents()


##Calculate real areas
#rpolys, rareas = plumes.calcManualAreas()


#Import areal data from file
rpolys, rareas, pxpolys, pxareas = importAreaData(plumes, destination)


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
    plotPX(plumes, i, target2, crop=None, show=False)              #Pixel areas
    plotXYZ(plumes, i, target2, crop=None, show=False, dem=True)   #Real areas


#---Alternative method for plotting image extents using original RGB images----                                                                                                                                                                                                                                                                                                      

#Get original images in directory
ims = sorted(glob.glob(camimgs), key=os.path.getmtime)

#Get pixel areas from Area object
pxs = plumes._pxpoly

#Get camera correction variables
cameraMatrix=cameraenvironment.getCamMatrixCV2()
distortP=cameraenvironment.getDistortCoeffsCv2()
newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortP, 
                                            (5184,3456),1,(5184,3456))                                                                                                 
    
#Get corresponding xy pixel areas and images                                            
count=1
for p,i in zip(pxs,ims):
    x=[]
    y=[]
    for ps in p[0]:    
        x.append(ps[0])
        y.append(ps[1])

    #Read image and undistort 
    im1=mpimg.imread(i)
    im1 = cv2.undistort(im1, cameraMatrix, distortP, newCameraMatrix=newMat)
    
    #Plot pixel area over image
    plt.figure(figsize=(20,10))             #Figure size set
    plt.imshow(im1)                         #Plot image
    plt.axis([0,5184,3456,0])               #Set axis dimensions to image size
    plt.xticks([])                          #Hide axis ticks
    plt.yticks([])
    plt.plot(x,y,'#fff544',linewidth=2)     #Plot pixel area
    
    #Save image to file
    target3 = destination + 'rgbimgs/'
    if not os.path.exists(target3):
        os.makedirs(target3)
            
    plt.savefig(target3 + 'plumeplotted' + str(count) + '.JPG', dpi=300)
    plt.show()
    count=count+1
    
    
#------------------------------------------------------------------------------    
    
print '\n\nFINISHED'