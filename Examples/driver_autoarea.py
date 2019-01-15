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
         Nick Hulton
         Lynne Buie
'''

#Import packages
import sys
import os

#Import PyTrx packages
sys.path.append('../')
from Area import Area
from CamEnv import CamEnv
from FileHandler import writeAreaFile, writeAreaSHP, writeCalibFile
from Utilities import plotAreaPX, plotAreaXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR3_2014.txt'
cammask = '../Examples/camenv_data/masks/KR3_2014_amask.jpg'
camimgs = '../Examples/images/KR3_2014_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/autoarea/'
if not os.path.exists(destination):
    os.makedirs(destination)


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)

#Report camera data and show corrected image
cameraenvironment.reportCamData()
cameraenvironment.showGCPs()
cameraenvironment.showPrincipalPoint()
cameraenvironment.showCalib()
             

#---------------------   Set area detection parameters   ----------------------             

#Define Area class initialisation variables
colour=False                #Define colour range for every image?
verify=False                #Manually verify detected areas?
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'R'                #Desired image band
equal = True                #Images with histogram equalisation?
threshold = 5               #Threshold for number of retained polygons
diff = 'light'              #Image enhancement parameter 1
phi = 50                    #Image enhancement parameter 2
theta = 20                  #Image enhancement parameter 3
maxcol = 8                  #Max value from which areas will be distinguished
mincol = 1                  #Min value from which areas will be distinguished
maxim = 3                   #Image number with maximum area of interest
pxext = [0,1200,2000,1500]  #Plotting extent for interactive plots


#Set up Area object, from which areal extent will be measured
lakes = Area(camimgs, cameraenvironment, calibFlag, imband, equal)


#Set image enhancement parameters. If these are undefined then they will be 
#set to a default enhancement of ('light', 50, 20)
lakes.setEnhance(diff, phi, theta)


#Set colour range, from which extents will be distinguished. If colour range 
#is not specified, it will be manually defined 
lakes.setColourrange(maxcol, mincol) 


#Set mask and image with maximum area of interest 
lakes.setMax(cammask, maxim)                                                                                                                                                     


#Set px plotting extent for easier colourrange definition and area verification
lakes.setPXExt(pxext[0], pxext[1], pxext[2], pxext[3])


#Set polygon threshold (i.e. number of polygons kept)
lakes.setThreshold(threshold)


#-------------------------   Calculate areas   --------------------------------

#Calculate real areas
areas = lakes.calcAutoAreas(colour, verify)


##Import data from file
#areas = lakes.importData(destination)


#----------------------------   Export data   ---------------------------------

#Get information for writing to files
matrix, tancorr, radcorr = cameraenvironment.getCalibdata()         #CV2 calib
imn = lakes.getImageNames()                                         #Img names
proj = 32633                                                        #Projection (WGS84)
xyzpts = [item[0][1] for item in areas]                             #XYZ coords
dem = cameraenvironment.getDEM()                                    #DEM
imgset=lakes._imageSet                                              #Images
uvpts = [item[1][1] for item in areas]                              #UV coords
cameraMatrix=cameraenvironment.getCamMatrixCV2()                    #Matrix
distortP=cameraenvironment.getDistortCoeffsCV2()                    #Distort


#Write out camera calibration info to .txt file
target1 = '../Examples/camenv_data/calib/KR3_2014_1.txt'
writeCalibFile(matrix, tancorr, radcorr, target1)

#Write results to file
writeAreaFile(areas, imn, destination)

#Create shapefiles
target1 = destination + 'shpfiles/'                 
writeAreaSHP(xyzpts, imn, target1, proj)            

#Write all image extents and dems 
target2 = destination + 'outputimgs/'

#Plot areas in image plane and as XYZ polygons  
for i in range(len(areas)):
    plotAreaPX(uvpts[i], imgset[i].getImageCorr(cameraMatrix, distortP), 
               show=True, save=target2+'uv_'+str(imn[i]))  
    plotAreaXYZ(xyzpts[i], dem, show=True, save=target2+'xyz_'+str(imn[i]))


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'