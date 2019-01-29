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
         Nick Hulton
         Lynne Buie
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
from Area import Area
from Velocity import Homography
from CamEnv import CamEnv
from FileHandler import writeAreaFile, writeAreaSHP, importAreaData, writeHomogFile
from Utilities import plotAreaPX, plotAreaXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR1_2014.txt'
caminvmask = '../Examples/camenv_data/invmasks/KR1_2014_inv.jpg'
camimgs = '../Examples/images/KR1_2014_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/manualarea/'
if not os.path.exists(destination):
    os.makedirs(destination)


#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)

##Show ground control points
#cameraenvironment.showGCPs()


#-----------------------   Calculate homography   -----------------------------

#Set homography parameters
hgback=1.0                      #Back-tracking threshold
hgmax=50000                     #Maximum number of points to seed
hgqual=0.1                      #Corner quality for seeding
hgmind=5.0                      #Minimum distance between seeded points
hgminf=4                        #Minimum number of seeded points to track

#Set up Homography object
homog = Homography(camimgs, cameraenvironment, caminvmask, calibFlag=True, 
                   band='L', equal=True)

#Calculate homography
hg = homog.calcHomographyPairs(hgback, hgmax, hgqual, hgmind, hgminf)             
homogmatrix = [item[0] for item in hg] 


#------------------------   Calculate Areas   ---------------------------------

#Define Area class initialisation variables
calibFlag = True            #Detect with corrected or uncorrected images
maxim = 0                   #Image number of maximum areal extent 
imband = 'R'                #Desired image band
equal = True                #Images with histogram equalisation?

#Set up Area object, from which areal extent will be measured
plumes = Area(camimgs, cameraenvironment, homogmatrix, calibFlag, imband, equal)


#-------------------------   Calculate areas   --------------------------------

#Calculate real areas
areas = plumes.calcManualAreas()

##Import areal data from file
#xyzfile=destination+'area_coords.txt'
#pxfile=destination+'px_coords.txt'
#areas = importAreaData(xyzfile, pxfile)


#----------------------------   Export data   ---------------------------------

#Write results to file
imn=plumes.getImageNames()
writeAreaFile(areas, imn, destination)

#Write homography to file
target1 = destination + 'homography.csv'
writeHomogFile(hg, imn, target1)


#Create shapefiles
target1 = destination + 'shpfiles/'    
proj = 32633
xyzpts = [item[0][1] for item in areas]
writeAreaSHP(xyzpts, imn, target1, proj) 
   
#Write all image extents and dems 
target2 = destination + 'outputimgs/'
uvpts = [item[1][1] for item in areas]
dem = cameraenvironment.getDEM()
imgset=plumes._imageSet
cameraMatrix=cameraenvironment.getCamMatrixCV2()
distortP=cameraenvironment.getDistortCoeffsCV2()

#Plot areas in image plane and as XYZ polygons (only if xyz areas calculated)   
for i in range(len(xyzpts)):
    plotAreaPX(uvpts[i], imgset[i].getImageCorr(cameraMatrix, distortP), 
               show=True, save=None)  
    plotAreaXYZ(xyzpts[i], dem, show=True, save=None)
    
    
#---Alternative method for plotting image extents using original RGB images----                                                                                                                                                                                                                                                                                                      

#Get original images in directory
ims = sorted(glob.glob(camimgs))

#Get camera correction variables
newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortP, 
                                            (5184,3456),1,(5184,3456))                                                                                                 
    
#Get corresponding xy pixel areas and images                                            
count=1
for p,i in zip(uvpts,ims):
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
    plt.close()
    count=count+1
    
    
#------------------------------------------------------------------------------    
    
print '\n\nFINISHED'