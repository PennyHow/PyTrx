'''
PYTRX EXAMPLE MANUAL LINE DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates terminus profiles (as line features) at Tunabreen, 
Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. 
Specifically this script performs manual detection of terminus position through 
sequential images of the glacier to derive line profiles which have been 
corrected for image distortion. 

Previously defined lines can also be imported from text or shape file (this 
can be changed by commenting and uncommenting commands in the "Calculate lines" 
section of this script).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton
         Lynne Buie
'''

#Import packages
import sys
import os

#Import PyTrx packages
sys.path.append('../')
from Line import Line
from Velocity import Homography
from CamEnv import CamEnv
from FileHandler import writeLineFile, writeLineSHP, writeHomogFile, importLineData
from Utilities import plotLinePX, plotLineXYZ


#---------------------------   Initialisation   -------------------------------

#Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU1_2015.txt'
invmask = '../Examples/camenv_data/invmasks/TU1_2015_inv.jpg'  
camimgs = '../Examples/images/TU1_2015_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/manualline/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)


#---------------------   Calculate homography   -------------------------------

#Set homography parameters
hgback=1.0                      #Back-tracking threshold
hgmax=50000                     #Maximum number of points to seed
hgqual=0.1                      #Corner quality for seeding
hgmind=5.0                      #Minimum distance between seeded points
hgminf=4                        #Minimum number of seeded points to track

#Set up Homography object
homog = Homography(camimgs, cam, invmask, calibFlag=True, band='L', equal=True)

#Calculate homography
hg = homog.calcHomographyPairs(hgback, hgmax, hgqual, hgmind, hgminf)             
homogmatrix = [item[0] for item in hg] 


#-----------------------   Calculate/import lines   ---------------------------

#Set up line object
terminus = Line(camimgs, cam, homogmatrix)


#Manually define terminus lines
lines = terminus.calcManualLines()


##Import lines from textfiles
#xyzfile=destination+'line_realcoords.txt'
#pxfile=destination+'line_pxcoords.txt'
#lines=importLineData(xyzfile, pxfile)

#----------------------------   Export data   ---------------------------------

#Write line data to txt file
imn=terminus.getImageNames()
writeLineFile(lines, imn, destination)

#Write homography data to .csv file
target1 = destination + 'homography.csv'
writeHomogFile(hg, imn, target1)

#Write shapefiles from line data
target2 = destination + 'shapefiles/'   
proj = 32633
xyzcoords = [item[0][1] for item in lines]
writeLineSHP(xyzcoords, imn, target2, proj)
#
#
##----------------------------   Show results   --------------------------------  

#Define destination
target3 = destination + 'outputimgs/'

#Get dem, images, camera matrix and distortion parameters
dem = cam.getDEM()
imgset=terminus._imageSet
cameraMatrix=cam.getCamMatrixCV2()
distortP=cam.getDistortCoeffsCV2()
pxcoords = [item[1][1] for item in lines]
#
#Plot lines in image plane and as XYZ lines 
for i in range(len(pxcoords)):
    plotLinePX(pxcoords[i], imgset[i].getImageCorr(cameraMatrix, distortP), 
               show=True, save=target3+'uv_'+str(imn[i]))  
    plotLineXYZ(xyzcoords[i], dem, show=True, save=target3+'xyz_'+str(imn[i]))

    
#------------------------------------------------------------------------------

print '\n\nFinished'