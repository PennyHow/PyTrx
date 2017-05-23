'''
PYTRX EXAMPLE DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates surface velocities at Kronebreen, Svalbard, for the 2014 
melt season using modules in PyTrx. Specifically this script performs feature
tracking through sequential daily images of the glacier to derived surface
velocities (spatial average, individual point displacements and interpolated
velocity maps) which have been corrected for image distortion and camera 
homography.

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
from CamEnv import CamEnv
from Images import TimeLapse
from FileHandler import writeHomographyFile, writeVelocityFile
from Utilities import plotVelocity, interpolateHelper, plotInterpolate


#--------------------------   Initialisation   --------------------------------

#Get data needed for processing
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_cam2_2014.txt'
camvmask = '../Examples/camenv_data/masks/c2_2014_vmask.JPG'
caminvmask = '../Examples/camenv_data/invmasks/c2_2014_inv.JPG'
camimgs = '../Examples/images/c2_2014_subset/*.JPG'


#Define data output directory
destination = '../Examples/results/c2_2014_subset/'


#Define camera environment
cameraenvironment = CamEnv(camdata, quiet=2)
#cameraenvironment.report()


#----------------------------   Prepare DEM   ---------------------------------

#Get DEM from camera environment object
dem=cameraenvironment.getDEM()

#Set extent
xmin=446000
xmax=451000
ymin=8754000
ymax=8760000

demex=dem.getExtent()
xscale=dem.getCols()/(demex[1]-demex[0])
yscale=dem.getRows()/(demex[3]-demex[2])

xdmin=(xmin-demex[0])*xscale
xdmax=((xmax-demex[0])*xscale)+1
ydmin=(ymin-demex[2])*yscale
ydmax=((ymax-demex[2])*yscale)+1
    
demred=dem.subset(xdmin,xdmax,ydmin,ydmax)
lims=demred.getExtent()
demred=demred.getZ()

#cameraenvironment._setInvProjVars()


#----------------------   Calculate velocities   ------------------------------

#Set up TimeLapse object
tl=TimeLapse(camimgs, cameraenvironment, camvmask, caminvmask, image0=0, 
             band='L', quiet=2) 

#Calculate homography and velocities    
hg, outputV = tl.calcVelocities()
   
   
#----------------------------   Plot Results   --------------------------------

print '\nData plotting...'
plotcams = False
plotcombined = False
plotspeed = False
plotmaps = False
save = False

span=[0,-1]
im1=tl.getImageObj(0)

for i in range(tl.getLength()-1)[span[0]:span[1]]:
    for vel in outputV:
        im0=im1
        im1=tl.getImageObj(i+1)
        plotVelocity(vel,im0,im1,cameraenvironment,demred,lims,None,
                     plotcams,plotcombined,plotspeed,plotmaps)

for vel in outputV:
    xy1 = vel[0][0]
    xy2 = vel[0][1]
    method='linear'

    grid, pointsextent = interpolateHelper(xy1,xy2,method,filt=False)
    fgrid, fpointsextent = interpolateHelper(xy1,xy2,method,filt=True)
    
    print 'Plotting unfiltered velocity map...'
    plotInterpolate(demred, lims, grid, pointsextent, 
                    save=destination+'interp1.jpg')
                    
    print 'Plotting filtered velocity map...'
    plotInterpolate(demred, lims, fgrid, fpointsextent, 
                    save=destination+'interp2.jpg')    


#---------------------------  Export data   -----------------------------------

print '\nBeginning file exporting...'

#Write homography data to .csv file
target1 = destination + 'homography.csv'
writeHomographyFile(hg,tl,target1)

#Write out velocity data to .csv file
target2 = destination + 'velo_output.csv'
writeVelocityFile(outputV, tl, target2) 


#------------------------------------------------------------------------------
print 'Finished'