'''
PYTRX EXAMPLE VELOCITY DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates surface velocities using modules in PyTrx at Kronebreen,
Svalbard, for a subset of the images collected during the 2014 melt season. 
Specifically this script performs feature-tracking through sequential daily 
images of the glacier to derive surface velocities (spatial average, 
individual point displacements and interpolated velocity maps) which have been 
corrected for image distortion and motion in the camera platform (i.e. image
registration).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)

'''

#Import packages
import sys
import os


#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv
from Measure import Velocity
from FileHandler import writeHomographyFile, writeVelocityFile
from Utilities import plotVelocity, interpolateHelper, plotInterpolate


#-------------------------   Map data sources   -------------------------------

#Get data needed for processing
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR2_2014.txt'
camvmask = '../Examples/camenv_data/masks/KR2_2014_vmask.JPG'
caminvmask = '../Examples/camenv_data/invmasks/KR2_2014_inv.JPG'
camimgs = '../Examples/images/KR2_2014_subset/demo/*.JPG'


#Define data output directory
destination = '../Examples/results/velocity/'
if not os.path.exists(destination):
    os.makedirs(destination)


#-----------------------   Create camera object   -----------------------------

#Define camera environment
cameraenvironment = CamEnv(camdata, quiet=2)


#----------------------   Calculate velocities   ------------------------------

#Set up Velocity object
velo=Velocity(camimgs, cameraenvironment, camvmask, caminvmask, image0=0, 
            band='L', quiet=2) 


#Calculate homography and velocities    
xyz, uv = velo.calcVelocities()

print 'Velocities calculated for ' + str(len(xyz[0])) + ' image pairs'  
 
  
#----------------------------   Plot Results   --------------------------------

print '\n\nPLOTTING DATA'
plotcams = True
plotcombined = True
plotspeed = True
plotmaps = True
save = True


for i in range(velo.getLength()-1):
    plotVelocity(velo, i, save=None, px=True, xyz=True)

for vel in xyz:
    xy1 = vel[0]
    xy2 = vel[1]
    method='linear'

    grid, pointsextent = interpolateHelper(xy1,xy2,method,filt=False)
    fgrid, fpointsextent = interpolateHelper(xy1,xy2,method,filt=True)
    
    colrange=[0,4]
    
    print 'Plotting unfiltered velocity map...'
    plotInterpolate(demred, lims, grid, pointsextent, show=True, 
                    save=destination+'interp.jpg')
                    
    print 'Plotting filtered velocity map...'
    plotInterpolate(demred, lims, fgrid, fpointsextent, show=True, 
                    save=destination+'interpfiltered.jpg')    


#---------------------------  Export data   -----------------------------------

print '\n\nWRITING DATA TO FILE'

#Write homography data to .csv file
target1 = destination + 'homography.csv'
writeHomographyFile(hg,vels,target1)

#Write out velocity data to .csv file
target2 = destination + 'velo_output.csv'
writeVelocityFile(outputV, hg, vels, target2) 


#------------------------------------------------------------------------------

print '\nFinished'