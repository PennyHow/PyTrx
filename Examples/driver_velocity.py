'''
PYTRX EXAMPLE VELOCITY DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates surface velocities at Kronebreen, Svalbard, for the 2014 
melt season using modules in PyTrx. Specifically this script performs feature
tracking through sequential daily images of the glacier to derive surface
velocities (spatial average, individual point displacements and interpolated
velocity maps) which have been corrected for image distortion and camera 
homography.

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)

'''

#Import packages
import sys


#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv
from Measure import Velocity
from FileHandler import writeHomographyFile, writeVelocityFile
from Utilities import plotVelocity, interpolateHelper, plotInterpolate


#-------------------------   Map data sources   -------------------------------

#Get data needed for processing
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_cam2_2014.txt'
camvmask = '../Examples/camenv_data/masks/c2_2014_vmask.JPG'
caminvmask = '../Examples/camenv_data/invmasks/c2_2014_inv.JPG'
camimgs = '../Examples/images/KR2_2014_subset/*.JPG'


#Define data output directory
destination = '../Examples/results/KR2_velocity/'


#-----------------------   Create camera object   -----------------------------

#Define camera environment
cameraenvironment = CamEnv(camdata, quiet=2)


#----------------------   Calculate velocities   ------------------------------

#Set up TimeLapse object
vels=Velocity(camimgs, cameraenvironment, camvmask, caminvmask, image0=0, 
            band='L', quiet=2) 

#Calculate homography and velocities    
hg, outputV = vels.calcVelocities()
   
   
#----------------------------   Plot Results   --------------------------------

print '\nData plotting...'
plotcams = False
plotcombined = False
plotspeed = False
plotmaps = False
save = False


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


span=[0,-1]
im1=vels.getImageObj(0)

for i in range(vels.getLength()-1)[span[0]:span[1]]:
    for vel in outputV:
        im0=im1
        im1=vels.getImageObj(i+1)
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
writeHomographyFile(hg,vels,target1)

#Write out velocity data to .csv file
target2 = destination + 'velo_output.csv'
writeVelocityFile(outputV, vels, target2) 


#------------------------------------------------------------------------------
print '\nFinished'