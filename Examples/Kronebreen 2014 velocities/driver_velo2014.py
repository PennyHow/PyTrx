# -*- coding: utf-8 -*-
"""
Created on Fri May 05 11:04:41 2017

@author: Penny How, p.how@ed.ac.uk


Driver for velocity calculator from camera 2 (2014) at Kronebreen
"""

#Import packages
import sys
import time
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import interpolate

#Define directory
fileDirectory = 'C:/Users/s0824923/Local Documents/python_workspace/pytrx_new/'

#Import PyTrx packages
sys.path.append(fileDirectory)
from CamEnv import CamEnv
from Images import TimeLapse
from FileHandler import writeHomographyFile
from Utilities import plotVelocity, arrowplot, filterSparse

#--------------------------   Initialisation   --------------------------------

#Get data needed for processing
camdata = fileDirectory + 'Data/GCPdata/CameraEnvironmentData_cam1_2014.txt'
camvmask = fileDirectory + 'Data/GCPdata/masks/c2_2014_vmask.JPG'
caminvmask = fileDirectory + 'Data/GCPdata/invmasks/c2_2014_inv.JPG'
camimgs = fileDirectory + 'Data/Images/Velocity/c2_2014_small/*.JPG'

#Define data output directory
destination = fileDirectory + 'Results/cam2_2014velo/'

#Define camera environment
cameraenvironment = CamEnv(camdata)
cameraenvironment.report()


#----------------------------   Prepare DEM   ---------------------------------

#Get DEM from camera environment object
dem=cameraenvironment.getDEM()
print dem.getExtent()

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

print lims
#cameraenvironment._setInvProjVars()


#----------------------   Calculate velocities   ------------------------------

#Set up TimeLapse object
tl=TimeLapse(camimgs, cameraenvironment, camvmask, caminvmask) 

#Calculate homography and velocities    
hg, outputV = tl.calcVelocities()

for vel in outputV:
    XYZs=vel[0][0]
    XYZd=vel[0][1]
    
    xd=XYZs[:,0]-XYZd[:,0]
    yd=XYZs[:,1]-XYZd[:,1]
    speed=np.sqrt(xd*xd+yd*yd)
    
    x1=XYZs[:,0]
    x2=XYZd[:,0]
    y1=XYZs[:,1]
    y2=XYZd[:,1]
    
    gridsize=10.
        
    #Define grid using point extent
    minx=divmod(min(x1),gridsize)[0]*gridsize
    miny=divmod(min(y1),gridsize)[0]*gridsize
    maxx=(divmod(max(x2),gridsize)[0]+1)*gridsize
    maxy=(divmod(max(y2),gridsize)[0]+1)*gridsize
    pointsextent=[minx,maxx,miny,maxy]   
    
    print pointsextent
    
    #Find the new point, with the adjusted origin
    newx = [(x-pointsextent[0]) for x in x1]
    newy = [(y-pointsextent[2]) for y in y1]
    newmaxx = math.floor(max(newx))+1
    newmaxy = math.floor(max(newy))+1  
    newpts=np.array([newx,newy]).T
    
    print 'newpoints',newpts.shape
    print 'vs',np.array(speed).shape
    #print len(xs),len(ys),len(vs),len(snrs)
    print newmaxx,newmaxy
    
    incrsx=((maxx-minx)/gridsize)+1
    incrsy=((maxy-miny)/gridsize)+1
    print 'increments x:y',incrsx,incrsy
    
    grid_y,grid_x = np.mgrid[miny:maxy:complex(incrsy),
                             minx:maxx:complex(incrsx)]
                             
    print grid_x.shape,grid_y.shape
    print grid_x[0:10],grid_y[0:10]
    
    #Interpolate the velocity and error points to the grid
    grid = interpolate.griddata(newpts, np.float64(speed), (grid_x, grid_y), method='linear')
    #error = interpolat.griddata(newpts, np.float64(snrs), (grid_x, grid_y), method=method)            


    
    plt.figure()
    plt.xlim(lims[0],lims[1])
    plt.ylim(lims[2],lims[3])
    plt.locator_params(axis = 'x', nbins=8)
    plt.tick_params(axis='both', which='major', labelsize=10)
#    img = plt.imshow(dem, origin='lower', extent=demex)
#    img.set_cmap('gray') 
    plt.imshow(grid, origin='lower', cmap=plt.get_cmap("gist_ncar"), 
               extent=[446980.0, 448770.0, 8756810.0, 8758790.0], alpha=0.5) #alpha=1 
#    plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=plt.get_cmap("gist_ncar"), s=10, edgecolors='none')
    plt.suptitle('Interpolated', fontsize=14)
    plt.colorbar()
    plt.show()     
   
#---------------------------   Plotting functions   ---------------------------

plotcams = True
plotcombined = True
plotspeed = True
plotmaps = True
save = False

span=[0,-1]
im1=tl.getImageObj(0)

for i in range(tl.getLength()-1)[span[0]:span[1]]:
    for vel in outputV:
        im0=im1
        im1=tl.getImageObj(i+1)
        plotVelocity(vel,im0,im1,cameraenvironment,demred,lims,None,
                     plotcams,plotcombined,plotspeed,plotmaps)


#---------------------------  Export data   -----------------------------------

#Write homography data to .csv file
print hg
print outputV
#writeHomographyFile(hg,tl,fname='homography.csv')

 

#--------------------------   Show results   ----------------------------------




#------------------------------------------------------------------------------
print 'Finished'
