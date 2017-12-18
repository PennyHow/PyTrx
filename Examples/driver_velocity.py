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
         Nick Hulton 
'''

#Import packages
import sys
import os
import numpy as np


#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv
from Measure import Velocity
from FileHandler import writeHomographyFile, writeVelocityFile, writeSHPFile
from Utilities import plotPX, plotXYZ, interpolateHelper, plotInterpolate


#-------------------------   Map data sources   -------------------------------

#Get data needed for processing
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR2_2014.txt'
camvmask = '../Examples/camenv_data/masks/KR2_2014_vmask.JPG'
caminvmask = '../Examples/camenv_data/invmasks/KR2_2014_inv.JPG'
camimgs = '../Examples/images/KR2_2014_subset/*.JPG'


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


#Set velocity parameters
hmg = True                      #Calculate homography?
err = True                      #Calculate errors?
bk = 1.0                        #Back-tracking threshold  
mpt = 50000                     #Maximum number of points to seed
ql = 0.1                        #Corner quality for seeding
mdis = 5.0                      #Minimum distance between seeded points
mfeat = 4                       #Minimum number of seeded points to track


#Calculate velocities and homography    
xyz, uv = velo.calcVelocities(homography=hmg, calcErrors=err, back_thresh=bk,
                              maxpoints=mpt, quality=ql, mindist=mdis, 
                              min_features=mfeat)
                                 
    
#---------------------------  Export data   -----------------------------------

print '\n\nWRITING DATA TO FILE'

#Write out velocity data to .csv file
target1 = destination + 'velo_output.csv'
writeVelocityFile(velo, target1) 

#Write homography data to .csv file
target2 = destination + 'homography.csv'
writeHomographyFile(velo, target2)

#Write points to shp file
target3 = destination + 'shpfiles/'     #Define file destination
if not os.path.exists(target3):
    os.makedirs(target3)                #Create file destination
proj = 32633                            #ESPG:32633 is projection WGS84
writeSHPFile(velo, target3, proj)       #Write shapefile


#----------------------------   Plot Results   --------------------------------

print '\n\nPLOTTING DATA'

#Set interpolation method ("nearest"/"cubic"/"linear")
method='linear' 

#Set DEM extent         
cr1 = [445000, 452000, 8754000, 8760000]            

#Set destination for file outputs
target4 = destination + 'imgfiles/'
if not os.path.exists(target4):
    os.makedirs(target4)
 
#Cycle through data from image pairs   
for i in range(velo.getLength()-1):
    
    #Get image0 name and print
    imn=velo._imageSet[i].getImagePath().split('\\')[1]
    print '\nVisualising data for ' + str(imn)


    #Plot uv velocity points on image plane   
    print 'Plotting image plane output'
    plotPX(velo, i, target4, crop=None, show=True)


    #Plot xyz velocity points on dem  
    print 'Plotting XYZ output'
    plotXYZ(velo, i, target4, crop=cr1, show=True, dem=True)

                
    #Plot interpolation map
    print 'Plotting interpolation map'
    grid, pointsextent = interpolateHelper(velo, i, method)
    plotInterpolate(velo, i, grid, pointsextent, show=True, save=target4, 
                    crop=cr1)                        


#--------   Example exporting raster grid of velocities as ASCII file   -------

#The text files generated here are ascii-formatted. These are recognised by
#many mapping software, such as ArcGIS and QGIS, and imported to create raster
#surfaces

print '\n\nWRITING ASCII FILES'

#Set destination for file outputs
target5 = destination + 'asciifiles/'
if not os.path.exists(target5):
    os.makedirs(target5)

#Cycle through velocity data from image pairs   
for i in range(velo.getLength()-1): 
    
    #Interpolate velocity points to grid
    grid, pointsextent = interpolateHelper(velo, i, method)
    
    #Change all the nans to -999.999 and flip the y axis
    grid[np.isnan(grid)] = -999.999     
    grid = np.flipud(grid)  

    
    #Open the fileName file with write permissions
    imn=velo._imageSet[i].getImagePath().split('\\')[1]
    afile = open(target5 + imn + '_interpmap.txt','w')
    print '\nWriting file: ' + target5 + imn + '_interpmap.txt'
    
    #Make a list for each raster header variable, with the label and value
    col = ["ncols", str(grid.shape[1])]
    row = ["nrows", str(grid.shape[0])]
    x = ["xllcorner", str(pointsextent[0])]
    y = ["yllcorner", str(pointsextent[2])]
    cell = ["cellsize", str(10.)]
    nd = ["NODATA_value", str(-999.999)]
    
    #Write each header line on a new line of the file
    header = [col,row,x,y,cell,nd]       
    for i in header:
        afile.write(" ".join(i) + "\n")
    
    #Iterate through each row and column value
    for i in range(grid.shape[0]): 
        for j in range(grid.shape[1]):
            
            #Write each data value to the row, separated by spaces
            afile.write(str(grid[i,j]) + " ")
            
        #New line at end of row
        afile.write("\n")
    
    #Close file
    afile.close() 


#------------------------------------------------------------------------------
print '\nFinished'