'''
PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie

PyTrx is licensed under a MIT License.

You should have received a copy of the license along with this
work. If not, see <https://choosealicense.com/licenses/mit/>.


PYTRX EXAMPLE SPARSE VELOCITY DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates sparse surface velocities using modules in PyTrx at 
Kronebreen, Svalbard, for a subset of the images collected during the 2014 melt 
season. Specifically this script performs feature-tracking through sequential 
daily images of the glacier to derive surface velocities (spatial average, 
individual point displacements and interpolated velocity maps) which have been 
corrected for image distortion and motion in the camera platform (i.e. image
registration).
'''

#Import packages
import os
import numpy as np

#Import PyTrx modules (from PyTrx file directory)
import sys
sys.path.append('../')
from CamEnv import CamEnv
from Velocity import Velocity, Homography
from FileHandler import writeHomogFile, writeVeloFile, writeVeloSHP, writeCalibFile
from Utilities import plotVeloPX, plotVeloXYZ, interpolateHelper, plotInterpolate

##If you have pip/conda installed PyTrx then comment out the PyTrx module
##imports above and uncomment these ones below
#from PyTrx.CamEnv import CamEnv
#from PyTrx.Velocity import Velocity, Homography
#from PyTrx.FileHandler import writeHomogFile, writeVeloFile, writeVeloSHP, writeCalibFile
#from PyTrx.Utilities import plotVeloPX, plotVeloXYZ, interpolateHelper, plotInterpolate


#-------------------------   Map data sources   -------------------------------

#Get data needed for processing
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR2_2014.txt'
camvmask = '../Examples/camenv_data/masks/KR2_2014_vmask.jpg'
caminvmask = '../Examples/camenv_data/invmasks/KR2_2014_inv.jpg'
camimgs = '../Examples/images/KR2_2014_subset/*.JPG'


#Define data output directory
destination = '../Examples/results/velocity1/'
if not os.path.exists(destination):
    os.makedirs(destination)


#-----------------------   Create camera object   -----------------------------

#Define camera environment
cameraenvironment = CamEnv(camdata)

#Optimise camera environment to refine camera pose
cameraenvironment.optimiseCamEnv('YPR')

#Report camera data and show corrected image
cameraenvironment.reportCamData()
cameraenvironment.showGCPs()
cameraenvironment.showPrincipalPoint()
cameraenvironment.showResiduals()


#----------------------   Calculate homography   ------------------------------

#Set homography parameters
hmethod='sparse'                #Method
hgwinsize=(25,25)               #Tracking window size
hgback=1.0                      #Back-tracking threshold
hgmax=50000                     #Maximum number of points to seed
hgqual=0.1                      #Corner quality for seeding
hgmind=5.0                      #Minimum distance between seeded points
hgminf=4                        #Minimum number of seeded points to track

#Set up Homography object
homog = Homography(camimgs, cameraenvironment, caminvmask, calibFlag=True, 
                band='L', equal=True)

#Calculate homography
hgout = homog.calcHomographies([hmethod, [hgmax, hgqual, hgmind], [hgwinsize, 
                                hgback, hgminf]])

    
#----------------------   Calculate velocities   ------------------------------

#Set velocity parameters
vmethod='sparse'                #Method
vwinsize=(25,25)                #Tracking window size
bk = 1.0                        #Back-tracking threshold  
mpt = 50000                     #Maximum number of points to seed
ql = 0.1                        #Corner quality for seeding
mdis = 5.0                      #Minimum distance between seeded points
mfeat = 4                       #Minimum number of seeded points to track

#Set up Velocity object
velo=Velocity(camimgs, cameraenvironment, hgout, camvmask, calibFlag=True, 
              band='L', equal=True) 

velocities = velo.calcVelocities([vmethod, [mpt, ql, mdis], [vwinsize, bk, 
                                  mfeat]])                                   
                                    
xyzvel=[item[0][0] for item in velocities] 
xyz0=[item[0][1] for item in velocities]
xyz1=[item[0][2] for item in velocities]
xyzerr=[item[0][3] for item in velocities]
uvvel=[item[1][0] for item in velocities]
uv0=[item[1][1] for item in velocities] 
uv1=[item[1][2] for item in velocities]
uv1corr=[item[1][3] for item in velocities]


#---------------------------  Export data   -----------------------------------

print('\n\nWRITING DATA TO FILE')

#Write out camera calibration info to .txt file
target1 = '../Examples/camenv_data/calib/KR2_2014_1.txt'
matrix, tancorr, radcorr = cameraenvironment.getCalibdata()
writeCalibFile(matrix, tancorr, radcorr, target1)

#Write out velocity data to .csv file
target2 = destination + 'velo_output.csv'
imn = velo.getImageNames()
writeVeloFile(xyzvel, uvvel, hgout, imn, target2) 

#Write homography data to .csv file
target3 = destination + 'homography.csv'
writeHomogFile(hgout, imn, target3)

#Write points to shp file
target4 = destination + 'shpfiles/'     #Define file destination
if not os.path.exists(target4):
    os.makedirs(target3)                #Create file destination
proj = 32633                            #ESPG:32633 is projection WGS84
writeVeloSHP(xyzvel, xyzerr, xyz0, imn, target4, proj)       #Write shapefile

  
#----------------------------   Plot Results   --------------------------------

print('\n\nPLOTTING DATA')

#Set interpolation method ("nearest"/"cubic"/"linear")
method='linear' 

#Set DEM extent         
cr1 = [445000, 452000, 8754000, 8760000]            

#Set destination for file outputs
target4 = destination + 'imgfiles/'
if not os.path.exists(target4):
    os.makedirs(target4)

cameraMatrix=cameraenvironment.getCamMatrixCV2()
distortP=cameraenvironment.getDistortCoeffsCV2() 
dem=cameraenvironment.getDEM()
imgset=velo._imageSet

#Cycle through data from image pairs   
for i in range(len(imn)-1):

    #Get image name and print
    print('\nVisualising data for ' + str(imn[i]))

    #Plot uv velocity points on image plane   
    print('Plotting image plane output')
    plotVeloPX(uvvel[i], uv0[i], uv1corr[i], 
               imgset[i].getImageCorr(cameraMatrix, distortP), 
               show=True, save=target4+'uv_'+imn[i])


    #Plot xyz velocity points on dem  
    print('Plotting XYZ output')
    plotVeloXYZ(xyzvel[i], xyz0[i], xyz1[i], 
                dem, show=True, save=target4+'xyz_'+imn[i])
    
                
    #Plot interpolation map
    print('Plotting interpolation map')
    grid, pointsextent = interpolateHelper(xyzvel[i], xyz0[i], xyz1[i], method)
    plotInterpolate(grid, pointsextent, dem, show=True, 
                    save=target4+'interp_'+imn[i])                        


#--------   Example exporting raster grid of velocities as ASCII file   -------

#The text files generated here are ascii-formatted. These are recognised by
#many mapping software, such as ArcGIS and QGIS, and imported to create raster
#surfaces

print('\n\nWRITING ASCII FILES')

#Set destination for file outputs
target5 = destination + 'asciifiles/'
if not os.path.exists(target5):
    os.makedirs(target5)

#Cycle through velocity data from image pairs   
for i in range(velo.getLength()-1): 
    
    #Change all the nans to -999.999 and flip the y axis
    grid[np.isnan(grid)] = -999.999     
    grid = np.flipud(grid)  
    
    #Open the fileName file with write permissions
    imn=velo._imageSet[i].getImageName()
    afile = open(target5 + imn + '_interpmap.txt','w')
    print('\nWriting file: ' + str(target5) + str(imn) + '_interpmap.txt')
    
    #Make a list for each raster header variable, with the label and value
    col = ['ncols', str(grid.shape[1])]
    row = ['nrows', str(grid.shape[0])]
    x = ['xllcorner', str(pointsextent[0])]
    y = ['yllcorner', str(pointsextent[2])]
    cell = ['cellsize', str(10.)]
    nd = ['NODATA_value', str(-999.999)]
    
    #Write each header line on a new line of the file
    header = [col,row,x,y,cell,nd]       
    for i in header:
        afile.write(' '.join(i) + '\n')
    
    #Iterate through each row and column value
    for i in range(grid.shape[0]): 
        for j in range(grid.shape[1]):
            
            #Write each data value to the row, separated by spaces
            afile.write(str(grid[i,j]) + ' ')
            
        #New line at end of row
        afile.write('\n')
    
    #Close file
    afile.close() 


#------------------------------------------------------------------------------
print('\nFinished')
