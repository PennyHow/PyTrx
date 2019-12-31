# -*- coding: utf-8 -*-
'''
PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie

PyTrx is licensed under a
Creative Commons Attribution 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.


PYTRX EXAMPLE DENSE VELOCITY DRIVER (EXTENDED VERSION)

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

This script is a class-independent version of 'driver_velocity1.py'. 
The functions used here do not depend on class object inputs and can be run as 
stand-alone functions.

This script has been included in order to provide the user with a more detailed 
overview of PyTrx's functionality beyond its object-oriented structure. It also 
allows flexible intervention and adaptation where needed. 
'''

#Import packages
import sys
import cv2
import glob
import numpy as np
from pathlib import Path

#Import PyTrx packages
sys.path.append('../')
from CamEnv import setProjection, optimiseCamera, computeResidualsXYZ
from DEM import load_DEM
import Velocity
import FileHandler
import Utilities 
 
#------------------------   Define inputs/outputs   ---------------------------

print('\nDEFINING DATA INPUTS')

#Camera name, location (XYZ) and pose (yaw, pitch, roll)
camname = 'KR2_2014'
camloc = np.array([447948.820, 8759457.100, 407.092])

#campose = np.array([4.80926, 0.05768, 0.14914]) 
campose = np.array([4.80926, 0.05768, 0.14914]) 

#Define image folder and image file type for velocity tracking
imgFiles = '../Examples/images/KR2_2014_subset/*.JPG'

#Define calibration images and chessboard dimensions (height, width)
calibPath = '../Examples/camenv_data/calib/KR2_2014_2.txt'

#Load DEM from path
#DEMpath = '../Examples/camenv_data/dem/KR_arcticdem_20140512.tif'        
DEMpath = '../Examples/camenv_data/dem/KR_demsmooth.tif'

#Define masks for velocity and homography point generation
vmaskPath_s = '../Examples/camenv_data/masks/KR2_2014_vmask.jpg'
vmaskPath_d = '../Examples/camenv_data/masks/KR2_2014_dem_vmask.jpg'      
hmaskPath = '../Examples/camenv_data/invmasks/KR2_2014_inv.jpg'    

#Define reference image (where GCPs have been defined)
refimagePath = '../Examples/camenv_data/refimages/KR2_2014.JPG'

#Define GCPs (world coordinates and corresponding image coordinates)
GCPpath = '../Examples/camenv_data/gcps/KR2_2014.txt'
#GCPpath = '../Examples/camenv_data/gcps/KR2_2014_arcticdem20140512.txt'

print('\nDEFINING DATA OUTPUTS')

#Shapefile output (with WGS84 projection)
target1 = '../Examples/results/velocity3/shpfiles_sparse/' 
target2 = '../Examples/results/velocity3/shpfiles_dense/'
projection = 32633


#--------------------------   Define parameters   -----------------------------

#DEM parameters 
DEMdensify = 2                      #DEM densification factor (for smoothing)

#Optimisation parameters
optparams = 'YPR'                   #Parameters to optimise
optmethod = 'trf'                   #Optimisation method

#Image enhancement paramaters
band = 'L'                          #Image band extraction (R, B, G, or L)
equal = True                        #Histogram equalisation?

#Sparse velocity parameters
vwin = (25,25)                      #Sparse corner matching window size
vback = 1.0                         #Back-tracking threshold  
vmax = 50000                        #Maximum number of corners to seed
vqual = 0.1                         #Corner quality for seeding
vmindist = 5.0                      #Minimum distance between points

#Dense velocity parameters
vgrid = [50,50]                     #Dense matching grid distance
vtemplate=10                        #Template size
vsearch=50                          #Search window size
vmethod='cv2.TM_CCORR_NORMED'       #Method for template matching
vthres=0.8                          #Threshold average template correlation

#General velocity parameters
vminfeat = 1                        #Minimum number of points to track
                         
#Homography parameters
hwin = (25,25)                      #Stable pt tracking window size
hmethod = cv2.RANSAC                #Homography calculation method 
                                    #(cv2.RANSAC, cv2.LEAST_MEDIAN, or 0)
hreproj = 5.0                       #Maximum allowed reprojection error
hback = 0.5                         #Back-tracking threshold
herr = True                         #Calculate tracking error?
hmax = 50000                        #Maximum number of points to seed
hqual = 0.1                         #Corner quality for seeding
hmindist = 5.0                      #Minimum distance between seeded points
hminfeat = 4                        #Minimum number of seeded points to track


#----------------------   Set up camera environment   -------------------------

print('\nLOADING DEM')
dem = load_DEM(DEMpath)
dem = dem.densify(DEMdensify)


print('\nLOADING GCPs')
GCPxyz, GCPuv = FileHandler.readGCPs(GCPpath)


print('\nLOADING CALIBRATION')
calib_out = FileHandler.readMatrixDistortion(calibPath)
matrix=np.transpose(calib_out[0])                          
tancorr = calib_out[1]                                     
radcorr = calib_out[2]                                      
focal = [matrix[0,0], matrix[1,1]]                          
camcen = [matrix[0,2], matrix[1,2]]                         
 

print('\nLOADING IMAGE FILES')
imagelist = sorted(glob.glob(imgFiles))
im1 = FileHandler.readImg(imagelist[0], band, equal)
imn1 = Path(imagelist[0]).name


print('\nOPTIMISING CAMERA ENVIRONMENT')
projvars = [camloc, campose, radcorr, tancorr, focal, camcen, refimagePath] 
new_projvars = optimiseCamera('YPR', projvars, GCPxyz, GCPuv, 
                              optmethod=optmethod, show=True)


print('\nCOMPILING TRANSFORMATION PARAMETERS')
camloc1, campose1, radcorr1, tancorr1, focal1, camcen1, refimagePath = new_projvars

matrix1 = np.array([focal1[0], 0, camcen[0], 0, focal1[1], 
                   camcen[1], 0, 0, 1]).reshape(3,3)

distort = np.hstack([radcorr1[0][0], radcorr1[0][1],          
                     tancorr1[0][0], tancorr1[0][1],          
                     radcorr1[0][2]])
 
new_invprojvars = setProjection(dem, camloc1, campose1, 
                            radcorr1, tancorr1, focal1, 
                            camcen1, refimagePath)

campars = [dem, new_projvars, new_invprojvars]                 

residuals = computeResidualsXYZ(new_invprojvars, GCPxyz, GCPuv, dem)

print('\nLOADING MASKS')
print('Defining velocity mask')
vmask1 = FileHandler.readMask(None, vmaskPath_s)
vmask2 = Velocity.readDEMmask(dem, im1, new_invprojvars, vmaskPath_d)

print('Defining homography mask')
hmask = FileHandler.readMask(None, hmaskPath)


#--------------------   Plot camera environment info   ------------------------

print('\nPLOTTING CAMERA ENVIRONMENT INFO')

#Load reference image
refimg = FileHandler.readImg(refimagePath) 
imn = Path(refimagePath).name    

#Show Prinicpal Point in image
Utilities.plotPrincipalPoint(camcen1, refimg, imn)

#Show corrected and uncorrected image
Utilities.plotCalib(matrix1, distort, refimg, imn)

#Show GCPs
Utilities.plotGCPs([GCPxyz, GCPuv], refimg, imn, 
                   dem, camloc, extent=None)    


#----------------------   Calculate velocities   ------------------------------

print('\nCALCULATING VELOCITIES')

#Create empty output variables
velo1 = [] 
velo2 = []                                    
homog = []

#Cycle through image pairs (numbered from 0)
for i in range(len(imagelist)-1):

    #Re-assign first image in image pair
    im0=im1
    imn0=imn1
                    
    #Get second image (corrected) in image pair
    im1 = FileHandler.readImg(imagelist[i+1], band, equal)
    imn1 = Path(imagelist[i+1]).name                                                       
    
    
    print('\nProcessing images: ' + str(imn0) + ' and ' + str(imn1))
        
    #Calculate homography between image pair
    print('Calculating homography...')

    hg = Velocity.calcSparseHomography(im0, im1, hmask, [matrix1,distort], 
                                       hmethod, hreproj, hwin, hback, hminfeat, 
                                       [hmax, hqual, hmindist])
    homog.append(hg)
                             
    #Calculate velocities between image pair
    print('Calculating sparse velocities...')
    vl1 = Velocity.calcSparseVelocity(im0, im1, vmask1, [matrix1,distort], 
                                      [hg[0],hg[3]], new_invprojvars, vwin, 
                                      vback, vminfeat, [vmax,vqual,vmindist])
    
    print('Calculating dense velocities...')    
    vl2 = Velocity.calcDenseVelocity(im0, im1, vgrid, vmethod, vtemplate, 
                                     vsearch, vmask2, [matrix1,distort], 
                                     [hg[0],hg[3]], campars, vthres, vminfeat)   
 
    velo1.append(vl1) 
    velo2.append(vl2)            

#---------------------------  Export data   -----------------------------------

print('\nWRITING DATA TO FILE')

#Get all image names
names=[]
for i in imagelist:
    names.append(str(Path(i).name).split('.JPG')[0])

#Extract xyz velocities, uv velocities, and xyz0 locations
xyzvel1=[item[0][0] for item in velo1] 
xyzerr1=[item[0][3] for item in velo1]
uvvel1=[item[1][0] for item in velo1]
xyz01=[item[0][1] for item in velo1]

xyzvel2=[item[0][0] for item in velo2] 
xyzerr2=[item[1][4] for item in velo2]
uvvel2=[item[1][0] for item in velo2]
xyz02=[item[0][1] for item in velo2]

xyzerr2 = xyzerr2[0].flatten()

#Write points to shp file                
FileHandler.writeVeloSHP(xyzvel1, xyzerr1, xyz01, names, target1, projection)       
FileHandler.writeVeloSHP(xyzvel2, xyzerr2, xyz02, names, target2, projection)
    
#------------------------------------------------------------------------------
print('\nFinished')