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
import numpy as np
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.ma as ma
import ogr
from PIL import Image, ImageDraw
from matplotlib import cm
from matplotlib import path

#Import PyTrx packages
sys.path.append('../')
from CamEnv import setProjection, projectXYZ, projectUV
from Velocity import calcDenseVelocity, calcHomography, seedGrid, readDEMmask
from DEM import load_DEM, voxelviewshed, ExplicitRaster
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
calibPath = '../Examples/camenv_data/calib/KR2_2014_1.txt'

#Load DEM from path
DEMpath = '../Examples/camenv_data/dem/KR_demsmooth.tif'        

#Define masks for velocity and homography point generation
vmaskPath = None       
#hmaskPath = '../Examples/camenv_data/invmasks/KR2_2014_inv.jpg'    

#Define reference image (where GCPs have been defined)
refimagePath = '../Examples/camenv_data/refimages/KR2_2014.JPG'

#Define GCPs (world coordinates and corresponding image coordinates)
GCPpath = '../Examples/camenv_data/gcps/KR2_2014.txt'


print('\nDEFINING DATA OUTPUTS')

#Velocity output
target1 = '../Examples/results/velocity3/velo_output.csv'

#Homography output
target2 = '../Examples/results/velocity3/homography.csv'

#Shapefile output (with WGS84 projection)
target3 = '../Examples/results/velocity3/shpfiles/'     
projection = 32633

#Plot outputs
target4 = '../Examples/results/velocity3/imgfiles/'
interpmethod='linear'                                 #nearest/cubic/linear
cr1 = [445000, 452000, 8754000, 8760000]              #DEM plot extent   


#--------------------------   Define parameters   -----------------------------

#DEM parameters 
DEMdensify = 2                      #DEM densification factor (for smoothing)

#Image enhancement paramaters
band = 'L'                          #Image band extraction (R, B, G, or L)
equal = True                        #Histogram equalisation?

#Velocity parameters
vwin = (25,25)                      #Tracking window size
vback = 1.0                         #Back-tracking threshold  
vmax = 50000                        #Maximum number of points to seed
vqual = 0.1                         #Corner quality for seeding
vmindist = 5.0                      #Minimum distance between seeded points
vminfeat = 4                        #Minimum number of seeded points to track
                           
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

print('\nLOADING MASKS')
#vmask = FileHandler.readMask(None, vmaskPath)
#hmask = FileHandler.readMask(None, hmaskPath)


print('\nLOADING DEM')
dem = load_DEM(DEMpath)
dem=dem.densify(DEMdensify)


print('\nLOADING GCPs')
GCPxyz, GCPuv = FileHandler.readGCPs(GCPpath)


print('\nLOADING CALIBRATION')
calib_out = FileHandler.readMatrixDistortion(calibPath)
matrix=np.transpose(calib_out[0])                               #Get matrix
tancorr = calib_out[1]                                      #Get tangential
radcorr = calib_out[2]                                      #Get radial
focal = [matrix[0,0], matrix[1,1]]                          #Focal length
camcen = [matrix[0,2], matrix[1,2]]                         #Principal point 

   
invprojvars = setProjection(dem, camloc, campose, radcorr, tancorr, focal, 
                            camcen, refimagePath) 


#--------------------   Plot camera environment info   ------------------------

print('\nPLOTTING CAMERA ENVIRONMENT INFO')

##Load reference image
#refimg = FileHandler.readImg(refimagePath) 
#imn = Path(refimagePath).name

##Show GCPs
#Utilities.plotGCPs([GCPxyz, GCPuv], refimg, imn, 
#                   dem, camloc, extent=None)          

##Show Prinicpal Point in image
#Utilities.plotPrincipalPoint(camcen, refimg, imn)

#Show corrected and uncorrected image
distort = np.hstack([radcorr[0][0], radcorr[0][1], tancorr[0][0], 
                     tancorr[0][1], radcorr[0][2]])
#Utilities.plotCalib(matrix, distort, refimg, imn)


#----------------------   Calculate velocities   ------------------------------

print('\nCALCULATING VELOCITIES')

#Get list of images
imagelist = sorted(glob.glob(imgFiles))

#Get first image in sequence and name
im1 = FileHandler.readImg(imagelist[0], band, equal)
imn1 = Path(imagelist[0]).name

#Make DEM mask
demz = dem.getZ()

demmask = readDEMmask(dem, im1, invprojvars, vmaskPath)

#Create empty output variables
velo = []                                     
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
#    print('Calculating homography...')
#    hg = calcHomography(im0, im1, hmask, [matrix,distort], hmethod, hreproj, 
#                        hwin, hback, hminfeat, [hmax, hqual, hmindist])
                             
    #Calculate velocities between image pair
    print('Calculating velocity...')
    griddistance = [100,100]
    projvars = [camloc, campose, radcorr, tancorr, focal, camcen, im0]
    xyz, uv0 = seedGrid(dem, griddistance, projvars, min_features=4, mask=demmask)
    
    #Create count for successful and failed tracking attempts
    success = 0
    fail = 0

    templatesize = 10
    searchsize=50
    
    method='opticalflow'
#    method='cv2.TM_CCORR_NORMED'  
    supersample=0.01
    threshold=2.0

    if method != 'opticalflow':
        
        maxcorr=[]
        avercorr=[]
        pu2=[]
        pv2=[]        
        
        #Get rows one by one
        for u,v in zip(uv0[:,0], uv0[:,1]):
             
            #Get template and search scene
            template = im0[int(v-(templatesize/2)):int(v+(templatesize/2)), 
                          int(u-(templatesize/2)):int(u+(templatesize/2))]
            search = im1[int(v-(searchsize/2)):int(v+(searchsize/2)), 
                        int(u-(searchsize/2)):int(u+(searchsize/2))]       
                   
            #Change array values from float64 to uint8
            template = template.astype(np.uint8)
            search = search.astype(np.uint8)
                          
            #Define method string as mapping object
            meth=eval(method)
                       
    #        try:
            #Try and match template in imageB 
            try:
                resz = cv2.matchTemplate(search, template, meth)
            except:
                print('Matching error')
                resz=None
            
            if resz.all() is not None:
                
                #Perform subpixel analysis if flag is True
                if supersample is not None:
                                        
                    #Create XY grid for correlation result 
                    resx = np.arange(0, resz.shape[1], 1)
                    resy = np.arange(0, resz.shape[0], 1)                    
                    resx,resy = np.meshgrid(resx, resy, sparse=True)
                                                            
                    #Create bicubic interpolation grid                                                                            
                    interp = interpolate.interp2d(resx, resy, resz, 
                                                  kind='cubic')                    
                    
                    #Create new XY grid to interpolate across
                    newx = np.arange(0, resz.shape[1], supersample)
                    newy = np.arange(0, resz.shape[0], supersample)
                            
                    #Interpolate 
                    resz = interp(newx, newy)
                                                            
                    #Get correlation values and coordinate locations        
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resz)
                                                                                                
                    #If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take min
                    if method == 'cv2.TM_SQDIFF':                            
                        location = min_loc
                        valm = min_val
                    elif method == 'cv2.TM_SQDIFF_NORMED':
                        location = min_loc
                        valm = min_val
                        
                    #Else, take maximum correlation and location
                    else:                 
                        location = max_loc
                        valm = max_val
                                        
                    #Calculate tracked point location                    
                    loc_x = ((u - ((resz.shape[1]*supersample)/2)) + 
                            (location[0]*supersample))
                    loc_y = ((v - ((resz.shape[1]*supersample)/2) + 
                            (location[1]*supersample)))                            
                    #ASSUMPTION: the origin of the template window is the same 
                    #as the origin of the correlation array (i.e. resz)                        
                        
                #If supersampling has not be specified
                else:
                    #Get correlation values and coordinate locations        
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resz)
                                                                            
                    #If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take min
                    if method == 'cv2.TM_SQDIFF':                            
                        location = min_loc
                        valm = min_val
                    elif method == 'cv2.TM_SQDIFF_NORMED':
                        location = min_loc
                        valm = min_val
                    else:
                        location = max_loc
                        valm = max_val                         
        
                    #Calculate tracked point location
                    loc_x = (u - (resz.shape[1]/2)) + (location[0])
                    loc_y = (v - (resz.shape[1]/2)) + (location[1])
        
                #Retain correlation and location            
                maxcorr.append(valm)
                avercorr.append(np.mean(resz))
                pu2.append(loc_x)
                pv2.append(loc_y)
                
                #Revise success count
                success = success+1                 
               
        fig, (ax1) = plt.subplots(1,1)         
        ax1.imshow(im0, origin='upper', cmap='gray')
        ax1.scatter(uv0[:,0], uv0[:,1], color='red')
        ax1.scatter(pu2, pv2, color='blue')
        plt.show()
        
        print('\nTemplate matching completed')
        print(str(success) + ' templates successfully matched')
       
    else:                     
        #Create empty lists for matching results
        error = []
        pu2 = []
        pv2 = []
        
        #Create count for successful and failed tracking attempts
        success = 0
        fail = 0
        
    
        #Optical Flow set-up parameters
        lk_params = dict( winSize  = (searchsize,searchsize),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | 
                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))    
                        
        #Rearrange pu, pv as 3d array in preparation for tracking 
        puv = [] 
        for u,v in zip(uv0[:,0], uv0[:,1]):
            puv.append(u)
            puv.append(v)
                
        puv = np.array(puv,dtype='float32').reshape((-1,1,2))
        
        #Track forward from Scene A to Scene B
        uv1, s1, err1  = cv2.calcOpticalFlowPyrLK(im0, im1, puv, None, 
                                                  **lk_params)     
    
        #Track backwards from Scene B to Scene A
        uv0r, s0, err0  = cv2.calcOpticalFlowPyrLK(im1, im0, uv1, None, 
                                                  **lk_params) 
    
        #Loop through track index    
        for i in range(len(s1)):
    
            #If tracking was successful            
            if s1[i][0] == 1:
                            
                #Find difference between the two points from Scene A                        
                diff=(puv[i]-uv0r[i])*(puv[i]-uv0r[i])
                diff=np.sqrt(diff[:,0]+diff[:,1])           
                
                #Filter by distance between the two point from Scene A
                if diff < threshold:
                    error.append(diff[0])
                    pu2.append(uv1[i][0][0])
                    pv2.append(uv1[i][0][1])
                    success = success+1
                
                #Else, append None
                else:                               
                    error.append(None)
                    pu2.append(None)
                    pv2.append(None)                
                    fail = fail+1
    
            #Else, append None
            else:
                error.append(None)
                pu2.append(None)
                pv2.append(None)                
                fail = fail+1 
                                                                                 
        print('\nTemplate matching completed')
        print(str(success) + ' templates successfully matched')
        print(str(fail) + ' templates failed to match')

        fig, (ax1) = plt.subplots(1,1)         
        ax1.imshow(im0, origin='upper', cmap='gray')
        ax1.scatter(uv0[:,0], uv0[:,1], color='red')
        ax1.scatter(pu2, pv2, color='blue')
        plt.show()
           
#        #Restructure results into 2d arrays        
#        error = np.array(error).reshape(pu.shape[0], pu.shape[1])
#        pu2 = np.array(pu2).reshape(pu.shape[0], pu.shape[1])
#        pv2 = np.array(pv2).reshape(pu.shape[0], pu.shape[1])
    


#    vl = calcDenseVelocity(im0, im1, vmask, [matrix,distort], [hg[0],hg[3]], 
#                           invprojvars, vwin, vback, vminfeat, [vmax, vqual, 
#                           vmindist])                                                                                                                     

                       

#---------------------------  Export data   -----------------------------------

#print('\nWRITING DATA TO FILE')
#
##Get all image names
#names=[]
#for i in imagelist:
#    names.append(Path(i).name)
#
##Extract xyz velocities, uv velocities, and xyz0 locations
#xyzvel=[item[0][0] for item in velo] 
#xyzerr=[item[0][3] for item in velo]
#uvvel=[item[1][0] for item in velo]
#xyz0=[item[0][1] for item in velo]
#
##Write out velocity data to .csv file
#FileHandler.writeVeloFile(xyzvel, uvvel, homog, names, target1) 
#
##Write homography data to .csv file
#FileHandler.writeHomogFile(homog, names, target2)
#
##Write points to shp file                
#FileHandler.writeVeloSHP(xyzvel, xyzerr, xyz0, names, target3, projection)       
#
#
##----------------------------   Plot Results   --------------------------------
#
#print('\nPLOTTING OUTPUTS')
#
##Extract uv0, uv1corr, xyz0 and xyz1 locations 
#uv0=[item[1][1] for item in velo]
#uv1corr=[item[1][3] for item in velo]
#uverr=[item[1][4] for item in velo]
#xyz0=[item[0][1] for item in velo]
#xyz1=[item[0][2] for item in velo]
#
#
##Cycle through data from image pairs   
#for i in range(len(xyz0)):
#    
#    #Get image from sequence
#    im=FileHandler.readImg(imagelist[i], band, equal)
#
#    #Correct image for distortion
#    newMat, roi = cv2.getOptimalNewCameraMatrix(matrix, distort, 
#                                                (im.shape[1],im.shape[0]), 
#                                                1, (im.shape[1],im.shape[0])) 
#    im = cv2.undistort(im1, matrix, distort, newCameraMatrix=newMat)
#    
#    #Get image name
#    imn = Path(imagelist[i]).name
#    print('Visualising data for ' + str(imn))
#        
#    #Plot uv velocity points on image plane  
#    Utilities.plotVeloPX(uvvel[i], uv0[i], uv1corr[i], im, show=True, 
#                         save=target4+'uv_'+imn)
#
##    Utilities.plotVeloPX(uverr[i], uv0[i], uv1corr[i], im, show=True, 
##                         save=target4+'uverr_'+imn)
##    
##    uvsnr=uverr[i]/uvvel[i]
##    Utilities.plotVeloPX(uvsnr, uv0[i], uv1corr[i], im, show=True, 
##                         save=target4+'uvsnr_'+imn)    
#
#
#    #Plot xyz velocity points on dem  
#    Utilities.plotVeloXYZ(xyzvel[i], xyz0[i], xyz1[i], dem, show=True, 
#                          save=target4+'xyz_'+imn)
#
##    Utilities.plotVeloXYZ(xyzerr[i], xyz0[i], xyz1[i], dem, show=True, 
##                          save=target4+'xyzerr_'+imn)    
#                
##    #Plot interpolation map
##    grid, pointsextent = Utilities.interpolateHelper(xyzvel[i], xyz0[i], 
##                                                     xyz1[i], interpmethod)
##    Utilities.plotInterpolate(grid, pointsextent, dem, show=True, 
##                              save=target4+'interp_'+imn)  
#
#    
#------------------------------------------------------------------------------
print('\nFinished')