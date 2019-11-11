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

#Import PyTrx packages
sys.path.append('../')
from CamEnv import setProjection, projectXYZ, projectUV
from Velocity import calcDenseVelocity, calcHomography, seedGrid, createMaskFromImg
from DEM import load_DEM, voxelviewshed, ExplicitRaster, createMaskFromImg
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
vmaskPath = '../Examples/camenv_data/masks/KR2_2014_demmask.jpg'       
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




##Plot dem mask manually
demz = dem.getZ()

#fig=plt.gcf()
#fig.canvas.set_window_title('Click to create mask. Press enter to record' 
#                            ' points.')
#imgplot = plt.imshow(im1, origin='upper')
#imgplot.set_cmap('gray')
#maskuv = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, 
#                mouse_stop=2)
#print('\n' + str(len(maskuv)) + ' points seeded')
#plt.show()
#plt.close()
#
##Close shape
#maskuv.append(maskuv[0])
#
##Generate polygon
#ring = ogr.Geometry(ogr.wkbLinearRing)
#for p in maskuv:
#    ring.AddPoint(p[0],p[1])
#p=maskuv[0]
#ring.AddPoint(p[0],p[1])    
#poly = ogr.Geometry(ogr.wkbPolygon)
#poly.AddGeometry(ring)
# 
##Rasterize polygon using PIL
#height = dem.getRows()
#width = dem.getCols()
#img1 = Image.new('L', (width,height), 0)
#draw=ImageDraw.Draw(img1)
#draw.polygon(maskuv, outline=1, fill=1)
#myMask=np.array(img1)
#
##Write to .jpg file    
#img1.save(vmaskPath, quality=75)
#
#maskuv = np.array(maskuv).reshape(-1,2)
#maskxyz = projectUV(maskuv, invprojvars)
#
##Generate polygon
#ring = ogr.Geometry(ogr.wkbLinearRing)
#for p in maskxyz:
#    ring.AddPoint(p[0],p[1])
#p=maskxyz[0]
#ring.AddPoint(p[0],p[1])    
#poly = ogr.Geometry(ogr.wkbPolygon)
#poly.AddGeometry(ring)
# 
##Rasterize polygon using PIL
#height = dem.getRows()
#width = dem.getCols()
#img1 = Image.new('L', (width,height), 0)
#draw=ImageDraw.Draw(img1)
#draw.polygon(maskxyz, outline=1, fill=1)
#myMask=np.array(img1)
#
##Plot image points    
#fig, (ax1, ax2) = plt.subplots(1,2)
#
#ax1.imshow(im1, cmap='gray')
#ax1.scatter(maskuv[:,0], maskuv[:,1], color='red')
#
#lims = dem.getExtent() 
#ax2.locator_params(axis = 'x', nbins=8)
#ax2.axis([lims[0],lims[1],lims[2],lims[3]])
#ax2.imshow(demz, origin='lower', 
#           extent=[lims[0],lims[1],lims[2],lims[3]], cmap='gray')
#ax2.scatter(maskxyz[:,0], maskxyz[:,1], color='red')
#
#plt.show()

imgmask, demmask = createMaskFromImg(dem, im1, invprojvars, imMaskPath=None, 
                                     demMaskPath=None)



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

    demx = dem.getData(0)    
    demx_uniq = demx[0,:]
    demx_uniq = demx_uniq.reshape(demx_uniq.shape[0],-1)
    
    demy = dem.getData(1)
    demy_uniq = demy[:,0]    
    demy_uniq = demy_uniq.reshape(demy_uniq.shape[0],-1)
    

    mz = ma.masked_array(demz, np.logical_not(demmask))
    mz = mz.filled(np.nan) 
    
    griddistance=[500,500]
    
    #Define grid as empty list    
    gridxyz=[]
    
    #Get DEM extent
    extent = dem.getExtent()
    
    #Define point spacings in dem space
    samplex = round((extent[1]-extent[0])/griddistance[0])
    sampley = round((extent[3]-extent[2])/griddistance[1])
    
    #Define grid in dem space
    linx = np.linspace(extent[0], extent[1], samplex)
    liny = np.linspace(extent[2], extent[3], sampley)
    
    #Create mesh
    meshx, meshy = np.meshgrid(linx, liny) 
    
    #Get Z values for mesh grid
    meshx2 = []
    meshy2 = []
    meshz2 = []
    
    for a,b in zip(meshx.flatten(), meshy.flatten()):

        both1 = (np.abs(demx_uniq-a)).argmin()
        both2 = (np.abs(demy_uniq-b)).argmin()
#        both1 = np.where(demx_uniq==a)
#        both2 = np.where(demy_uniq==b)
        
#        print('Found in x: ' + str(both1))
#        print('Found in y: ' + str(both2))

        if np.isnan(mz[both2,both1]) == False:
            np.array([a,b,mz[both2,both1]])
            meshx2.append(a)
            meshy2.append(b)
            meshz2.append(mz[both2,both1])
            print('Z value at ' + str(both1) + ' and ' + str(both2) + ': ' + str(mz[both2,both1]))

#        except:
#            pass

      
    
    
#    meshxy = np.array([meshx, meshy])
#    meshxmsk = ma.masked_array(meshx, np.logical_not(myMask))     
#    meshymsk = ma.masked_array(meshy, np.logical_not(myMask))  
    
#    gridx,gridy = seedGrid(dem, None, [500,500], 4, [camloc, campose, radcorr, tancorr, focal, camcen, refimagePath])
    
#    X = gridx.flatten()
#    Y = gridy.flatten()
#
#    demx = dem.getData(0)
#    demxflat = demx.flatten()
#    demy = dem.getData(1)
#    demyflat = demy.flatten()
#    demz = dem.getZ()
#    demzflat = demz.flatten()
#    
#    Z=[]
#    for a,b in zip(X,Y):
#
#        print(a)
#        print(b)      
#        
##        both1 = np.where(demxflat==a)          
##        both2 = np.where(demyflat==b)
#
#        both1 = (np.abs(demx-a)).argmin()
#        both2 = (np.abs(demy-b)).argmin()
#    
#        print(both1)
#        print(both2)
#
#        indx = np.intersect1d(both1, both2)
#        indx = int(indx)
#        
#        print(indx)
##        idxa = (np.abs(demx - a)).argmin()
##        idxb = (np.abs(demy - b)).argmin()
##        print(idxa)
##        print(idxb)
#        Z.append(demzflat[indx])
#        
    xyz=np.column_stack([meshx2,meshy2,meshz2])

    uv,depth,inframe = projectXYZ(camloc, campose, radcorr, tancorr, focal, camcen, im1, xyz)        

    #Plot image points    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    lims = dem.getExtent()
    ax1.axis([lims[0],lims[1],lims[2],lims[3]])    
    ax1.imshow(demz, origin='lower',
               extent=[lims[0],lims[1],lims[2],lims[3]], cmap='gray')
    ax1.scatter(meshx.flatten(), meshy.flatten(), color='red')
     
    ax2.locator_params(axis = 'x', nbins=8)
    ax2.axis([lims[0],lims[1],lims[2],lims[3]])
    ax2.imshow(mz, origin='lower', 
               extent=[lims[0],lims[1],lims[2],lims[3]], cmap='gray')
    ax2.scatter(meshx2, meshy2, color='red')

    ax3.imshow(im1, cmap='gray')
    ax3.scatter(uv[:,0], uv[:,1], color='red')
    plt.show()
    
#    #Create empty numpy array
#    griduv=np.zeros([475,2])
#    griduv[::]=float('NaN')
    
#    #Get XYZ real world coordinates and corresponding uv coordinates
#    X=invprojvars[0]
#    Y=invprojvars[1]
#    Z=invprojvars[2]
#    uv0=invprojvars[3]
#    
#    XYZ = np.column_stack([X,Y,Z])
#    uv0_u = uv0[:,0]
#    uv0_v = uv0[:,1]
    
#    #Snap uv and xyz grids together
#    u=interpolate.griddata(XYZ, uv0_u, (gridx, gridy), method='linear')
#    v=interpolate.griddata(XYZ, uv0_v, (gridx, gridy), method='linear')

        
#    print(s)
#    print(type(s))

#    vl = calcDenseVelocity(im0, im1, vmask, [matrix,distort], [hg[0],hg[3]], 
#                           invprojvars, vwin, vback, vminfeat, [vmax, vqual, 
#                           vmindist])                                                                                                                     
#    
#    #Append velocity and homography information
#    velo.append(vl)
#    homog.append(hg)
                       

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