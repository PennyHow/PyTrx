'''
PYTRX EXAMPLE VELOCITY DRIVER (EXTENDED VERSION)

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

This script is an extended breakdown of 'driver_velocity.py'. The functions 
have been expanded out to show what PyTrx is doing step-by-step. Functions that 
have been expanded in this script are:
    Camera calibration
    Plotting GCPs, principal point and camera calibration
    Velocity feature-tracking
    Homography calculation
    Writing raw output to file (.csv)
    Writing output to shapefile (.shp)
This breakdown script has been included in order to provide the user with a 
more detailed overview of PyTrx's functionality. It also allows flexible
intervention and adaptation where needed. 

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton 
         Lynne Buie
'''

#Import packages
import sys
import os
from osgeo import ogr, osr
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv
from Measure import Velocity
from Utilities import arrowplot


#------------------------   Define input parameters   -------------------------

print '\nDEFINING DATA INPUTS'

#Camera name, location (XYZ) and pose (yaw, pitch, roll)
camname = 'KR2_2014'
camloc = np.array([447948.820, 8759457.100, 407.092])
campose = np.array([4.80926, 0.05768, 0.14914]) 

#Define image folder and image file type 
imgFiles = '../Examples/images/KR2_2014_subset/*.JPG'

#Define calibration images and chessboard dimensions
#Dimensions are number of chessboard corners for chessboard height and width
calibPath = '../Examples/camenv_data/calib/KR2_calibimgs/*.JPG'
chessboard = [6, 9]

#Define GCPs (world coordinates and corresponding image coordinates)
GCPpath = '../Examples/camenv_data/gcps/KR2_2014.txt'

#Define reference image (where GCPs have been defined)
imagePath = '../Examples/camenv_data/refimages/KR2_2014.JPG'

#Load DEM from path
DEMpath = '../Examples/camenv_data/dem/KR_demsmooth.tif'        

#Densify DEM 
DEMdensify = 2

#Define point seeding mask files 
#Mask is automatically generated if file not found
#No mask generated if input is None
camvmask = '../Examples/camenv_data/masks/KR2_2014_vmask.JPG'       #Velocity
caminvmask = '../Examples/camenv_data/invmasks/KR2_2014_inv.JPG'    #Homography

#Set velocity feature-tracking parameters
verr = True                     #Calculate tracking error?
vback = 1.0                     #Back-tracking threshold  
vmax = 50000                    #Maximum number of points to seed
vqual = 0.1                     #Corner quality for seeding
vmindist = 5.0                  #Minimum distance between seeded points
vminfeat = 4                    #Minimum number of seeded points to track

                            
#Set homography feature-tracking parameters
hmethod = cv2.RANSAC            #Homography calculation method 
                                #(cv2.RANSAC, cv2.LEAST_MEDIAN, or 0)
hreproj = 5.0                   #Maximum allowed reprojection error
hback = 1.0                     #Back-tracking threshold
herr = True                     #Calculate tracking error?
hmax = 50000                    #Maximum number of points to seed
hqual = 0.1                     #Corner quality for seeding
hmindist = 5.0                  #Minimum distance between seeded points
homogerr = True                 #Calculate homography matrix error?
hminfeat = 4                    #Minimum number of seeded points to track

#------------------------   Calibrate camera   --------------------------------

print '\nCALIBRATING CAMERA'

#Define shape of chessboard array
objp = np.zeros((chessboard[0]*chessboard[1],3), np.float32)           
objp[:,:2] = np.mgrid[0:chessboard[1],0:chessboard[0]].T.reshape(-1,2) 
    
#Array to store object pts and img pts from all images
objpoints = []                                   
imgpoints = []                                   
    
#Define location of calibration photos
imgs = glob.glob(calibPath)

#Set image counter for loop
imageCount = 0

#Loop to determine if each image contains a chessboard pattern and 
#store corner values if it does
for fname in imgs:
    
    #Read file as an image using OpenCV
    img = cv2.imread(fname)   

    #Change RGB values to grayscale             
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    
    #Find chessboard corners in image
    patternFound, corners = cv2.findChessboardCorners(gray,
                                                      (chessboard[1],
                                                       chessboard[0]),
                                                      None)
    
    #Cycle through images, print if chessboard corners have been found 
    #for each image
    imageCount += 1
    print str(imageCount) + ': ' + str(patternFound) + ' ' + fname
    
    #If found, append object points to objp array
    if patternFound == True:
        objpoints.append(objp)
        
        #Determine chessboard corners to subpixel accuracy
        #Inputs: winSize specified 11x11, zeroZone is nothing (-1,-1), 
        #opencv criteria
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,
                         30,0.001))
                         
        imgpoints.append(corners)
        
        #Draw and display corners
        cv2.drawChessboardCorners(img,(chessboard[1],chessboard[0]),corners,
                                  patternFound)

#Calculate initial camera matrix and distortion
err,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,
                                               imgpoints,
                                               gray.shape[::-1],
                                               None,
                                               5)
#Retain principal point coordinates
pp = [mtx[0][2],mtx[1][2]]

#Optimise camera matrix and distortion using fixed principal point
err,mtxcv2,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,
                                                  imgpoints,
                                                  gray.shape[::-1],
                                                  mtx,
                                                  5,
                                                  flags=cv2.CALIB_FIX_PRINCIPAL_POINT)

#Change matrix structure for compatibility
mtx = np.array([mtxcv2[0][0],mtxcv2[0][1],0,
               0,mtxcv2[1][1],0,
               pp[0],pp[1],1]).reshape(3,3)
                       
#Restructure distortion parameters for compatibility with PyTrx
rad = np.array([dist[0],dist[1],dist[4], 0.0, 0.0, 0.0]).reshape(6)
tan = np.array(dist[2:4]).reshape(2)


#-----------------------   Create PyTrx objects   -----------------------------

print '\nINITIALISING OBJECTS'
            
#Define camera environment
camenv=CamEnv([camname,                     #Camera name
              GCPpath,                      #GCP file path
              DEMpath,                      #DEM file path
              imagePath,                    #Reference image file path
              (mtx,tan,rad),                #Calibration matrix and distortion
              camloc,                       #Camera location XYZ
              campose,                      #Camera pose YPR
              DEMdensify])                  #DEM densification factor


#Set up Velocity object
velo=Velocity(imgFiles,                     #Path to image files
              camenv,                       #Camera environment object
              camvmask,                     #Mask for velocity tracking
              caminvmask,                   #Mask for homography tracking
              image0=0,                     #Image start (i.e. image 0)
              band='L',                     #Image band extraction (L,R,G,B)
              equal=True,                   #Histogram equalisation for imgs?
              quiet=2)                      #Level of commentary (0-3)



#----------------   Plot camera environment parameters   ----------------------
           
#Plot GCPs in image plane and DEM scene
print '\nPLOTTING GROUND CONTROL POINTS (GCPs)'

#Get reference image dimension 
h = camenv._refImage.getImageSize()[0]
w = camenv._refImage.getImageSize()[1]

#Get GCPs from camera environment object    
worldgcp, imgcp = camenv._gcp.getGCPs()
        
#Prepare DEM for plotting
demobj=camenv.getDEM()
demextent=demobj.getExtent()
dem=demobj.getZ()
                
#Initialise subplot window   
fig, (ax1,ax2) = plt.subplots(1,2)
fig.canvas.set_window_title('GCP locations of KR2_2014')

#Plot GCPs in image plane
ax1.axis([0,w,h,0])  
ax1.imshow(camenv._refImage.getImageArray(), origin='lower', cmap='gray')
ax1.scatter(imgcp[:,0], imgcp[:,1], color='red')
        
#Plot GCPs in DEM scene
ax2.locator_params(axis = 'x', nbins=8)
ax2.axis([demextent[0],demextent[1],demextent[2],demextent[3]])
ax2.imshow(dem, origin='lower', extent=demextent, cmap='gray')
ax2.scatter(worldgcp[:,0], worldgcp[:,1], color='red')
ax2.scatter(camloc[0], camloc[1], color='blue')

#Show plot
plt.show()
        
 
#Plot image with principle point
print '\nPLOTTING PRINCIPLE POINT'

#Initialise plotting environment
fig, (ax1) = plt.subplots(1)
fig.canvas.set_window_title('Principal Point of KR2_2014')

#Set image axes to image dimensions
ax1.axis([0,w,h,0])        

#Plot gray image
ax1.imshow(camenv._refImage.getImageArray(), 
           origin='lower', cmap='gray')

#Overlay principal point and marker lines
ax1.scatter(pp[0], pp[1], color='yellow', s=100)
ax1.axhline(y=pp[1])
ax1.axvline(x=pp[0])

#Show plot
plt.show()


#Plot calibration (uncorrected and corrected image)
print '\nPLOTTING CALIBRATION'

#Calculate optimal camera matrix
newMat, roi = cv2.getOptimalNewCameraMatrix(mtxcv2, dist, (w,h), 1, 
                                            (w,h))

#Correct reference image for distortion                                                
corr_image = cv2.undistort(camenv._refImage.getImageArray(), mtxcv2, 
                           dist, newCameraMatrix=newMat)
   
#Initialise subplot window                        
fig, (ax1,ax2) = plt.subplots(1,2)
fig.canvas.set_window_title('Calibration output of KR2_2014')

#Plot uncorrected image
ax1.imshow(camenv._refImage.getImageArray(), cmap='gray')    
ax1.axis([0,w,h,0])

#Plot corrected image
ax2.imshow(corr_image, cmap='gray')            
ax2.axis([0,w,h,0])  

#Show plot  
plt.show()


#----------------------   Calculate velocities   ------------------------------

print '\nCALCULATING VELOCITIES'                               

#Create object attributes
xyzvel = []                                     
xyzhomog = []

#Get first image (image0) file path and array data for initial tracking
imn1=velo._imageSet[0].getImagePath().split('\\')[1]
im1=velo._imageSet[0].getImageArray()

#Cycle through image pairs (numbered from 0)
for i in range(velo.getLength()-1):

    #Re-assign first image in image pair
    im0=im1
    imn0=imn1
                    
    #Get second image in image pair (and subsequently clear memory)
    im1=velo._imageSet[i+1].getImageArray()
    imn1=velo._imageSet[i+1].getImagePath().split('\\')[1]       
    velo._imageSet[i].clearAll()
   
    print '\nFeature-tracking for images: ', imn0,' and ', imn1
        
    #Calculate homography between image pair. Output is as follows: 
    #[[homography matrix], [pts0, pts1, ptscorrected], [pts error], [homography error]]
    homog = velo._calcHomography(im0, im1, 
                                 method=hmethod, 
                                 ransacReprojThreshold=hreproj, 
                                 back_thresh=hback, 
                                 calcErrors=herr, 
                                 maxpoints=hmax, 
                                 quality=hqual, 
                                 mindist=hmindist, 
                                 calcHomogError=homogerr, 
                                 min_features=hminfeat) 
                                 
    #Calculate velocities between image pair
    #Output is as follows: [[xyzvel, xyz0, xyz1],[pxvel,uv0,uv1,uv1corrected]]
    vel = velo.calcVelocity(im0, im1, 
                            homog[0], 
                            homog[3],
                            back_thresh=vback, 
                            calcErrors=verr, 
                            maxpoints=vmax, 
                            quality=vqual, 
                            mindist=vmindist, 
                            min_features=vminfeat)                       
                                                                                

    #Append velocity and homography information
    xyzvel.append(vel)
    xyzhomog.append(homog)
                       
        
#---------------------------  Export data   -----------------------------------

#Define data output directory
destination = '../Examples/results/velocity_extended/'
if not os.path.exists(destination):
    os.makedirs(destination)
            

print '\nEXPORTING RAW VELOCITY DATA'

#Write out velocity data to .csv file
target2 = destination + 'velo_output.csv'
f=open(target2,'w')

#Write active directory to file
dirname = velo._imageSet[0].getImagePath().split('\\')[0]
f.write(dirname + '\n')

#Define column headers
header=('Image 0, Image 1, Average xyz velocity, Features tracked , '
        'Average px velocity, Homography RMS Error, SNR')    
f.write(header + '\n')

#Get name of first image in sequence
fn1 = velo._imageSet[0].getImagePath().split('\\')[1]

#Iterate through timeLapse object
for i in range(velo.getLength()-1):
    
    #Re-define image0 for each iteration
    fn0=fn1
        
    #Write image file names to file        
    fn1 = velo._imageSet[i+1].getImagePath().split('\\')[1]
    out=fn0+','+fn1
        
    #Get velocity data                     
    xyz = xyzvel[i][0][0]
    uv = xyzvel[i][1][0]

    #Calculate average unfiltered velocity
    xyzvelav = sum(xyz)/len(xyz)
    pxvelav= sum(uv)/len(uv)
                   
#    #Determine number of features (unfiltered) tracked
#    numtrack = len(xyz)

    #Write unfiltered velocity information
    f.write(out + ',' +str(xyzvelav) + ',' +str(len(xyz)) + ',')
                          
    #Get homography information                         
    hpt0 = xyzhomog[i][1][0]            #Seeded pts in im0
    hpt1 = xyzhomog[i][1][1]            #Tracked pts in im1
    hpt1corr = xyzhomog[i][1][2]        #Corrected pts in im1
    herr = xyzhomog[i][3]               #Homography error

    #Get xyz homography errors                      
    xd=herr[1][0]
    yd=herr[1][1]
    
    #Get uv point positions                
    psx=hpt0[:,0,0]
    psy=hpt0[:,0,1]
    
    if hpt1corr is not None:
        pfx=hpt1corr[:,0,0]
        pfy=hpt1corr[:,0,1] 
    else:                   
        pfx=hpt1[:,0,0]
        pfy=hpt1[:,0,1]    
        
    #Determine uv point position difference
    pdx=pfx-psx
    pdy=pfy-psy
    
    #Calculate homography and homography error            
    homogdist=np.sqrt(pdx*pdx+pdy*pdy)
    errdist=np.sqrt(xd*xd+yd*yd)
    
    #Calculate mean homography and mean error 
    meanerrdist=np.mean(errdist)
    
    #Calculate SNR between pixel velocities and error
    snr=meanerrdist/pxvelav
    
    #Write pixel velocity and homography information
    f.write((str(pxvelav) + ',' + str(meanerrdist) + ','  +
        str(snr)))
                     
    #Break line in output file
    f.write('\n')
        

print '\nEXPORTING RAW HOMOGRAPHY DATA'

#Initialise file writing
target3 = destination + 'homography.csv'
f=open(target3,'w')

#Write active directory to file
dirname = velo._imageSet[0].getImagePath().split('\\')[0]
f.write(dirname + '\n')

#Define column headers
header=('Image 0, Image 1,"Homography Matrix[0,0]","[0,1]","[0,2]",'
        '"[1,0]","[1,1]","[1,2]","[2,0]","[2,1]","[2,2]",Features Tracked,'
        'xmean,ymean,xsd,ysd,"Mean Error Magnitude",'
        '"Mean Homographic displacement","Homography SNR"')    
f.write(header+'\n')

#Get name of first image in sequence
fn1 = velo._imageSet[0].getImagePath().split('\\')[1]

#Iterate through timeLapse object
for i in range(velo.getLength()-1):
    
    #Re-define image0 for each iteration
    fn0=fn1

    #Get name of image1       
    fn1 = velo._imageSet[i+1].getImagePath().split('\\')[1]
    out=fn0+','+fn1
    
    #Get homography information from velocity object
    hmatrix = xyzhomog[i][0]                            #Homography matrix
    hpt0 = xyzhomog[i][1][0]                            #Seeded pts in im0
    hpt1 = xyzhomog[i][1][1]                            #Tracked pts in im1
    hpt1corr = xyzhomog[i][1][2]                        #Corrected pts im1
    herr = xyzhomog[i][3]                               #Homography error    
    
    #Get xyz homography errors
    xd = herr[1][0]
    yd = herr[1][1] 
    
    #Get uv point positions
    psx = hpt0[:,0,0]
    psy = hpt0[:,0,1]
    
    if hpt1corr is None:
        pfx = hpt1[:,0,0]
        pfy = hpt1[:,0,1]
    else:
        pfx = hpt1corr[:,0,0]
        pfy = hpt1corr[:,0,1]
    
    #Determine uv point position difference
    pdx=pfx-psx
    pdy=pfy-psy
    
    #Calculate signal-to-noise ratio            
    errdist=np.sqrt(xd*xd+yd*yd)
    homogdist=np.sqrt(pdx*pdx+pdy*pdy)
    sn=errdist/homogdist
    
    #Calculate mean homography, mean error and mean SNR 
    meanerrdist=np.mean(errdist)
    meanhomogdist=np.mean(homogdist)
    meansn=np.mean(sn)
    
    #Define output homography matrix
    if hmatrix is not None:
        hmatrix.shape=(9)
        for val in hmatrix:
            out=out+','+str(val)
    
    #Determine number of points tracked in homography calculations
    tracked=len(hpt0)
    out=out+','+str(tracked)
    
    #Define output homography matrix errors
    for val in herr[0]:
        out=out+','+str(val)
    
    #Compile all data for output file
    out = (out+','+str(meanerrdist)+','+str(meanhomogdist)+','
           +str(meansn))
    
    #Write to output file
    f.write(out+'\n') 
 

#-------------------   Generate velocity shape files   ------------------------

print '\nWRITING VELOCITIES TO SHAPE FILE'   

#Write points to shp file
target4 = destination + 'shpfiles/'     #Define file destination
if not os.path.exists(target4):
    os.makedirs(target4)                #Create file destination
projection = 32633                      #ESPG:32633 is projection WGS84

        
#Get driver and create shapeData in shp file directory        
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)
    pass
    

#Cycle through velocities
for i in range(velo.getLength()-1): 
    
    #Get velocity, pt and image name for time step
    vel = xyzvel[i][0][0]    
    pt0 = xyzvel[i][0][1]
    imn = velo._imageSet[i].getImagePath().split('\\')[1] 
    
    #Create file space            
    shp = target4 + str(imn) + '_vel.shp'
    if os.path.exists(shp):
        print '\nDeleting pre-existing datasource'
        driver.DeleteDataSource(shp)
    ds = driver.CreateDataSource(shp)
    if ds is None:
        print 'Could not create file %s' %shp
    
    #Set projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(projection)
    layer = ds.CreateLayer(' ', proj, ogr.wkbPoint)
   
    #Add attributes to layer (ID and velocity)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))        
    layer.CreateField(ogr.FieldDefn('velocity', ogr.OFTReal))  
    
    #Get xy coordinates
    x0 = pt0[:,0]
    y0 = pt0[:,1]
    
    #Create point features with data attributes in layer           
    for v,x,y in zip(vel, x0, y0):
        count=1
    
        #Create feature    
        feature = ogr.Feature(layer.GetLayerDefn())
    
        #Create feature attributes    
        feature.SetField('id', count)
        feature.SetField('velocity', v)
    
        #Create feature location
        wkt = "POINT(%f %f)" %  (float(x) , float(y))
        point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
    
        #Free up data space
        feature.Destroy()                       
        count=count+1

    #Free up data space                          
    ds.Destroy()
            

#----------------------------   Plot Results   --------------------------------

print '\n\nPLOTTING DATA'

#Set interpolation method ("nearest"/"cubic"/"linear")
method='linear' 

#Set DEM extent         
cr1 = [445000, 452000, 8754000, 8760000]            

#Set destination for file outputs
target5 = destination + 'imgfiles/'
if not os.path.exists(target5):
    os.makedirs(target5)
 
#Cycle through data from image pairs   
for i in range(velo.getLength()-1):
 
    #Get image name
    imn=velo._imageSet[i].getImagePath().split('\\')[1] 
    
          
    #Create plotting window for image scene
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': UV output')
    
    #Get image corrected for distortion
    img=velo._imageSet[i].getImageArray()        
    
    #Plot image
    ax1.imshow(img, cmap='gray')            
    ax1.axis([0, velo._imageSet[i].getImageSize()[1],
              velo._imageSet[i].getImageSize()[0], 0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Get UV velocities and pt0
    velocity = xyzvel[i][1][0]         #Get UV velocities
    pt0 = xyzvel[i][1][1]                #Get UV point positions from im0
    
    #Get point positions from im1
    if xyzvel[i][1][3] is None:           
        pt1 = xyzvel[i][1][2]        #Get uncorrected pts if corrected is None
    else:
        pt1 = xyzvel[i][1][3]        
            
    pt0x=pt0[:,0,0]                     #pt0 x values
    pt0y=pt0[:,0,1]                     #pt0 y values
    pt1x=pt1[:,0,0]                     #pt1 x values
    pt1y=pt1[:,0,1]                     #pt1 y values
    
    #Plot xy positions onto images
    uvplt = ax1.scatter(pt0x, pt0y, c=velocity, s=50, vmin=0,
                        vmax=max(velocity), cmap=plt.get_cmap("gist_ncar"))
    plt.colorbar(uvplt, ax=ax1)

    #Plot arrows
    xar,yar=arrowplot(pt0x, pt0y, pt1x, pt1y, scale=5.0,headangle=15)
    ax1.plot(xar,yar,color='black')
     
    #Save figure
    plt.savefig(target5 + 'uv_' + imn, dpi=300) 
    
  
    #Set-up XYZ velocity plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': XYZ output')
    ax1.set_xticks([])
    ax1.set_yticks([])
        
    #Prepare DEM if desired
    demobj=velo._camEnv.getDEM()
    demextent=demobj.getExtent()
    dem=demobj.getZ()           
    
    #Plot DEM and set cmap
    implot = ax1.imshow(dem, origin='lower', cmap='gray', extent=demextent)
    ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
        
    #Plot camera location
    ax1.scatter(camloc[0], camloc[1], c='g')
        
    #Get xyz points and velocities   
    xyzvelo = xyzvel[i][0][0]                  #xyz velocity
    xyzstart = xyzvel[i][0][1]                 #xyz pt0 position
    xyzend = xyzvel[i][0][2]                   #xyz pt1 position
       
    #Get xyz positions from image0 and image1
    xyz_xs = xyzstart[:,0]                      #pt0 x values
    xyz_xe = xyzend[:,0]                        #pt1 x values
    xyz_ys = xyzstart[:,1]                      #pt0 y values
    xyz_ye = xyzend[:,1]                        #pt1 y values
                              
    #Scatter plot velocity                 
    xyzplt = ax1.scatter(xyz_xs, xyz_ys, c=xyzvelo, s=50, 
                         cmap=plt.get_cmap('gist_ncar'), 
                         vmin=0, vmax=max(xyzvelo))  

    #Plot vector arrows denoting direction                             
    xar,yar=arrowplot(xyz_xs,xyz_ys,xyz_xe,xyz_ye,scale=5.0,headangle=15)
    ax1.plot(xar,yar,color='black')

    #Plot color bar
    plt.colorbar(xyzplt, ax=ax1)
    
    #Save figure
    plt.savefig(target5  + 'xyz_' + imn, dpi=300)
     
    #Show figures
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
 
    #Close figures  
    plt.close()
    
    
#------------------------------------------------------------------------------
print '\nFinished'