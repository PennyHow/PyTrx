'''
PYTRX EXAMPLE AUTOMATED AREA DRIVER (EXTENDED)

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates supraglacial lake surface area at Kronebreen, Svalbard, 
for a small subset of the 2014 melt season using modules in PyTrx. Specifically 
this script performs automated detection of supraglacial lakes through 
sequential images of the glacier to derive surface areas which have been 
corrected for image distortion.

Previously defined areas can also be imported from file (this can be changed 
by commenting and uncommenting commands in the "Calculate areas" section of 
this script).

This script is an extended breakdown of 'driver_velocity.py'. The functions 
have been expanded out to show what PyTrx is doing step-by-step. Functions that 
have been expanded in this script are:
    Image enhancement
    Automated area detection
    Exporting raw data to text file
    Exporting data to shape file
    Plotting area features
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
import cv2
import numpy as np
from pylab import array, uint8
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import ogr, osr

#Import PyTrx packages
sys.path.append('../')
from Measure import Area
from CamEnv import CamEnv
from FileHandler import readMask


#------------------------   Define input parameters   -------------------------

print '\nDEFINING DATA INPUTS'

#Camera name, location (XYZ) and pose (yaw, pitch, roll)
camname = 'KR3_2014'
camloc = np.array([451632.669, 8754648.786, 624.699])
campose = np.array([1.57904, 0.11871, -0.07796]) 

#Define image folder and image file type 
imgFiles = '../Examples/images/KR3_2014_subset/*.JPG'


#Define calibration images and chessboard dimensions
#Dimensions are number of chessboard corners for chessboard height and width
calibPath = '../Examples/camenv_data/calib/KR3_calibimgs/*.JPG'
chess = [6, 9]

#Define GCPs (world coordinates and corresponding image coordinates)
GCPpath = '../Examples/camenv_data/gcps/KR3_2014.txt'

#Define reference image (where GCPs have been defined)
imagePath = '../Examples/camenv_data/refimages/KR3_2014.JPG'

#Load DEM from path
DEMpath = '../Examples/camenv_data/dem/KR_demsmooth.tif'        

#Densify DEM 
DEMdensify = 2

#Define detection mask file 
#Mask is automatically generated if file not found
#No mask generated if input is None
maskPath = '../Examples/camenv_data/masks/KR3_2014_amask.JPG'


#Define Area class initialisation variables
maxim = 0                   #Image number of maximum areal extent 
thresh = 5                  #Maximum number of detected lakes
maxim = 0                   #Image number with maximum extents
band = 'R'                  #Image band
equal = True                #Images with histogram equalisation?
quiet = 2                   #Level of commentary

#Image enhancement for detection variables
maxIntensity = 255.0        #Maximum pixel intensity
phi = 50                    #Phi image enhancement parameter
theta = 20                  #Theta image enhancement parameter


#--------------------   Create camera and area objects   ----------------------

print '\nINITIALISING OBJECTS'

#Define camera environment
camera=CamEnv([camname,                         #Camera name
              GCPpath,                          #GCP file path
              DEMpath,                          #DEM file path
              imagePath,                        #Reference image file path
              ([calibPath,chess[0],chess[1]]),  #Calibration file path
              camloc,                           #Camera location XYZ
              campose,                          #Camera pose YPR
              DEMdensify])                      #DEM densification factor
             

#Define area object
lakes = Area(imgFiles,                          #Image file paths
             camera,                            #Camera environment
             True,                              #Calibrated image flag 
             None,                              #Mask file path
             maxim,                             #Image number with max extent
             band,                              #Image band 
             quiet,                             #Level of commentary
             False,                             #Load all images?
             'EXIF')                            #Where img time is derived from


#--------------------   Define detection parameters   -------------------------

print '\nDEFINING RGB PIXEL RANGE'

#Get camera matrix and distortion
cameraMatrix = camera.getCamMatrixCV2()
distortP = camera.getDistortCoeffsCv2()

#Get image with maximum lake extent
maximg = lakes._imageSet[maxim].getImageCorr(cameraMatrix, distortP)        
maximn = lakes._imageSet[maxim].getImagePath().split('\\')[1]

          
#Read mask from file
mask = readMask(maximg, maskPath)
        
#Mask the glacier
booleanMask = np.array(mask, dtype=bool)
booleanMask = np.invert(booleanMask)

#Mask extent image with boolean array
np.where(booleanMask, 0, maximg) #fit arrays to each other
maximg[booleanMask] = 0 #mask image with boolean mask object
        
        
#Increase intensity such that dark pixels become much brighter
#and bright pixels become slightly brighter
maximg = (maxIntensity/phi)*(maximg/(maxIntensity/theta))**0.5
maximg = array(maximg, dtype = uint8)   

     
#Initialise figure window
fig=plt.gcf()
fig.canvas.set_window_title(maximn + ': Click lightest colour and darkest' 
                            ' colour')
        
#Plot image
plt.imshow(maximg, origin='upper')
        
#Manually interact to select lightest and darkest part of the region            
colours = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, 
                    mouse_pop=3, mouse_stop=2)

#Print clicked points        
print '\n' + maximn + ': you clicked ', colours
        
#Show plot
plt.show()
            
#Obtain coordinates from selected light point
col1_y = colours[0][0]
col1_x = colours[0][1]

#Obtain coordinates from selected dark point
col2_y = colours[1][0]
col2_x = colours[1][1]
    
#Get RGB values from given coordinates
col1_rbg = lakes._getRBG(maximg, col1_x, col1_y)
col2_rbg = lakes._getRBG(maximg, col2_x, col2_y) 
        
#Assign RGB range based on value of the chosen RGB values
if col2_rbg > col1_rbg:
    upper_boundary = col2_rbg
    lower_boundary = col1_rbg
else:
    upper_boundary = col1_rbg
    lower_boundary = col2_rbg
    
print 'Colour range found from manual selection'
print 'Upper RBG boundary: ' + str(upper_boundary)
print 'Lower RBG boundary: ' + str(lower_boundary)    


#------------------   Detect lakes from image sequence   ----------------------
    
#Set up output datasets
uvcoords = []
uvareas = []
xyzcoords = []
xyzareas = []
               
#Cycle through image sequence (numbered from 0)
for i in range(lakes.getLength()):
    
    #Get corrected image and image name
    img1 = lakes._imageSet[i].getImageCorr(cameraMatrix, distortP)
    imn = lakes._imageSet[i].getImagePath().split('\\')[1]
       
    #Make a copy of the image array
    img2 = np.copy(img1)
    
    #Mask extent image with mask as boolean array
    np.where(booleanMask, 0, img2)
    img2[booleanMask] = 0
    
    #Enhance image if enhancement parameters are present
    #Increase intensity such that dark pixels become much brighter
    #and bright pixels become slightly brighter
    img2 = (maxIntensity/phi)*(img2/(maxIntensity/theta))**0.5
    img2 = array(img2, dtype = uint8) 

    #Extract extent based on RBG range
    detectrange = cv2.inRange(img2, np.array(lower_boundary,dtype='uint8'), 
                              np.array(upper_boundary,dtype='uint8'))
    
    #Polygonize extents using OpenCV findContours function        
    polyimg, line, hier = cv2.findContours(detectrange, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)
    
    print 'Detected %d regions' % (len(line))
    
    #Append all polygons from the polys list that have more than 
    #a given number of points     
    pxpoly = []
    for c in line:
        if len(c) >= 40:
            pxpoly.append(c)
    
    #Only keep the nth longest polygons
    if len(pxpoly) >= thresh:
        pxpoly.sort(key=len)
        pxpoly = pxpoly[-(thresh):]        
      
    print 'Kept %d regions' % (len(pxpoly))
    
    #Fill polygons and extract polygon extent
    px_im = Image.new('L', (img2.shape[1], img2.shape[0]), 'black')
    px_im = np.array(px_im) 
    cv2.drawContours(px_im, pxpoly, -1, (255,255,255), 4)
    for p in pxpoly:
        cv2.fillConvexPoly(px_im, p, color=(255,255,255))           
    output = Image.fromarray(px_im)
    pixels = output.getdata()
    values = []    
    for px in pixels:
        if px > 0:
            values.append(px)
    
    #Total target extent (px)
    pxextent = len(values)
    
    #Total image extent (px)
    pxcount = len(pixels)
    
    print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px)'
                    
    #Calculate real-world areas using camera environment        
    pts, a = lakes._getRealArea(pxpoly)

    #Append pixel coordinates and areas, and xyz coordinates and areas
    uvcoords.append(pxpoly)
    uvareas.append(pxextent)
    xyzcoords.append(pts)
    xyzareas.append(a)


#-------------------------   Export raw data to file   ------------------------

#Define data output directory
destination = '../Examples/results/autoarea_extended/'
if not os.path.exists(destination):
    os.makedirs(destination)

    
#Write out camera calibration info to .txt file
target1 = '../Examples/camenv_data/calib/KR3_2014_1.txt'
f=open(target1,'w')                            
f.write('RadialDistortion' + '\n' + str(camera._radCorr[:3]) + '\n' +
        'TangentialDistortion' + '\n' + str(camera._tanCorr) + '\n' +
        'IntrinsicMatrix' + '\n' + str(camera._intrMat) + '\n' +
        'End')


#Write sum of all pixel extents to file 
f = open(destination + 'px_sum.txt', 'w') 
for i in range(lakes.getLength()):
    imn = lakes._imageSet[i].getImagePath().split('\\')[1]           
    f.write(str(imn) + '_polysum(px)\t')
    f.write(str(uvareas[i]) + '\n')


#Write pixel coordinates to file
polycount=1
f = open(destination + 'px_coords.txt', 'w')
for i in range(lakes.getLength()):
    imn = lakes._imageSet[i].getImagePath().split('\\')[1] 
    f.write(str(imn) + '\t')                
    for pol in uvcoords[i]:
        f.write('Poly' + str(polycount) + '\t')                    
        for pts in pol:
            if len(pts)==1:
                f.write(str(pts[0][0]) + '\t' + str(pts[0][1]) + '\t')
            elif len(pts)==2:
                f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')
        polycount = polycount+1
    f.write('\n\n')
    polycount=1
    
    
#Write xyz areas to file
f = open(destination + 'area_all.txt', 'w')
for i in range(lakes.getLength()):
    imn = lakes._imageSet[i].getImagePath().split('\\')[1]
    f.write(str(imn) + '_polyareas\t')                
    for area in xyzareas[i]:
        f.write(str(area) + '\t')
    f.write('\n')
            
            
#Write sum of all xyz areas to file
f = open(destination + 'area_sum.txt', 'w')                           
for i in range(lakes.getLength()):
    imn = lakes._imageSet[i].getImagePath().split('\\')[1]
    f.write(str(imn) + '_totalpolyarea\t' + str(sum(xyzareas[i])) + '\n')
    
     
#Write XYZ coordinates to file
polycount=1  
f = open(destination + 'area_coords.txt', 'w')
for i in range(lakes.getLength()):
    imn = lakes._imageSet[i].getImagePath().split('\\')[1] 
    f.write(str(imn) + '\t')   
    for pol in xyzcoords[i]:
        f.write('Poly' + str(polycount) + '\t')
        for pts in pol:
            f.write(str(pts[0]) + '\t' + str(pts[1]) 
                    + '\t' + str(pts[2]) + '\t')
        polycount=polycount+1
    f.write('\n\n')
    polycount=1
            
            
#----------------------   Write data to shapefile   ---------------------------
            
#Create shapefiles (only applicable for xyz areas)
target2 = destination + 'shpfiles/'                 #Set destination
if not os.path.exists(target2):
    os.makedirs(target2)                            #Create destination
    
projection = 32633                                  #WGS84 projection

        
#Get driver and create shapeData in shp file directory        
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)
    pass
                        
#Set projection and initialise area layer            
for i in range(lakes.getLength()): 
    
    #Get image name for each time step         
    imn = lakes._imageSet[i].getImagePath().split('\\')[1] 
                  
    #Create datasource in shapefile
    shp = target2 + str(imn) + '_area.shp'    
    if os.path.exists(shp):
        print '\nDeleting pre-existing datasource'
        driver.DeleteDataSource(shp)
    ds = driver.CreateDataSource(shp)
    if ds is None:
        print 'Could not create file %s' %shp

    #Assign projection
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(projection)
    layer = ds.CreateLayer(' ', proj, ogr.wkbPolygon)

    #Add attributes
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('area', ogr.OFTReal))
    
    #Get ogr polygons from xyz areal data at given image 
    ogrpolys = []        
    for shape in xyzcoords[i]:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in shape:
            if np.isnan(pt[0]) == False:                   
                ring.AddPoint(pt[0],pt[1],pt[2])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        ogrpolys.append(poly)
        
    polycount=1

    #Create feature            
    for p in ogrpolys:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(p)
        feature.SetField('id', polycount)
        feature.SetField('area', p.Area())
        layer.CreateFeature(feature)
        feature.Destroy()                
        polycount=polycount+1
    
    ds.Destroy()
        
        
#---------------------------   Plot results   ---------------------------------

#Write all image extents and dems 
target2 = destination + 'outputimgs/'
if not os.path.exists(target2):
    os.makedirs(target2)

#Cycle through image sequence
for i in range(lakes.getLength()): 
    
    #Get image and image name for each time step         
    imn = lakes._imageSet[i].getImagePath().split('\\')[1] 
    img = lakes._imageSet[i].getImageCorr(cameraMatrix, distortP)
          
    #Set-up UV plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': UV output')
    ax1.imshow(img, cmap='gray')           
    ax1.axis([0,img.shape[1],img.shape[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Plot areal extents from coordinates
    for p in uvcoords[i]:                   #Get all coordinates
        x=[]
        y=[]
        for xy in p:                        #Get coordinates for each polygon
            if len(xy)==1:
                x.append(xy[0][0])          #Get X coordinates
                y.append(xy[0][1])          #Get Y coordinates
            elif len(xy)==2:
                x.append(xy[0])
                y.append(xy[1])
        ax1.plot(x,y,'w-')                  #Plot polygons
     
    #Save figure
    plt.savefig(target2 + 'uv_' + imn, dpi=300) 
    

    #Set-up XYZ plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': XYZ output')
    ax1.set_xticks([])
    ax1.set_yticks([])
        
    #Prepare and plot DEM
    demobj = lakes._camEnv.getDEM()
    demextent = demobj.getExtent()
    dem = demobj.getZ()
    ax1.imshow(dem, origin='lower', cmap='gray', extent=demextent)
    ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
    
    #Plot camera location
    ax1.scatter(camloc[0], camloc[1], c='g')
            
    #Extract xy data from poly pts
    count=1                
    for shp in xyzcoords[i]: 
        xl=[]
        yl=[]
        for pt in shp:
            xl.append(pt[0])
            yl.append(pt[1])
        lab = 'Area ' + str(count)
        ax1.plot(xl, yl, c=np.random.rand(3,1), linestyle='-', label=lab)
        count=count+1
    
    #Add legend
    ax1.legend()
              
    #Save figure
    plt.savefig(target2  + 'xyz_' + imn, dpi=300)
     
    #Show figures
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
 
    #Close plot   
    plt.close()


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print '\n\nFINISHED'