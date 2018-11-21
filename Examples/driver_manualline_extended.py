'''
PYTRX EXAMPLE MANUAL LINE DRIVER (EXTENDED VERSION)

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates terminus profiles (as line features) at Tunabreen, 
Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. 
Specifically this script performs manual detection of terminus position through 
sequential images of the glacier to derive line profiles which have been 
corrected for image distortion. 

This script is an extended breakdown of 'driver_manualline.py'. The functions 
have been expanded out to show what PyTrx is doing step-by-step. Functions that 
have been expanded in this script are:
    Manual line definition
    Line projection
    Exporting output data
    Shapefile generation
    Plotting
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
from osgeo import ogr,osr
import numpy as np
import matplotlib.pyplot as plt

#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv
from Measure import Line


#---------------------------   Define data inputs   ---------------------------

print '\nDEFINING DATA INPUTS'

#Define camera name, location and pose
camname = 'TU1_2015'
camloc = np.array([551141.156, 8710426.44, 480.658])    #UTM meters
camDirection = np.array([5.9929, 0.174, 0])             #Yaw, pitch, roll

#Define image folder and image file type 
imgFiles = '../Examples/images/TU1_2015_subset/*.JPG'

#Define calibration (Camera matrix, radial distortion, tangential distortion)
calibPath = '..\Examples\camenv_data\calib\TU1_2015_1.txt'

#Define GCPs (world coordinates and corresponding image coordinates)
GCPpath = '..\Examples\camenv_data\gcps\TU1_2015.txt'

#Define reference image (where GCPs have been defined)
imagePath = '..\Examples\camenv_data\refimages\TU1_2015.JPG'

#Load DEM from path
DEMpath = '..\Examples\camenv_data\dem\TU_demzero.tif'        

#Densify DEM 
DEMdensify = 2

#--------------------------   Initialise objects   ----------------------------

print '\nINITIALISING PYTRX OBJECTS'

#Initialise camera environment
camera = CamEnv([camname,GCPpath,DEMpath,imagePath,
                calibPath,camloc,camDirection,DEMdensify])
              
              
#Initialise line object              
terminus = Line(imgFiles, camera)            
        
#-------------------------  Define lines in images   --------------------------

print '\nMANUAL PIXEL LINE DEFINITION' 

#Initialise outputs
px_pts = []
px_lines = []
#Cycle through image files
for i in range(terminus.getLength()):  

    #Get camera matrix and distortion parameters
    cameraMatrix = camera.getCamMatrixCV2()
    distortP = camera.getDistortCoeffsCv2()    
    
    #Get corrected image
    img = terminus._imageSet[i].getImageCorr(cameraMatrix, distortP)
    
    #Get image name
    imn = terminus._imageSet[i].getImagePath().split('\\')[1] 

    #Define line
    pt, line = terminus._calcManualLinePX(img, imn)
    
    px_pts.append(pt)
    px_lines.append(line)
    
    print '\nUV Line defined in ' + imn                
    print 'UV line length: %i pixels' % (line.Length())
    print 'UV line contains %i points' % (line.GetPointCount())


#--------------------------   Project uv points   -----------------------------

print '\n\nCOMMENCING GEORECTIFICATION OF LINES'

#Set output variables and counter
xyz_pts = []
xyz_lines = []        
count=1


#Project pixel coordinates to obtain real world coordinates and lines
for p in px_pts:              
    
    #Project image coordinates
    xyz = terminus._camEnv.invproject(p)
    
    #Initially construct geometry object             
    linexyz = ogr.Geometry(ogr.wkbLineString)
    
    #Append points to geometry object
    for q in xyz:
        if len(q)==2:
            linexyz.AddPoint(q[0],q[1])
        elif len(q)==3:
            linexyz.AddPoint(q[0],q[1],q[2])
        else:
            pass


    #Optional commentary
    print '\nXYZ line defined in ' + imn                
    print 'XYZ line length: %i metres' % (linexyz.Length())
    print 'XYZ line contains %i points' % (linexyz.GetPointCount())
    
    
    #Append coordinates and distances            
    xyz_pts.append(xyz)
    xyz_lines.append(linexyz)
    
    #Update counter
    count=count+1


#----------------------   Export raw data to txt file   -----------------------

print '\nEXPORTING RAW DATA'

#Define data output directory
destination = '../Examples/results/manualline_extended/'

#Generate directory
if not os.path.exists(destination):
    os.makedirs(destination)
 
          
#Pixel line coordinates file generation                        
imgcount=1
f = open(destination + 'line_pxcoords.txt', 'w')
for i in range(terminus.getLength()):
    imn = terminus._imageSet[i].getImagePath().split('\\')[1]
    f.write(str(imn) + '\t')                                   
    for pts in px_pts[i]:
        f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')    
    f.write('\n\n')
    imgcount = imgcount+1


#Pixel line length file generation            
imgcount=1 
f = open(destination + 'line_pxlength.txt', 'w')            
for i in range(terminus.getLength()):
    imn = terminus._imageSet[i].getImagePath().split('\\')[1]
    f.write(str(imn) + '_length(px)\t')
    f.write(str(px_lines[i].Length()) + '\n')
    imgcount=imgcount+1 


#Real line coordinates file generation                  
imgcount=1  
f = open(destination + 'line_realcoords.txt' , 'w')
for i in range(terminus.getLength()):
    imn = terminus._imageSet[i].getImagePath().split('\\')[1]                
    f.write(str(imn) + '\t')
    for pts in xyz_pts[i]:
        f.write(str(pts[0]) + '\t' + 
                str(pts[1]) + '\t' + 
                str(pts[2]) + '\t')
    f.write('\n\n')
    imgcount=imgcount+1


#Real line length file generation            
imgcount=1
f = open(destination + 'line_reallength.txt', 'w')
for i in range(terminus.getLength()):
    imn = terminus._imageSet[i].getImagePath().split('\\')[1]
    f.write(str(imn) + '_length(m)\t')                
    f.write(str(xyz_lines[i].Length()) + '\n')
    imgcount=imgcount+1


#----------------------   Export data as shape files   ------------------------

print '\nEXPORTING SHAPEFILES'

#Define destination folder for shapefiles  
target1 = destination + 'shapefiles/'

#Generate destination folder
if not os.path.exists(target1):
    os.makedirs(target1) 

    
#Get driver and create shapeData in shp file directory        
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)
    pass

        
#Set projection and initialise line layer                   
for i in range(terminus.getLength()):
    
    #Get polygon coordinates and image name for each time step
    line = xyz_lines[i]  
    imn = terminus._imageSet[i].getImagePath().split('\\')[1] 

    #Create datasource in shapefile
    shp = target1 + str(imn) + '_line.shp'    
    if os.path.exists(shp):
        driver.DeleteDataSource(shp)
    ds = driver.CreateDataSource(shp)
    if ds is None:
        print 'Could not create file %s' %shp

    #Assign projection   
    projection = 32633
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(projection)
    layer = ds.CreateLayer(' ', proj, ogr.wkbLineString)
    
    #Add line attributes
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('length', ogr.OFTReal))
        
    #Create feature from data
    lcount=1            
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(xyz_lines[i])
    feature.SetField('id', lcount)
    feature.SetField('length', xyz_lines[i].Length())
    layer.CreateFeature(feature)
    feature.Destroy() 
    lcount=lcount+1
    
    #Destroy datasource to free up memory
    ds.Destroy()


#----------------------------   Show results   --------------------------------

print '\nPLOTTING OUTPUTS'

#Define destination location for output plots
target2 = destination + 'outputimgs/'

#Generate destination
if not os.path.exists(target2):
    os.makedirs(target2)
    
#Prepare DEM for plotting
demobj=terminus._camenv.getDEM()
demextent=demobj.getExtent()
dem=demobj.getZ()
    

#Cycle through image sequence
for i in range(terminus.getLength()):
    
    #Get corrected image
    img=terminus._imageSet[i].getImageCorr(terminus._camEnv.getCamMatrixCV2(), 
                                           terminus._camEnv.getDistortCoeffsCv2())      
    
    #Get image name
    imn = terminus._imageSet[i].getImagePath().split('\\')[1] 
    
    #Get image size
    imsz = terminus._imageSet[i].getImageSize()
          
    #Create plotting window for UV line output
    fig1, (ax1) = plt.subplots(1, figsize=(20,10))
    fig1.canvas.set_window_title(imn + ': UV output')
    ax1.imshow(img, cmap='gray')            
    ax1.axis([0,imsz[1],imsz[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])
       
    #Plot pixel lines                 
    x=[]
    y=[]
    for xy in px_pts[i]:
        x.append(xy[0])                 #Get X coordinates
        y.append(xy[1])                 #Get Y coordinates
    ax1.plot(x,y,'w-')                  #Plot line
    
    #Save figure
    plt.savefig(target2 + 'uv_' + imn, dpi=300) 

       
    #Create plotting window for XYZ line output
    fig2, (ax2) = plt.subplots(1, figsize=(20,10))
    fig2.canvas.set_window_title(imn + ': XYZ output')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    #Plot DEM
    ax2.imshow(dem, origin='lower', cmap='gray', extent=demextent)
    ax2.axis([demextent[0], demextent[1], demextent[2], demextent[3]])
        
    #Plot camera location
    ax2.scatter(camloc[0], camloc[1], c='g')
                 
    #Get xy data from line pts
    xl=[]
    yl=[]        
    for pt in xyz_pts[i]:
        xl.append(pt[0])
        yl.append(pt[1])

    #Plot line points and camera position on to DEM image        
    ax2.plot(xl, yl, 'y-')
    
    #Save figure    
    plt.savefig(target2  + 'xyz_' + imn, dpi=300)
    
    #Show figures
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()    
    
    #Close figure                        
    plt.close()   
    
#------------------------------------------------------------------------------

print '\n\nFINISHED'