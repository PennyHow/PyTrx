'''
PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie

PyTrx is licensed under a MIT License.

You should have received a copy of the license along with this
work. If not, see <https://choosealicense.com/licenses/mit/>.


PYTRX EXAMPLE AUTOMATED AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates snow/bare ice extent at Qasigiannguit Glacier, West 
Greenland, for a small subset of the 2020/21 melt season using modules in 
PyTrx. Specifically this script performs automated detection of snow regions 
through images of the glacier to derive surface areas which have been 
corrected for image distortion.

Previously defined areas can also be imported from file (this can be changed 
by commenting and uncommenting commands in the "Calculate areas" section of 
this script).
'''

#Import packages
import os, sys

##Import PyTrx modules
try:
    sys.path.append('../')
    from Area import Area
    import FileHandler as FileHandler
    from Velocity import Homography
    from CamEnv import CamEnv
    from Utilities import plotAreaPX, plotAreaXYZ
except:
    from PyTrx.Area import Area
    import PyTrx.FileHandler as FileHandler
    from PyTrx.Velocity import Homography
    from PyTrx.CamEnv import CamEnv
    from PyTrx.Utilities import plotAreaPX, plotAreaXYZ


#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironment_QAS_2020.txt'
camamask = '../Examples/camenv_data/masks/QAS_2020_amask.jpg'
invmask = '../Examples/camenv_data/invmasks/QAS_2020_inv.jpg'      
camimgs = '../Examples/images/QAS_2020_subset/*.CR2'                           

#Define data output directory
destination = '../Examples/results/QAS_autoarea/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)

#Optimisation parameters
optflag = 'YPR'                 #Parameters to optimise (YPR/INT/EXT/ALL)
optmethod = 'trf'               #Optimisation method (trf/dogbox/lm)
show=True                       #Show refined camera environment

#Optimise camera environment for YPR
cam.optimiseCamEnv(optflag, optmethod, show)
 
##Report camera data and show corrected image
#cam.reportCamData()
#cam.showGCPs()
#cam.showPrincipalPoint()
#cam.showCalib()
#cam.showResiduals()

#---------------------   Calculate homography   -------------------------------

#Set homography parameters
hmethod='sparse'                #Method
hgwinsize=(25,25)               #Tracking window size
hgback=1.0                      #Back-tracking threshold
hgmax=50000                     #Maximum number of points to seed
hgqual=0.1                      #Corner quality for seeding
hgmind=5.0                      #Minimum distance between seeded points
hgminf=4                        #Minimum number of seeded points to track

#Set up Homography object
homog = Homography(camimgs, cam, invmask, calibFlag=True, band='L', equal=True)

#Calculate homography
hg = homog.calcHomographies([hmethod, [hgmax, hgqual, hgmind], [hgwinsize, 
                             hgback, hgminf]])    
      
homogmatrix = [item[0] for item in hg] 


#----------------------   Detect and calculate areas  -------------------------             

#Set up Area object, from which areal extent will be measured
imband = 'R'                                                                   #PARAMETER: Desired image band (R/G/B/L)
equal = True                                                                   #PARAMETER: Images with histogram equalisation? (True/False)
calibFlag = False                                                              #PARAMETER: Use images with lens distortion or corrected? (True/False)
snow = Area(camimgs, cam, homogmatrix, calibFlag, imband, equal)

#Set image enhancement parameters. If these are undefined then they will be 
#set to a default enhancement of ('light', 50, 20)
diff = 'light'                                                                 #PARAMETER: Image enhancement parameter 1 (light/dark)
phi = 50                                                                       #PARAMETER: Image enhancement parameter 2 
theta = 20                                                                     #PARAMETER: Image enhancement parameter 3
snow.setEnhance(diff, phi, theta)


#Set colour range, from which extents will be distinguished. If colour range 
#is not specified, it will be manually defined 
define = False                                                                 #PARAMETER: Define colour range with clicks or pre-defined? (True/False)
if define is False:
    maxcol = 17                                                                #PARAMETER: Max value from which areas will be distinguished
    mincol = 14                                                                #PARAMETER: Min value from which areas will be distinguished
    snow.setColourrange(maxcol, mincol)                             


#Set mask and image with maximum area of interest 
maxim = 0                                                                      #PARAMETER: Image number of maximum areal extent 
snow.setMax(camamask, maxim)                                                                                                                                                     


# #Set px plotting extent for easier colourrange definition and area verification
# pxext = [0,1200,2000,1500]  
# lakes.setPXExt(pxext[0], pxext[1], pxext[2], pxext[3])


#Set polygon threshold (i.e. number of polygons kept)
threshold = 10                                                                 #PARAMETER: Threshold for number of retained polygons
snow.setThreshold(threshold)


#Calculate real areas
colour=False                                                                   #PARAMETER: Define colour range in each image? (True/False)
verify=False                                                                   #PARAMETER: Manually verify detected areas? (True/False)
areas = snow.calcAutoAreas(colour, verify)


##Import data from file
#areas = lakes.importData(destination)


#----------------------------   Export data   ---------------------------------

#Get area data for writing to files
xyzareas = [item[0][0] for item in areas]                           #XYZ areas
xyzpts = [item[0][1] for item in areas]                             #XYZ coords
uvareas = [item[1][0] for item in areas]                            #UV areas
uvpts = [item[1][1] for item in areas]                              #UV coords

#Get camera and dem information for writing to files
matrix, tancorr, radcorr = cam.getCalibdata()                       #CV2 calib
imn = snow.getImageNames()                                          #Img names
proj = 32622                                                        #Projection (WGS84)
dem = cam.getDEM()                                                  #DEM
imgset=snow._imageSet                                               #Images
cameraMatrix=cam.getCamMatrixCV2()                                  #Matrix
distortP=cam.getDistortCoeffsCV2()                                  #Distort


#Write homography data to .csv file
FileHandler.writeHomogFile(hg, imn, destination+'homography.csv')


#Write results to file
FileHandler.writeAreaFile(uvareas, xyzareas, imn, destination+'areas.csv')
FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                            destination+'uvcoords.txt', 
                            destination+'xyzcoords.txt')

#Create shapefiles                
FileHandler.writeAreaSHP(xyzpts, imn, destination+'shpfiles/', proj)            


#Write all image extents and dems 
target4 = destination + 'outputimgs/'
if not os.path.exists(target4):
    os.makedirs(target4)
    
for i in range(len(areas)):
    plotAreaPX(uvpts[i], imgset[i].getImageCorr(cameraMatrix, distortP),        
                show=True, save=target4+'uv_'+str(imn[i]).split('.CR2')[0]+'.jpg')  
    plotAreaXYZ(xyzpts[i], dem, show=True, save=target4+'xyz_'+
                str(imn[i]).split('.CR2')[0] + '.jpg')


#------------------------------------------------------------------------------                                                                                                                                                                                                                                                                                                      

print('\n\nFINISHED')
