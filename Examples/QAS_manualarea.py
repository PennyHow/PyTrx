'''
PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie

PyTrx is licensed under a MIT License.

You should have received a copy of the license along with this
work. If not, see <https://choosealicense.com/licenses/mit/>.


PYTRX EXAMPLE MANUAL AREA DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates meltwater plume surface extent at Kronebreen (camera 
site 1, KR1) for a small subset of the 2014 melt season using modules in PyTrx. 
Specifically this script performs manual detection of supraglacial lakes 
through sequential images of the glacier to derive surface areas which have 
been corrected for image distortion. Previously defined pixel areas can also 
be imported from file (this can be changed by commenting and uncommenting 
commands in the "Calculate areas" section of this script).
'''

#Import packages
import os

#If you have downloaded PyTrx directly from GitHub then comment out the PyTrx
#module imports above and uncomment these ones below
import sys
sys.path.append('../')
from Area import Area
from Velocity import Homography
from CamEnv import CamEnv
import FileHandler 
from Utilities import plotAreaPX, plotAreaXYZ

#-----------------------------   Map data files   -----------------------------

#Define data inputs
camdata = '../Examples/camenv_data/camenvs/CameraEnvironment_QAS_2020.txt'
invmask = None  

camimgs = '../Examples/images/pho_test/*.CR2'                                  #Image path
#camimgs = '../Examples/images/QAS_EOS/*.CR2' #End of Season

#Define data output directory
destination = '../Examples/results/manualarea/'
if not os.path.exists(destination):
    os.makedirs(destination)
    

#--------------------   Create camera and area objects   ----------------------

#Define camera environment
cam = CamEnv(camdata)

# #Optimise camera environment
# optparams = 'YPR'
# cam.optimiseCamEnv(optparams)

##Show ground control points
#cameraenvironment.showGCPs()
#cameraenvironment.showResiduals()


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


#------------------------   Calculate Areas   ---------------------------------

#Define Area class initialisation variables
calibFlag = False            #Detect with corrected or uncorrected images. Default = true
maxim = 0                   #Image number of maximum areal extent 
imband = 'L'                #Desired image band
equal = True                #Images with histogram equalisation?

#Set up Area object, from which areal extent will be measured
plumes = Area(camimgs, cam, homogmatrix, calibFlag, imband, equal)


#-------------------------   Calculate areas   --------------------------------

#Calculate real areas
areas = plumes.calcManualAreas()

##Import areal data from file
#xyzfile=destination+'area_coords.txt'
#pxfile=destination+'px_coords.txt'
#areas = importAreaData(xyzfile, pxfile)


#----------------------------   Export data   ---------------------------------

#Get area data
xyzareas = [item[0][0] for item in areas]
xyzpts = [item[0][1] for item in areas]
uvareas = [item[1][0] for item in areas]
uvpts = [item[1][1] for item in areas]

#Get relevant camera and dem data
imn=plumes.getImageNames()
dem = cam.getDEM()
imgset=plumes._imageSet
cameraMatrix=cam.getCamMatrixCV2()
distortP=cam.getDistortCoeffsCV2()

#Write results to file
FileHandler.writeAreaFile(uvareas, xyzareas, imn, destination+'areas.csv')
FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                            destination+'uvcoords.txt', 
                            destination+'xyzcoords.txt')

#Write homography to file
FileHandler.writeHomogFile(hg, imn, destination+'homography.csv')

#Create shapefiles
target1 = destination + 'shpfiles/'    
proj = 32622
FileHandler.writeAreaSHP(xyzpts, imn, target1, proj) 

#Write results to file
FileHandler.writeAreaFile(uvareas, xyzareas, imn, destination+'areas.csv')
FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                            destination+'uvcoords.txt', 
                            destination+'xyzcoords.txt')
   
# #Plot areas in image plane and as XYZ polygons  
# target2 = destination + 'outputimgs/'
# for i in range(len(xyzpts)):
#     plotAreaPX(uvpts[i], imgset[i].getImageCorr(cameraMatrix, distortP), 
#                show=True, save=None)  
#     plotAreaXYZ(xyzpts[i], dem, show=True, save=None)
    
    
#------------------------------------------------------------------------------    
    
print('\n\nFINISHED')
