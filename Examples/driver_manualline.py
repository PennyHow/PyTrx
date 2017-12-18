'''
PYTRX EXAMPLE MANUAL LINE DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates terminus profiles (as line features) at Tunabreen, 
Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. 
Specifically this script performs manual detection of terminus position through 
sequential images of the glacier to derive line profiles which have been 
corrected for image distortion. 

Previously defined lines can also be imported from text or shape file (this 
can be changed by commenting and uncommenting commands in the "Calculate lines" 
section of this script).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton
'''

#Import packages
import sys
import os
from osgeo import ogr
import glob

#Import PyTrx packages
sys.path.append('../')
from Measure import Line
from CamEnv import CamEnv
from FileHandler import writeLineFile, writeSHPFile, importLineData
from Utilities import plotPX, plotXYZ


#---------------------------   Initialisation   -------------------------------

#Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU2_2015.txt'   
camimgs = '../Examples/images/TU2_2015_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/manualline/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)

#Set up line object
terminus = Line(camimgs, cam)


#-----------------------   Calculate/import lines   ---------------------------

#Choose action "plot", "importtxt" or "importshp". Plot proceeds with the 
#manual  definition of terminus lines, importtxt imports line data from text 
#files, and  importshp imports line data from shape file (.shp)
action = 'importtxt'      


#Manually define lines from imagery
if action == 'plot':
    rline, rlength = terminus.calcManualLinesXYZ()
    pxlength = terminus._pxline
    pxline = terminus._pxpts


#Import line data from text files   
elif action == 'importtxt':
    #Import lines to terminus object
    rline, rlength, pxline, pxlength = importLineData(terminus, destination)


#Import line data from shape files (only imports real line data, not pixel)
elif action == 'importshp':
    shpfiles = destination + 'shapefiles/*.SHP'
    xyz_line=[]    
    xyz_corr=[]
    xyz_len=[]
   
    #Get shape object from files
    for i in glob.glob(shpfiles):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(i, 0)
        layer = dataSource.GetLayer()
        
        #Append line length and xyz coordinates to lists           
        for feature in layer:
            geom = feature.GetGeometryRef()
            xyz_line.append(geom)
            xyz_len.append(geom.Length())
            ptpos = geom.Centroid().ExportToWkt()
            pt2=[ptpos]
            pt2 = ptpos.split('(')
            pt3 = pt2[1].split(')')
            pt3 = pt3[0].split(' ')
            xyz_corr.append(pt3)
        
    #Append data to Line object
    terminus._realpts = xyz_corr
    terminus._realline = xyz_line


#Program will terminate if an invalid string is inputted as the action variable        
else:
    print 'Invalid action. Please re-define.'
    sys.exit(1)
   
   
#----------------------------   Export data   ---------------------------------

#Change flags to write text and shape files
write = True
shp = True

#Write line data to txt file
if write is True:   
    writeLineFile(terminus, destination)

#Write shapefiles from line data
if shp is True:   
    target1 = destination + 'shapefiles/'
    if not os.path.exists(target1):
        os.makedirs(target1)    
    proj = 32633
    writeSHPFile(terminus, target1, proj)


#----------------------------   Show results   --------------------------------

#Generate destination location
target2 = destination + 'outputimgs/'
if not os.path.exists(target2):
    os.makedirs(target2)

#Plot and save all extent and area images
length=len(pxline)
for i in range(len(pxline)):
    plotPX(terminus, i, target2, crop=None, show=True)
    plotXYZ(terminus, i, target2, crop=None, show=True, dem=True)
    
    
#------------------------------------------------------------------------------

print '\n\nFinished'