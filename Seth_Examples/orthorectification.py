"""
Created on Fri Mar  5 17:07:47 2021

@author: sethn
"""

#Import packages
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
sys.path.append('../')
import osgeo.ogr as ogr
import osgeo.osr as osr
from scipy import interpolate

#from pyproj import Proj
from CamEnv import GCPs, CamEnv, setProjection, projectUV, projectXYZ, optimiseCamera, computeResidualsXYZ
import DEM

# =============================================================================
# Define camera environment
# =============================================================================

directory = os.getcwd()
ingleCamEnv = directory + '/cam_env/Inglefield_CAM_camenv.txt'
ingle_calibimg = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.JPG'

#Define data output directory
destination = directory + '/results'
if not os.path.exists(destination):
    os.makedirs(destination)
print(destination)

# Define camera environment
ingleCam = CamEnv(ingleCamEnv)

# Get DEM from camera environment
dem = ingleCam.getDEM()

# Get GCPs
inglefield_gcps = directory + '/cam_env/GCPs_20190712.txt'
gcps = GCPs(dem, inglefield_gcps, ingle_calibimg)
ingle_xy = gcps.getGCPs()

# Report calibration data
ingleCam.reportCalibData()


##Show GCPs                           
#ingleCam.showGCPs()
#
##Show Camera plots                          
#ingleCam.showPrincipalPoint()
#ingleCam.showCalib()


# Optimise camera environment
ingleCam.optimiseCamEnv('YPR', 'trf', show=False)

#Get inverse projection variables through camera info            
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
                            ingleCam._camCen, ingleCam._refImage, viewshed=False)

#Inverse project image coordinates using function from CamEnv object                       
ingle_xyz = projectUV(ingle_xy[1], invprojvars)


# #------------------   Export xyz locations as .shp file   ---------------------

print('\n\nSAVING TEXT FILE')


#Write xyz coordinates to .txt file
target1 = destination + '/ingle_xyz.txt'
f = open(target1, 'w')
f.write('x' + '\t' + 'y' + '\t' + 'z' + '\n')
for i in ingle_xyz:
    f.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')                                  
f.close()


#------------------   Export xyz locations as .shp file   ---------------------

print('\n\nSAVING SHAPE FILE')


#Get ESRI shapefile driver     
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)


#Create data source
shp = destination + '/ingle_xyz.shp'   
if os.path.exists(shp):
    driver.DeleteDataSource(shp)
ds = driver.CreateDataSource(shp)
if ds is None:
    print('Could not create file ' + str(shp))
 
       
#Set WGS84 projection
proj = osr.SpatialReference()
proj.ImportFromEPSG(32633)          #CHANGE TO VALID EPSG projection


#Create layer in data source
layer = ds.CreateLayer('ingle_xyz', proj, ogr.wkbPoint)
  
  
#Add attributes to layer
layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      #ID    
    
  
#Create point features with data attributes in layer           
for a in ingle_xyz:
    count=1

    #Create feature    
    feature = ogr.Feature(layer.GetLayerDefn())

    #Create feature attributes    
    feature.SetField('id', count)
      
    #Create feature location
    wkt = "POINT(%f %f)" %  (float(a[0]) , float(a[1]))
    point = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(point)
    layer.CreateFeature(feature)

    #Free up data space
    feature.Destroy()                       
    count=count+1


#Free up data space    
ds.Destroy() 