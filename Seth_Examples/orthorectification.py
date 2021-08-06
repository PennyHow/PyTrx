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

from pyproj import Proj
from CamEnv import GCPs, CamEnv, setProjection, projectUV, projectXYZ, optimiseCamera
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

# Define GCP class
inglefield_gcps = directory + '/cam_env/GCPs_20190712.txt'
gcps = GCPs(dem, inglefield_gcps, ingle_calibimg)
ingle_xy = gcps.getGCPs()

# Report calibration data
ingleCam.reportCalibData()

#Show GCPs                           
ingleCam.showGCPs()

#Show Camera plots                          
ingleCam.showPrincipalPoint()
ingleCam.showCalib()

#Get inverse projection variables through camera info            
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
                            ingleCam._camCen, ingleCam._refImage)

#Inverse project image coordinates using function from CamEnv object                       
ingle_xyz = projectUV(ingle_xy[1], invprojvars)


# # Optimise camera environment
# ingleCam.optimiseCamEnv('YPR')
# opt_projvars = optimiseCamera('YPR', [ingleCam._camloc, ingleCam._camDirection, 
#                               ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
#                               ingleCam._camCen, ingleCam._refImage], ingle_xy[0], 
#                               ingle_xy[1], 'trf', show=False)



#Retrieve DEM from CamEnv object
demobj=ingleCam.getDEM()
demextent=demobj.getExtent()
dem=demobj.getZ()

#Get camera position (xyz) from CamEnv object
post = ingleCam._camloc

# #------------------   Export xyz locations as .shp file   ---------------------

print('\n\nSAVING SHAPE FILE')


#Get ESRI shapefile driver     
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)


#Create data source
shp = destination + 'orthorectify.shp'   
if os.path.exists(shp):
    driver.DeleteDataSource(shp)
ds = driver.CreateDataSource(shp)
if ds is None:
    print('Could not create file ' + str(shp))
 
       
# #Set WGS84 projection
# proj = osr.SpatialReference()
# proj.ImportFromEPSG(32619)


# #Create layer in data source
# layer = ds.CreateLayer('orthorectify', proj, ogr.wkbPoint)
  
  
# #Add attributes to layer
# layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      #ID    
# layer.CreateField(ogr.FieldDefn('time', ogr.OFTReal))       #Time
# field_region = ogr.FieldDefn('region', ogr.OFTString)        
# field_region.SetWidth(8)    
# layer.CreateField(field_region)                             #Calving region
# field_style = ogr.FieldDefn('style', ogr.OFTString)        
# field_style.SetWidth(10)    
# layer.CreateField(field_style)                              #Calving size    