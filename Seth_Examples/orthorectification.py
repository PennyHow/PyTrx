"""
Created on Fri Mar  5 17:07:47 2021

@author: sethn
"""

#Import packages
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyproj import Proj
from PyTrx.CamEnv import GCPs, CamEnv, setProjection, projectUV, projectXYZ
import PyTrx.DEM
# from Images import CamImage

# =============================================================================
# Define camera environment
# =============================================================================

ingleCamEnv = 'C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data/cam_env/Inglefield_CAM_camenv.txt'
ingle_calibimg = 'C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data/cam_env/CAM_20190712.JPG'

#Define data output directory
destination = 'C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data'
if not os.path.exists(destination):
    os.makedirs(destination)
    
print(destination)

# Define camera environment
ingleCam = CamEnv(ingleCamEnv)

# Get DEM from camera environment
dem = ingleCam.getDEM()

# Define GCP class
inglefield_gcps = 'C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield_Data/cam_env/GCPs_20190712.txt'
gcps = GCPs(dem, inglefield_gcps, ingle_calibimg)
ingle_xy = gcps.getGCPs()

# Report calibration data
ingleCam.reportCalibData()
ingleCam.getCamMatrix()

#Get inverse projection variables through camera info               
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
                            ingleCam._camCen, ingleCam._refImage)

#Inverse project image coordinates using function from CamEnv object                       
ingle_xyz = projectUV(ingle_xy, invprojvars)

#Retrieve DEM from CamEnv object
demobj=ingleCam.getDEM()
demextent=demobj.getExtent()
dem=demobj.getZ()

#Get camera position (xyz) from CamEnv object
post = ingleCam._camloc

#Show GCPs                           
ingleCam.showGCPs()

#Plot DEM 
fig,(ax1) = plt.subplots(1, figsize=(15,15))
fig.canvas.set_window_title('TU1 calving event locations')
ax1.locator_params(axis = 'x', nbins=8)
ax1.tick_params(axis='both', which='major', labelsize=0)
ax1.imshow(dem, origin='lower', extent=demextent, cmap='gray')
ax1.axis([demextent[0], demextent[1], demextent[2], demextent[3]])
cloc = ax1.scatter(post[0], post[1], c='g', s=10, label='Camera location')



# # =============================================================================
# # Define camera location coordinates 
# #   Project camera lat/long coordinates to UTM (Z 19N) and then use PyTrx DEM
# #   functionality to get elevation (Z) at that location
# # =============================================================================

# # Inglefield CAM (Right side)
# # project camera coordinates to utm
# camlocx = -68.988806
# camlocy = 78.591213
# camlocproj = Proj("+proj=utm +zone=19 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# camlocproj(camlocx, camlocy)

# rowcoords = dem.getData(0)[0,:]    
# colcoords = dem.getData(1)[:,0]
             
# camxcoord = (np.abs(rowcoords-500247.18789250357)).argmin()
# camycoord = (np.abs(colcoords-8724350.67101476)).argmin()

# X = dem.getData(0)
# Y = dem.getData(1)
# Z = dem.getData(2)


# xsize = np.linspace(0, 1, Z.shape[0])
# ysize = np.linspace(0, 1, Z.shape[1])
# yy, xx = np.meshgrid(ysize, xsize)
# # Identify NaNs
# Z[Z == -9999] = np.nan
# vals = ~np.isnan(Z)


# # Sample DEM
# xx_sub = xx[2000:3000, 200:2000]
# yy_sub = yy[2000:3000, 200:2000]
# dem_sub = Z[2000:3000, 200:2000]
# vals_sub = ~np.isnan(dem_sub)
# z_griddata = griddata(np.array([xx_sub[vals_sub].ravel(),
#                                              yy_sub[vals_sub].ravel()]).T, 
#                                              dem_sub[vals_sub].ravel(),                    
                                                # (xx_sub,yy_sub), 
                                                # method='nearest')
#------------------   Export xyz locations as .txt file   ---------------------

# print('\n\nSAVING TEXT FILE')


# #Write xyz coordinates to .txt file
# target1 = destination + 'TU1_calving_xyz.txt'
# f = open(target1, 'w')
# f.write('x' + '\t' + 'y' + '\t' + 'z' + '\n')
# for i in tu1_xyz:
#     f.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')                                  
# f.close()


# #------------------   Export xyz locations as .shp file   ---------------------

# print('\n\nSAVING SHAPE FILE')


# #Get ESRI shapefile driver     
# typ = 'ESRI Shapefile'        
# driver = ogr.GetDriverByName(typ)
# if driver is None:
#     raise IOError('%s Driver not available:\n' % typ)


# #Create data source
# shp = destination + 'tu1_calving.shp'   
# if os.path.exists(shp):
#     driver.DeleteDataSource(shp)
# ds = driver.CreateDataSource(shp)
# if ds is None:
#     print('Could not create file ' + str(shp))
 
       
# #Set WGS84 projection
# proj = osr.SpatialReference()
# proj.ImportFromEPSG(32633)


# #Create layer in data source
# layer = ds.CreateLayer('tu1_calving', proj, ogr.wkbPoint)
  
  
# #Add attributes to layer
# layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      #ID    
# layer.CreateField(ogr.FieldDefn('time', ogr.OFTReal))       #Time
# field_region = ogr.FieldDefn('region', ogr.OFTString)        
# field_region.SetWidth(8)    
# layer.CreateField(field_region)                             #Calving region
# field_style = ogr.FieldDefn('style', ogr.OFTString)        
# field_style.SetWidth(10)    
# layer.CreateField(field_style)                              #Calving size    
 
  
# #Create point features with data attributes in layer           
# for a,b,c,d in zip(tu1_xyz, time, region, style):
#     count=1

#     #Create feature    
#     feature = ogr.Feature(layer.GetLayerDefn())

#     #Create feature attributes    
#     feature.SetField('id', count)
#     feature.SetField('time', b)
#     feature.SetField('region', c) 
#     feature.SetField('style', d)         

#     #Create feature location
#     wkt = "POINT(%f %f)" %  (float(a[0]) , float(a[1]))
#     point = ogr.CreateGeometryFromWkt(wkt)
#     feature.SetGeometry(point)
#     layer.CreateFeature(feature)


