# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:50:08 2021

@author: sethn
"""

# Import modules
import gdal
import pyproj
import numpy as np
import scipy.interpolate as interpolate


def geotiff_read(infile):
    """
    Function to read a Geotiff file and convert to numpy array.
    """
    # Allow GDAL to throw Python exceptions
    gdal.UseExceptions()
    # Read tiff and convert to a numpy array
    tiff = gdal.Open(infile)
    if tiff.RasterCount == 1:
        array = tiff.ReadAsArray()
    if tiff.RasterCount > 1:
        array = np.zeros((tiff.RasterYSize, tiff.RasterXSize, tiff.RasterCount))
        for i in range(tiff.RasterCount):
            band = tiff.GetRasterBand(i + 1)
            array[:, :, i] = band.ReadAsArray()
    # Get parameters
    geotransform = tiff.GetGeoTransform()
    projection = tiff.GetProjection()
    band = tiff.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    return array, geotransform, projection, nodata

def geotiff_write(outfile, geotransform, projection, data, nodata=None):
    """
    Function to write a numpy array as a GeoTIFF file.
    
    IMPORTANT: I've edited this function so it writes the data as byte format.
    
    """
    # Produce numpy to GDAL conversion dictionary    
    print('Writing %s' % outfile)
    driver = gdal.GetDriverByName('GTiff')
    
    if data.ndim == 2:
        (x,y) = data.shape
        tiff = driver.Create(outfile, y, x, 1, gdal.GDT_Byte)
        tiff.GetRasterBand(1).WriteArray(data)
       
    if data.ndim > 2:
        bands = data.shape[2]
        (x,y,z) = data.shape
        tiff = driver.Create(outfile, y, x, bands, gdal.GDT_Byte)
        for band in range(bands):
            array = data[:, :, band + 1]
            tiff.GetRasterBand(band).WriteArray(array)
    
    if nodata:
        tiff.GetRasterBand(1).SetNoDataValue(nodata)
    tiff.SetGeoTransform(geotransform)
    tiff.SetProjection(projection) 
    tiff = None	
    
    return 1

# Import DEM
dem, gt, proj, nodata = geotiff_read('C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield Data/201907-Minturn-Elv-5cm-octree_dem.tif')
x = np.linspace(0, 1, dem.shape[0])
y = np.linspace(0, 1, dem.shape[1])
yy, xx = np.meshgrid(y, x)
# Identify NaNs
dem[dem == -9999] = np.nan
vals = ~np.isnan(dem)
# Sample DEM
xx_sub = xx[1950:2150,3650:3850]
yy_sub = yy[1950:2150,3650:3850]
dem_sub = dem[1950:2150,3650:3850]
vals_sub = ~np.isnan(dem_sub)
z_dense_smooth_griddata = interpolate.griddata(np.array([xx_sub[vals_sub].ravel(),
                                                         yy_sub[vals_sub].ravel()]).T, 
                                                         dem_sub[vals_sub].ravel(), (xx_sub,yy_sub), 
                                                         method='cubic')

