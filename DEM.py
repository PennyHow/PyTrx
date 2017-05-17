# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 00:44:33 2013

@author: nrjh
"""
import numpy as np
import scipy.io as sio
import gdal
import math
from scipy import interpolate

from gdalconst import * 
import struct

#from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
#from scipy.interpolate import griddata,interp2d

class ExplicitRaster(object):
    
    '''A class to represent a numeric Raster with explicit X-Y Cell referencing
    in each grid cell'''
    

# Basic constuctor method
    def __init__(self,X,Y,Z,nodata=float('nan')):
        
        if not (X.shape==Y.shape and X.shape==Z.shape):
            print 'Raster data and/or co-ordinate arrays are differently sized'
            print 'X-shape',X.shape
            print 'Y-shape',Y.shape
            print 'Z-shape',Z.shape
            return 
        self._data=np.array([X,Y,Z]) 
        self._nodata=nodata
        #print 'raw dims', X[0][0],X[-1][-1],Y[-1][0],Y[0][-1]
        self._extents=[X[0][0]-0.5*(X[0][1]-X[0][0]),X[-1][-1]+0.5*(X[-1][-1]-X[-1][-2]),Y[0][0]-0.5*(Y[1][0]-Y[0][0]),Y[-1][-1]+0.5*(Y[-1][-1]-Y[-2][-1])]
        #print self._extents
        
    def getData(self,dim=None):
        if dim==None:
            return self._data
        elif (dim==0 or dim==1 or dim==2):
            return self._data[dim]
        else:
            return None
            
    def getZ(self):
        return self.getData(2)
        
#return the shape of the data array      
    def getShape(self):
        return self._data[0].shape
    
    def getRows(self):
        return self._data[0].shape[0]
        
    def getCols(self):
        return self._data[0].shape[1]
    
    def getNoData(self):
        return self._nodata
        
    def getExtent(self):
             
        return self._extents
        
    def subset(self,cmin,cmax,rmin,rmax):
        cmin=max(0,cmin)
        rmin=max(0,rmin)
        cmax=min(self._data[0].shape[1],cmax)
        rmax=min(self._data[0].shape[0],rmax)
        X=self._data[0][rmin:rmax,cmin:cmax]
        Y=self._data[1][rmin:rmax,cmin:cmax]
        Z=self._data[2][rmin:rmax,cmin:cmax]
        return ExplicitRaster(X,Y,Z)
        
    def densify(self,densefac=2):
        print 'Densifying DEM'
    #    print self._data[0:2,:].shape
        #linsiz=self.getRows()*self.getCols()
        x=self._data[0,0,:]
        y=self._data[1,:,0]
        
        z=np.transpose(self._data[2])
        nx=((x.size-1)*densefac)+1
        ny=((y.size-1)*densefac)+1
        
        xd = np.linspace(x[0], x[-1], nx)
        yd = np.linspace(y[0], y[-1], ny)
        
        yv,xv = np.meshgrid(yd,xd)

        #import time
        #t=time.time()
        f=RectBivariateSpline(x, y, z, bbox=[None, None, None, None], kx=1, ky=1, s=0)
        #t=t-time.time()
        #print '\n Calc time 1: ',t,'\n'
        zv=np.zeros((nx,ny))
        
        #print xv.shape,yv.shape,zv.shape
        xv=np.reshape(xv,(nx*ny))
        yv=np.reshape(yv,(nx*ny))
        zv=np.reshape(zv,(nx*ny))
             
        #t=time.time()
        for i in range(xv.size):
            zv[i]=f(xv[i],yv[i])
        #t=t-time.time()
        #print '\n Calc time 2: ',t,'\n'
        
        xv=np.transpose(np.reshape(xv,(nx,ny)))
        yv=np.transpose(np.reshape(yv,(nx,ny)))
        zv=np.transpose(np.reshape(zv,(nx,ny)))
       # xv=np.transpose(xv)
       # yv=np.transpose(yv)
       # zv=np.transpose(zv)
        
        return ExplicitRaster(xv,yv,zv)
        
        #zv=f(xv,yv)
        #griddata(self.data[:,0:1], values, (grid_x, grid_y), method='linear')        
        
    def reportDEM(self):
        print '\nDEM object reporting:\n'
        print 'Data has ',self.getRows(),' rows by ',self.getCols(),' columns'
        print 'No data item is: ',self.getNoData()
        print 'Data Extent Coordinates are [xmin,xmax,ymin,ymax]: ',self.getExtent()
    
        
def load_DEM(demfile):
    
    suffix=demfile.split('.')[-1].upper()
    if suffix==("MAT"):
        return DEM_FromMat(demfile)
    elif suffix==("TIF") or suffix==("TIFF"):
        return DEM_FromTiff(demfile)
    else:
        print 'DEM format (suffix) not supported'
        print 'DEM file: ',demfile,' not read'
        return None
    
def DEM_FromMat(matfile):
    
    print 'Creating DEM object'
    mat = sio.loadmat(matfile)

#==============================================================================
    X=np.ascontiguousarray(mat['X'])
    Y=np.ascontiguousarray(mat['Y'])
    Z=np.ascontiguousarray(mat['Z'])
    print Y[0][0],Y[-1][0]
    if Y[0][0]>Y[-1][0]:
        print 'Flipping input DEM'
        X = np.flipud(X)
        Y = np.flipud(Y)
        Z = np.flipud(Z)
    dem=ExplicitRaster(X,Y,Z)
    return dem 

def DEM_FromTiff(tiffFile):

    dataset = gdal.Open(tiffFile, GA_ReadOnly) 
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    
    geotransform = dataset.GetGeoTransform() 
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    
    band = dataset.GetRasterBand(1)
    
    scanline = band.ReadRaster( 0, 0, band.XSize, band.YSize,band.XSize, band.YSize, band.DataType)
    value = struct.unpack('f' * band.XSize *band.YSize, scanline)
    Z=np.array(value).reshape(rows,cols)
    
    X=np.zeros((rows,cols))
    Y=np.zeros((rows,cols))
    
    originX=originX+(pixelWidth*0.5)
    originY=originY+(pixelWidth*0.5)
    for i in range(rows):
        for j in range(cols):
            X[i,j]=(j*pixelWidth)+originX
            Y[i,j]=(i*pixelHeight)+originY
    
    if Y[0,0]>Y[-1,0]:
        print '\nFlipping input DEM\n'
    
        X=np.flipud(X) 
        Y=np.flipud(Y)
        Z=np.flipud(Z)
        
    dem=ExplicitRaster(X,Y,Z)
    
    return dem
    
def voxelviewshed(dem,viewpoint):
#Calculate a viewshed over a DEM (fast)
#
# USAGE: vis=voxelviewshed(X,Y,Z,camxyz)
#
# INPUTS:
# X,Y,Z: input DEM (regular grid).
# camxyz: 3-element vector specifying viewpoint.
#
# OUTPUT:
#    vis: boolean visibility matrix (same size as Z)
#
    print '\nDoing Voxel viewshed\n'
    
    X=dem.getData(0)
    Y=dem.getData(1)
    Z=dem.getData(2)
    
   # print X.shape
   # print Y.shape
   # print Z.shape
    sz=Z.shape

    dx=abs(X[1,1]-X[0,0])
    dy=abs(Y[1,1]-Y[0,0])

#linearise the grid
    X=np.reshape(X,X.shape[0]*X.shape[1],order='F')
    Y=np.reshape(Y,Y.shape[0]*Y.shape[1],order='F')
    Z=np.reshape(Z,Z.shape[0]*Z.shape[1],order='F')
          
    X=(X-viewpoint[0])/dx
    Y=(Y-viewpoint[1])/dy        
    Z=Z-viewpoint[2]

    d=np.zeros(len(X))
    
    for i in range(len(X)):
        if (np.isnan(X[i]) or np.isnan(Y[i]) or np.isnan(Z[i])):
            d[i]=float('NaN')
        else:
            d[i]=np.sqrt(X[i]*X[i]+Y[i]*Y[i]+Z[i]*Z[i]) #X,Y alone is OK. %+Z.^2;
    
    dint=np.round(np.sqrt(X*X+Y*Y))
    
#  in matlab.  x=atan2(Y,X)+math.pi)/(math.pi*2)
    x=np.empty(X.shape[0])
    #print X.shape[0]
    for i in xrange(X.shape[0]):
        x[i]=(math.atan2(Y[i],X[i])+math.pi)/(math.pi*2)
    
    y=Z/d
    
# the following produces the equivalent of Matlab:
# [~,ix]=sortrows([round(sqrt(X.^2+Y.^2)) x]); 

    ix=np.lexsort((x,dint)).tolist()

# the following produces the equivalent of Matlab:        
#  loopix=find(diff(x(ix))<0);        
    loopix=np.nonzero(np.diff(x[ix])<0)[0]

#As Matlab: vis=true(size(X,1),1);
    vis=np.ones(x.shape, dtype=bool)        
    maxd=np.nanmax(d)        
    N=np.ceil(2.*math.pi/(dx/maxd))

#As Matlab: voxx=(0:N)'/N;
    voxx=np.zeros(N+1)
    n=voxx.shape[0]
    for i in range(n):
        voxx[i]=i*1./(n-1)
        
#As Matlab: voxy=zeros(size(voxx))-inf;
    voxy=np.zeros(n)-1.e+308


    for k in range(loopix.size-1):            
        lp=ix[loopix[k]+1:loopix[k+1]+1]
        lp=lp[-1:]+lp[:]+lp[:1]
        yy=y[lp]
        xx=x[lp]     
        xx[0]=xx[0]-1
        xx[-1]=xx[-1]+1       
        f = interpolate.interp1d(voxx,voxy)
        vis[lp[1:-1]]=f(xx[1:-1])<yy[1:-1]        
        f=interpolate.interp1d(xx,yy)
        voxy=np.maximum(voxy,f(voxx))


    vis=np.reshape(vis,sz,order='F')
    vis.shape=sz
   # print vis.shape
    return vis


    
#Tester code to run if main. Requires suitable files in ..\Data\Images\Velocity test sets
if __name__ == "__main__":    
    from PyTrx_Tests import doDEMTests
    
    doDEMTests() 