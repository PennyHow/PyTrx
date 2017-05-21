# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 21:36:04 2016

@author: nrjh
"""
from CamEnv import CamCalib,CamEnv
from Images import CamImage,ImageSequence,TimeLapse
from DEM import ExplicitRaster,DEM_FromMat
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from Utilities import plotTrackRose,apply_persp_homographyPts
from FileHandler import writeHomographyFile
import cv2
import sys

from scipy import spatial

def FeatureTrackTester():
    
    '''This routine checks the raw feature track testing, and plots
    results of that for a single image.  Soure/ identified points on 'image 0',
    are in red, destination points in green, and backtraked location in orange'''
    
#  define camera soure data, images, mask paths etc    
    camEnvSource='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'
    imagesPath='./Data/Images/Velocity/c1_2014/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

# set up the camera environment
    camEnv1=CamEnv(camEnvSource)
# create a timelapse object
    tl=TimeLapse(imagesPath, camEnv1, maskpath, invmaskpath)
# specifiy the first two images in the sequence
    im0=tl.getImageArrNo(0)
    im1=tl.getImageArrNo(1)
    
    invmask=tl.getInverseMask()
    
# time on
    t=time.time()
# access the featureTrack routine in timelapse directly
# back_thresh is the threshold (in pixels) for excluding points that
# can't be sucessfully back-tracked
# maxpoints is the maximum number of points that will be seeded in 
# identifying features to track
# calcError switches on-off the calculation of tracking error magnitudes 
# (signal to noise ratio), so can be turned off for speed
# quality affect the sharpness of the features originally tracked, a lower
# number means only better intrinsic features are tracked
# mindist is the minimum distance any feature tacked can be from another,
# thus features will always be this far apart.
    outputT=tl.featureTrack(im0,im1,invmask,back_thresh=0.1,calcErrors=True,maxpoints=2000,quality=0.1,mindist=5.0)
# time off
    t=time.time()-t
    print 'Calc track time = ',t,' seconds'
        
    start=np.asarray(outputT[0][0]).T
    end=np.asarray(outputT[0][1]).T
    back=np.asarray(outputT[0][2]).T
       
    xst=start[0,:].tolist()
    yst=start[1,:].tolist()
    xen=end[0,:].tolist()
    yen=end[1,:].tolist()
    xbk=back[0,:].tolist()
    ybk=back[1,:].tolist()
    
    plt.matshow(im0)
    plt.set_cmap('gray')
    plt.scatter(xst,yst,color='red')
    plt.scatter(xen,yen,color='green')
    plt.scatter(xbk,ybk,color='magenta')
    plt.show()
    
def HomographyTester(span=[0,-1]):
    camEnvSource='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'
    imagesPath='./Data/Images/Velocity/KR1_2014_hourly/060000Good/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

    camEnv1=CamEnv(camEnvSource)
    tl=TimeLapse(imagesPath, camEnv1, maskpath, invmaskpath)
    cameraMatrix=camEnv1.getCamMatrixCV2()
    distortP=camEnv1.getDistortCoeffsCv2()
    
    
    imn1=tl._imageSet[span[0]].getImagePath().split('\\')[1]
    im1=tl._imageSet[span[0]].getImageArray()
    im1_dist=tl._imageSet[span[0]].getImageCorr(cameraMatrix, distortP)
    
    for i in range(tl.getLength()-1)[span[0]:span[1]]:
       # im0=self.getImageArrNo(i)
       # im1=self.getImageArrNo(i+1)
        im0=im1
        imn0=imn1
        im0_dist=im1_dist
        im1=tl._imageSet[i+1].getImageArray()
        imn1=tl._imageSet[i+1].getImagePath().split('\\')[1]
        im1_dist=tl._imageSet[i+1].getImageCorr(cameraMatrix, distortP)

        tl._imageSet[i].clearAll()
            
    #im0=tl.getImageArrNo(0)
    #im1=tl.getImageArrNo(1)
        
        t=time.time()
    
# set calcErrors true otherwise we can't calculate/ plot homography
# points

        outputH=tl.homography(im0,im1,back_thresh=1.0,calcErrors=True,maxpoints=2000,quality=0.1,mindist=5.0)
        t=time.time()-t 
    #print outputH
    
#    homogp=tl.apply_persp_homographyPts_array(outputT[0][0],homog,False)
    
#        print 'debug' , len(outputH[1][0]),outputH[1][0].shape
        start=outputH[1][0]
        end=outputH[1][1]
        homog=outputH[1][2]
    
#    end=np.asarray(outputT[0][1]).T
#    back=np.asarray(outputT[0][2]).T
#    homogp=np.asarray(homogp).T
    
#    print start.shape,end.shape,back.shape,homogp.shape
        xst=start[:,0,0].tolist()
        yst=start[:,0,1].tolist()
        xen=end[:,0,0].tolist()
        yen=end[:,0,1].tolist()
    
        xhom=homog[:,0,0].tolist()
        yhom=homog[:,0,1].tolist()    
        
        

    
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        imgplot1 = ax1.matshow(im0_dist)
        ax1.set_title(imn0)
        imgplot2 = ax2.matshow(im1_dist)
        ax2.set_title(imn1)
        
        imgplot1.set_cmap('gray')
        imgplot2.set_cmap('gray')

        ax1.scatter(xst,yst,color='red')
        ax1.scatter(xen,yen,color='green')
        ax1.scatter(xhom,yhom,color='magenta')
        ax2.scatter(xst,yst,color='red')
        ax2.scatter(xen,yen,color='green')
        ax2.scatter(xhom,yhom,color='magenta')
        
        xar,yar=arrowplot(xen,yen,xst,yst,scale=5.0,headangle=15)

        ax1.plot(xar,yar,color='orange')
        ax2.plot(xar,yar,color='orange')
        xar,yar=arrowplot(xen,yen,xhom,yhom,scale=5.0,headangle=15)
        ax1.plot(xar,yar,color='black')
        ax2.plot(xar,yar,color='black')
        
        plt.show()
    
    

def allHomogTest(back_thresh=1.0,calcErrors=True,maxpoints=2000,quality=0.1,mindist=5.0,min_features=4,span=[0,-1]):    
    
    camEnvSource='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'
    imagesPath='./Data/Images/Velocity/KR1_2014_hourly/060000Good/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

    camEnv1=CamEnv(camEnvSource)
    tl=TimeLapse(imagesPath, camEnv1, maskpath, invmaskpath)
    
    
    hgpairs=tl.calcHomographyPairs(back_thresh=back_thresh,calcErrors=calcErrors,maxpoints=maxpoints,quality=quality,mindist=mindist,min_features=min_features,span=span)
    
    
#    for hgp in hgpairs:
#        homogMatrix, points, ptserrors, homogerrors=hgp
#        print homogMatrix
        
    writeHomographyFile(hgpairs,tl,fname='homography_KR1060000c.csv',span=span)
        
    
#def plotHomogImage(image1,image2,CamEnv):
    
def plotDistortion(imageNos=-1,back_thresh=1.0,calcErrors=True,maxpoints=2000,quality=0.1,mindist=5.0,min_features=4):

    '''This is a routine to test homography.  It plots the raw data
    points and also those that are distortion corrected'''
# various input soureces
    camEnvSource='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'
    imagesPath='./Data/Images/Velocity/KR1_2014_hourly/060000Good/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

# create a new camera environment
    camEnv1=CamEnv(camEnvSource)
# create a new timelapse object
    tl=TimeLapse(imagesPath, camEnv1, maskpath, invmaskpath)
    #hgpairs=tl.calcHomographyPairs(back_thresh=back_thresh,calcErrors=calcErrors,maxpoints=maxpoints,quality=quality,mindist=mindist,min_features=min_features)
 
#for each image pair set in the sequence (i.e. one less than total set)
#These are number from zero forward ( so 0 is for image pair 0-1, and the last
# pair is for n-2, n-1

# get the camera matrix and disortion coefficients
    cameraMatrix=camEnv1.getCamMatrixCV2()
    distortP=camEnv1.getDistortCoeffsCv2()
    
    print '\n\n Cam Matric distort p ',cameraMatrix,distortP
    
# This is set up to loop through image pairs, so set a dummy 'next'
# set of image references here first.  This includes, the source
# image path, the image array (data) and a corrected version
# of the image using the correction values supplied
    imn1=tl._imageSet[0].getImagePath()
    im1=tl._imageSet[0].getImageArray()
    im1_dist=tl._imageSet[0].getImageCorr(cameraMatrix, distortP)
    
# iterate for the number of images we have    
    for i in range(tl.getLength()-1)[0:imageNos]:
        im0=im1
        imn0=imn1
        im0_dist=im1_dist

# get the next set of image data        
        im1=tl._imageSet[i+1].getImageArray()
        imn1=tl._imageSet[i+1].getImagePath()
        im1_dist=tl._imageSet[i+1].getImageCorr(cameraMatrix, distortP)

#clear the previous image data to stop gobbling memory
        tl._imageSet[i].clearImage()
        tl._imageSet[i].clearImageArray()
        
        print '\nProcessing homograpy for images: ',imn0,' and ',imn1
        
# grabs the homography between the image pair
        hg=tl.homography(im0,im1,back_thresh=back_thresh,calcErrors=calcErrors,maxpoints=maxpoints,quality=quality,mindist=mindist,calcHomogError=True,min_features=min_features)
        
        homogMatrix, points, ptserrors, homogerrors=hg
        
        #props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        imgplot1 = ax1.matshow(im0)
        ax1.set_title('Raw Image ')
#        p0=cv2.goodFeaturesToTrack(im0,maxpoints,quality,mindist,mask=self.getInverseMask())
        p0=cv2.goodFeaturesToTrack(im0,maxpoints,quality,mindist)
        x=p0[:,0,0]
        y=p0[:,0,1]
        ax1.scatter(x,y,color='red')
                     
        imgplot2 = ax2.matshow(im0_dist)
        ax2.set_title('Image undistorted')
        #print '\n\n Cam Matric distort p ',cameraMatrix,distortP
        cameraMatrix=camEnv1.getCamMatrixCV2()
        distortP=camEnv1.getDistortCoeffsCv2()
        
        size=tl._imageSet[i].getImageSize()
        #print 'image size',size
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortP, (w,h), 1, (w,h))

        
        p0u=cv2.undistortPoints(p0, cameraMatrix, distortP,P=newMat)
               
        
        #print p0u.shape
        x1=p0u[:,0,0]
        y1=p0u[:,0,1]
        #ax2.scatter(x1,y1,color='green')
        ax2.scatter(x1,y1,color='blue')

        imgplot1.set_cmap('gray')
        imgplot2.set_cmap('gray')
        plt.show() 


def VelocityTester(plotcams=True,plotcombined=True,plotmaps=True,plotspeed=True,span=[0,-1]):
    camEnvSource='./Data/GCPdata/CameraEnvironmentData_cam1_test.txt'
    imagesPath='./Data/Images/Velocity/KR1_2014_hourly/060000Good/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

    camEnv1=CamEnv(camEnvSource)
    dem=camEnv1.getDEM()
    
    print dem.getExtent()
    
#    xdmin=285
#    xdmax=365
#    ydmin=140
#    ydmax=220

    xmin=446000
    xmax=451000
    ymin=8754000
    ymax=8760000
    
    demex=dem.getExtent()
    xscale=dem.getCols()/(demex[1]-demex[0])
    yscale=dem.getRows()/(demex[3]-demex[2])
    
    xdmin=(xmin-demex[0])*xscale
    xdmax=((xmax-demex[0])*xscale)+1
    ydmin=(ymin-demex[2])*yscale
    ydmax=((ymax-demex[2])*yscale)+1
        
    demred=dem.subset(xdmin,xdmax,ydmin,ydmax)
    lims=demred.getExtent()
    demred=demred.getZ()
    #demred=demred.getZ()
    
    #camEnv1._setInvProjVars()
    tl=TimeLapse(imagesPath, camEnv1, maskpath, invmaskpath)
    cameraMatrix=camEnv1.getCamMatrixCV2()
    distortP=camEnv1.getDistortCoeffsCv2()  
    
    imn1=tl._imageSet[span[0]].getImagePath().split('\\')[1]
    im1=tl._imageSet[span[0]].getImageArray()
    im1_dist=tl._imageSet[span[0]].getImageCorr(cameraMatrix, distortP)
    
    for i in range(tl.getLength()-1)[span[0]:span[1]]:
       # im0=self.getImageArrNo(i)
       # im1=self.getImageArrNo(i+1)
        im0=im1
        imn0=imn1
        im0_dist=im1_dist
        im1=tl._imageSet[i+1].getImageArray()
        imn1=tl._imageSet[i+1].getImagePath().split('\\')[1]
        im1_dist=tl._imageSet[i+1].getImageCorr(cameraMatrix, distortP)

        tl._imageSet[i].clearAll()
            
    #im0=tl.getImageArrNo(0)
    #im1=tl.getImageArrNo(1)
        
        t=time.time()
    
# set calcErrors true otherwise we can't calculate/ plot homography
# points

        hg=tl.homography(im0,im1,back_thresh=1.0,calcErrors=False,maxpoints=2000,quality=0.1,mindist=5.0,calcHomogError=True,min_features=4)
         

        outputV=tl.calcVelocity(im0,im1,hg,back_thresh=2.0,maxpoints=2000,quality=0.1,mindist=5.0)
        t=time.time()-t    
        
        
        if (plotcams):
            start=outputV[1][0]
            end=outputV[1][1]
            back=outputV[1][2]
            
            xs=start[:,0,0]
            xe=end[:,0,0]
            xb=back[:,0,0]
            ys=start[:,0,1]
            ye=end[:,0,1]
            yb=back[:,0,1]
    
        
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
            imgplot1 = ax1.matshow(im0_dist)
            ax1.set_title(imn0)
            imgplot2 = ax2.matshow(im1_dist)
            ax2.set_title(imn1)
            
            imgplot1.set_cmap('gray')
            imgplot2.set_cmap('gray')
    
            ax1.scatter(xs,ys,color='red')
            ax1.scatter(xe,ye,color='green')
            ax1.scatter(xb,yb,color='magenta')
            ax2.scatter(xs,ys,color='red')
            ax2.scatter(xe,ye,color='green')
            ax2.scatter(xb,yb,color='magenta')
            
            
            
    #        xar,yar=arrowplot(xen,yen,xst,yst,scale=5.0,headangle=15)
    
    #        ax1.plot(xar,yar,color='orange')
    #        ax2.plot(xar,yar,color='orange')
    #       xar,yar=arrowplot(xen,yen,xhom,yhom,scale=5.0,headangle=15)
    #        ax1.plot(xar,yar,color='black')
    #        ax2.plot(xar,yar,color='black')
            
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            
            plt.show()
        
        if (plotcombined):
            start=outputV[1][0]
            end=outputV[1][1]
            back=outputV[1][2]
            
            xs=start[:,0,0]
            xe=end[:,0,0]
            xb=back[:,0,0]
            ys=start[:,0,1]
            ye=end[:,0,1]
            yb=back[:,0,1]
            
            #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
            f, (ax1, ax2) = plt.subplots(1, 2)
            
            imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)
            #imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
            
            Xs=outputV[0][0][:,0]
            Ys=outputV[0][0][:,1]
            Xd=outputV[0][1][:,0]
            Yd=outputV[0][1][:,1]            
                        
            ax2.scatter(Xs,Ys)
            
            xar,yar=arrowplot(Xs,Ys,Xd,Yd,scale=5.0,headangle=15)

            ax2.plot(xar,yar,color='orange')
     
            ilims=tl._imageSet[0].getImageSize()
            ilims=[0,ilims[1],ilims[0],0]      
            
            imgplot1 = ax1.imshow(im0_dist,extent=ilims)
            ax1.set_title(imn0)
                       
            imgplot1.set_cmap('gray')
    
            ax1.scatter(xs,ys,color='red')
            ax1.scatter(xe,ye,color='green')
            ax1.scatter(xb,yb,color='magenta')
            xar,yar=arrowplot(xs,ys,xb,yb,scale=5.0,headangle=15)
            ax1.plot(xar,yar,color='black')
            
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()
            
        if (plotspeed):
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            
            imgplot1 = ax1.imshow(demred, origin='lower', extent=lims)
            imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
            
            XYZs=outputV[0][0]
            XYZd=outputV[0][1]
            
            xd=XYZs[:,0]-XYZd[:,0]
            yd=XYZs[:,1]-XYZd[:,1]
            speed=np.sqrt(xd*xd+yd*yd)

            v_all=np.vstack((XYZs[:,0],XYZs[:,1],XYZd[:,0],XYZd[:,1],speed))
            print 'vshape',v_all.shape
            
            v_all=v_all.transpose()
            print 'vshape',v_all.shape
            
            filtered=filterSparse(v_all,numNearest=12,threshold=2,item=4)
            print 'filtered',filtered.shape
            
            ax1.scatter(filtered[:,0],filtered[:,1])
            
            Xs=filtered[:,0]
            Ys=filtered[:,1]
            Xd=filtered[:,2]
            Yd=filtered[:,3]
            
            ax2.scatter(Xs,Ys)
            
            xar,yar=arrowplot(Xs,Ys,Xd,Yd,scale=5.0,headangle=15)

            ax2.plot(xar,yar,color='orange')
            
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()

        if (plotmaps):
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            
            imgplot1 = ax1.imshow(demred, origin='lower', extent=lims)
            imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
            
            XYZs=outputV[0][0]
            XYZd=outputV[0][1]
            
            
            ax1.scatter(XYZs[:,0],XYZs[:,1])
            ax2.scatter(XYZd[:,0],XYZd[:,1])
            
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()
            

def filterSparse(data,numNearest=12,threshold=2,item=2):
    """ A function to apply a filter to remove noise from a sparse dataset
    as an array. Removes points if they are within a specified value 
    (threshold) from the mean of a specified number of nearest neighbour 
    points (numNearest). The item field identifies which column of the 
    array holds the field to be filtered on.
        
    This function works best if called iteratively, as more than one point 
    may be anomolous compared to neighbouring ranges. """
 
    # Get x and ys and set up as KD tree

    XY=data[:,0:2]
    print 'xy',XY.shape
    tree=spatial.KDTree(XY)
    
    goodset=[]
    for point in XY:
        # Get the number of nearest neighbours
        d,k=tree.query(point, numNearest)
       
        # Get mean and standard deviation of data
        stdev=np.std(data[k[1:],item])
        m=np.mean(data[k[1:],item])
       
        # Get data value for this point
        value=data[k[0],item]
       
        # Append point to goodset if is within threshold range of neighbours
        if (value>(m-(stdev*threshold)) and value<(m+(stdev*threshold))):
            goodset.append(data[k[0]])
        
    return np.array(goodset)


    def filterDensity(self, data,numNearest=5,threshold=10.,absthres=float("inf")):
        """ A function to apply a filter to remove noise from a sparse dataset
        as an array. Removes points if they are within a specified value 
        (threshold) from the mean of a specified number of nearest neighbour 
        points (numNearest). The item field identifies which column of the 
        array holds the field to be filtered on.
            
        This function works best if called iteratively, as more than one point 
        may be anomolous compared to neighbouring ranges. """
 
        # Get x and ys and set up as KD tree

        XY=data[:,0:2]
        tree=spatial.KDTree(XY)
        
        nearestd=[]
        for point in XY:
            d,k=tree.query(point, numNearest)
            nearestd.append(np.mean(d[1:]))
        meand=np.mean(nearestd)
        print 'meand',meand
                
        goodset=[]
        for point in XY:
            # Get the number of nearest neighbours
            d,k=tree.query(point, numNearest)
            locmean=np.mean(d[1:])

            if (locmean<meand*threshold and locmean<absthres):
                goodset.append(data[k[0]])
            
        return np.array(goodset)
        
### if main run all the test sequences 
if __name__ == "__main__":
   
    #FeatureTrackTester()
    #HomographyTester ([54,68]) 
    #allHomogTest()
    
    VelocityTester(plotcams=True,plotcombined=True,plotmaps=True,span=[40,41])   
    #VelocityTester(plotcams=False,plotcombined=True,plotmaps=False,plotspeed=False,span=[50,60]) 
    
    #FlowTester()
    #allHomogTest(min_features=50,maxpoints=2000,span=[0,-1])
    #plotDistortion(imageNos=3,min_features=50,maxpoints=2000)

    print '\nProgram finished'

