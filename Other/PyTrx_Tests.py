# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:03:01 2016

@author: nrjh

This Modules contains test sequences that are accessed by other modules

They're all here to reduce the clutter in the base class modules

"""

from CamEnv import CamCalib,CamEnv
from Images import CamImage,ImageSequence,TimeLapse
from DEM import load_DEM

import matplotlib.pyplot as plt
import numpy as np

### Tester code.  For camera calibration class
### Requires suitable files in ..\Data\Images\Velocity test sets

def doCalibrationTests():
    calibtextfile='./Data/GCPdata/calib/c2_2015_1.txt'
    testCalibrationFromText(calibtextfile)
    calibtextfiles=['./Data/GCPdata/calib/c2_2015_1.txt','./Data/GCPdata/calib/c2_2015_2.txt','./Data/GCPdata/calib/c2_2015_3.txt']
    testCalibrationFromList(calibtextfiles)
    #missing test for mat file calibration read

def testCalibrationFromText(sourceimagefile):
    '''Test calibration reading from single text calibration file'''
    print '\nTesting creation of Calibration object from calibraition text file'
    calib=CamCalib(sourceimagefile)
    calib.reportCalibData()

def testCalibrationFromMat(sourceimagefile):
    '''Test calibration reading from single matlab calibration file'''
    print '\nTesting creation of Calibration object from calibraition text file'
    calib=CamCalib(sourceimagefile)
    calib.reportCalibData()

def testCalibrationFromList(sourceimagefiles):
    '''Test calibration reading from mutiple(list) text calibration file'''
    print '\nTesting creation of Calibration object from calibraition text file'
    calib=CamCalib(sourceimagefiles)
    calib.reportCalibData()        
 
##############################################################
###  Image Module Testing sequences
##############################################################

def doImageTests():
#define the sosource image to use for testing
#modify this path if you don't have this or want to use a different image    
    source='./Data/Images/Velocity/c2_2014_test/IMG_2254_D14-05-08__T12_0.JPG'
    testConfirmImage('lalal')
    testConfirmImage('./Data/GCPdata/CameraEnvironmentData_cam2_2014.txt')
    testRawImage(source)
    testImageBands(source)
    imagesPath='./Data/Images/Velocity/c1_2014/*'
    testImagesequence(imagesPath) 
 
def testConfirmImage(sourceimagefile):
    im=CamImage(sourceimagefile)
    
    if not im.imageGood():
        print 'Image object not good'
    else:
        print 'Image object fully created'
        print type(im)
            
def testRawImage(sourceimagefile):
    print '\nCreating camera image object using greyscale\n'
    im=CamImage(sourceimagefile,'l')
    im.reportCamImageData()
    image=im.getImage()
    plt.imshow(image,origin='lower')
    plt.show()
    
def testImageBands(sourceimagefile):
    #create 4 CamImage objects, one for each band
    print '\nCreating camera image object using greyscale\n'
    im=CamImage(sourceimagefile,'l')
    im.reportCamImageData()

    
    #generate a 4-way test plot to show each band
    print '\nGenerating 4-way test plot of each image band\n'
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)
    imgplot1 = ax1.matshow(im.getImageArray())
    ax1.set_title('Greyscale')
    textstr='Size: '+str(im.getImageSize())+'\nDatetime: '+str(im.getImageTime())
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',bbox=props)
             
    print '\n Changing band to red\n'
    im.changeBand('r')
    imgplot2 = ax2.matshow(im.getImageArray())
    ax2.set_title('Red Band')
    
    print '\n Changing band to green\n'
    im.changeBand('g')
    imgplot3 = ax3.matshow(im.getImageArray())        
    ax3.set_title('Green Band')
    
    print '\n Changing band to blue\n'
    im.changeBand('b')
    imgplot4 = ax4.matshow(im.getImageArray())
    ax4.set_title('Blue Band')
    imgplot1.set_cmap('gray')
    imgplot2.set_cmap('gray')
    imgplot3.set_cmap('gray')        
    imgplot4.set_cmap('gray')
    #plt.suptitle('Image showing calibrated (right) and uncalibrated (left) image', fontsize=14)
    plt.show() 

    print '\nPlotting Finished\n'
    

def testImagesequence(sequence):
    seq=ImageSequence(sequence)
    ims=seq.getImages()
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)

    for im in ims:      
        plt.close()
        im.reportCamImageData()
        image=im.getImage()
        plt.imshow(image,origin='lower')
        textstr='Size: '+str(im.getImageSize())+'\nDatetime: '+str(im.getImageTime())
        plt.text(0.1, 0.9, textstr, fontsize=10,
             verticalalignment='bottom',bbox=props)
        
        plt.show()
        im.clearImage()
        im.clearImageArray()   
        
def doDEMTests():
    
    print 'Testing .mat read'
    source='./Data/GCPdata/dem/dem.mat'
   # testDEM(source)
   # plotDEM(source)
    densifyDEM(source)
    
    print 'Testing .tif read'
    source='./Data/GCPdata/dem/TunaZero1.tif'
    densifyDEM(source)
    
def testDEM(demFile):
    '''Basic DEM load, report and display'''
    
    dem=load_DEM(demFile)
    dem.reportDEM()
    
def plotDEM(demFile):

    dem=load_DEM(demFile)
    demim=dem.getZ()
    plt.figure()
    plt.locator_params(axis = 'x', nbins=8)
    plt.tick_params(axis='both', which='major', labelsize=10)
    imgplot = plt.imshow(demim, origin='lower', extent=dem.getExtent())
    imgplot.set_cmap('gray') 
    plt.show()
    
def densifyDEM(demFile):
    dem=load_DEM(demFile)
    print '\nAttempting to densify DEM by 2\n'
    print 'Original DEM is:'
    dem.reportDEM()
    dem2=dem.densify(2)
    print 'New DEM is:\n'
    dem2.reportDEM()
    
def doCamEnvTests():
    source='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'

    CamEnvFileSetup(source)
    CamEnvTestProjection(source)

def CamEnvFileSetup(camenvfile):       
    '''Simple test to check a CamEnv can be successfully set'''
    CamEnv1=CamEnv(camenvfile)
    CamEnv1.report()

def CamEnvTestProjection(camenvfile):
    CamEnv1=CamEnv(camenvfile)
    print 'Initially CamEnv Projection variables set to: ',CamEnv1._invProjVars
    print 'Setting CamEnv Projection variables'
    #CamEnv1._setInvProjVars()
    xy=CamEnv1.getRefImageSize()
    uv=[[1,1],[1,xy[1]-1],[xy[0]-1,xy[1]-1],[xy[0]-1,1],[xy[0]*.5,xy[1]*.5]]
    uv=np.array(uv)
    xyz=CamEnv1.invproject(uv)
    print uv
    print xyz
    xyz=CamEnv1.invproject(uv)
    print uv
    print xyz

def doTimeLapseTests():
    CamEnvSource='./Data/GCPdata/CameraEnvironmentData_cam2_test.txt'
    imagesPath='./Data/Images/Velocity/c1_2014/*'
    maskpath='./Data/GCPdata/masks/c1_2014_vmask.jpg'
    invmaskpath='./Data/GCPdata/invmasks/c1_2014_inv.jpg'

    testTimeLapseSetupBasic(CamEnvSource,imagesPath)
    testTimeLapseSetupWithMasks(CamEnvSource,imagesPath,maskpath,invmaskpath)
    testTimeLapseTimeSequencing(CamEnvSource,imagesPath)
    
def testTimeLapseSetupBasic(camenvfile,imageSequence):
    print 'Testing basic time laspse setup'
    camEnv1=CamEnv(camenvfile)
    tl=TimeLapse(imageSequence, camEnv1)
    tl.report()

def testTimeLapseSetupWithMasks(camenvfile,imageSequence,maskpath,invmaskpath):
    print 'Testing time laspse setup with masks'
    camEnv1=CamEnv(camenvfile)
    tl=TimeLapse(imageSequence, camEnv1,maskpath,invmaskpath)
    tl.report()
    
def testTimeLapseTimeSequencing(camenvfile,imageSequence):
    print 'Testing time laspse timing settings'
    camEnv1=CamEnv(camenvfile)
    tl=TimeLapse(imageSequence, camEnv1)
    tl.set_Timings()
    print 'TimeLapse timings are:'
    tims=tl.get_Timings()
    imfiles=tl.getFileList()
    for i in range(tl.getLength()):
        print imfiles[i],tims[i]

    
### if main run all the test sequences 
if __name__ == "__main__":
    
#    doCalibrationTests()
#    doImageTests()
    doDEMTests()
    doCamEnvTests()
    doTimeLapseTests()