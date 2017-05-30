'''
PYTRX IMAGES MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This is the Images module of PyTrx. This module contains the object-constructors and
functions for:
(1) Importing and handling image data, specifically RBG, one-band (R, B or G), 
    and grayscale images 
(2) Handling image sequences (i.e. a set of multiple images)
(3) Camera registration from static point feature tracking (referred to here as 
    homography)
(4) Feature tracking and calculating the associated errors

Classes:
CamImage:      A class to represent raw images and holds information on image 
               properties (image size, exif data, bands for subsequent 
               processing).
ImageSequence: A class to model a raw collection of CamImage objects.
TimeLapse:     A class for the processing of an ImageSet to find glacier 
               velocity as points, with methods to track in the xy image plane 
               and project tracks to real-world (xyz) coordinates.


@author: Nick Hulton (Nick.Hulton@ed.ac.uk), 
         Lynne Addison
         Penny How (p.how@ed.ac.uk)
'''

#Import packages
import numpy as np
import operator
from PIL import Image 
from PIL.ExifTags import TAGS
from datetime import datetime
import glob
import imghdr
import os
import cv2
import math

#Import PyTrx modules
from FileHandler import readMask

#------------------------------------------------------------------------------

class CamImage(object):    
    '''A class representing a raw single band (optical RGB or greyscale). This 
    CamImage object is used in subsequent timelapse analysis. The object 
    contains the image data, image path, image dimensions and timestamp 
    (derived from the image Exif data, if available).
        
    Optionally the user can specify whether the red, blue or green values 
    should be used, or whether the images should be converted to grey scale 
    which is the default.  
        
    No image calibration is undertaken.'''
    
    def __init__(self, imagePath, band='l',quiet=2):
        '''CamImage constructor to set image path, read in image data in the 
        specified band and access Exif data. 
        
        Inputs:
        imagePath:      The file path to a given image.
        band:           Specified image band to pass forward
                        'r': red band
                        'b': blue band
                        'g': green band
                        'l': grayscale (default)
        quiet:          Level of commentary during processing. This can be a 
                        integer value between 0 and 2.
                        0: No commentary.
                        1: Minimal commentary.
                        2: Detailed commentary.
                          
        The default grayscale band option ('l') applies an equalization filter 
        on the image whereas the RGB splits are raw RGB. This could be modified 
        to permit more sophisticated settings of RGB combinations and/or 
        filters with file reading.
        
        Class properties:
            self._imageGood:  Boolean denoting whether image file path is 
                              correct.
            self._impath:     String of image path.
            self._band:       String denoting the desired image band.
            self._imageArray: Image data as numpy array
            self._imsize:     Image size as list [rows,columns].
            self._image:      Floating point array of image data [rows,columns].
            self._timestamp:  Python datetime object derived from exif data, if 
                              present (otherwise set as None).
            self._quiet:      Integer value denoting amount of commentary 
                              whilst processing.
        '''
        #Define class properties
        self._imageGood=True
        self._impath = imagePath
        self._band = band.upper()
        self._imageArray=None
        self._image=None
        self._imsize=None
        self._timestamp=None
        self._quiet=quiet
        
        # Checks image file paths 
        success=self._checkImage(imagePath)
        if not success:
            self._imageGood=False
            return

                    
    def imageGood(self):
        '''Return image file path status.'''
        return self._imageGood

        
    def clearImage(self):
        '''Clear memory of image data.'''
        self._image=None


    def clearImageArray(self):
        '''Clear memory of image array data.'''
        self._imageArray=None     

        
    def clearAll(self):
        '''Clear memory of all retained data.'''
        self._image=None
        self._imageArray=None      

     
    def _checkImage(self,path):
        '''Check that the given image file path is correct.'''
        if self._quiet>2:
            print '\nChecking image file ',path
        
        #Check file path using os package
        exists=os.path.isfile(path) 
        if exists:
            
            #Check file type
            ftype=imghdr.what(path)
            if ftype is None:
                if self._quiet>2:
                    print 'File exists but not image type'
                return False
            else:
                if self._quiet>2:
                    print 'File found of image type: ', ftype
                return True

        else:
            if self._quiet>0:            
                print 'File does not exist',path
            return False

        
    def getImageType(self):
        '''Return the image file type.'''
        return imghdr.what(self._impath)        


    def getImagePath(self):
        '''Return the file path of the image.'''        
        return self._impath


    def getImage(self):
        '''Return the image.'''
        if self._image is None:
            self._readImage()
        return self._image

    
    def getImageCorr(self,cameraMatrix, distortP):
        '''Return the image array that is corrected for the specificied 
        camera matrix and distortion parameters.'''
        #Get image array        
        if self._imageArray is None:
            self._readImageData()
            
        if self._quiet>2:    
            print '\nMatrix: ' + str(cameraMatrix)
            print 'Distortion: ' + str(distortP)
            
        size=self.getImageSize()
        if self._quiet>2:        
            print 'Image size: ', size
        
        #Calculate optimal camera matrix 
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, 
                                                    distortP, 
                                                    (w,h), 
                                                    1, 
                                                    (w,h))
        #Correct image for distortion                                                
        corr_image = cv2.undistort(self._imageArray, 
                                   cameraMatrix, 
                                   distortP, 
                                   newCameraMatrix=newMat)
        return corr_image

        
    def getImageArray(self):
        '''Return the image as an array.'''
        if self._imageArray is None:
            self._readImageData()   
        return self._imageArray
        
    def getImageSize(self):
        '''Return the size of the image (which is obtained from the image Exif 
        information).'''        
        if self._imsize is None:
            self._imsize,self._timestamp=self.getExif()
        return self._imsize

        
    def getImageTime(self):
        '''Return the time of the image (which is obtained from the image Exif
        information).'''        
        if self._timestamp is None:
            self._imsize,self._timestamp=self.getExif()
        return self._timestamp
 
       
    def getExif(self):
        '''Return the exif image size and time stamp data from the image. Image
        size is returned as a string (height, width). The time stamp is
        returned as a Python datetime object.'''
        #Get the Exif data
        exif = {}
        if self._image is None:
            self._image=Image.open(self._impath)
        
        info = self._image._getexif()
        
        #Put each item into the Exif dictionary
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exif[decoded] = value            
        imsize=[exif['ExifImageHeight'], exif['ExifImageWidth']]
        
        #Construct datetime object from Exif string
        try:
            timestr = exif['DateTime']
            items=timestr.split()
            date=items[0].split(':')
            time=items[1].split(':')
            timestamp=datetime(int(date[0]),int(date[1]),int(date[2]),
                               int(time[0]),int(time[1]),int(time[2]))
        except:
            if self._quiet>0:
                print ('\nUnable to get valid timestamp for image file: '
                        + self._impath)
            timestamp=None
            
        return imsize, timestamp      

        
    def changeBand(self,band):
        '''Change the band you want the image to represent ('r', 'b', 'g' or
        'l').'''
        self._band=band.upper()
        self._readImageData()
        
        
    def reportCamImageData(self):
        '''Report image data (file path, image size and datetime).'''
        print '\nImage source path: ',self.getImagePath()
        print 'Image size: ',self.getImageSize()
        print 'Image datetime: ',self.getImageTime()

        
    def _readImage(self):
        '''Read image from file path using PIL.'''
        self._image=Image.open(self._impath)
    
    
    def _readImageData(self):
        '''Function to prepare an image by opening, equalising, converting to 
        a desired band or grayscale, then returning a copy.'''        
        #Open image from file using PIL
        if self._image is None:
            self._image=Image.open(self._impath)
        
        #Apply histogram equalisation
        h = self._image.convert("L").histogram()
        lut = []
        for b in range(0, len(h), 256):
            # step size
            step = reduce(operator.add, h[b:b+256]) / 255
            # create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i+b]
        
        #Convert to grayscale or desired band
        gray = self._image.point(lut*self._image.layers)
        if self._band=='R':
            gray,g,b=gray.split()
        elif self._band=='G':
            r,gray,b=gray.split() 
        elif self._band=='B':
            r,g,gray=gray.split() 
        else:
            gray = gray.convert('L')
        
        #Copy image array
        self._imageArray = np.array(gray).copy()
                

#------------------------------------------------------------------------------
        
class ImageSequence(object):
    '''A class to model a raw collection of CamImage objects, which can 
    subsequently be used for making photogrammetric measurements from.
      
    Inputs:
        imageList: The list of images, which can be passed in 3 ways:
                   1) As a list of CamImage objects e.g.
                   imageSet = ImageSet(["image1.JPG", "image2.JPG"])
                   2) As a list of image paths e.g.
                   imageSet = ImageSet([Image("image1.JPG"), 
                              Image("image2.JPG")])
                   imageSet = ImageSet([ImageObject1, ImageObject2])
                   3) As a folder containing images e.g.
                   imageSet = ImageSet("Folder/*")
                   4) (To implement) A file containing the sequence
                   of images to be processed
        band:      Specified image band to pass forward
                    'r': red band
                    'b': blue band
                    'g': green band
                    'l': grayscale (default)
        loadall:   Flag which, if true, will force all images in the sequence
                   to be loaded as images (array) initially and thus not 
                   re-loaded in subsequent processing. This is only advised
                   for small image sequences.
        quiet:     Level of commentary during processing. This can be a integer 
                   value between 0 and 2.
                   0: No commentary.
                   1: Minimal commentary.
                   2: Detailed commentary.
                                               
    Class variables:
        self._quiet:          Integer value denoting amount of commentary 
                              whilst processing.
        self._imageList:      Inputted list of images.
        self._imageSet:       Sequence of CamImage objects.
        self._band:           String denoting the desired image band.
    '''
    def __init__(self, imageList, band='L', loadall=False, quiet=2):
        
        self._quiet=quiet        
        if self._quiet>0:
            print '\nCONSTRUCTING IMAGE SEQUENCE'
        
        self._band=band
        self._imageList=imageList
        
        #Construct image set (as CamImage objects)
        if isinstance(imageList, list): 
            
            #Construction from list of CamImage objects
            if isinstance(imageList[0],CamImage):
                if self._quiet>1:
                    print '\nList of camera images assumed in image sequence'
                    print ' Attempting to add all to sequence'
                self._imageSet = []
                for item in list:
                    if isinstance(item,CamImage):
                        self._imageSet.append(item)
                    else:
                        if self._quiet>1:
                            print ('\nWarning non-image item found in image' 
                                   ' set list specification - item not added')
                return
                
            #Construction from list containing file name strings                
            elif isinstance(imageList[0],str):
                if self._quiet>1:                
                    print '\nList of camera images assumed of image sequence'
                    print ' Attempting to add all to sequence'
                self._loadImageStringSequence(imageList,loadall)
                
            else:
                if self._quiet>1:                
                    print ('\nList item type used to define image list neither' 
                           ' image nor strings (filenames)')
                return None
        
        #Construction from string of file paths
        if isinstance(imageList, str):
            if self._quiet>1:
                print ('\nImage directory path assumed. Searching for images.' 
                       ' Attempting to add all to sequence')
                print imageList
            self._imageList = glob.glob(imageList)
            self._loadImageStringSequence(self._imageList,loadall)
            
            
    def getImageArrNo(self,i):
        '''Get image array i from image sequence.'''
        im=self._imageSet[i]
        arr=im.getImageArray()
        im.clearAll()
        return arr

    
    def getImageObj(self,i):
        '''Get CamImage object i from image sequence.'''
        imo=self._imageSet[i] 
        return imo

        
    def _loadImageStringSequence(self,imageList,loadall):
        '''Function for generating an image set (of CamImage objects) from a 
        list of images. Sequence of image arrays will be loaded if the loadall 
        flag is set to true.'''       
        #Construct CamImage objects
        self._imageSet = []
        for imageStr in imageList:
            im=CamImage(imageStr, self._band)
            
            #Load image arrays if loadall is true
            if im.imageGood():
                self._imageSet.append(im)
                if loadall:
                    im.getImageArray(self)
                    
            else:
                if self._quiet>0:
                    print '\nProblem reading image: ',imageStr
                    print 'Image:',imageStr,' not added to sequence'

                
    def getImages(self):
        '''Return image set (i.e. a sequence of CamImage objects).'''
        return self._imageSet

        
    def getFileList(self):
        '''Return list of image file paths.'''
        return self._imageList

        
    def getLength(self):
        '''Return length of image set.'''
        return len(self._imageSet)


#------------------------------------------------------------------------------

class TimeLapse(ImageSequence):
    '''A class that handles the processing of an ImageSet to find glacier 
    velocity as points, with methods to track and project tracks from the uv
    image plane to xyz real-world coordinates.
    
    This class treats the images as a contigous sequence of name references by
    default
    
    Inputs:
        imageList:      List of images, for the ImageSet object.
        camEnv:         The Camera Environment corresponding to the images, 
                        for the ImageSequence object.
        maskPath:       The file path for the mask indicating the target area
                        for deriving velocities from. If this file exists, the 
                        mask will be loaded. If this file does not exist, then 
                        the mask generation process will load, and the result 
                        will be saved with this path.
        invmaskPath:    As above, but the mask for the stationary feature 
                        tracking (for camera registration/determining
                        camera homography).
        image0:         The image number in the ImageSet from which the 
                        analysis will commence. This is set to the first image 
                        in the ImageSet by default.
        band:           String denoting the desired image band.
        quiet:          Level of commentary during processing. This can be a 
                        integer value between 0 and 2.
                        0: No commentary.
                        1: Minimal commentary.
                        2: Detailed commentary.                          
        loadall:        Flag which, if true, will force all images in the sequence
                        to be loaded as images (array) initially and thus not 
                        re-loaded in subsequent processing. This is only advised
                        for small image sequences. 
        timingMethod:   Method for deriving timings from imagery. By default, 
                        timings are extracted from the image EXIF data.
    
    Class properties:
        self._camEnv:   The camera environment object (CamEnv).
        self._image0:   Integer denoting the image number in the ImageSet from
                        which the analysis will commence.
        self._imageN:   Length of the ImageSet.
        self._mask:     Mark array.
        self._invmask:  Inverse mask array.
        self._timings:  Timing between images.
        self._quiet:    Integer value between denoting amount of commentary 
                        whilst processing.
    '''   
        
    def __init__(self, imageList, camEnv, maskPath=None, invmaskPath=None, 
                 image0=0, band='L', quiet=2, loadall=False, 
                 timingMethod='EXIF'):
        
        ImageSequence.__init__(self, imageList, band, loadall, quiet)
        
        #Set initial class properties
        self._camEnv = camEnv
        self._image0 = image0
        self._imageN = self.getLength()-1
        
        if maskPath is None:
            self._mask = None
        else:
            self._mask = readMask(self.getImageArrNo(0), maskPath)
           
        if invmaskPath is None:
            self._invmask = None
        else:
            self._invmask = readMask(self.getImageArrNo(0), invmaskPath)

        self._timings=None
        #self.set_Timings(timingMethod)

    
    def set_Timings(self,method='EXIF'):
        '''Method to explictly set the image timings that can be used for
        any offset time calculations (e.g. velocity) and also used for any
        output and plots.
        
        For now, only one method exists (EXIF derived) but this is here to 
        permit other ways to define image timings.'''        
        self._timings=[]
        for im in self.getImages():
            self._timings.append(im.getImageTime())
            

    def get_Timings(self):
        '''Return image timings.'''
        return self._timings
  
      
    def getMask(self):
        '''Return image mask.'''
        return self._mask

    
    def getInverseMask(self):
        '''Return inverse mask.'''
        return self._invmask
 
 
    def getCamEnv(self):
        '''Return camera environment object (CamEnv).'''
        return self._camEnv

       
    def featureTrack(self, i0, iN, Mask, back_thresh=1.0, calcErrors=True,
                     maxpoints=50000, quality=0.1, mindist=5.0, 
                     min_features=1):
        '''Function to feature track between two masked images. The
        Shi-Tomasi algorithm with OpenCV's goodFeaturesToTrack function is used
        to initially seed points in the first image. Then, the Lucas Kanade 
        optical flow algorithm is applied using the OpenCV function 
        calcOpticalFlowPyrLK to find these tracked points in the second image.
        
        A backward tracking then tracks back from these to the original
        points, checking if this is within one pixel as a validation measure.
        
        The error associated with this tracking is a signal-to-noise ratio. 
        This is defined as the ratio between the distance 
        originally tracked and the error between the original and back-tracked 
        point.        
        
        This class returns the points in both images as a list, along with the 
        corresponding list of SNR measures.'''
        #Feature tracking set-up parameters
        lk_params = dict( winSize  = (25,25),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | 
                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                                      
        #Find corners of the first image. p0 is returned as an array of shape 
        #(n,1,2), where n is the number of features identified        
        p0=cv2.goodFeaturesToTrack(i0,maxpoints,quality,mindist, mask=Mask)

        #tracked is the number of features returned by goodFeaturesToTrack        
        tracked=p0.shape[0]
                
        #Check if there are enough points to initially track 
        if tracked<min_features:
 
            #Optional commentary
            if self._quiet>0:
                print '\nNot enough features found to track.  Found: ',len(p0)
            return None
            
        else:
            #Track forward from im0 to im1. p1 is returned as an array of shape
            #(n,1,2), where n is the number of features tracked
            p1, status1, error1  = cv2.calcOpticalFlowPyrLK(i0, iN, p0, 
                                                            None, **lk_params) 
                                                            
            #Track backwards from im1 to im0 using the forward-tracked points
            p0r, status0, error0  = cv2.calcOpticalFlowPyrLK(iN, i0, p1, 
                                                             None, **lk_params)         
           
           #Find euclidian pixel distance beween original(p0) and backtracked 
           #(p0r) points and discard point greater than the threshold. This is 
           #a way of checking tracking robustness
            dist=(p0-p0r)*(p0-p0r)
            dist=np.sqrt(dist[:,0,0]+dist[:,0,1])            
            tracked=len(dist)
            good = dist < back_thresh
            
            #Points are boolean filtered by the backtracking success   
            p0=p0[good]
            p1=p1[good]
            p0r=p0r[good]

            #Return None if number of tracked features is under the 
            #min_features threshold
            if p0.shape[0]<min_features:
                if self._quiet>0:
                    print ('\nNot enough features successfully tracked.' 
                           'Tracked: ',len(p0))
                return None
        
       #Optional commentary
        if self._quiet>1:        
            print '\n'+str(tracked)+' features tracked'
            print (str(p0.shape[0]) + ' features remaining after' 
                   'forward-backward error')
           
        #Optional step: Calculate signal-to-noise error               
        #Signal-to-noise is defined as the ratio between the distance 
        #originally tracked and the error between the original and back-tracked 
        #point.
        if calcErrors is True:
            if self._quiet>1:          
                print '\nCalculating tracking errors'
            dist=dist[good]
            length,snr=self._calcTrackErrors(p0,p1,dist)
            
            #Error to contain the original lengths, back-tracking error and snr
            error=[length,dist,snr]
        else:
            error=None
            
        return [p0,p1,p0r], error

        
    def _calcTrackErrors(self,p0,p1,dist):
        '''Function to calculate signal-to-noise ratio with forward-backward 
        tracking data. The distance between the backtrack and original points
        (dist) is assumed to be pre-calcuated.'''               
        #Determine length between the two sets of points
        length=(p0-p1)*(p0-p1)
        length=np.sqrt(length[:,0,0]+length[:,0,1])
        
        #Calculate signal-to-noise ratio
        snr = dist/length
        
        return length,snr

        
    def homography(self, img1, img2, method=cv2.RANSAC,
                   ransacReprojThreshold=5.0, back_thresh=1.0, calcErrors=True,
                   maxpoints=50000, quality=0.1, mindist=5.0,
                   calcHomogError=True, min_features=4):
        '''Function to supplement correction for movement in the camera 
        platform given an image pair (i.e. image registration). Returns the 
        homography representing tracked image movement, and the tracked 
        features from each image.
        Returns: 
            homogMatrix:      The calculated homographic shift for the image 
                                pair (homogMatrix).
            src_pts_corr,
            dst_pts_corr,
            homog_pts:        The original, tracked and back-tracked homography 
                              points.  
            ptserror:         Difference between the original homography points
                              and the back-tracked points.
            homogerror:       Difference between the interpolated homography
                              matrix and the equivalent tracked points
        '''        
        # Feature track between images
        trackdata = self.featureTrack(img1, img2, 
                                      self.getInverseMask(),
                                      back_thresh=1.0, 
                                      calcErrors=calcErrors,
                                      maxpoints=maxpoints, 
                                      quality=quality,
                                      mindist=mindist, 
                                      min_features=min_features) 

        #Pass empty object if tracking insufficient
        if trackdata==None:
            if self._quiet>0:
                print '\nNo features to undertake Homography'
            return None

        #Separate raw tracked points and errors            
        points, ptserrors=trackdata
        
        #Call camera matrix and distortion coefficients from camera environment
        cameraMatrix=self._camEnv.getCamMatrixCV2()
        distortP=self._camEnv.getDistortCoeffsCv2()
        
        #Calculate optimal camera matrix 
        size=img1.shape
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, 
                                                    distortP, 
                                                    (w,h), 1, (w,h))
               
        #Correct tracked points for image distortion. The homgraphy here is 
        #defined forwards (i.e. the points in image 1 are first corrected, 
        #followed by those in image 2)        
        #Correct points in first image  
        src_pts_corr=cv2.undistortPoints(points[0], 
                                         cameraMatrix, 
                                         distortP,P=newMat)
        
        #Correct tracked points in second image
        dst_pts_corr=cv2.undistortPoints(points[1], 
                                         cameraMatrix, 
                                         distortP,P=newMat) 
        
        #Find the homography between the two sets of corrected points
        homogMatrix, mask = cv2.findHomography(src_pts_corr, 
                                               dst_pts_corr, 
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=5.0)
        
        #Optional: calculate homography error
        #Homography error calculated from equivalent set of homography points
        #from original, uncorrected images
        if calcHomogError:
            
            #Optional commentary
            if self._quiet>1:
                print '\nCalculating Homography errors'

            #Apply global homography to source points
            homog_pts=self.apply_persp_homographyPts_array(src_pts_corr,
                                                           homogMatrix,
                                                           False)          
        
            #Calculate offsets between tracked points and the modelled points 
            #using the global homography
            xd=dst_pts_corr[:,0,0]-homog_pts[:,0,0]
            yd=dst_pts_corr[:,0,1]-homog_pts[:,0,1]
            
            #Calculate mean magnitude and standard deviations of the model 
            #homography (i.e. actual point errors)          
            xmean=np.mean(xd)       
            ymean=np.mean(yd)       #Mean should approximate to zero
            xsd=np.std(xd)          
            ysd=np.std(yd)          #SD indicates overall scale of error

            #Compile all error measures    
            homogerrors=([xmean,ymean,xsd,ysd],[xd,yd])
            
        else:
            homogerrors=None
            homog_pts=None
            
        return homogMatrix, [src_pts_corr,dst_pts_corr,homog_pts], ptserrors, homogerrors


    def apply_persp_homographyPts_array(self, pts, homog, inverse=False):    
       '''Funtion to apply a perspective homography to a sequence of 2D 
       values held in X and Y. The perspective homography is represented as a 
       3 X 3 matrix (homog). The source points are inputted as an array. The 
       homography perspective matrix is modelled in the same manner as done so 
       in OpenCV.'''
       #Get empty array that is the same size as the number of points given
       n=pts.shape[0]
       hpts=np.zeros(pts.shape)
       
       if inverse:
           val,homog=cv2.invert(homog)       

       for i in range(n):
           div=1./(homog[2][0]*pts[i][0][0] + homog[2][1]*pts[i][0][1] + 
                   homog[2][2])
           hpts[i][0][0]=(homog[0][0]*pts[i][0][0] + homog[0][1]*pts[i][0][1] +
                          homog[0][2])*div
           hpts[i][0][1]=(homog[1][0]*pts[i][0][0] + homog[1][1]*pts[i][0][1] +
                          homog[1][2])*div
                          
       return hpts 

           
    def apply_persp_homographyPts_list(self, pts, homog, inverse=False):    
       '''Funtion to apply a perspective homography to a sequence of 2D 
       values held in X and Y. The perspective homography is represented as a
       3 X 3 matrix (homog). The source points are inputted as a list. The 
       homography perspective matrix is modelled in the same manner as done so 
       in OpenCV. The output points are returned as a list.'''
       #Create empty output list
       hpts=[]
       
       if inverse:
           val,homog=cv2.invert(homog) 
            
       for p in pts:
           div=1./(homog[2][0]*p[0]+homog[2][1]*p[1]+homog[2][2])
           xh=(homog[0][0]*p[0]+homog[0][1]*p[1]+homog[0][2])*div
           yh=(homog[1][0]*p[0]+homog[1][1]*p[1]+homog[1][2])*div
           hpts.append([xh,yh])
              
       return hpts
       
       ###MERGE apply_persp_homographyPts_list with apply_persp_homographyPts_array


#def homographyCheck(timelapse):
#    ''' Perform homogprahy check for a given timelapse sequence'''

          
    def apply_cam_correction_points(self, points):
        '''Method to apply camera correction to point locations in the XY pixel 
        space.'''
        
        print 'Correcting point locations for ', len(points), ' points'
        
        intrMat,tanCorr,radCorr=self._camEnv.getCalibdata()
        
        ###INCOMPLETE    
    
       
    def calcHomographyPairs(self, back_thresh=1.0, calcErrors=True, 
                            maxpoints=50000, quality=0.1, mindist=5.0,
                            calcHomogError=True, min_features=4, span=[0,-1]):
        '''Method to calculate homography between succesive image pairs in an 
        image sequence.''' 
        #Optional commentary
        if self._quiet>0:
            print 'CALCULATING HOMOGRAPHY'
        
        #Create empty list for output
        pairwiseHomography=[]
        
        #Get first image (image0) path and array data
        imn1=self._imageSet[0].getImagePath()
        im1=self._imageSet[0].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1)[span[0]:span[1]]:
            
            #Re-assign first image in image pair
            im0=im1
            imn0=imn1
            
            #Get second image in image pair (clear memory subsequently)
            im1=self._imageSet[i+1].getImageArray()
            imn1=self._imageSet[i+1].getImagePath()
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
            
            #Optional commentary
            if self._quiet>1:
                print '\nProcessing homograpy for images: ',imn0,' and ',imn1
                
            #Calculate homography and errors from image pair
            hg=self.homography(im0, im1, back_thresh=back_thresh, 
                               calcErrors=calcErrors, maxpoints=maxpoints,
                               quality=quality, mindist=mindist, 
                               calcHomogError=calcHomogError, 
                               min_features=min_features)

            #Assemble all homography information for every image pair set            
            pairwiseHomography.append(hg)
        
        #Returns homography matrix, associated points, point error and 
        #homography error
        return pairwiseHomography

        
    def report(self):
        '''Reporter for TimeLapse object. Returns information concerning:
        - Number of images in image sequence.
        - Mask status.
        - Inverse mask status.
        '''        
        print '\nTimelapse object:'
        
        #Image sequence length
        print self.getLength,' images are defined in the sequence'
        print 'Image list:',self._imageList
        
        #Mask status
        if self._mask==None:
            print ('Mask file (to define mask area in which to track features)' 
                   'not set')
        else:
            print ('Mask file (to define mask area in which to track features)' 
                   'set to: ', self._mask)
        
        #Inverse mask status
        if self._invmask==None:
            print ('Inverse Mask File (to define mask area in which to track' 
                   'features) not set')
        else:
            print ('Inverse Mask File (to define mask area in which to derive' 
                   'background area for homography) set to: ', self._invmask)

        ###ADD MORE CHECKS


    def calcVelocity(self, img1, img2, homography=None, method=cv2.RANSAC, 
                     ransacReprojThreshold=5.0, back_thresh=1.0, 
                     calcErrors=True, maxpoints=50000, quality=0.1, 
                     mindist=5.0, min_features=4):
        '''Function to measure the velocity between a pair of images.'''       
        #Set threshold difference for point tracks
        displacement_tolerance_rel=2.0
        
        #Track points between the image pair
        trackdata = self.featureTrack(img1, img2, self.getMask(),
                                      back_thresh=1.0, calcErrors=calcErrors,
                                      maxpoints=maxpoints, quality=quality,
                                      mindist=mindist, 
                                      min_features=min_features) 
        
        #Pass empty object if tracking was insufficient
        if trackdata==None:
            if self._quiet>0:
                print '\nNo features to undertake velocity measurements'
            return None        
            
        #Separate raw tracked points and errors            
        points, ptserrors=trackdata

        #Get camera matrix and distortion parameters from CamEnv object
        cameraMatrix=self._camEnv.getCamMatrixCV2()
        distortP=self._camEnv.getDistortCoeffsCv2()
        
        #Calculate optimal camera matrix 
        size=img1.shape
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, 
                                                    distortP, 
                                                    (w,h), 1, (w,h))
        
        #Correct tracked points for image distortion. The displacement here is 
        #defined forwards (i.e. the points in image 1 are first corrected, 
        #followed by those in image 2)        
        #Correct points in first image 
        src_pts_corr=cv2.undistortPoints(points[0], 
                                         cameraMatrix, 
                                         distortP,P=newMat)
        
        #Correct points in second image                                         
        dst_pts_corr=cv2.undistortPoints(points[1], 
                                         cameraMatrix, 
                                         distortP,P=newMat) 

        #Calculate homography if desired
        if homography!=None:
            
            #Optional commentary
            if self._quiet>1:
                print '\nHomography not found. Calculating homography.'
            
            #Get homography matrix
            homogMatrix=homography[0]
            
            #Apply perspective homography matrix to tracked points
            tracked=dst_pts_corr.shape[0]
            dst_pts_homog=self.apply_persp_homographyPts_array(dst_pts_corr,
                                                               homogMatrix,
                                                               inverse=True)
            
            #Calculate difference between original and tracked points
            dispx=dst_pts_homog[:,0,0]-src_pts_corr[:,0,0]
            dispy=dst_pts_homog[:,0,1]-src_pts_corr[:,0,1]
            
            #Use pythagoras' theorem to obtain distance
            disp_dist=np.sqrt(dispx*dispx+dispy*dispy)
            
            #Determine threshold for good points using a given displacement 
            #tolerance (defined earlier)
            xsd=homography[3][0][2]
            ysd=homography[3][0][3]
            sderr=math.sqrt(xsd*xsd+ysd*ysd)
            good=disp_dist > sderr * displacement_tolerance_rel
            
            #Keep good points
            src_pts_corr=src_pts_corr[good]
            dst_pts_corr=dst_pts_corr[good]
            dst_pts_homog=dst_pts_homog[good]
            
            #Determine number of points kept
            retained=dst_pts_corr.shape[0]
            
            #Optional commentary
            if self._quiet>1:
                print 'Points removed because of homography uncertainty:'
                print 'Before: ', tracked, ' After: ', retained

        else:
            dst_pts_homog=None
        
        #Project good points (original and tracked) to obtain XYZ coordinates
        uvs=src_pts_corr[:,0,:]
        uvd=dst_pts_homog[:,0,:]

        if self._quiet>1:
            print '\nUndertaking inverse projection'
        xyzs=self._camEnv.invproject(uvs)
        xyzd=self._camEnv.invproject(uvd)

        #Return real-world point positions (original and tracked points),
        #and xy pixel positions (original, tracked, and homography)
        return [[xyzs,xyzd],[src_pts_corr,dst_pts_corr,dst_pts_homog]]
        
        
    def calcVelocities(self, span=[0,-1], homography=True, method=cv2.RANSAC, 
                       ransacReprojThreshold=5.0, back_thresh=1.0, 
                       calcErrors=True, maxpoints=50000, quality=0.1, 
                       mindist=5.0, min_features=4):
        '''Function to calculate velocities between succesive image pairs.'''
        #Optional commentary
        if self._quiet>0:
            print '\nCALCULATING VELOCITIES'
        
        #Create empty lists for velocities and homography
        pairwiseVelocities=[]
        pairwiseHomogs=[]
        
        #Get first image (image0) file path and array data for intial tracking
        imn1=self._imageSet[span[0]].getImagePath().split('\\')[1]
        im1=self._imageSet[span[0]].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1)[span[0]:span[1]]:

            #Re-assign first image in image pair
            im0=im1
            imn0=imn1
                            
            #Get second image in image pair (and subsequently clear memory)
            im1=self._imageSet[i+1].getImageArray()
            imn1=self._imageSet[i+1].getImagePath().split('\\')[1]       
            self._imageSet[i].clearAll()
           
            #Optional commentary
            if self._quiet>0:
                print '\nFeature-tracking for images: ',imn0,' and ',imn1

            #Determine homography between image pairif required
            #Set calcErrors true otherwise we can't calculate/ plot homography
            #points
            if homography is True:        
                hg=self.homography(im0,im1,back_thresh=1.0,calcErrors=True,
                                   maxpoints=2000,quality=0.1,mindist=5.0,
                                   calcHomogError=True,min_features=4)
                 
            #Calculate velocities between image pair
            vel=self.calcVelocity(im0,im1,hg,back_thresh=2.0,maxpoints=2000,
                                  quality=0.1,mindist=5.0)            

            #Append homography and velocity data to output lists            
            pairwiseHomogs.append(hg)            
            pairwiseVelocities.append(vel)
        
        return pairwiseHomogs, pairwiseVelocities
        
        
#------------------------------------------------------------------------------

#Testing code. Requires suitable files in ..\Data\Images\Velocity test sets 
#if __name__ == "__main__":
#    from Development import allHomogTest
#    from PyTrx_Tests import doImageTests,doTimeLapseTests
#
#    #Test image loading capabilities    
#    doImageTests()
#    
#    #Test TimeLapse object initialisation
#    doTimeLapseTests()
#    
#    #Test homography
#    allHomogTest(min_features=50,maxpoints=2000)
    
#    print '\nProgram finished'        

#------------------------------------------------------------------------------