'''
PYTRX MEASURE MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This is the Measure module of PyTrx.

@author: Penny How, p.how@ed.ac.uk
'''

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2
import sys
from PIL import Image as Img
from pylab import array, uint8
import math
import ogr

from FileHandler import readMask
from Images import ImageSequence


#------------------------------------------------------------------------------

class Velocity(ImageSequence):
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
            print '\n\nCALCULATING HOMOGRAPHY'
        
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
            print '\n\nCALCULATING VELOCITIES'
        
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

class Area(Velocity):
    '''A class for processing change in area (i.e. a lake or plume) through an 
    image sequence, with methods to calculate extent change in an image plane 
    (px) and real areal change via georectification.
    
    Inputs:
    imageList:          List of images, for the ImageSet object        
    cameraenv:          Camera environment parameters which can be read into 
                        the CamEnv class as a text file (see CamEnv 
                        documentation for exact information needed)
    method:
    calibFlag:          An indicator of whether images are calibrated, for the 
                        ImageSet object.     
    maxMaskPath:        The file path for the mask indicating the region where 
                        areal extent should be recorded. If this does exist, 
                        the mask will be if this is inputted as a file 
                        directory to a jpg file. If the file does not exist, 
                        then the mask generation process will load and the mask 
                        will be saved to the given directory. If no directory 
                        is specified (i.e. maxMaskPath=None), the mask 
                        generation process will load but the result will not be 
                        saved.
    maxim:              Image with maximum areal extent to be detected (for 
                        mask generation).     
    band:
    quiet:
    loadall:
    timingMethod:
    '''
    
    #Initialisation of Area class object          
    def __init__(self, imageList, cameraenv, method='auto', calibFlag=True, 
                 maxMaskPath=None, maxim=0, band='L', quiet=2, loadall=False, 
                 timingMethod='EXIF'):
                     
        #Initialise and inherit from the TimeLapse class object
        Velocity.__init__(self, imageList, cameraenv, None, None, 0, band,
                           quiet, loadall, timingMethod)

        #Optional commentary        
        if self._quiet>0:
            print '\n\nAREA DETECTION COMMENCING'
            
        self._maximg = maxim
        self._pxplot = None
        self._colourrange = None
        self._threshold = None
        self._method = method
        self._enhance = None
        self._calibFlag = calibFlag
        self._quiet = quiet
        
        self._pxpoly = None
        self._pxextent = None
        self._realpoly = None
        self._area = None
               
        if maxMaskPath is None:
            self._maxMaskPath=None
        else:
            self._maxMaskPath=maxMaskPath
            self._setMaxMask()


    def calcAreas(self, color=False, verify=False):
        '''Get real world areas from an image set. Calculates the polygon 
        extents for each image and the area of each given polygon.'''                
        #Get pixel polygons
        if self._method is 'auto':
            if self._pxpoly is None:
                self.calcExtents(color, verify)
        
        elif self._method is 'manual':
            if self._pxpoly is None:
                self.manualExtents()
        
        #Optional commentary
        if self._quiet>0:
            print '\n\nCOMMENCING GEORECTIFICATION OF AREAS'
            
        xyz = []
        area = []
        
        for p in self._pxpoly:
            pts, a = self.calcArea(p)
            xyz.append(pts)
            area.append(a)
        
        self._realpoly = xyz
        self._area = area
        
        return self._realpoly, self._area


    def calcArea(self, pxpoly):
        '''Get real world areas from px polygons defined in one image.'''                    
        #Create outputs
        xyz = []   
        area = []                               

        #Project image coordinates
        for p in pxpoly:                        
            allxyz = self._camEnv.invproject(p)
            xyz.append(allxyz)                  
        
        #Create polygons
        rpoly = self._ogrPoly(xyz)              
        
        #Determine area of each polygon
        for r in rpoly:
            area.append(r.Area())               
           
        return xyz, area  
          

    def calcExtents(self, color=False, verify=False):
        '''Get pixel extent from a series of images. Return the extent polygons
        and cumulative extent values (px).'''       
        #Only define color range once
        if color is False:        
            #Get image with maximum extent
            if self._calibFlag is True:
                cameraMatrix=self._camEnv.getCamMatrixCV2()
                distortP=self._camEnv.getDistortCoeffsCv2()
                maximg=self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                                 distortP)        
            else:
                maximg = self._imageSet[self._maximg].getImageArray() 

            #Get image name
            maximn=self._imageSet[self._maximg].getImagePath().split('\\')[1]
              
            #Get mask and mask image if present
            if self._mask is not None:
                maximg = self._maskImg(maximg)        
                   
            #get colour range for extent detection
            if self._colourrange is None:
                if self._enhance is not None:
                    maximg = self.enhanceImg(maximg)
                self.defineColourrange(maximg, maximn)    
            
        #Set up output datasets
        areas=[]        
        px=[]
                       
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()):
            if self._calibFlag is True:
                cameraMatrix=self._camEnv.getCamMatrixCV2()
                distortP=self._camEnv.getDistortCoeffsCv2()
                img1 = self._imageSet[i].getImageCorr(cameraMatrix, 
                                                      distortP)
            else:
                img1=self._imageSet[i].getImageArray()

            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
               
            img2 = np.copy(img1)
            if self._mask is not None:
                img2 = self._maskImg(img2)
            if self._enhance is not None:
                img2 = self._enhanceImg(img2)
            if color is True:
                self.defineColourrange(img2, imn)
            if self._quiet>0:
                print '\nCalculating extent for ' + imn
            polys,extent = self.calcExtent(img2)        
            areas.append(polys)
            px.append(extent)

            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
            
        #Retain pixel areas in object
        self._pxpoly = areas
        self._pxextent = px
        
        #Manually verify polygons if verify flag is True
        if verify is True:
            self.verifyExtents()

        #Return all xy coordinates and pixel extents                 
        return self._pxpoly, self._pxextent

        
    def verifyExtents(self):
        '''Manually verify all polygons in images.'''
        #Create output
        verified = []
        update_ext = []
                
        #Verify pixel polygons in each image        
        for i,px in zip((range(self.getLength())),self._pxpoly):
            
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img1=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                    self._camEnv.getDistortCoeffsCv2())      
            else:
                img1=self._imageSet[i].getImageArray()            
            
            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
            
            #Verify polygons
            img2 = np.copy(img1)
            
            if 1:             
                if self._quiet>0:                
                    print '\n\nVERIFYING DETECTED AREAS'
                verf = []
                
                def onpick(event):
                    v = []
                    thisline = event.artist
                    xdata = thisline.get_xdata()
                    ydata = thisline.get_ydata()
                    for x,y in zip(xdata,ydata):
                        v.append([x,y])
                    v2=np.array(v, dtype=np.int32).reshape((len(xdata)),2)
                    verf.append(v2)
                    ind=event.ind
                    print ('Verified extent at ' + 
                           str(np.take(xdata, ind)[0]) + ', ' + 
                           str(np.take(ydata, ind)[0]))
                    
                fig, ax1 = plt.subplots()
                fig.canvas.set_window_title(imn + ': Click on valid areas.')
                ax1.imshow(img2, cmap='gray')
                if self._pxplot is not None:
                    ax1.axis([self._pxplot[0],self._pxplot[1],
                              self._pxplot[2],self._pxplot[3]])
                for a in px:
                    x=[]
                    y=[]
                    for b in a:
                        for c in b:
                            x.append(c[0])
                            y.append(c[1])
                    line = Line2D(x, y, linestyle='-', color='y', picker=True)
                    ax1.add_line(line)
                fig.canvas.mpl_connect('pick_event', onpick)
            
            plt.show()           
            
            vpx=[]
            vpx=verf
            verified.append(vpx)
            
            #Update extents
            h = img2.shape[0]
            w = img2.shape[1]
            px_im = Img.new('L', (w,h), 'black')
            px_im = np.array(px_im) 
            cv2.drawContours(px_im, px, -1, (255,255,255), 4)
            for p in px:
                cv2.fillConvexPoly(px_im, p, color=(255,255,255))           
            output = Img.fromarray(px_im)
            pixels = output.getdata()
            values = []    
            for px in pixels:
                if px > 0:
                    values.append(px)
            pxext = len(values) #total lake extent
            update_ext.append(pxext)
    
            #Clear memory            
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
        
        #Reset method (which indicates how the data is structured)
        self._method='manual'
        
        #Rewrite pixel polygon and extent data
        self._pxpoly = verified
        self._pxextent = update_ext
    
        
    def manualExtents(self):
        '''Get manual pixel extent from a series of images. Return the 
        extent polygons and cumulative extent values (px).'''                 
        #Set up output dataset
        areas=[]        
        px=[]
                
        #Crop all images, find extent in all images        
        for i in (range(self.getLength())):           
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                   self._camEnv.getDistortCoeffsCv2())      
            else:
                img=self._imageSet[i].getImageArray()          

            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
            
            polys,extent = self.manualExtent(img, imn)        
            areas.append(polys)
            px.append(extent)
            
            #Clear memory
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
    
        #Return all extents, all cropped images and corresponding image names
        self._pxpoly = areas
        self._pxextent = px
        
        return self._pxpoly, self._pxextent


    def calcExtent(self, img):
        '''Get extent from a given image using a predefined RBG colour range. 
        The colour range is then used to extract pixels within that range 
        using the OpenCV function inRange. If a threshold has been set (using
        the setThreshold function) then only nth polygons will be retained.'''         
        #Get upper and lower RBG boundaries from colour range
        upper_boundary = self._colourrange[0]
        lower_boundary = self._colourrange[1]
    
        #Transform RBG range to array    
        upper_boundary = np.array(upper_boundary, dtype='uint8')
        lower_boundary = np.array(lower_boundary, dtype='uint8')
    
        #Extract extent based on rbg range
        mask = cv2.inRange(img, lower_boundary, upper_boundary)
        
        #Speckle filter to remove noise
        cv2.filterSpeckles(mask, 1, 30, 2)
        
        #Polygonize extents        
        polyimg, line, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)
        
        #Optional commentary
        if self._quiet>1:
            print 'Detected %d regions' % (len(line))
        
        #Append all polygons from the polys list that have more than 
        #a given number of points     
        pxpoly = []
        for c in line:
            if len(c) >= 40:
                pxpoly.append(c)
        
        #If threshold has been set, only keep the nth longest polygons
        if self._threshold is not None:
            if len(pxpoly) >= self._threshold:
                pxpoly.sort(key=len)
                pxpoly = pxpoly[-(self._threshold):]        

        #Optional commentary
        if self._quiet>1:        
            print 'Kept %d regions' % (len(pxpoly))
        
        #Get image dimensions
        h = img.shape[0]
        w = img.shape[1]
        
        #THIS DOESN'T SEEM TO WORK PROPERLY YET. DOES NOT FILL POLYGONS 
        #CORRECTLY
        px_im = Img.new('L', (w,h), 'black')
        px_im = np.array(px_im) 
        cv2.drawContours(px_im, pxpoly, -1, (255,255,255), 4)
        for p in pxpoly:
            cv2.fillConvexPoly(px_im, p, color=(255,255,255))           
        output = Img.fromarray(px_im)
        pixels = output.getdata()
        values = []    
        for px in pixels:
            if px > 0:
                values.append(px)
        pxextent = len(values) #total lake extent
        pxcount = len(pixels) #total img area
        
        #Optional commentary
        if self._quiet>1:
            print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px)'
        
        return pxpoly, pxextent
        

    def manualExtent(self, img, imn):
        '''Manually define extent by clicking around region in target image.'''
         #Manual interaction to select lightest and darkest part of the region
#        pts=[]        
#        while len(pts) < 3:
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Click around region. Press enter '
                                    'to record points.')
        plt.imshow(img, origin='upper', cmap='gray')
        if self._pxplot is not None:
            plt.axis([self._pxplot[0],self._pxplot[1],
                      self._pxplot[2],self._pxplot[3]])                   
        pts = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, 
                        mouse_pop=3, mouse_stop=2)
        
        #Optional commentary
        if self._quiet>1:
            print '\n' + imn + ': you clicked ' + str(len(pts)) + ' points'
        
        plt.show()
        plt.close()
        
#            #Reboot window if =<2 points are recorded
#            if len(pts) > 3:
#                pts = []
            
        #Create polygon if area has been recorded   
        try:
            #Complete the polygon             
            pts.append(pts[0])        
            ring = ogr.Geometry(ogr.wkbLinearRing)   
            for p in pts:
                ring.AddPoint(p[0],p[1])
            p=pts[0]
            ring.AddPoint(p[0],p[1])
            pxpoly = ogr.Geometry(ogr.wkbPolygon)
            
            #Create polygon ring with calculated area           
            pxpoly.AddGeometry(ring) 
            pxextent = pxpoly.Area()
        except:
            pxextent = 0
                   
        #Get image dimensions
        h = img.shape[0]
        w = img.shape[1]
        pxcount = h*w
        
        #Optional commentary
        if self._quiet>1:
            print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px)'    
        
        #Convert pts list to array
        pts = np.array(pts)           
        pts=[pts]
        
        return pts, pxextent


    def setEnhance(self, diff, phi, theta):
        '''Set image enhancement parameters. See enhanceImg function for 
        detailed explanation of the parameters.
        -diff: 'light' or 'dark'
        -phi: a value between 0 and 1000
        -theta: a value between 0 and 1000'''
        self._enhance = diff, phi, theta
        

    def seeEnhance(self):
        '''Enhance image using an interactive plot. WARNING: this function will
        significantly slow down your computer. Only use if your computer can
        handle it.'''
        #Get image with maximum areal extent to detect
        if self._calibFlag is True:
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCv2()
            img = self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                               distortP)
        else:
            img = self._imageSet[self._maximg].getImageArray()         

        fig,ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax = plt.subplot(111)
        ax.imshow(img)
        
        diff = 'light'  
                
        #Inititalise sliders for phi and theta
        axcolor = 'lightgoldenrodyellow'
        axphi  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        axtheta = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
        sphi = Slider(axphi, 'Phi', 1, 100.0, valinit=1)
        stheta = Slider(axtheta, 'Theta', 1, 50.0, valinit=1)
        
        def update(val):        #Update image when phi and theta are changed
            phi = sphi.val
            theta = stheta.val
            self.setEnhance(diff, phi, theta)
            img1 = self.enhanceImg(img)
            ax.imshow(img1)
                        
        sphi.on_changed(update)
        stheta.on_changed(update)
               
        #Initialise reset button
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        
        def reset(event):
            sphi.reset()
            stheta.reset()            
        button.on_clicked(reset)

#        #Initialise toggle button for diff variable
#        rax = mp.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
#        radio = RadioButtons(rax, ('light', 'dark'), active=0
#        
#        def colorfunc(label):
#            hzdict = {'light': 'light', 'dark': 'dark'}
#            diff = hzdict[label]
#            self.setEnhance 
#            print 'Enhancement set to preference ' + diff + ' pixels'           
#        radio.on_clicked(colorfunc)   
                
        plt.show()
        plt.close()
        
        self.setEnhance('light', sphi.val, stheta.val)
         
        
    def setPXExt(self,xmin,xmax,ymin,ymax):
        '''Set plotting extent (makes it easier for defining colour ranges and
        verifying areas).'''
        self._pxplot = [xmin,xmax,ymin,ymax]


    def setThreshold(self, number):
        '''Set threshold for number of polgons kept from an image.'''
        self._threshold = number
                                

    def setColourrange(self, upper, lower):
        '''Manually define the RBG colour range that will be used to filter
        the image/images. Input the upper boundary (i.e. the highest value) 
        first.'''
        if self._quiet>0:        
            print '\nColour range defined from given values:'
            print 'Upper RBG boundary: ', upper
            print 'Lower RBG boundary: ', lower            
        self._colourrange = [upper, lower]
        
                
    def defineColourrange(self, img, imn):
        '''Define colour range manually by clicking on the lightest and 
        darkest regions of the extent that will be tracked. Click the lightest
        part first, then the darkest.
        Operations:
        -Left click to select
        -Right click to undo
        -Close the image window to continue
        -The window automatically times out after two clicks'''
        #Manual interaction to select lightest and darkest part of the region
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Click lightest colour and darkest' 
                                    'colour')
        
        plt.imshow(img, origin='upper')

        if self._pxplot is not None:
            plt.axis([self._pxplot[0],self._pxplot[1],
                      self._pxplot[2],self._pxplot[3]])
            
        colours = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, 
                            mouse_pop=3, mouse_stop=2)
        
        #Optional commentary
        if self._quiet>0:
            print '\n' + imn + ': you clicked ', colours
        
        plt.show()
            
        #Obtain coordinates from selected light point
        col1_y = colours[0][0]
        col1_x = colours[0][1]
    
        #Obtain coordinates from selected dark point
        col2_y = colours[1][0]
        col2_x = colours[1][1]
    
        #Get RBG values from given coordinates
        col1_rbg = self._getRBG(img, col1_x, col1_y)
        col2_rbg = self._getRBG(img, col2_x, col2_y) 
        
        #Assign RBG range based on value of the chosen RBG values
        if col1_rbg > col2_rbg:
            upper_boundary = col1_rbg
            lower_boundary = col2_rbg
        elif col2_rbg > col1_rbg:
            upper_boundary = col2_rbg
            lower_boundary = col1_rbg
        elif col1_rbg == col2_rbg:
            upper_boundary = col1_rbg
            lower_boundary = col2_rbg
        else:
            print 'Unrecognised RBG range.'
            sys.exit(1)
            
        if self._quiet>1:
            print 'Colour range found from manual selection'
            print 'Upper RBG boundary: ' + str(upper_boundary)
            print 'Lower RBG boundary: ' + str(lower_boundary)

        #Store RBG range
        self._colourrange = [upper_boundary, lower_boundary]
        
        
    def _getRBG(self, img, x, y):
        '''Return the RBG value for a given point (x,y) in an image (img).'''
        RBG = img[x,y]    
        return RBG


    def _setMaxMask(self):
        '''Set mask for tracking areal extent using the image with the 
        largest extent function. Click around the target extent using the left
        click on a mouse. Right click will delete the previous point. Press
        Enter when you have completed the mask that you wish to use - this will
        save the points. Then exit the window to continue the program.
        '''
        #Get image with maximum areal extent to detect
        if self._calibFlag is True:
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCv2()
            maxi = self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                               distortP)
        else:
            maxi = self._imageSet[self._maximg].getImageArray()
            
        #Define mask on image with maximum areal extent
        self._mask = readMask(maxi, self._maxMaskPath)

        
    def _maskImg(self, img):
        '''Mask images using the largest extent mask (boolean object). Unlike 
        the masking function in the TimeLapse class, the boolean mask is used
        to reassign overlapping image pixels to zero. The masking function in 
        the TimeLapse class uses a numpy masking object (numpy.ma).'''            
        #Mask the glacier
        booleanMask = np.array(self._mask, dtype=bool)
        booleanMask = np.invert(booleanMask)
        
        #Copy properties of img
#        img2 = np.copy(img)
        
        #Mask extent image with boolean array
        np.where(booleanMask, 0, img) #fit arrays to each other
        img[booleanMask] = 0 #mask image with boolean mask object

        return img
        
       
    def _enhanceImg(self, img):
        '''Change brightness and contrast of image using phi and theta variables.
        Change phi and theta values accordingly.
        Enhancement parameters:
        -diff: Inputted as either 'light or 'dark', signifying the intensity of
        the image pixels. 'light' increases the intensity such that dark pixels 
        become much brighter and bright pixels become slightly brighter. 
        'dark' decreases the intensity such that dark pixels become much darker
        and bright pixels become slightly darker.
        -phi: defines the intensity of all pixel values
        -theta: defines the number of "colours" in the image, e.g. 3 signifies
        that all the pixels will be grouped into one of three pixel values'''                          
        #Extract enhancement parameters from enhance object
        diff = self._enhance[0]
        phi = self._enhance[1]
        theta = self._enhance[2]
        
        #Define maximum pixel intensity
        maxIntensity = 255.0 #depends on dtype of image data 
        
        if diff == 'light':        
            #Increase intensity such that dark pixels become much brighter
            #and bright pixels become slightly brighter
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
            img1 = array(img1, dtype = uint8)
        
        elif diff == 'dark':        
            #Decrease intensity such that dark pixels become much darker and 
            #bright pixels become slightly darker          
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
            img1 = array(img1, dtype=uint8)
        
        else:
            if self._quiet>0:            
                print '\nInvalid diff variable' 
                print 'Re-assigning diff variable to "light"'
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
            img1 = array(img1, dtype = uint8)
            
        return img1   
   
    
    def _ogrPoly(self, xyz):
        '''Get real world OGR polygons (.shp) from xyz poly pts with real world 
        points which are compatible with mapping software (e.g. ArcGIS)'''                       
        #Create geometries from xyz coordinates using ogr        
        polygons = []        
        for shape in xyz:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for pt in shape:
                if np.isnan(pt[0]) == False:                   
                    ring.AddPoint(pt[0],pt[1],pt[2])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            polygons.append(poly)

        return polygons



class Line(Area):
    '''Class for point and line measurements.'''             
    def __init__(self, imageList, cameraenv, method='manual', calibFlag=True,
                 band='L', quiet=2, loadall=False, timingMethod='EXIF'):

        Area.__init__(self, imageList, cameraenv, method, calibFlag, None, 0, 
                      band, quiet, loadall, timingMethod)
                     
        self._method = method
        self._enhance = None
        
        self._pxpts = None
        self._pxline = None
        self._realpts = None
        self._realline = None
      

    def manualLinesPX(self):
        '''Get manual pixel line from a series of images. Return the 
        line pixel coordinates and pixel length.'''                 
        #Set up output dataset
        pts=[]        
        lines=[]
        count=1

        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()):
            if self._calibFlag is True:
                cameraMatrix=self._camEnv.getCamMatrixCV2()
                distortP=self._camEnv.getDistortCoeffsCv2()
                img1 = self._imageSet[i].getImageCorr(cameraMatrix, 
                                                      distortP)
            else:
                img1=self._imageSet[i].getImageArray()

            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
            
            #Define line
            pt,length = self.manualLinePX(img1, imn)
            if self._quiet>1:            
                print '\nLine defined in ' + imn                
                print 'Img%i line length: %d px' % (count, length.Length())
                print 'Line contains %i points\n' % (length.GetPointCount())
            
            #Append line
            pts.append(pt)
            lines.append(length)
            count=count+1
    
        #Return all line coordinates and length
        self._pxpts = pts
        self._pxline = lines        
        return self._pxpts, self._pxline
        

    def manualLinePX(self, img, imn):
        '''Manually define a line by clicking in the target image.'''
         #Manual interaction to define line
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Define line.' 
                                    'Press enter to record points.')
        plt.imshow(img, origin='upper',cmap='gray')        
        pts = plt.ginput(n=0, timeout=0, show_clicks=True, 
                         mouse_add=1, mouse_pop=3, mouse_stop=2)

        #Optional commentary
        if self._quiet>2:            
            print ('\nYou clicked ' + str(len(pts)) + ' in image ' + imn)
        
        #Plot
        plt.show()
        plt.close()
        
        #Create OGR line object
        line = self._ogrLine(pts)
        
        #Re-format point coordinates
        pts = np.squeeze(pts)

        return pts, line


    def _ogrLine(self, pts):
        '''Create OGR line from a set of pts.'''              
        line = ogr.Geometry(ogr.wkbLineString)   
        for p in pts:
            line.AddPoint(p[0],p[1])
        
        return line

   
    def calcLinesXYZ(self):
        '''Get real world lines from an image set. Calculates the line 
        coordinates and length of each given set of pixel points.'''
        #Get pixel points if not already defined
        if self._pxpts is None:
            self.manualLinesPX()
        
        #Set output variables and counter
        rpts = []
        rline = []        
        count=1
        
        #Project pixel coordinates to obtain real world coordinates and lines
        for p in self._pxpts:
            pts, line = self.calcLength(p)
            print 'Img ' + str(count) + ' line length: ' + str(line.Length())
            rpts.append(pts)
            rline.append(line)
            count=count+1

        #Return real line coordinates and line objects                
        self._realpts = rpts
        self._realline = rline        
        return self._realpts, self._realline


    def calcLineXYZ(self, px):
        '''Get real world line from px points defined in one image.'''                    
        #Create outputs        
        rpts = []  

        #Project image coordinates           
        xyz = self._CamEnv.invproject(px)
        rpts.append(xyz)              
        rpts = np.squeeze(rpts)

        #Create polygons        
        rline = self._ogrLine(rpts)    
           
        return rpts, rline  
        

#------------------------------------------------------------------------------


#Testing code. Requires suitable files in ..\Data\Images\Velocity test sets 
#if __name__ == "__main__":
#    from Development import allHomogTest
#    from PyTrx_Tests import doTimeLapseTests
    
#    #Test TimeLapse object initialisation
#    doTimeLapseTests()
#    
#    #Test homography
#    allHomogTest(min_features=50,maxpoints=2000)
    
#    print '\nProgram finished'  


#------------------------------------------------------------------------------