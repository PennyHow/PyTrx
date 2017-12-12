'''
PYTRX MEASURE MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This is the Measure module of PyTrx. It handles the functionality for obtaining 
measurements from oblique time-lapse imagery, namely velocities, areas and
distances. Specifically, this module contains functions for:
(1) Performing camera registration from static point feature tracking (referred 
    to here as homography).
(2) Calculating surface velocities derived from feature tracking, with 
    associated errors and signal-to-noise ratio calculated.
(3) Performing automated and manual detection of areal extents in oblique 
    imagery.
(4) Performing manual detection of lines in oblique imagery.
(5) Determining real-world surface areas and distances from oblique imagery.

Classes
Velocity:                       A class for the processing of an ImageSet to 
                                determine pixel displacements and real-world 
                                velocities from a sparse set of points, with 
                                methods to track in the xy image plane and 
                                project tracks to real-world (xyz) coordinates.
Area:                           A class for processing change in area (i.e. a 
                                lake or plume) through an image sequence, with 
                                methods to calculate extent change in an image 
                                plane (px) and real areal change via 
                                georectification.  
Line:                           A class for handling lines/distances (e.g. 
                                glacier terminus position) through an image 
                                sequence, with methods to manually define pixel 
                                lines in the image plane and georectify them to 
                                generate real-world coordinates and distances. 
                                The Line class object primarily inherits from 
                                the Area class.

Key functions in Velocity
calcHomographyPairs():          Method to calculate homography between 
                                succesive image pairs in an image sequence.
calcVelocities():               Method to calculate velocities between 
                                succesive image pairs in an image sequence.
                               
Key functions in Area
calcAutoAreas():                Method to obtain real world areas from an image 
                                set. Calculates the polygon extents for each 
                                image and the area of each given polygon.
calcManualAreas():              Method to obtain real world areas from an image 
                                set by manually detecting them in a point-and-
                                click manner.        
calcAutoExtents():              Method to obtain pixel extent from a series of 
                                images. Return the extent polygons and 
                                cumulative extent values (px).
calcManualExtents():            Method to manually select pixel extents from a 
                                series of images. Return the extent polygons 
                                and cumulative extent values (px).
verifyExtents():                Method to manuall verify all detected polygons 
                                in an image sequence.
                                                           
Key functions in Line
calcManualLinesXYZ():           Method to obtain real world lines/distances 
                                from an image set. Calculates the line 
                                coordinates and length of each given set of 
                                pixel points.
calcManualLinesPX():            Method to manually define a pixel line through
                                a series of images. Returns the line pixel 
                                coordinates and pixel length.
                                                               
@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton 
         Lynne Addison
'''

#Import packages
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

#Import PyTrx functions and classes
from FileHandler import readMask
from Images import ImageSequence

#------------------------------------------------------------------------------

class Velocity(ImageSequence):
    '''A class for the processing of an ImageSet to determine pixel 
    displacements and real-world velocities from a sparse set of points, with 
    methods to track in the xy image plane and project tracks to real-world 
    (xyz) coordinates.
    
    This class treats the images as a contigous sequence of name references by
    default.
    
    Args
    imageList:          List of images, for the ImageSet object.
    camEnv:             The Camera Environment corresponding to the images, 
                        for the ImageSequence object.
    maskPath:           The file path for the mask indicating the target area
                        for deriving velocities from. If this file exists, the 
                        mask will be loaded. If this file does not exist, then 
                        the mask generation process will load, and the result 
                        will be saved with this path.
    invmaskPath:        As above, but the mask for the stationary feature 
                        tracking (for camera registration/determining
                        camera homography).
    image0:             The image number in the ImageSet from which the 
                        analysis will commence. This is set to the first image 
                        in the ImageSet by default.
    band:               String denoting the desired image band.
    quiet:              Level of commentary during processing. This can be a 
                        integer value between 0 and 2.
                        0: No commentary.
                        1: Minimal commentary.
                        2: Detailed commentary.                          
    loadall:            Flag which, if true, will force all images in the 
                        sequence to be loaded as images (array) initially and 
                        thus not re-loaded in subsequent processing. This is 
                        only advised for small image sequences. 
    timingMethod:       Method for deriving timings from imagery. By default, 
                        timings are extracted from the image EXIF data.
    
    Class properties
    self._xyzvel:       XYZ velocities. 
    self._xyz0:         XYZ positions of points in the first of each image 
                        pair.
    self._xyz1:         XYZ positions of points in the second of each image
                        pair.
    self._uvvel:        UV velocities (px).
    self._uv0:          UV positions of points in the first of each image pair.
    self._uv1:          UV positions of points in the second of each image 
                        pair.
    self._uv1corr:      UV positions of corrected points in the second of each 
                        image pair that have been (i.e. corrected using image
                        registration).               
    self._homogmatrix:  Homography matrix.
    self._homogpts0:    Seeded points (UV) in the first of each image pair.
    self._homogpts1:    Tracked points (UV) in the second of each image pair.
    self._homogpts1corr:Corrected points (UV) in the second of each image pair.
    self._homogptserr:  Error associated with the tracked points.
    self._homogerr:     Error associated with the homography models.    
    self._camEnv:       The camera environment object (CamEnv).
    self._image0:       Integer denoting the image number in the ImageSet from
                        which the analysis will commence.
    self._imageN:       Length of the ImageSet.
    self._mask:         Mask array.
    self._invmask:      Inverse mask array.
    self._timings:      Timing between images.
    self._quiet:        Integer value between denoting amount of commentary 
                        whilst processing.
    '''   
        
    def __init__(self, imageList, camEnv, maskPath=None, invmaskPath=None,
                 calibFlag=True, image0=0, band='L', quiet=2, loadall=False, 
                 timingMethod='EXIF'):
        
        ImageSequence.__init__(self, imageList, band, loadall, quiet)
        
        #Set initial class properties
        self._camEnv = camEnv
        self._image0 = image0
        self._imageN = self.getLength()-1
        self._timings = None
        self._calibFlag = True
        
        #Set mask 
        if maskPath is None:
            self._mask = None
        else:
            if self._quiet > 0:
                print '\n\nSETTING VELOCITY MASK'
            self._mask = readMask(self.getImageArrNo(0), maskPath)
            if self._quiet > 1:
                print 'Velocity mask set'
         
        #Set inverse mask
        if invmaskPath is None:
            self._invmask = None
        else:
            if self._quiet > 0:
                print '\n\nSETTING HOMOGRAPHY MASK'
            self._invmask = readMask(self.getImageArrNo(0), invmaskPath)
            if self._quiet > 1:
                print 'Homography mask set'
           
    
    def setTimings(self, method='EXIF'):
        '''Method to explictly set the image timings that can be used for
        any offset time calculations (e.g. velocity) and also used for any
        output and plots.
        
        For now, only one method exists (EXIF derived) but this is here to 
        permit other ways to define image timings.
        
        Input
        method (str):           Method for setting timings (default set to 
                                derive timings from image EXIF information).
        '''       
        #Initialise list        
        self._timings=[]
        
        #Get EXIF time information from images
        for im in self.getImages():
            self._timings.append(im.getImageTime())
            

    def getTimings(self):
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


    def calcVelocities(self, homography=True, calcErrors=True, back_thresh=1.0,  
                       maxpoints=50000, quality=0.1, mindist=5.0, 
                       min_features=4):
        '''Function to calculate velocities between succesive image pairs. 
        Image pairs are called from the ImageSequence object. Points are seeded
        in the first of these pairs using the Shi-Tomasi algorithm with 
        OpenCV's goodFeaturesToTrack function. 
        
        The Lucas Kanade optical flow algorithm is applied using the OpenCV 
        function calcOpticalFlowPyrLK to find these tracked points in the 
        second image of each image pair. A backward tracking method then tracks 
        back from these to the first image in the pair, checking if this is 
        within a certain distance as a validation measure.
        
        Tracked points are corrected for image distortion and camera platform
        motion (if needed). The points in each image pair are georectified 
        subsequently to obtain xyz points. The georectification functions are 
        called from the Camera Environment object, and are based on those in
        ImGRAFT (Messerli and Grinsted, 2015). Velocities are finally derived 
        from these using a simple Pythagoras' theorem method.
        
        This function returns the xyz velocities and points from each image 
        pair, and their corresponding uv velocities and points in the image 
        plane.
        
        Inputs
        homography:                 Flag to denote whether homography should
                                    be calculated and images should be 
                                    corrected for image registration.
        calcErrors:                 Flag to denote whether tracked point errors 
                                    should be calculated.
        back_thesh:                 Threshold for back-tracking distance (i.e.
                                    the difference between the original seeded
                                    point and the back-tracked point in im0).
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.                 
        min_features:               Minimum number of seeded points to track.
        
        Outputs
        xyz:                        List containing the xyz velocities for each 
                                    point (xyz[0]), the xyz positions for the 
                                    points in the first image (xyz[1]), and the 
                                    xyz positions for the points in the second 
                                    image(xyz[2]). 
        uv:                         List containing the uv velocities for each
                                    point (uv[0], the uv positions for the 
                                    points in the first image (uv[1]), the
                                    uv positions for the points in the second
                                    image (uv[2]), and the corrected uv points 
                                    in the second image if they have been 
                                    calculated using the homography model for
                                    image registration (uv[3]). If the 
                                    corrected points have not been calculated 
                                    then an empty list is merely returned.                                 
        '''
        #Calculate homography if flag is true        
        if homography is True:
            self.calcHomographyPairs()
            
        #Optional commentary
        if self._quiet>0:
            print '\n\nCALCULATING VELOCITIES'
        
        #Create object attributes
        self._xyzvel = []
        self._xyz0 = []
        self._xyz1 = [] 
        self._uvvel = []           
        self._uv0 = []
        self._uv1 = []            
        self._uv1corr = []
        
        #Get first image (image0) file path and array data for initial tracking
        imn1=self._imageSet[0].getImagePath().split('\\')[1]
        im1=self._imageSet[0].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1):

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

            #Determine homography between image pair if required
            #Set calcErrors true otherwise we can't calculate/ plot homography
            #points
            if homography is True:                
                #Calculate velocities between image pair
                vel=self.calcVelocity(im0, im1, self._homogmatrix[i], 
                                      self._homogerr[i],
                                      back_thresh=back_thresh, 
                                      calcErrors=calcErrors, 
                                      maxpoints=maxpoints, 
                                      quality=quality, 
                                      mindist=mindist, 
                                      min_features=min_features)                       
                       
            else:               
                #Calculate velocities between image pair without homography
                vel=self.calcVelocity(im0, im1, None, None, 
                                      back_thresh=back_thresh, 
                                      calcErrors=calcErrors, 
                                      maxpoints=maxpoints, 
                                      quality=quality, 
                                      mindist=mindist, 
                                      min_features=min_features)                                      
                    
            #Assign important info as object attributes
            self._xyzvel.append(vel[0][0])         #xyz velocities
            self._xyz0.append(vel[0][1])           #xyz locations in im0
            self._xyz1.append(vel[0][2])           #xyz locations in im1
            
            self._uvvel.append(vel[1][0])          #uv velocities
            self._uv0.append(vel[1][1])            #uv locations in im0
            self._uv1.append(vel[1][2])            #uv locations in im1
            
            #Append corrected uv1 points if homography info was present
            if homography is True:
                self._uv1corr.append(vel[1][3])    #corrected uv1 locations
            else:
                self._uv1corr = None
            
        return ([self._xyzvel, self._xyz0, self._xyz1], 
                [self._uvvel, self._uv0, self._uv1, self._uv1corr])

        

    def calcVelocity(self, img1, img2, hmatrix=None, hpts=None,  
                     back_thresh=1.0, calcErrors=True, maxpoints=50000, 
                     quality=0.1, mindist=5.0, min_features=4):
        '''Function to calculate the velocity between a pair of images. Points 
        are seeded in the first of these using the Shi-Tomasi algorithm with 
        OpenCV's goodFeaturesToTrack function. 
        
        The Lucas Kanade optical flow algorithm is applied using the OpenCV 
        function calcOpticalFlowPyrLK to find these tracked points in the 
        second image. A backward tracking method then tracks back from these to 
        the original points, checking if this is within a certain distance as a 
        validation measure.
        
        Tracked points are corrected for image distortion and camera platform
        motion (if needed). The points in the image pair are georectified 
        subsequently to obtain xyz points.  The georectification functions are 
        called from the Camera Environment object, and are based on those in
        ImGRAFT (Messerli and Grinsted, 2015). Velocities are finally derived
        from these using a simple Pythagoras' theorem method.
        
        This function returns the xyz velocities and points, and their 
        corresponding uv velocities and points in the image plane.
        
        Inputs
        img1:                       Image 1 in the image pair.
        img2:                       Image 2 in the image pair.
        hmatrix:                    Homography matrix.
        hpts:                       Homography points.
        back_thesh:                 Threshold for back-tracking distance (i.e.
                                    the difference between the original seeded
                                    point and the back-tracked point in im0).
        calcErrors:                 Flag to denote whether tracked point errors 
                                    should be calculated.
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.                 
        min_features:               Minimum number of seeded points to track.
        
        Outputs
        xyz:                        List containing the xyz velocities for each 
                                    point (xyz[0]), the xyz positions for the 
                                    points in the first image (xyz[1]), and the 
                                    xyz positions for the points in the second 
                                    image(xyz[2]). 
        uv:                         List containing the uv velocities for each
                                    point (uv[0], the uv positions for the 
                                    points in the first image (uv[1]), the
                                    uv positions for the points in the second
                                    image (uv[2]), and the corrected uv points 
                                    in the second image if they have been 
                                    calculated using the homography model for
                                    image registration (uv[3]). If the 
                                    corrected points have not been calculated 
                                    then an empty list is merely returned.                                 
        '''       
        #Set threshold difference for point tracks
        displacement_tolerance_rel=2.0
        
        #Track points between the image pair
        points, ptserrors = self._featureTrack(img1, img2, self.getMask(),
                                               back_thresh=back_thresh, 
                                               calcErrors=calcErrors,
                                               maxpoints=maxpoints, 
                                               quality=quality,
                                               mindist=mindist, 
                                               min_features=min_features) 
        
        #Pass empty object if tracking was insufficient
        if points==None:
            if self._quiet>0:
                print '\nNo features to undertake velocity measurements'
            return None        
            
        if self._calibFlag is True:
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
            
            #Correct tracked points for image distortion. The displacement here 
            #is defined forwards (i.e. the points in image 1 are first 
            #corrected, followed by those in image 2)      
            #Correct points in first image 
            src_pts_corr=cv2.undistortPoints(points[0], 
                                             cameraMatrix, 
                                             distortP,P=newMat)
            
            #Correct points in second image                                         
            dst_pts_corr=cv2.undistortPoints(points[1], 
                                             cameraMatrix, 
                                             distortP,P=newMat) 
        else:
            src_pts_corr = points[0]
            dst_pts_corr = points[1]

        #Calculate homography if desired
        if hmatrix is not None:
            #Optional commentary
            if self._quiet>1:
                print '\nCorrecting for homography.'
            
            #Apply perspective homography matrix to tracked points
            tracked=dst_pts_corr.shape[0]
            dst_pts_homog = self.apply_persp_homographyPts(dst_pts_corr,
                                                           hmatrix,
                                                           'array',
                                                           inverse=True)
            
            #Calculate difference between points corrected for homography and
            #those uncorrected for homography
            dispx=dst_pts_homog[:,0,0]-src_pts_corr[:,0,0]
            dispy=dst_pts_homog[:,0,1]-src_pts_corr[:,0,1]
            
            #Use pythagoras' theorem to obtain distance
            disp_dist=np.sqrt(dispx*dispx+dispy*dispy)
            
            #Determine threshold for good points using a given displacement 
            #tolerance (defined earlier)
            xsd=hpts[0][2]
            ysd=hpts[0][3]
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
            #Optional commentary
            if self._quiet>1:
                print ('\nHomography matrix not supplied. Original tracked '
                       'points kept')
                       
            #Original tracked points assigned if homography not given
            dst_pts_homog=dst_pts_corr

        #Optional commentary
        if self._quiet>1:
            print '\nUndertaking inverse projection'
            
        #Project good points (original and tracked) to obtain XYZ coordinates
        if src_pts_corr is not None:
            uvs=src_pts_corr[:,0,:]
            xyzs=self._camEnv.invproject(uvs)
        else:
            xyzs=None
        
        if dst_pts_homog is not None:
            uvd=dst_pts_homog[:,0,:]
            xyzd=self._camEnv.invproject(uvd)
        else:
            xyzd=None
        
        xyzvel=[]
        pxvel=[]       
        for a,b,c,d in zip(xyzs, xyzd, src_pts_corr, dst_pts_homog):                        
            xyzvel.append(np.sqrt((b[0]-a[0])*(b[0]-a[0])+
                          (b[1]-a[1])*(b[1]-a[1])))
            pxvel.append(np.sqrt((d[0][0]-c[0][0])*(d[0][0]-c[0][0])+
                         (d[0][1]-c[0][1])*(d[0][1]-c[0][1])))
                
        #Return real-world point positions (original and tracked points),
        #and xy pixel positions (original, tracked, and homography-corrected)
        if hmatrix is not None:
            return [[xyzvel, xyzs, xyzd],
                    [pxvel, src_pts_corr, dst_pts_corr, dst_pts_homog]]
        
        else:
            return [[xyzvel, xyzs, xyzd], 
                    [pxvel, src_pts_corr, dst_pts_corr, None]]
        
        
    def calcHomographyPairs(self, back_thresh=1.0, calcErrors=True, 
                            maxpoints=50000, quality=0.1, mindist=5.0,
                            calcHomogError=True, min_features=4):
        '''Function to generate a homography model through a sequence of 
        images, and perform for image registration. Points that are assumed 
        to be static in the image plane are tracked between image pairs, and 
        movement in these points are used to generate sequential homography 
        models.
        
        The homography models are held in the Velocity object and can be called
        in subsequent velocity functions, such as calcVelocities and
        calcVelocity.
        
        Inputs
        back_thesh:                 Threshold for back-tracking distance (i.e.
                                    the difference between the original seeded
                                    point and the back-tracked point in im0).
        calcErrors:                 Flag to denote whether tracked point errors 
                                    should be calculated.
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.                 
        min_features:               Minimum number of seeded points to track.
        ''' 
        #Optional commentary
        if self._quiet>0:
            print '\n\nCALCULATING HOMOGRAPHY'
        
        #Create empty list for outputs
        self._homogmatrix = []       
        self._homogpts0 = []      
        self._homogpts1 = []        
        self._homogpts1corr = [] 
        self._homogptserr = []
        self._homogerr = []             
        
        #Get first image (image0) path and array data
        imn1=self._imageSet[0].getImagePath().split('\\')[1]
        im1=self._imageSet[0].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1):
            
            #Re-assign first image in image pair
            im0=im1
            imn0=imn1
            
            #Get second image in image pair (clear memory subsequently)
            im1=self._imageSet[i+1].getImageArray()
            imn1=self._imageSet[i+1].getImagePath().split('\\')[1]
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
            
            #Optional commentary
            if self._quiet>1:
                print '\nProcessing homograpy for images: ',imn0,' and ',imn1
                
            #Calculate homography and errors from image pair
            hg=self._calcHomography(im0, im1, back_thresh=back_thresh, 
                                    calcErrors=calcErrors, maxpoints=maxpoints,
                                    quality=quality, mindist=mindist, 
                                    calcHomogError=calcHomogError, 
                                    min_features=min_features)
        
            #Assign homography information as object attributes
            self._homogmatrix.append(hg[0])           #Homography matrix
            self._homogpts0.append(hg[1][0])          #Seeded pts in im0
            self._homogpts1.append(hg[1][1])           #Tracked pts in im1
            self._homogpts1corr.append(hg[1][2])       #Corrected pts im1
            self._homogptserr.append(hg[2])            #Tracked pts error
            self._homogerr.append(hg[3])              #Homography error
        
       
    def _featureTrack(self, i0, iN, Mask, back_thresh=1.0, calcErrors=True,
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
        corresponding list of SNR measures.

        Inputs
        i0:                         Image 1 in the image pair.
        iN:                         Image 2 in the image pair.
        Mask:                       Image mask to seed points in.
        back_thesh:                 Threshold for back-tracking distance (i.e.
                                    the difference between the original seeded
                                    point and the back-tracked point in im0).
        calcErrors:                 Flag to denote whether tracked point errors 
                                    should be calculated.
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.                 
        min_features:               Minimum number of seeded points to track.
        
        Outputs
        p0:                         Point coordinates for points seeded in 
                                    image 1.
        p1:                         Point coordinates for points tracked to
                                    image 2.
        p0r:                        Point coordinates for points back-tracked
                                    from image 2 to image 1.
        error:                      SNR measurements for the corresponding 
                                    tracked point. The signal is the magnitude
                                    of the displacement from p0 to p1, and the 
                                    noise is the magnitude of the displacement
                                    from p0r to p0.
        '''
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
                
                #Error commentary
                if self._quiet>0:
                    print '\nNot enough features successfully tracked.' 
                return None
        
       #Optional commentary
        if self._quiet>1:        
            print '\n'+str(tracked)+' features tracked'
            print (str(p0.shape[0]) + ' features remaining after ' 
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


    def _calcHomography(self, img1, img2, method=cv2.RANSAC,
                   ransacReprojThreshold=5.0, back_thresh=1.0, calcErrors=True,
                   maxpoints=50000, quality=0.1, mindist=5.0,
                   calcHomogError=True, min_features=4):
        '''Function to supplement correction for movement in the camera 
        platform given an image pair (i.e. image registration). Returns the 
        homography representing tracked image movement, and the tracked 
        features from each image.
        
        Inputs
        img1:                       Image 1 in the image pair.
        img2:                       Image 2 in the image pair.
        method:                     Method used to calculate homography model,
                                    which plugs into the OpenCV function
                                    cv2.findHomography: 
                                    cv2.RANSAC: RANSAC-based robust method.
                                    cv2.LEAST_MEDIAN: Least-Median robust 
                                    0: a regular method using all the points.                                   
        ransacReprjThreshold:       Maximum allowed reprojection error.
        back_thesh:                 Threshold for back-tracking distance (i.e.
                                    the difference between the original seeded
                                    point and the back-tracked point in im0).
        calcErrors:                 Flag to denote whether tracked point errors 
                                    should be calculated.
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.
        calcHomogError:             Flag to denote whether homography errors
                                    should be calculated.                 
        min_features:               Minimum number of seeded points to track.
        
        Outputs
        homogMatrix:                The calculated homographic shift for the 
                                    image pair (homogMatrix).
        src_pts_corr,
        dst_pts_corr,
        homog_pts:                  The original, tracked and back-tracked 
                                    homography points.  
        ptserror:                   Difference between the original homography 
                                    points and the back-tracked points.
        homogerror:                 Difference between the interpolated 
                                    homography matrix and the equivalent 
                                    tracked points
        '''         
        # Feature track between images
        trackdata = self._featureTrack(img1, img2, 
                                      self.getInverseMask(),
                                      back_thresh=back_thresh, 
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
        
        if self._calibFlag is True:
            #Call camera matrix and distortion coefficients from camenv
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
        else:
            src_pts_corr = points[0]
            dst_pts_corr = points[1]
        
        #Find the homography between the two sets of corrected points
        homogMatrix, mask = cv2.findHomography(src_pts_corr, 
                                               dst_pts_corr, 
                                               method=method,
                                               ransacReprojThreshold=ransacReprojThreshold)
        
        #Optional: calculate homography error
        #Homography error calculated from equivalent set of homography points
        #from original, uncorrected images
        if calcHomogError:
            
            #Optional commentary
            if self._quiet>1:
                print '\nCalculating Homography errors'

            #Apply global homography to source points
            homog_pts=self.apply_persp_homographyPts(src_pts_corr, homogMatrix,
                                                     'array', False)          
        
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
            
        return (homogMatrix, [src_pts_corr,dst_pts_corr,homog_pts], ptserrors, 
                homogerrors)

        
    def apply_persp_homographyPts(self, pts, homog, typ, inverse=False):        
        '''Funtion to apply a perspective homography to a sequence of 2D 
        values held in X and Y. The perspective homography is represented as a 
        3 X 3 matrix (homog). The source points are inputted as an array. The 
        homography perspective matrix is modelled in the same manner as done so 
        in OpenCV.
        
        Inputs
        pts:                  Input point positions to correct.
        homog:                Perspective homography matrix.
        typ:                  Format of input points (either can be an 'array 
                              or 'list'.                                   
        inverse:              Flag to denote if perspective homography matrix 
                              needs inversing.
        
        Output
        hpts:                 Corrected point positions.
        '''         
        if typ is 'array':
            #Get empty array that is the same size as pts
            n=pts.shape[0]
            hpts=np.zeros(pts.shape)
           
            if inverse:
               val,homog=cv2.invert(homog)       
    
            for i in range(n):
                div=1./(homog[2][0]*pts[i][0][0] + homog[2][1]*pts[i][0][1] + 
                        homog[2][2])
                hpts[i][0][0]=((homog[0][0]*pts[i][0][0] + 
                               homog[0][1]*pts[i][0][1] + homog[0][2])*div)
                hpts[i][0][1]=((homog[1][0]*pts[i][0][0] + 
                                homog[1][1]*pts[i][0][1] + homog[1][2])*div)
                              
            return hpts 
           
        elif typ is 'list':
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
                
        
    def _calcTrackErrors(self,p0,p1,dist):
        '''Function to calculate signal-to-noise ratio with forward-backward 
        tracking data. The distance between the backtrack and original points
        (dist) is assumed to be pre-calcuated.
        
        Inputs
        p0:                         Point coordinates for points seeded in 
                                    image 1.
        p1:                         Point coordinates for points tracked to
                                    image 2.
        dist:                       Distance between p0 and p0r (i.e. points
                                    back-tracked from image 2 to image 1).
                                                        
        Outputs:
        length:                     Displacement between p0 and p1 (i.e. a
                                    velocity, or signal).
        snr:                        Signal-to-noise ratio, the signal being
                                    the variable 'length' and the noise being
                                    the variable 'dist'.
        '''               
        #Determine length between the two sets of points
        length=(p0-p1)*(p0-p1)
        length=np.sqrt(length[:,0,0]+length[:,0,1])
        
        #Calculate signal-to-noise ratio
        snr = dist/length
        
        return length,snr
 
       
#------------------------------------------------------------------------------

class Area(Velocity):
    '''A class for processing change in area (i.e. a lake or plume) through an 
    image sequence, with methods to calculate extent change in an image plane 
    (px) and real areal change via georectification.
    
    Args:
    imageList (str, list):     List of images, for the ImageSet object.        
    cameraenv (str):           Camera environment parameters which can be read 
                               into the CamEnv class as a text file (see CamEnv 
                               documentation for exact information needed).
    calibFlag (bool):          An indicator of whether images are calibrated, 
                               for the ImageSet object.     
    maxMaskPath (str):         The file path for the mask indicating the region 
                               where areal extent should be recorded. If this 
                               does exist, the mask will be if this is inputted 
                               as a file directory to a jpg file. If the file 
                               does not exist, then the mask generation process 
                               will load and the mask will be saved to the 
                               given directory. If no directory is specified 
                               (i.e. maxMaskPath=None), the images will not be 
                               masked.
    maxim (int):               Image with maximum areal extent to be detected 
                               (for mask generation).     
    band (str):                String denoting the desired image band.
    quiet (int):               Level of commentary during processing. This can 
                               be an integer value between 0 and 2.
                               0: No commentary.
                               1: Minimal commentary.
                               2: Detailed commentary.                          
    loadall (bool):            Flag which, if true, will force all images in 
                               the sequence to be loaded as images (array) 
                               initially and thus not re-loaded in subsequent 
                               processing. 
                               This is only advised for small image sequences. 
    timingMethod (str):        Method for deriving timings from imagery. By 
                               default, timings are extracted from the image 
                               EXIF data.

    Class properties:
    self._maximg (int):        Reference number for image with maximum areal 
                               extent.
    self._pxplot (list):       Pixel plotting extent for easier colourrange 
                               definition and area verification 
                               [xmin, xmax, ymin, ymax].
    self._colourrange (list):  Colour range for automated area detection.
    self._threshold (int):     Maximum number of polygons retained after 
                               initial detection (based on size of polygons).
    self._enhance (list):      Image enhancement parameters for changing 
                               brightness and contrast. This is defined by 
                               three variables, held as a list: 
                               [diff, phi, theta].
                               (1) diff: Inputted as either 'light or 'dark', 
                               signifying the intensity of the image pixels. 
                               'light' increases the intensity such that dark 
                               pixels become much brighter and bright pixels 
                               become slightly brighter. 'dark' decreases the 
                               intensity such that dark pixels become much 
                               darker and bright pixels become slightly darker.
                               (2) phi: Defines the intensity of all pixel 
                               values.
                               (3) theta: Defines the number of "colours" in 
                               the image, e.g. 3 signifies that all the pixels 
                               will be grouped into one of three pixel values.
                               The default enhancement parameters are 
                               ['light', 50, 20].
    self._calibFlag (bool):    Boolean flag denoting whether measurements 
                               should be made on images corrected/uncorrected 
                               for distortion.
    self._quiet (int):         Integer value between denoting amount of 
                               commentary whilst processing.
    self._pxpoly (arr):        Output pixel coordinates (uv) for detected areas 
                               in an image sequence.
    self._pxextent (list):     Output pixel extents for detected areas in an 
                               image sequence.
    self._realpoly (arr):      Output real-world coordinates (xyz) for detected 
                               areas in an image sequence.
    self._area (list):         Output real-world surface areas for detected 
                               areas in an image sequence.
    '''
    
    #Initialisation of Area class object          
    def __init__(self, imageList, cameraenv, calibFlag=True, 
                 maxMaskPath=None, maxim=0, band='L', quiet=2, loadall=False, 
                 timingMethod='EXIF'):
                     
        #Initialise and inherit from the TimeLapse class object
        Velocity.__init__(self, imageList, cameraenv, None, None, calibFlag, 
                          0, band, quiet, loadall, timingMethod)
                 
        #Optional commentary        
        if self._quiet>0:
            print '\n\nCOMMENCING AREA DETECTION'
        
        #Set up class properties
        self._maximg = maxim
        self._calibFlag = calibFlag
        self._quiet = quiet
        self._pxplot = None
        
        #Create mask if required
        if maxMaskPath is None:
            self._maxMaskPath=None
        else:
            self._maxMaskPath=maxMaskPath
            self._setMaxMask()


    def calcAutoAreas(self, px=None, colour=False, verify=False):
        '''Get real world areas from an image set. Calculates the polygon 
        extents for each image and the area of each given polygon using 
        automated detection.
        
        Args
        colour (boolean):           Flag to denote whether colour range for 
                                    detection should be defined for each image
                                    or only once.
        verify (boolean):           Flag to denote whether detected polygons
                                    should be manually verified by user.
        
        Returns
        self._realpoly (arr):       XYZ coordinates for all detected polygon
                                    areas in all images.
        self._area (list):          List of surface areas for all detected 
                                    polygons in all images.
        '''                           
        #Get pixel polygons using automated extent detection method
        if px is None:
            self.calcAutoExtents(colour, verify)
        
        #Optional commentary
        if self._quiet>0:
            print '\n\nCOMMENCING GEORECTIFICATION OF AREAS'
        
        #Create empty output lists
        xyz = []
        area = []
        
        #Calculate real-world areas with inverse projection        
        for p in self._pxpoly:
            pts, a = self._getRealArea(p)
            xyz.append(pts)
            area.append(a)

        #Assign real-world areas to Area object        
        self._realpoly = xyz
        self._area = area        
        return self._realpoly, self._area


    def calcManualAreas(self, px=None):
        '''Get real world areas from an image set. Calculates the polygon 
        extents for each image and the area of each given polygon using manual
        detection.
        
        Returns
        self._realpoly (arr):       XYZ coordinates for all detected polygon
                                    areas in all images.
        self._area (list):          List of surface areas for all detected 
                                    polygons in all images.
        '''                          
        #Get pixel polygons using manual extent detection method
        if px is None:
            pxpoly, pxextent = self.calcManualExtents()
        
        #Optional commentary
        if self._quiet>0:
            print '\n\nCOMMENCING GEORECTIFICATION OF AREAS'
        
        #Create empty output lists
        xyz = []
        area = []
        
        #Calculate real-world areas with inverse projection
        for p in self._pxpoly:
            pts, a = self._getRealArea(p)
            xyz.append(pts)
            area.append(a)
        
        #Assign real-world areas to Area object
        self._realpoly = xyz
        self._area = area        
        return self._realpoly, self._area
          

    def calcAutoExtents(self, colour=False, verify=False):
        '''Get pixel extents from a series of images using automated detection. 
        Return the extent (px) polygons and cumulative extent values (px).
        
        Args
        colour (boolean):           Flag to denote whether colour range for 
                                    detection should be defined for each image
                                    or only once.
        verify (boolean):           Flag to denote whether detected polygons
                                    should be manually verified by user.
        
        Returns
        self._pxpoly (arr):         Pixel coordinates (uv) for all detected 
                                    polygon extents in all images.
        self._pxextent (list):      List of pxel extents for all detected 
                                    polygons in all images.
        '''               
        #Optional commentary
        if self._quiet>0:
            '\n\nCOMMENCING AUTOMATED AREA DETECTION' 
            
        #If user is only defining the color range once
        if colour is False: 
            
            #Define colour range if none is given
            if self._colourrange is None:

                #Get image with maximum extent (either corrected or distorted)
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
                       
                #Enhance image if enhancement parameters given
                if self._enhance is not None:
                    maximg = self._enhanceImg(maximg)
                
                #Define colour range
                self.defineColourrange(maximg, maximn)    
            
        #Set up output datasets
        areas=[]        
        px=[]
                       
        #Cycle through image sequence (numbered from 0)
        for i in range(self.getLength()):
            
            #Get corrected/distorted image
            if self._calibFlag is True:
                cameraMatrix=self._camEnv.getCamMatrixCV2()
                distortP=self._camEnv.getDistortCoeffsCv2()
                img1 = self._imageSet[i].getImageCorr(cameraMatrix, 
                                                      distortP)
            else:
                img1=self._imageSet[i].getImageArray()

            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
               
            #Make a copy of the image array
            img2 = np.copy(img1)
            
            #Mask image if mask is present
            if self._mask is not None:
                img2 = self._maskImg(img2)
            
            #Enhance image if enhancement parameters are present
            if self._enhance is not None:
                img2 = self._enhanceImg(img2)
            
            #Define colour range if required
            if colour is True:
                self.defineColourrange(img2, imn)
            
            #Optional commentary
            if self._quiet>0:
                print '\nCalculating extent for ' + imn
            
            #Calculate extent
            polys,extent = self._calcAutoExtent(img2)        
            areas.append(polys)
            px.append(extent)

            #Clear memory
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
        '''Method to manually verify all polygons in images. Plots sequential
        images with detected polygons and the user manually verifies them by 
        clicking them.
        
        Returns
        self._pxpoly   (arr):       Pixel coordinates (uv) for all verfied 
                                    polygon extents in all images.
        self._pxextent (list):      List of pxel extents for all verified 
                                    polygons in all images.'''
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
                    print '\n\nVERIFYING DETECTED AREAS FOR ' + imn
                
                #Set up empty output list                
                verf = []
                
                #Function for click verification within a plot
                def onpick(event):
                    
                    #Get XY coordinates for clicked point in a plot
                    v = []
                    thisline = event.artist
                    xdata = thisline.get_xdata()
                    ydata = thisline.get_ydata()
                    
                    #Append XY coordinates
                    for x,y in zip(xdata,ydata):
                        v.append([x,y])
                    v2=np.array(v, dtype=np.int32).reshape((len(xdata)),2)
                    verf.append(v2)
                    
                    #Verify extent if XY coordinates coincide with a
                    #detected area
                    ind=event.ind
                    print ('Verified extent at ' + 
                           str(np.take(xdata, ind)[0]) + ', ' + 
                           str(np.take(ydata, ind)[0]))
                
                #Plot image
                fig, ax1 = plt.subplots()
                fig.canvas.set_window_title(imn + ': Click on valid areas.')
                ax1.imshow(img2, cmap='gray')
                
                #Chane plot extent if pxplot variable is present
                if self._pxplot is not None:
                    ax1.axis([self._pxplot[0],self._pxplot[1],
                              self._pxplot[2],self._pxplot[3]])
                
                #Plot all detected areas
                for a in px:
                    x=[]
                    y=[]
                    for b in a:
                        for c in b:
                            x.append(c[0])
                            y.append(c[1])
                    line = Line2D(x, y, linestyle='-', color='y', picker=True)
                    ax1.add_line(line)
                
                #Verify extents using onpick function
                fig.canvas.mpl_connect('pick_event', onpick)
            
            #Show plot
            plt.show()           
            
            #Append all verified extents
            vpx=[]
            vpx=verf
            verified.append(vpx)
            
            #Get areas of verified extents
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
            pxext = len(values)
            update_ext.append(pxext)
    
            #Clear memory            
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
        
        #Rewrite pixel polygon and extent data
        self._pxpoly = verified
        self._pxextent = update_ext
    
        
    def calcManualExtents(self):
        '''Method to manually define pixel extents from a series of images. 
        Returns the extent polygons and cumulative extent values (px).
        
        Returns
        self._pxpoly   (arr):       Pixel coordinates (uv) for all detected 
                                    polygon extents in all images.
        self._pxextent (list):      List of pxel extents for all detected 
                                    polygons in all images.
                                    '''                 
        #Optional commentary        
        if self._quiet>0:
            '\n\nCOMMENCING MANUAL AREA DETECTION'
            
        #Set up output dataset
        areas=[]        
        px=[]
                
        #Cycle through images        
        for i in (range(self.getLength())):
            
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                   self._camEnv.getDistortCoeffsCv2())      
            else:
                img=self._imageSet[i].getImageArray()          

            #Get image name
            imn=self._imageSet[i].getImagePath().split('\\')[1]
            
            #Manually define extent and append
            polys,extent = self._calcManualExtent(img, imn)        
            areas.append(polys)
            px.append(extent)
            
            #Clear memory
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
    
        #Return all extents, all cropped images and corresponding image names
        self._pxpoly = areas
        self._pxextent = px        
        return self._pxpoly, self._pxextent


    def setEnhance(self, diff, phi, theta):
        '''Set image enhancement parameters. Change brightness and contrast of 
        image using phi and theta variables.
        Change phi and theta values accordingly. See _enhanceImg function for 
        detailed explanation of the parameters.
        
        Args
        diff (str):               Inputted as either 'light or 'dark', 
                                  signifying the intensity of the image pixels. 
                                  'light' increases the intensity such that 
                                  dark pixels become much brighter and bright 
                                  pixels become slightly brighter. 
                                  'dark' decreases the intensity such that dark 
                                  pixels become much darker and bright pixels 
                                  become slightly darker.
        phi (int):                A value between 0 and 1000 which defines the
                                  intensity of all pixel values.
        theta (int):              A value between 0 and 1000 which defines the 
                                  number of "colours" in the image, e.g. 3
                                  signifies that all the pixels will be grouped
                                  into one of three pixel values.
        '''
        self._enhance = diff, phi, theta
        

    def seeEnhance(self):
        '''Enhance image using an interactive plot and assign enhance 
        parameters based on user preferences. 
        WARNING: this function will significantly slow down your computer. 
        Only use if your computer can handle it.
        '''
        #Get image with maximum areal extent to detect
        if self._calibFlag is True:
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCv2()
            img = self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                               distortP)
        else:
            img = self._imageSet[self._maximg].getImageArray()         

        #Plot image
        fig,ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax = plt.subplot(111)
        ax.imshow(img)
        
        #Initially assign image enhance diff variable
        diff = 'light'  
                
        #Inititalise sliders for phi and theta
        axcolor = 'lightgoldenrodyellow'
        axphi  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        axtheta = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
        sphi = Slider(axphi, 'Phi', 1, 100.0, valinit=1)
        stheta = Slider(axtheta, 'Theta', 1, 50.0, valinit=1)
        
        #Function to update image when phi and theta are changed
        def update(val):        
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
        
        #Function for reset button event
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
        
        #Show plot        
        plt.show()
        plt.close()
        
        #Assign image enhancement parameters
        self.setEnhance('light', sphi.val, stheta.val)
         
        
    def setPXExt(self,xmin,xmax,ymin,ymax):
        '''Set plotting extent. Setting the plot extent will make it easier to 
        define colour ranges and verify areas.
        
        Args
        xmin (int):               X-axis minimum value.
        xmax (int):               X-axis maximum value.
        ymin (int):               Y-axis minimum value.
        ymax (int):               Y-axis maximum value.
        '''
        self._pxplot = [xmin,xmax,ymin,ymax]


    def setThreshold(self, number):
        '''Set threshold for number of polgons kept from an image.
        
        Args
        number (int):             Number denoting the number of detected 
                                  polygons that will be retained.
        '''
        self._threshold = number
                                

    def setColourrange(self, upper, lower):
        '''Manually define the RBG colour range that will be used to filter
        the image/images.
        
        Args
        upper (int):              Upper value of colour range.
        lower (int):              Lower value of colour range.
        '''
        #Optional commentary
        if self._quiet>0:        
            print '\nColour range defined from given values:'
            print 'Upper RBG boundary: ', upper
            print 'Lower RBG boundary: ', lower
        
        #Assign colour range
        self._colourrange = [upper, lower]
        
                
    def defineColourrange(self, img, imn):
        '''Define colour range manually by clicking on the lightest and 
        darkest regions of the target extent that will be defined.
        
        Args
        img (arr):          Image array (for plotting the image).
        imn (str):          Image name.
        
        Plot interaction information:
        Left click to select.
        Right click to undo selection.
        Close the image window to continue.
        The window automatically times out after two clicks.
        '''
        #Initialise figure window
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Click lightest colour and darkest' 
                                    ' colour')
        
        #Plot image
        plt.imshow(img, origin='upper')
        
        #Define plotting extent if required
        if self._pxplot is not None:
            plt.axis([self._pxplot[0],self._pxplot[1],
                      self._pxplot[2],self._pxplot[3]])

        #Manually interact to select lightest and darkest part of the region            
        colours = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, 
                            mouse_pop=3, mouse_stop=2)
        
        #Optional commentary
        if self._quiet>0:
            print '\n' + imn + ': you clicked ', colours
        
        #Show plot
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
        
        #Optional commentary
        if self._quiet>1:
            print 'Colour range found from manual selection'
            print 'Upper RBG boundary: ' + str(upper_boundary)
            print 'Lower RBG boundary: ' + str(lower_boundary)

        #Store RBG range
        self._colourrange = [upper_boundary, lower_boundary]


    def _calcAutoExtent(self, img):
        '''Get extent from a given image using a predefined RBG colour range. 
        The colour range is then used to extract pixels within that range 
        using the OpenCV function inRange. If a threshold has been set (using
        the setThreshold function) then only nth polygons will be retained.
        
        Args
        img (arr):            Image array.
        
        Returns
        pxpoly:               Pixel coordinates (uv) for all detected polygons
                              within the image.
        pxextent:             Pixel extents for all detected polygons within 
                              the image.
        '''                       
        #Get upper and lower RBG boundaries from colour range
        upper_boundary = self._colourrange[0]
        lower_boundary = self._colourrange[1]
    
        #Transform RBG range to array    
        upper_boundary = np.array(upper_boundary, dtype='uint8')
        lower_boundary = np.array(lower_boundary, dtype='uint8')
    
        #Extract extent based on RBG range
        mask = cv2.inRange(img, lower_boundary, upper_boundary)

#        #Speckle filter to remove noise - needs fixing
#        mask = cv2.filterSpeckles(mask, 1, 30, 2)
        
        #Polygonize extents using OpenCV findContours function        
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
        
        #Fill polygons and extract polygon extent
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
        
        #Total target extent (px)
        pxextent = len(values)
        
        #Total image extent (px)
        pxcount = len(pixels)
        
        #Optional commentary
        if self._quiet>1:
            print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px)'
        
        #Return pixel coordinates and extents
        return pxpoly, pxextent
        

    def _calcManualExtent(self, img, imn):
        '''Method to Manually define an extent by clicking around a region in a 
        given image.
        
        Args
        img (arr):          Image array (for plotting the image).
        imn (str):          Image name.
        
        Returns
        pts (arr):          List of manually defined points which make up the 
                            polygon.
        pxextent (int):    Pixel extent of manually defined polygon 
        '''
         #Manual interaction to select lightest and darkest part of the region
#        pts=[]        
#        while len(pts) < 3:
        
        #Initialise figure window
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Click around region. Press enter '
                                    'to record points.')
        
        #Plot image
        plt.imshow(img, origin='upper', cmap='gray')
        
        #Set plotting extent if required
        if self._pxplot is not None:
            plt.axis([self._pxplot[0],self._pxplot[1],
                      self._pxplot[2],self._pxplot[3]]) 
        
        #Manual input of points from clicking on plot using pyplot.ginput
        pts = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, 
                        mouse_pop=3, mouse_stop=2)
        
        #Optional commentary
        if self._quiet>1:
            print '\n' + imn + ': you clicked ' + str(len(pts)) + ' points'
        
        #Show plot
        plt.show()
        plt.close()
        
#            #Reboot window if =<2 points are recorded
#            if len(pts) > 3:
#                pts = []
            
        #Create polygon if area has been recorded   
        try:
            
            #Complete the polygon ring             
            pts.append(pts[0]) 
            
            #Create geometry
            ring = ogr.Geometry(ogr.wkbLinearRing) 
            
            #Append clicked points to geometry
            for p in pts:
                ring.AddPoint(p[0],p[1])
            p=pts[0]
            ring.AddPoint(p[0],p[1])
            
            #Creat polygon ring
            pxpoly = ogr.Geometry(ogr.wkbPolygon)          
            pxpoly.AddGeometry(ring) 
            
            #Calculate area of polygon area
            pxextent = pxpoly.Area()
        
        #Create zero object if no polygon has been recorded 
        except:
            pxextent = 0
                   
        #Get image dimensions and calculated total image extent (px)
        h = img.shape[0]
        w = img.shape[1]
        pxcount = h*w
        
        #Optional commentary
        if self._quiet>1:
            print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px)'    
        
        #Convert pts list to array
        pts = np.array(pts)           
        pts=[pts]
        
        #Return pixel coordinates and area of polygon
        return pts, pxextent


    def _getRealArea(self, pxpoly):
        '''Get real world areas from pixel polygons (defined in one image).
        
        Args
        pxpoly (arr):           Image coordinates (uv) of polygon/polygons.
        
        Returns
        xyz (arr):              Real-world coordinates (xyz) of polygon/
                                polygons.
        area (list):            Surface area/areas associated with 
                                polygon/polygons.
        '''                    
        #Create outputs
        xyz = []   
        area = []                               

        #Inverse project image coordinates using function from CamEnv object
        for p in pxpoly:                        
            allxyz = self._camEnv.invproject(p)
            xyz.append(allxyz)                  
        
        #Create polygons
        rpoly = self._ogrPoly(xyz)              
        
        #Determine area of each polygon
        for r in rpoly:
            area.append(r.Area())               
        
        #Return xyz coordinates and surface areas
        return xyz, area         

        
    def _getRBG(self, img, x, y):
        '''Return the compressed RBG value for a given point in an image.
        
        Args
        img (arr):              Image from which the RBG value will be taken
                                from.        
        x (int):                X coordinate of image.
        y (int):                Y coordinate of image.
        
        Returns
        RBG (int):              Single value denoting pixel intensity for the 
                                image XY coordinate. 
        '''
        #Get pixel intensity value        
        RBG = img[x,y]
        
        #Change pixel intensity value to 1 if 0 has been initially assigned
        if RBG == 0:
            RBG == 1
        
        #Return pixel intensity value
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
        the TimeLapse class uses a numpy masking object (numpy.ma).
        
        Args
        img (arr):                Input image array for masking.
        
        Returns
        img (arr):                Masked image array.
        '''            
        #Mask the glacier
        booleanMask = np.array(self._mask, dtype=bool)
        booleanMask = np.invert(booleanMask)
        
        #Mask extent image with boolean array
        np.where(booleanMask, 0, img) #fit arrays to each other
        img[booleanMask] = 0 #mask image with boolean mask object

        return img
        
       
    def _enhanceImg(self, img):
        '''Change brightness and contrast of image using phi and theta 
        variables. Change phi and theta values accordingly.
        
        Args
        img (arr):                    Input image array for enhancement.
        
        Returns
        img1 (arr):                   Enhanced image.
        
        Enhancement parameters (self._enhance):
        diff:                   Inputted as either 'light or 'dark', signifying 
                                the intensity of the image pixels. 'light' 
                                increases the intensity such that dark pixels 
                                become much brighter and bright pixels become 
                                slightly brighter. 
                                'dark' decreases the intensity such that dark 
                                pixels become much darker and bright pixels 
                                become slightly darker.
        phi:                    Defines the intensity of all pixel values.
        theta:                  Defines the number of "colours" in the image, 
                                e.g. 3 signifies that all the pixels will be 
                                grouped into one of three pixel values.
        '''                          
        #Extract enhancement parameters from enhance object
        diff = self._enhance[0]
        phi = self._enhance[1]
        theta = self._enhance[2]
        
        #Define maximum pixel intensity
        maxIntensity = 255.0 #depends on dtype of image data 
        
        #If diff variable is light
        if diff == 'light':        

            #Increase intensity such that dark pixels become much brighter
            #and bright pixels become slightly brighter
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
            img1 = array(img1, dtype = uint8)
        
        #If diff variable is dark
        elif diff == 'dark':        

            #Decrease intensity such that dark pixels become much darker and 
            #bright pixels become slightly darker          
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
            img1 = array(img1, dtype=uint8)
        
        #If diff variable not assigned then reassign to light
        else:
            if self._quiet>0:            
                print '\nInvalid diff variable' 
                print 'Re-assigning diff variable to "light"'
            img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
            img1 = array(img1, dtype = uint8)
        
        #Return enhanced image
        return img1   
   
    
    def _ogrPoly(self, xyz):
        '''Get real world OGR polygons (.shp) from xyz poly pts with real world 
        points which are compatible with mapping software (e.g. ArcGIS).
        
        Args
        xyz (arr):                XYZ coordinates of all shapes in a given
                                  image (i.e. not an entire sequence).
        
        Returns
        polygons (list):          List of ogr geometry polygons.                           
        '''                       
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


#------------------------------------------------------------------------------

class Line(Area):
    '''A class for handling lines/distances (e.g. glacier terminus position)
    through an image sequence, with methods to manually define pixel lines in 
    the image plane and georectify them to generate real-world coordinates and 
    distances. The Line class object primarily inherits from the Area class.
    
    Args
    imageList (str, list):     List of images, for the ImageSet object.        
    cameraenv (str):           Camera environment parameters which can be read 
                               into the CamEnv class as a text file (see CamEnv 
                               documentation for exact information needed).
    calibFlag (bool):          An indicator of whether images are calibrated, 
                               for the ImageSet object.          
    band (str):                String denoting the desired image band.
    quiet (int):               Level of commentary during processing. This can 
                               be an integer value between 0 and 2.
                               0: No commentary.
                               1: Minimal commentary.
                               2: Detailed commentary.                          
    loadall (bool):            Flag which, if true, will force all images in 
                               the sequence to be loaded as images (array) 
                               initially and thus not re-loaded in subsequent 
                               processing. 
                               This is only advised for small image sequences. 
    timingMethod (str):        Method for deriving timings from imagery. By 
                               default, timings are extracted from the image 
                               EXIF data.
                               
    Class properties:    
    self._pxpts:        Output pixel coordinates (uv) for detected lines in an 
                        image sequence.
    self._pxline:       Output pixel lengths for detected lines in an image 
                        sequence.
    self._realpts:      Output real-world coordinates (xyz) for detected lines 
                        in an image sequence.
    self._realline:     Output real-world distances for detected lines in 
                        an image sequence.
    '''
     
    #Object initialisation        
    def __init__(self, imageList, cameraenv, calibFlag=True,
                 band='L', quiet=2, loadall=False, timingMethod='EXIF'):
        
        #Initialise and inherit from Area class
        Area.__init__(self, imageList, cameraenv, calibFlag, None, 0, 
                      band, quiet, loadall, timingMethod)
        
        #Optional commentary
        if self._quiet>0:
            '\n\nCOMMENCING LINE DETECTION.'
            

    def calcManualLinesXYZ(self, px=None):
        '''Method for calculating real world lines from an image sequence. 
        Lines are manually defined by the user in the image plane. These are 
        subsequently georectified to obtain real-world coordinates and 
        distances.
        
        Returns
        self._realpts (arr):      Output real-world coordinates (xyz) for 
                                  detected lines in an image sequence.
        self._realline (list):    Output real-world distances for detected 
                                  lines in an image sequence.
        '''
        #Get pixel points if not already defined
        if px is None:
            self.calcManualLinesPX()
        
        #Set output variables and counter
        rpts = []
        rline = []        
        count=1
        
        #Optional commentary
        if self._quiet>0:
            print '\n\nCOMMENCING GEORECTIFICATION OF LINES'
        
        #Project pixel coordinates to obtain real world coordinates and lines
        for p in self._pxpts:              
            
            #Project image coordinates
            xyz = self._camEnv.invproject(p)
#            rp = np.squeeze(rpts)
            
            #Create ogr line object
            rl = self._ogrLine(xyz)
            
            #Optional commentary
            if self._quiet>1:
                print ('\nImg ' + str(count) + ' line length: ' 
                       + str(rl.Length()) + ' m')
            
            #Append coordinates and distances            
            rpts.append(xyz)
            rline.append(rl)
            
            #Update counter
            count=count+1

        #Return real line coordinates and line length objects                
        self._realpts = rpts
        self._realline = rline        
        return self._realpts, self._realline
     

    def calcManualLinesPX(self):
        '''Method to manually define pixel lines from an image sequence. The 
        lines are manually defined by the user on an image plot. Returns the 
        line pixel coordinates and pixel length.
        
        Returns
        self._pxpts (arr):        Output pixel coordinates (uv) for detected 
                                  lines in an image sequence.
        self._pxline (list):      Output pixel lengths for detected lines in an 
                                  image sequence.
        '''                         
        #Optional commentary 
        if self._quiet>0:
            print '\n\nMANUAL PX LINE DEFINITION'
            
        #Set up output dataset
        pts=[]        
        lines=[]
        count=1

        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()):
            
            #Get corrected/distorted image
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
            pt,length = self._calcManualLinePX(img1, imn)
            
            #Optional commentary
            if self._quiet>1:            
                print '\n\nLine defined in ' + imn                
                print 'Img%i line length: %d px' % (count, length.Length())
                print 'Line contains %i points' % (length.GetPointCount())
            
            #Append line
            pts.append(pt)
            lines.append(length)
            count=count+1
    
        #Return all line coordinates and length
        self._pxpts = pts
        self._pxline = lines        
        return self._pxpts, self._pxline
        

    def _calcManualLinePX(self, img, imn):
        '''Function to manually define a line by clicking in the target image
        plot. This primarily operates via the pyplot.ginput function which 
        allows users to define coordinates through plot interaction.
        
        Args
        img (arr):              Image array for plotting.
        imn (str):              Image name.
        
        Returns
        pts (arr):              Pixel coordinates (uv) for plotted line.
        line (ogr.Geometry):    Geometry object for plotted line from which 
                                length and other information can be derived.
        '''
        #Initialise figure window
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Define line. ' 
                                    'Press enter to record points.')
        
        #Plot image
        plt.imshow(img, origin='upper',cmap='gray')        
        pts = plt.ginput(n=0, timeout=0, show_clicks=True, 
                         mouse_add=1, mouse_pop=3, mouse_stop=2)

        #Optional commentary
        if self._quiet>2:            
            print ('\nYou clicked ' + str(len(pts)) + ' in image ' + imn)
        
        #Show plot
        plt.show()
        plt.close()
        
        #Create OGR line object
        line = self._ogrLine(pts)
        
        #Re-format point coordinates
        pts = np.squeeze(pts)

        #Return coordinates and geometry object
        return pts, line


    def _ogrLine(self, pts):
        '''Function to construct an OGR line from a set of uv coordinates.
        
        Args
        pts (arr):                A series of uv coordinates denoting a line.
        
        Returns
        line (ogr.Geometry):      A line object constructed from the input 
                                  coordinates.
        ''' 
        #Initially construct geometry object             
        line = ogr.Geometry(ogr.wkbLineString)
        
        #Append points to geometry object
        for p in pts:
            if len(p)==2:
                line.AddPoint(p[0],p[1])
            elif len(p)==3:
                line.AddPoint(p[0],p[1],p[2])
            else:
                print 'LINE ERROR: Point not recognised.' 
        #Return geometry line
        return line 
        

#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   