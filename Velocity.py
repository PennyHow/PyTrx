'''
PYTRX VELOCITY MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This is the Velocity module of PyTrx. It handles the functionality for 
obtaining velocity measurements from oblique time-lapse imagery. Specifically, 
this module contains functions for:
(1) Performing camera registration from static point feature tracking (referred 
    to here as homography).
(2) Calculating surface velocities derived from feature tracking, with 
    associated errors and signal-to-noise ratio calculated.
(3) Determining real-world surface areas and distances from oblique imagery.

Classes
Velocity:                       A class for the processing of an image Sequence 
                                to determine pixel displacements and real-world 
                                velocities from a sparse set of points, and 
                                correct for camera platform motion

Key class functions 
calcVelocities:                 Calculate velocities between succesive image 
                                pairs in an image sequence
calcHomographyPairs:            Calculate homography between succesive image 
                                pairs in an image sequence
                               
Key standalone functions
calcVelocity:                   Calculate velocities between an image pair
calcHomography:                 Calculate homography between an image pair
                                                               
@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton 
         Lynne Buie
'''

#Import packages
import numpy as np
import cv2
import math

#Import PyTrx functions and classes
from FileHandler import readMask
from Images import ImageSequence
from CamEnv import projectUV, setProjection

#------------------------------------------------------------------------------
class Homography(ImageSequence):
    '''A class for the processing the homography of an image sequence to 
    determine motion in a camera platform.
    
    This class treats the images as a contigous sequence of name references by
    default.
    
    Args
    imageList:          List of images, for the ImageSet object.
    camEnv:             The Camera Environment corresponding to the images, 
                        for the ImageSequence object.
    invmaskPath:        As above, but the mask for the stationary feature 
                        tracking (for camera registration/determining
                        camera homography).
    band:               String denoting the desired image band.
    equal:              Flag denoting whether histogram equalisation is applied 
                        to images (histogram equalisation is applied if True). 
                        Default is True.                        
    '''
        
    def __init__(self, imageList, camEnv, invmaskPath=None, calibFlag=True, 
                 band='L', equal=True):
        
        ImageSequence.__init__(self, imageList, band, equal)
        
        #Set initial class properties
        self._camEnv = camEnv
        self._imageN = self.getLength()-1
        self._calibFlag = calibFlag
         
        #Set mask
        if invmaskPath is None:
            self._invmask = None
        else:
            self._invmask = readMask(self.getImageArrNo(0), invmaskPath)
            print '\nHomography mask set'


    def calcHomographyPairs(self, back_thresh=1.0, maxpoints=50000, 
                            quality=0.1, mindist=5.0, min_features=4):
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
        maxpoints:                  Maximum number of points to seed in im0
        quality:                    Corner feature quality.
        mindist:                    Minimum distance between seeded points.                 
        min_features:               Minimum number of seeded points to track.
        ''' 
        print '\n\nCALCULATING HOMOGRAPHY'
        
        #Create empty list for outputs
        homog=[]   
        
        #Get first image (image0) path and array data
        imn1=self._imageSet[0].getImageName()
        im1=self._imageSet[0].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1):
            
            #Re-assign first image in image pair
            im0=im1
            imn0=imn1
            
            #Get second image in image pair (clear memory subsequently)
            im1=self._imageSet[i+1].getImageArray()
            imn1=self._imageSet[i+1].getImageName()
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
            
            print '\nProcessing homograpy for images: ',imn0,' and ',imn1
            
            #Get inverse mask and calibration parameters
            invmask = self.getInverseMask()
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCV2()
               
            #Calculate homography and errors from image pair
            hg=calcHomography(im0, im1, invmask, [cameraMatrix, distortP], 
                              back_thresh=back_thresh,
                              method=cv2.RANSAC,
                              ransacReprojThreshold=5.0,
                              maxpoints=maxpoints,
                              quality=quality, 
                              mindist=mindist, 
                              min_features=min_features)
        
            #Assign homography information as object attributes
            homog.append(hg)
            
        return homog            


    def getInverseMask(self):
        '''Return inverse mask.'''
        return self._invmask

            
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
    equal:              Flag denoting whether histogram equalisation is applied 
                        to images (histogram equalisation is applied if True). 
                        Default is True.                        
    loadall:            Flag which, if true, will force all images in the 
                        sequence to be loaded as images (array) initially and 
                        thus not re-loaded in subsequent processing. This is 
                        only advised for small image sequences. 
    timingMethod:       Method for deriving timings from imagery. By default, 
                        timings are extracted from the image EXIF data. 
    '''
        
    def __init__(self, imageList, camEnv, homography=None, maskPath=None, 
                 calibFlag=True, band='L', equal=True):
        
        ImageSequence.__init__(self, imageList, band, equal)
        
        #Set initial class properties
        self._camEnv = camEnv
        self._homog = homography
        self._imageN = self.getLength()-1
        self._calibFlag = calibFlag
        
        #Set mask 
        if maskPath is None:
            self._mask = None
        else:
            self._mask = readMask(self.getImageArrNo(0), maskPath)
            print '\nVelocity mask set'


    def calcVelocities(self, back_thresh=1.0, maxpoints=50000, 
                       quality=0.1, mindist=5.0, min_features=4):
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
           
        print '\n\nCALCULATING VELOCITIES'
        velocity=[]
        
        #Get camera environment 
        camenv = self.getCamEnv()
        
        #Get DEM from camera environment
        dem = camenv.getDEM() 

        #Get inverse projection variables through camera info               
        invprojvars = setProjection(dem, camenv._camloc, camenv._camDirection, 
                                    camenv._radCorr, camenv._tanCorr, 
                                    camenv._focLen, camenv._camCen, 
                                    camenv._refImage) 
        
        #Get camera matrix and distortion parameters for calibration
        mtx=self._camEnv.getCamMatrixCV2()
        distort=self._camEnv.getDistortCoeffsCV2()
        
        #Get mask
        mask=self.getMask()
        
        #Get first image (image0) file path and array data for initial tracking
        imn1=self._imageSet[0].getImageName()
        im1=self._imageSet[0].getImageArray()
        
        #Cycle through image pairs (numbered from 0)
        for i in range(self.getLength()-1):

            #Re-assign first image in image pair
            im0=im1
            imn0=imn1
                            
            #Get second image in image pair (and subsequently clear memory)
            im1=self._imageSet[i+1].getImageArray()
            imn1=self._imageSet[i+1].getImageName()       
            self._imageSet[i].clearAll()
           
            print '\nFeature-tracking for images: ',imn0,' and ',imn1

            #Calculate velocities between image pair with homography
            if self._homog is not None:
                pts=calcVelocity(im0, im1, mask, [mtx,distort], 
                                 [self._homog[i][0],self._homog[i][3]], 
                                 invprojvars, back_thresh, maxpoints, quality, 
                                 mindist, min_features)                      
            else:
                pts=calcVelocity(im0, im1, mask, [mtx,distort], 
                                 [None, None], invprojvars, back_thresh, 
                                 maxpoints, quality, mindist, min_features)
                
            #Append output
            velocity.append(pts)         
        
        #Return XYZ and UV velocity information
        return velocity
        
        
    def getMask(self):
        '''Return image mask.'''
        return self._mask
 
 
    def getCamEnv(self):
        '''Return camera environment object (CamEnv).'''
        return self._camEnv
    

#------------------------------------------------------------------------------    

def calcVelocity(img1, img2, mask, calib=None, homog=None, invprojvars=None, 
                 back_thresh=1.0, maxpoints=50000, quality=0.1, mindist=5.0, 
                 min_features=4):
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
    points, ptserrors = featureTrack(img1, img2, mask,
                                     back_thresh=back_thresh, 
                                     maxpoints=maxpoints, 
                                     quality=quality,
                                     mindist=mindist, 
                                     min_features=min_features) 
    
    #Pass empty object if tracking was insufficient
    if points==None:
        print '\nNo features to undertake velocity measurements'
        return None        
        
    if calib is not None:        
        #Calculate optimal camera matrix 
        size=img1.shape
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(calib[0], 
                                                    calib[1], 
                                                    (w,h), 1, (w,h))
        
        #Correct tracked points for image distortion. The displacement here 
        #is defined forwards (i.e. the points in image 1 are first 
        #corrected, followed by those in image 2)      
        #Correct points in first image 
        src_pts_corr=cv2.undistortPoints(points[0], 
                                         calib[0], 
                                         calib[1],P=newMat)
        
        #Correct points in second image                                         
        dst_pts_corr=cv2.undistortPoints(points[1], 
                                         calib[0], 
                                         calib[1],P=newMat) 
    else:
        src_pts_corr = points[0]
        dst_pts_corr = points[1]

    #Calculate homography-corrected pts if desired
    if homog is not None:
        
        #Get homography matrix and homography points
        hmatrix=homog[0]
        hpts=homog[1]
        
        #Apply perspective homography matrix to tracked points
        tracked=dst_pts_corr.shape[0]
        dst_pts_homog = apply_persp_homographyPts(dst_pts_corr,
                                                  hmatrix,
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
        
        print (str(dst_pts_corr.shape[0]) + 
               ' Points remaining after homography correction')

    else:
        #Original tracked points assigned if homography not given
        print 'Homography matrix not supplied. Original tracked points kept'
        dst_pts_homog=dst_pts_corr
    
    #Calculate pixel velocity
    pxvel=[]       
    for c,d in zip(src_pts_corr, dst_pts_homog):                        
        pxvel.append(np.sqrt((d[0][0]-c[0][0])*(d[0][0]-c[0][0])+
                     (d[0][1]-c[0][1])*(d[0][1]-c[0][1])))
        
    #Project good points (original and tracked) to obtain XYZ coordinates
    if invprojvars is not None:        
        #Project good points from image0
        uvs=src_pts_corr[:,0,:]
        xyzs=projectUV(uvs, invprojvars)
        
        #Project good points from image1
        uvd=dst_pts_homog[:,0,:]
        xyzd=projectUV(uvd, invprojvars)
        
        #Calculate xyz velocity
        xyzvel=[]
        for a,b in zip(xyzs, xyzd):                        
            xyzvel.append(np.sqrt((b[0]-a[0])*(b[0]-a[0])+
                          (b[1]-a[1])*(b[1]-a[1])))
    else:
        xyzs=None
        xyzd=None
        xyzvel=None
            
    #Return real-world point positions (original and tracked points),
    #and xy pixel positions (original, tracked, and homography-corrected)
    if hmatrix is not None:
        return [[xyzvel, xyzs, xyzd], 
                [pxvel, src_pts_corr, dst_pts_corr, dst_pts_homog]]
    
    else:
        return [[xyzvel, xyzs, xyzd], 
                [pxvel, src_pts_corr, dst_pts_corr, None]]
        
        
def calcHomography(img1, img2, mask, correct, method=cv2.RANSAC, 
                   ransacReprojThreshold=5.0, back_thresh=1.0, maxpoints=50000, 
                   quality=0.1, mindist=5.0, min_features=4):
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
    trackdata = featureTrack(img1, img2,
                             mask,
                             back_thresh=back_thresh, 
                             maxpoints=maxpoints, 
                             quality=quality,
                             mindist=mindist, 
                             min_features=min_features) 

    #Pass empty object if tracking insufficient
    if trackdata==None:
        print '\nNo features to undertake Homography'
        return None

    #Separate raw tracked points and errors            
    points, ptserrors=trackdata
    
    if correct is not None:
        
        #Calculate optimal camera matrix 
        size=img1.shape
        h = size[0]
        w = size[1]
        newMat, roi = cv2.getOptimalNewCameraMatrix(correct[0], 
                                                    correct[1], 
                                                    (w,h), 1, (w,h))
               
        #Correct tracked points for image distortion. The homgraphy here is 
        #defined forwards (i.e. the points in image 1 are first corrected, 
        #followed by those in image 2)        
        #Correct points in first image  
        src_pts_corr=cv2.undistortPoints(points[0], 
                                         correct[0], 
                                         correct[1],P=newMat)
        
        #Correct tracked points in second image
        dst_pts_corr=cv2.undistortPoints(points[1], 
                                         correct[0], 
                                         correct[1],P=newMat) 
    else:
        src_pts_corr = points[0]
        dst_pts_corr = points[1]
    
    #Find the homography between the two sets of corrected points
    homogMatrix, mask = cv2.findHomography(src_pts_corr, dst_pts_corr, 
                                           method, ransacReprojThreshold)
    
    #Calculate homography error
    #Apply global homography to source points
    homog_pts = apply_persp_homographyPts(src_pts_corr, homogMatrix, False)          

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
                
    return (homogMatrix, [src_pts_corr,dst_pts_corr,homog_pts], ptserrors, 
            homogerrors)


def apply_persp_homographyPts(pts, homog, inverse=False):        
    '''Funtion to apply a perspective homography to a sequence of 2D 
    values held in X and Y. The perspective homography is represented as a 
    3 X 3 matrix (homog). The source points are inputted as an array. The 
    homography perspective matrix is modelled in the same manner as done so 
    in OpenCV.
    
    Variables
    pts:                  Input point positions to correct.
    homog:                Perspective homography matrix.                                   
    inverse:              Flag to denote if perspective homography matrix 
                          needs inversing.
    
    Returns
    hpts:                 Corrected point positions.
    '''         
    if isinstance(pts,np.ndarray):
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
       
    elif isinstance(pts, list):
        hpts=[]
               
        if inverse:
            val,homog=cv2.invert(homog) 

        for p in pts:
            div=1./(homog[2][0]*p[0]+homog[2][1]*p[1]+homog[2][2])
            xh=(homog[0][0]*p[0]+homog[0][1]*p[1]+homog[0][2])*div
            yh=(homog[1][0]*p[0]+homog[1][1]*p[1]+homog[1][2])*div
            hpts.append([xh,yh])
    else:
        print 'PERPECTIVE INPUT:'
        print type(pts)
        hpts=None
              
        return hpts 
        

def featureTrack(i0, iN, mask, back_thresh=1.0, maxpoints=50000, quality=0.1, 
                 mindist=5.0, min_features=1):
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

    Variables
    i0 (arr):                   Image 1 in the image pair
    iN (arr):                   Image 2 in the image pair
    mask (arr):                 Image mask to seed points in
    back_thesh (int):           Threshold for back-tracking distance (i.e.
                                the difference between the original seeded
                                point and the back-tracked point in im0)
    maxpoints (int):            Maximum number of points to seed in im0
    quality (int):              Corner feature quality
    mindist (int):              Minimum distance between seeded points                
    min_features (int):         Minimum number of seeded points to track
    
    Returns
    p0 (arr):                   Point coordinates for points seeded in image 1
    p1 (arr):                   Point coordinates for points tracked to image 2
    p0r (arr):                  Point coordinates for points back-tracked
                                from image 2 to image 1
    error (arr):                SNR measurements for the corresponding tracked 
                                point. The signal is the magnitude of the 
                                displacement from p0 to p1, and the noise is 
                                the magnitude of the displacement from p0r to 
                                p0
    '''
    #Feature tracking set-up parameters
    lk_params = dict( winSize  = (25,25),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | 
                                  cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                                  
    #Find corners of the first image. p0 is returned as an array of shape 
    #(n,1,2), where n is the number of features identified 
    if mask is not None:       
        p0=cv2.goodFeaturesToTrack(i0,maxpoints,quality,mindist,mask=mask)
    else:
        p0=cv2.goodFeaturesToTrack(i0,maxpoints,quality,mindist)
        
    #tracked is the number of features returned by goodFeaturesToTrack        
    tracked=p0.shape[0]
            
    #Check if there are enough points to initially track 
    if tracked<min_features:
        print 'Not enough features found to track.  Found: ',len(p0)
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
        print 'Average back-tracking difference: ' + str(np.mean(good))

        #Return None if number of tracked features is under the 
        #min_features threshold
        if p0.shape[0]<min_features:
            print 'Not enough features successfully tracked.' 
            return None
           
    print str(tracked)+' features tracked'
    print str(p0.shape[0]) + ' features remaining after forward-backward error'
       
    #Calculate signal-to-noise error               
    #Signal-to-noise is defined as the ratio between the distance 
    #originally tracked and the error between the original and back-tracked 
    #point.
    dist=dist[good]
    length,snr=calcTrackErrors(p0,p1,dist)
    
    #Error to contain the original lengths, back-tracking error and snr
    error=[length,dist,snr]
        
    return [p0,p1,p0r], error


def calcTrackErrors(p0,p1,dist):
    '''Function to calculate signal-to-noise ratio with forward-backward 
    tracking data. The distance between the backtrack and original points
    (dist) is assumed to be pre-calcuated.
    
    Variables
    p0 (arr):                   Point coordinates for points seeded in 
                                image 1
    p1 (arr):                   Point coordinates for points tracked to
                                image 2
    dist (arr):                 Distance between p0 and p0r (i.e. points
                                back-tracked from image 2 to image 1)
                                                    
    Returns
    length (arr):               Displacement between p0 and p1 (i.e. a
                                velocity, or signal)
    snr (arr):                  Signal-to-noise ratio, the signal being
                                the variable 'length' and the noise being
                                the variable 'dist'
    '''               
    #Determine length between the two sets of points
    length=(p0-p1)*(p0-p1)
    length=np.sqrt(length[:,0,0]+length[:,0,1])
    
    #Calculate signal-to-noise ratio
    snr = dist/length
    
    return length,snr 