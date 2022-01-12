#PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie
#
#PyTrx is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

"""
The Images module contains the object-constructors and functions for: (1) 
Importing and handling image data, specifically RBG, one-band (R, B or G), and 
grayscale images; and (2) Handling image sequences (i.e. a set of multiple 
images).
"""

#Import packages
from pathlib import Path
import numpy as np
import operator
from PIL import Image 
from PIL.ExifTags import TAGS
from datetime import datetime
from pylab import array, uint8
from functools import reduce
import glob
import imghdr
import os
import cv2

#------------------------------------------------------------------------------

class CamImage(object):    
    """A class representing a raw single band (optical RGB or greyscale). This 
    CamImage object is used in subsequent timelapse analysis. The object 
    contains the image data, image path, image dimensions and timestamp 
    (derived from the image Exif data, if available).       
    Optionally the user can specify whether the red, blue or green values 
    should be used, or whether the images should be converted to grey scale 
    which is the default. No image calibration is undertaken at this point.
    The default grayscale band option ('l') applies an equalization filter 
    on the image whereas the RGB splits are raw RGB. This could be modified 
    to permit more sophisticated settings of RGB combinations and/or 
    filters with file reading.
    
    :param imagePath: The file path to a given image
    :type imagePath: str
    :param band: Specified image band to pass forward: 'r' - red band; 'b' - blue band; 'g' - green band; and 'l' - grayscale. Default to 'l'
    :type band: str
    :param equal: Flag denoting whether histogram equalisation is applied to images (histogram equalisation is applied if True). Default to True
    :type equal: bool, optional
    """   
    def __init__(self, imagePath, band='l', equal=True):
        '''CamImage constructor to set image path, read in image data in the 
        specified band and access Exif data.         
        '''
        #Define class properties
        self._imageGood = True
        self._band = band.upper()
        self._equal = equal
        self._imageArray = None
        self._image = None
        self._imsize = None
        self._timestamp = None
        self._impath = imagePath
        
        #Check image file path
        success=self._checkImage(imagePath)
        if not success:
            self._imageGood=False
            return

                    
    def imageGood(self):
        """Return image file path status."""
        return self._imageGood

        
    def clearImage(self):
        """Clear memory of image data."""
        self._image=None


    def clearImageArray(self):
        """Clear memory of image array data."""
        self._imageArray=None     

        
    def clearAll(self):
        """Clear memory of all retained data."""
        self._image=None
        self._imageArray=None      

     
    def _checkImage(self, path):
        """Check that the given image file path is correct."""
        print('\nChecking image file ' + str(path))
        
        #Check file path using os package
        exists=os.path.isfile(path) 
        if exists:
            
            #Check file type
            ftype=imghdr.what(path)
            if ftype is None:
                print('File exists but not image type: ' + str(ftype))
                return False
            else:
                print('File found of image type: ' + str(ftype))
                return True

        else:           
            print('File does not exist: ' + str(path))
            return False

        
    def getImageType(self):
        """Return the image file type."""
        return imghdr.what(self._impath)        


    def getImagePath(self):
        """Return the file path of the image."""        
        return self._impath


    def getImageName(self):
        """Return image name."""
        imn = self.getImagePath()
        imn = Path(imn)
        return imn.name   


    def getImage(self):
        """Return the image."""
        if self._image is None:
            self._readImage()
        return self._image

    
    def getImageCorr(self, cameraMatrix, distortP):
        """Return the image array that is corrected for the specificied 
        camera matrix and distortion parameters.
        
        :param cameraMatrix: Intrinsic camera matrix
        :type cameraMatrix: arr
        :param distortP: Lens distortion parameters
        :type distortP: arr
        :returns: Image corrected for image distortion (arr)
        :rtype: arr
        """
        #Get image array        
        if self._imageArray is None:
            self._readImageData()
            
        size=self.getImageSize()      
        
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
        """Return the image as an array."""
        if self._imageArray is None:
            self._readImageData()   
        return self._imageArray
 

    def getImageEnhance(self, diff, phi, theta):
        """Return enhanced image using :func:`PyTrx.Images.enhanceImg` function.

        :param diff: Inputted as either 'light or 'dark', signifying the intensity of the image pixels. 'light' increases the intensity such that dark pixels become much brighter and bright pixels become slightly brighter. 'dark' decreases the intensity such that dark pixels become much darker and bright pixels become slightly darker.
        :type diff: str
        :param phi: Defines the intensity of all pixel values
        :type phi: int
        :param theta: Defines the number of "colours" in the image, e.g. 3 signifies that all the pixels will be grouped into one of three pixel values
        :type theta: int         
        :returns: Enhanced image
        :rtype: arr
        """
        image = self.getImageArray()
        image1 = enhanceImage(image, diff, phi, theta)
        return image1

       
    def getImageSize(self):
        """Return the size of the image (which is obtained from the image Exif 
        information)."""        
        if self._imsize is None:
            self._imsize,self._timestamp=self.getExif()
        return self._imsize

        
    def getImageTime(self):
        """Return the time of the image (which is obtained from the image Exif
        information)."""       
        if self._timestamp is None:
            self._imsize,self._timestamp=self.getExif()
        return self._timestamp
 
       
    def getExif(self):
        """Return the exif image size and time stamp data from the image. Image
        size is returned as a string (height, width). The time stamp is
        returned as a Python datetime object."""
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
            print ('\nUnable to get valid timestamp for image file: '
                   + self._impath)
            timestamp=None
            
        return imsize, timestamp      

        
    def changeBand(self,band):
        """Change the band you want the image to represent ('r', 'b', 'g' or 'l')
        
        :param band: Image band ('r', 'b', 'g', or 'l')
        :type band: str
        """
        self._band=band.upper()
        self._readImageData()
        
        
    def reportCamImageData(self):
        """Report image data (file path, image size and datetime)."""
        print('\nImage source path: ' + str(self.getImagePath()))
        print('Image size: ' + str(self.getImageSize()))
        print('Image datetime: ' + str(self.getImageTime()))

        
    def _readImage(self):
        """Read image from file path using PIL."""
        self._image=Image.open(self._impath)
    
    
    def _readImageData(self):
        """Function to prepare an image by opening, equalising, converting to 
        a desired band or grayscale, then returning a copy."""               
        #Open image from file using PIL        
        if self._image is None:
            self._image = Image.open(self._impath)
        
        img = self._image
        
        if self._equal is True:
            
            #Apply histogram equalisation
            h = img.convert('L').histogram()
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
            img = img.point(lut*img.layers)
        
        if self._band == 'R':
            img,g,b = img.split()
        elif self._band == 'G':
            r,img,b = img.split() 
        elif self._band == 'B':
            r,g,img = img.split() 
        else:
            img = img.convert('L')
        
        #Copy image array
        self._imageArray = np.array(img).copy()
                

#------------------------------------------------------------------------------
        
class ImageSequence(object):
    """A class to model a raw collection of CamImage objects, which can 
    subsequently be used for making photogrammetric measurements from.
      
    :param imageList: The list of images, which can be passed in 3 ways: 1) As a list of :class:`PyTrx.Image.CamImage` objects; 2) As a list of image paths; and 3) As a folder containing images
    :type imageList: str
    :param band: Image band ('r', 'b', 'g', or 'l'), default to 'l'
    :type band: str, optional
    :param equal: Flag denoting whether histogram equalisation is applied to images (histogram equalisation is applied if True). Default to True
    :type equal: bool, optional
    """
    def __init__(self, imageList, band='L', equal=True):
        print('\n\nCONSTRUCTING IMAGE SEQUENCE')
        
        self._band=band
        self._equal=equal
        self._imageList=imageList
        
        #Construct image set (as CamImage objects)
        if isinstance(imageList, list): 
            
            #Construction from list of CamImage objects
            if isinstance(imageList[0],CamImage):
                print('\nList of camera images assumed in image sequence')
                print(' Attempting to add all to sequence')
                self._imageSet = []
                for item in list:
                    if isinstance(item,CamImage):
                        self._imageSet.append(item)
                    else:
                        print('\nWarning non-image item found in image set' 
                              ' list specification - item not added')
                return
                
            #Construction from list containing file name strings                
            elif isinstance(imageList[0],str):               
                print('\nList of camera images assumed of image sequence')
                print(' Attempting to add all to sequence')
                self._loadImageStringSequence(imageList)
                
            else:                
                print ('\nList item type used to define image list neither' 
                           ' image nor strings (filenames)')
                return None
        
        #Construction from string of file paths
        if isinstance(imageList, str):
            print('\nImage directory path assumed. Searching for images.' + 
                  ' Attempting to add all to sequence')
            print(str(imageList))
            self._imageList = sorted(glob.glob(imageList))
#                                     key=os.path.getmtime)
            self._loadImageStringSequence(self._imageList)
            
            
    def getImageArrNo(self,i):
        """Get image array i from image sequence
        
        :param i: Image number in sequence
        :type i: int
        :returns: Image array
        :rtype: arr
        """
        im=self._imageSet[i]
        arr=im.getImageArray()
        im.clearAll()
        return arr

    
    def getImageObj(self,i):
        """Get :class:`PyTrx.Images.CamImage` object i from image sequence
        
        :param i: Image number in sequence
        :type i: int
        :returns: :class:`PyTrx.Images.CamImage` 
        :rtype: arr
	    """
        imo=self._imageSet[i] 
        return imo

        
    def _loadImageStringSequence(self,imageList):
        """Function for generating an image set (of :class:`PyTrx.Images.CamImage`
        objects) from a list of images."""      
        #Construct CamImage objects
        self._imageSet = []
        for imageStr in imageList:
            im=CamImage(imageStr, self._band, self._equal)
            
            #Append image filepath if filepath is true
            if im.imageGood():
                self._imageSet.append(im)
                    
            else:
                print('\nProblem reading image: ' + str(imageStr))
                print('Image: ' + str(imageStr) + ' not added to sequence')

                
    def getImages(self):
        """Return image set (i.e. a sequence of CamImage objects)."""
        return self._imageSet

        
    def getImageFileList(self):
        """Return list of image file paths."""
        return self._imageList


    def getImageNames(self):
        """Return list of image file names."""
        imgf = self.getImageFileList()
        imns = []
        for i in imgf:
            imn=Path(i)
            imns.append(imn.name)
        return imns

        
    def getLength(self):
        """Return length of image set."""
        return len(self._imageSet)


def enhanceImage(img, diff, phi, theta):
    """Change brightness and contrast of image using phi and theta variables. 
    Change phi and theta values accordingly.
    
    :param img: Input image array for enhancement
    :type img: arr
    :param diff: Inputted as either 'light or 'dark', signifying the intensity of the image pixels. 'light' increases the intensity such that dark pixels become much brighter and bright pixels become slightly brighter. 'dark' decreases the intensity such that dark pixels become much darker and bright pixels become slightly darker.
    :type diff: str
    :param phi: Defines the intensity of all pixel values
    :type phi: int
    :param theta: Defines the number of "colours" in the image, e.g. 3 signifies that all the pixels will be grouped into one of three pixel values
    :type theta: int               .
    :returns: Enhanced image.
    :rtype: arr
    """                          
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
        print('\nInvalid diff variable')
        print('Re-assigning diff variable to "light"')
        img1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
        img1 = array(img1, dtype = uint8)
    
    #Return enhanced image
    return img1


#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'        

#------------------------------------------------------------------------------
