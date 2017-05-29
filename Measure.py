'''
PYTRX MEASURE MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

The Measure module 

@author: Penny How, p.how@ed.ac.uk
'''

import matplotlib.pyplot as mp
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import cv2
import sys
from PIL import Image as Img
from pylab import array, uint8
from osgeo import ogr, osr
import os

from FileHandler import readMask
from Images import TimeLapse
from CamEnv import CamEnv


#------------------------------------------------------------------------------

class Area(TimeLapse):
    '''A class for processing change in area (i.e. a lake or plume) through an 
    image sequence, with methods to calculate extent change in an image plane 
    (px) and real areal change via georectification.
    
    Inputs:
        -imageList: List of images, for the ImageSet object
        
        -CamEnv: Camera environment parameters:= which can be read into the 
         CamEnv class as a text file (see CamEnv documentation for exact 
         information needed)
         
        -maskPath: The file path for the mask inidcating the region where areal 
         extent should be recorded. If this does exist, the mask will be if
         this is inputted as a filedirectory to a jpg file. If the file does
         not exist, then the mask generation process will load and the mask 
         will be saved to the given directory. If no directory is specified 
         (i.e. maxMaskPath=None), the mask generation process will load but the
         result will not be saved
         
        -image0: The image number from the ImageSet for which the analysis
         should commence
         
        -calibFlag: An indicator of whether images are calibrated, for the 
         ImageSet object'''
         

#---------------------------   INITIALISATION  --------------------------------
#The Area class is initialised using a list of image filepaths and a Camera
#Environment object (see CamEnv.py for more information on how to set up a 
#Camera Environment). An optional input is also a filepath to a mask that will 
#be used to initially mask images for automated detection of target areas.
     
    def __init__(self, imageList, cameraenv, maxMaskPath=None, calibFlag=True, 
                 band='L', loadall=False, timingMethod='EXIF'):
        
        TimeLapse.__init__(self, imageList, cameraenv, None, None, 0, band, 
                           loadall, timingMethod)
        
        self._maximg = None
        self._pxplot = None
        self._colourrange = None
        self._threshold = None
        self._method = None
        self._pxpoly = None
        self._pxextent = None
        self._realpoly = None
        self._area = None
        self._enhance = None
        self._calibFlag = calibFlag
        
        if maxMaskPath==None:
            self._maxMaskPath=None
        else:
            self._maxMaskPath=maxMaskPath
            self.setMaxMask()


#--------------------------   MASTER FUNCTIONS  -------------------------------
#If you just want to calculate pixel extents and real world areas then these 
#functions will get you there without having to delve into the nitty gritty
#sections of code. Pixel and real world measurements are determined using an
#automated algorithm that defines areas based on RBG pixel intensity. Change
#the method variable to 'manual' to define areas manually.
    
    def getRealPoly(self, method='auto'):
        '''Return real world point coordinates of all extent polygons in all 
        images. Extents are defined automatically unless the method variable
        is changed for the string "manual".'''
        if self._realpoly is None:
            self.calcAreas(method)
        return self._realpoly

    
    def getRealPolyN(self, number, method='auto'):
        '''Return real world point coordinates of all extent polygons in a given 
        image (specified by the number variable as a single value). Extents are 
        defined automatically unless the method variable is changed for the 
        string "manual".'''
        if self._realpoly is None:
            self.calcAreas(method)
        return self._realpoly[number]
    

    def getArea(self, method='auto'):
        '''Return real world area of all extent polygons in all images 
        (calculate if not already determined). Extents are defined 
        automatically unless the method variable is changed for the string 
        "manual".'''
        if self._area is None:
            self.calcAreas(method)
        return self._area


    def getAreaN(self, number, method='auto'):
        '''Return real world area of all extent polygons in a given image 
        (specified by the number variable as a single value). Extents are 
        defined automatically unless the method variable is changed for the 
        string "manual".'''
        if self._area is None:
            self.calcAreas(method)
        return self._area[number]
    
    
    def getSumArea(self, method='auto'):
        '''Return cumulative area of all extent polygons in all images 
        (calculate if not already determined). Extents are defined 
        automatically unless the method variable is changed for the string 
        "manual".'''
        if self._area is None:
            self.calcAreas(method)            
        sumarea = []
        for a in self._area:
            all_areas = sum(a)
            sumarea.append(all_areas)        
        return sumarea

    
    def getSumAreaN(self, number, method='auto'):
        '''Return cumulative area of all extent polygons in a given image 
        (specified by the number variable as a single value). Extents are 
        defined automatically unless the method variable is changed for the 
        string "manual".'''
        if self._area is None:
            self.calcAreas(method)
        all_areas = sum(self._area[number])
        return all_areas    

           
    def getExtent(self, method='auto'):
        '''Return all pixel extent values (calculate if not already 
        determined).Extents are defined automatically unless the method 
        variable is changed for the string "manual".'''
        if self._pxextent is None:
            if method is 'auto' or 'Auto' or 'a' or 'A' or 'automated' or 'Automated':
                self.calcExtents()
            elif method is 'manual' or 'Manual' or 'm' or 'M':
                self.manualExtents()
            else:
                print 'Invalid method for extent defintion'
                print "Method input is either 'auto' or 'manual'"
                sys.exit(1)
        return self._pxextent


    def getExtentN(self, number, method='auto'):
        '''Return extent value for an image number. Extents are defined 
        automatically unless the method variableis changed for the string 
        "manual".'''
        if self._pxextent is None:
            if method is 'auto' or 'Auto' or 'a' or 'A' or 'automated' or 'Automated':
                self.calcExtents()
            elif method is 'manual' or 'Manual' or 'm' or 'M':
                self.manualExtents()
            else:
                print 'Invalid method for extent defintion'
                print "Method input is either 'auto' or 'manual'"
                sys.exit(1)
        return self._pxextent[number]
     
    
    def getPxPoly(self, method='auto'):
        '''Return all pixel extent polygons (calculate if not already 
        determined).Extents are defined automatically unless the method 
        variable is changed for the string "manual".'''
        if self._pxpoly is None:
            if method is 'auto' or 'Auto' or 'a' or 'A' or 'automated' or 'Automated':
                self.calcExtents()
            elif method is 'manual' or 'Manual' or 'm' or 'M':
                self.manualExtents()
            else:
                print 'Invalid method for extent defintion'
                print "Method input is either 'auto' or 'manual'"
                sys.exit(1)
        return self._pxpoly
        
        
    def getPxPolyN(self, number, method='auto'):
        '''Return pixel extent polygons for an image number. Extents are 
        defined automatically unless the method variable is changed for the 
        string "manual".'''
        if self._pxpoly is None:
            if method is 'auto' or 'Auto' or 'a' or 'A' or 'automated' or 'Automated':
                self.calcExtents()
            elif method is 'manual' or 'Manual' or 'm' or 'M':
                self.manualExtents()
            else:
                print 'Invalid method for extent defintion'
                print "Method input is either 'auto' or 'manual'"
                sys.exit(1)
        return self._pxpoly[number]
           
            
#-----------------------   IMAGE MASKING FUNCTIONS  ---------------------------
#This collection of functions deal with masking the image for automated
#detection of target areas. By masking the image, it makes it easier to 
#distinguish an area of interest. If a .jpg mask has been included in the Area
#class initialisation, then that mask will be used. If not, the mask is set 
#using the image in the sequence with the largest area of interest. This can 
#be set using the setMaxImg function (if it is not set then it is automatically
#set to the first image of the sequence). The mask is manually defined using 
#this image.
        
    def setMaxImg(self, number):
        '''Set image with largest extent. This image is used for mask definition 
        and setting enhancement parameters. If maximg is undefined, it is 
        automatically set to the first image in the sequence.'''
        #Get image sequence
        img_len = self.getLength() 
        
        #Exit programme if image sequence is not found or assigned maximum 
        #image reference exceeds the number of images in the sequence
        if img_len == 0:
            print 'Images not found'
            sys.exit(1)
        elif number >= img_len:
            print 'Please choose a maximum image number less than ' + str(img_len)
            sys.exit(1)
        
        #Assign image with maximum extents
        self._maximg = number
        
    
    def getMaxImgData(self):
        '''Get data for the image with the largest extent.'''       
        #If MaxImg is not set, then it is automatically assigned to the first
        #image in the sequence
        if self._maximg is None:
            self.setMaxImg(0)

        #Call image (correct or uncorrected)        
        if self._calibFlag is True:
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCv2()
            return self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                            distortP)
        
        else:
            return self._imageSet[self._maximg].getImageArray()

    
    def setMaxMask(self):
        ''' Set mask for tracking areal extent using the image with the 
        largest extent function. Click around the target extent using the left
        click on a mouse. Right click will delete the previous point. Press
        Enter when you have completed the mask that you wish to use - this will
        save the points. Then exit the window to continue the program.'''
        #If a mask has not been set in the initialisation of the Extents class
        #then the mask will be manually defined using the assigned maxImg
        self._mask = readMask(self.getMaxImgData(), self._maxMaskPath)

        
    def maskImg(self, img):
        '''Mask images using the largest extent mask (boolean object). Unlike 
        the masking function in the TimeLapse class, the boolean mask is used
        to reassign overlapping image pixels to zero. The masking function in 
        the TimeLapse class uses a numpy masking object (numpy.ma).'''
        #Get mask from TimeLapse class
        if self._mask is None:
            self.setMaxMask()
            
        #Mask the glacier
        booleanMask = np.array(self._mask, dtype=bool)
        booleanMask = np.invert(booleanMask)
        
        #Copy properties of img
#        img2 = np.copy(img)
        
        #Mask extent image with boolean array
        np.where(booleanMask, 0, img) #fit arrays to each other
        img[booleanMask] = 0 #mask image with boolean mask object

        return img
        

#---------------------   IMAGE MODIFICATION FUNCTIONS  ------------------------
#The following functions are used in the automated extent detection to enhance 
#an image in order to better distinguish the target extent. This enhancement 
#largely involves changing the brightness and contrast of an image using phi 
#and theta variables. These can either be set directly using the setEnhance
#function, or can be previewed using the seeEnhance function.
       
    def enhanceImg (self, img):
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
            print 'Invalid diff variable' 
            sys.exit(1)
            
        return img1


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
        #Get maximum extent image and plot
        img=self.getMaxImgData()          

        fig,ax = mp.subplots()
        mp.subplots_adjust(left=0.25, bottom=0.25)
        ax = mp.subplot(111)
        ax.imshow(img)
        
        diff = 'light'  
                
        #Inititalise sliders for phi and theta
        axcolor = 'lightgoldenrodyellow'
        axphi  = mp.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        axtheta = mp.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
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
        resetax = mp.axes([0.8, 0.025, 0.1, 0.04])
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
                
        mp.show()
        mp.close()
        
        print sphi.val
        print stheta.val
        self.setEnhance('light', sphi.val, stheta.val)
 
       
#------------------------ EXTENT DETECTION FUNCTIONS  -------------------------
#Functions for manual and automated detection of areas in the image plane. The
#majority of the heavy-lifting is done by either the calcExtents or 
#manualExtents functions, which define the extents through an image sequence.
#These two functions also directly feed into the Areas class, defining the
#pxpoly and pxextents objects in the class which can be subsequently used in
#the transformation functions, plotting functions and exporting functions.
#The other functions are associated with the extent detection method 
#(setMethod, getMethod), threshold for the number of areas detected 
#(setThreshold, getThreshold), and the pixel colourrange that is used to define
#target areas (setColourrange, getColourrange, defineColourrange and getRBG).
    
    def calcExtents(self, color=False, verify=False):
        '''Get pixel extent from a series of images. Return the extent polygons
        and cumulative extent values (px).'''
        #Set extent detection method
        self.setMethod('auto')
        
        #Only define color range once
        if color is False:        
            #Get image with maximum extent. If this is undefined, the image is
            #automatically set to the first image
            maximg = self.getMaxImgData() 
              
            #Get mask. If mask is undefined, the mask is manually set from the 
            #first image in the sequence
            self.getMask()
            if self._mask is None:
                self.setMaxMask()        
                   
            #get colour range for extent detection
            if self._colourrange is None:
                maxmsk = self.maskImg(maximg)
                if self._enhance is not None:
                    maxmsk = self.enhanceImg(maxmsk)
                self.defineColourrange(maxmsk)    
            
            #Set up output dataset
            areas=[]        
            px=[]
                    
            #Crop all images, find extent in all images        
            for i in range(self._imageN):
                img1 = self._imageSet[i].getImageArray() 
                img2 = np.copy(img1)
                img2 = self.maskImg(img2)
                if self._enhance is not None:
                    img2 = self.enhanceImg(img2)
                polys,extent = self.calcExtent(img2)        
                areas.append(polys)
                px.append(extent)
        
            #Return all extents, all cropped images and corresponding image names
            self._pxpoly = areas
            self._pxextent = px
        
        #Define color range for each image
        elif color is True:
            #Set up output dataset
            areas=[]        
            px=[]             
            
            #Get mask. If mask is undefined, the mask is manually set from the 
            #first image in the sequence
            if self._mask is None:
                self.setMaxMask()        
                    
            #For all images in the sequence        
            for i in range(self._imageN):

                #Call corrected/uncorrected image
                if self._calibFlag is True:
                    img1=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                        self._camEnv.getDistortCoeffsCv2())      
                else:
                    img1=self._imageSet[i].getImageArray()
                
                #Mask and enhance image
                img2 = np.copy(img1)
                img2 = self.maskImg(img2)
                if self._enhance is not None:
                    img2 = self.enhanceImg(img2)
                
                #Define extent
                self.defineColourrange(img2)
                polys,extent = self.calcExtent(img2)        
                areas.append(polys)
                px.append(extent)
            
            #Retain polygons and extents
            self._pxpoly = areas
            self._pxextent = px
        
        #Manually verify polygons if verify flag is True
        if verify is True:
            self.verifyExtents()
                
        return self._pxpoly, self._pxextent

        
    def verifyExtents(self):
        '''Manually verify all polygons in images.'''
        #Create output
        verified = []
        update_ext = []
        
        #Get pixel polygons
        pxpoly = self.getPxPoly()
        
        #Verify pixel polygons in each image        
        for i,px in zip((range(self._imageN)),pxpoly):
            
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img1=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                    self._camEnv.getDistortCoeffsCv2())      
            else:
                img1=self._imageSet[i].getImageArray()            

            #Verify polygons
            img2 = np.copy(img1)
            print self.getImagePath(i)
            verf = self.verifyExtent(img2,px)
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
        
        #Reset method (which indicates how the data is structured)
        self.setMethod('manual')
        
        #Rewrite pixel polygon and extent data
        self._pxpoly = verified
        self._pxextent = update_ext
    
        
    def manualExtents(self):
        '''Get manual pixel extent from a series of images. Return the 
        extent polygons and cumulative extent values (px).''' 
        #Set extent detection method
        self.setMethod('manual')
                
        #Set up output dataset
        areas=[]        
        px=[]
                
        #Crop all images, find extent in all images        
        for i in range(self._imageN):           
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                    self._camEnv.getDistortCoeffsCv2())      
            else:
                img=self._imageSet[i].getImageArray()          
            polys,extent = self.manualExtent(img)        
            areas.append(polys)
            px.append(extent)
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
        #Define colour range if not already set           
        if self._colourrange is None:
            self.defineColourrange(img)
        
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
        
        print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px) \n'
        
        return pxpoly, pxextent
        

    def manualExtent(self, img):
        '''Manually define extent by clicking around region in target image.'''
         #Manual interaction to select lightest and darkest part of the region
#        pts=[]        
#        while len(pts) < 3:
        fig=mp.gcf()
        fig.canvas.set_window_title('Click around region. Press enter to record points.')
        mp.imshow(img, origin='upper', cmap='gray')
        if self._pxplot is not None:
            mp.axis([self._pxplot[0],self._pxplot[1],self._pxplot[2],self._pxplot[3]])                   
        pts = mp.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, 
                        mouse_pop=3, mouse_stop=2)
        print 'you clicked:', pts
        mp.show()
        mp.close()
        
#            #Reboot window if =<2 points are recorded
#            if len(pts) > 3:
#                pts = []
            
        #Create polygon if area has been recorded   
        try:
            pts.append(pts[0])    #completes the polygon    
            ring = ogr.Geometry(ogr.wkbLinearRing)   
            for p in pts:
                ring.AddPoint(p[0],p[1])
            p=pts[0]
            ring.AddPoint(p[0],p[1])
            pxpoly = ogr.Geometry(ogr.wkbPolygon)
            pxpoly.AddGeometry(ring) #create polygon ring
            pxextent = pxpoly.Area()
        except:
            pxextent = 0
                   
        #Get image dimensions
        h = img.shape[0]
        w = img.shape[1]
        pxcount = h*w
        print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px) \n'    #print polygon area
        
        #convert pts list to array
        pts = np.array(pts)           
        pts=[pts]
        
        return pts, pxextent


    def verifyExtent(self, img, pxpoly):
        '''Verify detected areas.'''
        if 1:             
            print 'Verifying detected areas...'
            verified = []
            
            def onpick(event):
                v = []
                thisline = event.artist
                xdata = thisline.get_xdata()
                ydata = thisline.get_ydata()
                for x,y in zip(xdata,ydata):
                    v.append([x,y])
                v2=np.array(v, dtype=np.int32).reshape((len(xdata)),2)
                verified.append(v2)
                ind=event.ind
                print 'Verified extent at ' + str(np.take(xdata, ind)[0]) + ', ' + str(np.take(ydata, ind)[0])
                
            fig, ax1 =mp.subplots()
            fig.canvas.set_window_title('Click on valid areas.')
            ax1.imshow(img, cmap='gray')
            if self._pxplot is not None:
                ax1.axis([self._pxplot[0],self._pxplot[1],self._pxplot[2],self._pxplot[3]])
            for a in pxpoly:
                x=[]
                y=[]
                for b in a:
                    for c in b:
                        x.append(c[0])
                        y.append(c[1])
                line = Line2D(x, y, linestyle='-', color='y', picker=True)
                ax1.add_line(line)
            fig.canvas.mpl_connect('pick_event', onpick)
        
        mp.show()
        return verified
    
    
    def setMethod(self, method):
        '''Set method for extent detection.
        Inputs:
        -method: String signifiying method for extent detection. 
         'auto' = automated extent detection
         'manual' = manual extent detection'''
        self._method = method

    
    def getMethod(self):
        '''Return method for extent detection - automated ('auto') or manual
        ('manual')'''
        return self._method


    def setThreshold(self, number):
        '''Set threshold for number of polgons kept from an image.'''
        self._threshold = number
         
         
    def getThreshold(self):
        '''Return threshold for number of polgons kept from an image.'''
        return self._threshold
                       

    def setColourrange(self, upper, lower):
        '''Manually define the RBG colour range that will be used to filter
        the image/images. Input the upper boundary (i.e. the highest value) 
        first.'''
        print 'Colour range defined from given values:'
        print 'Upper RBG boundary: ', upper
        print 'Lower RBG boundary: ', lower, '\n'            
        self._colourrange = [upper, lower]
        
        
    def getColourrange(self):
        '''Return the set colour range for extent.'''
        return self._colourrange

                
    def defineColourrange(self, img):
        '''Define colour range manually by clicking on the lightest and 
        darkest regions of the extent that will be tracked. Click the lightest
        part first, then the darkest.
        Operations:
        -Left click to select
        -Right click to undo
        -Close the image window to continue
        -The window automatically times out after two clicks'''
        #Manual interaction to select lightest and darkest part of the region
        fig=mp.gcf()
        fig.canvas.set_window_title('Click lightest colour and darkest colour')
        mp.imshow(img, origin='upper')

        if self._pxplot is not None:
            mp.axis([self._pxplot[0],self._pxplot[1],self._pxplot[2],self._pxplot[3]])
            
        colours = mp.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, 
                            mouse_pop=3, mouse_stop=2)
        print 'you clicked:', colours
        mp.show()
    
        col1 = colours[0]
        col2 = colours[1]
        
        #Obtain coordinates from selected light point
        col1_y = col1[0]
        col1_x = col1[1]
    
        #Obtain coordinates from selected dark point
        col2_y = col2[0]
        col2_x = col2[1]
    
        #Get RBG values from given coordinates
        col1_rbg = self.getRBG(img, col1_x, col1_y)
        col2_rbg = self.getRBG(img, col2_x, col2_y) 
        
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
            print 'Unregonised RBG range.'
            sys.exit(1)
            
        print 'Colour range found from manual selection'
        print 'Upper RBG boundary: ', upper_boundary
        print 'Lower RBG boundary: ', lower_boundary, '\n'

        #Store RBG range
        self._colourrange = [upper_boundary, lower_boundary]
        
        
    def getRBG(self, img, x, y):
        '''Return the RBG value for a given point (x,y) in an image (img).'''
        RBG = img[x,y]    
        return RBG


#-------------------------   PROJECTION FUNCTIONS  ----------------------------
#Functions for transforming pixel area information into real world areas and
#coordinates. The function calcAreas will calculate real world areas and 
#coordinates for all pixel areas detected from an image sequence. Use the 
#calcAreas function to also set up the real world polygon coordinates object
#(self._realpoly) and real world polygon areas object (self._area). 
#The inversion is performed by the invproject function in the CamEnv, which is 
#called in the calcXYZ function in this section (see CamEnv.py for more 
#information on the georectification method).
   
    def calcAreas(self, method='auto', color=False, verify=False):
        '''Get real world areas from an image set. Calculates the polygon 
        extents for each image and the area of each given polygon.'''
        if method is 'auto':        
            print 'Automated method set'
            self.setMethod('auto')
        elif method is 'manual':
            print 'Manual method set'
            self.setMethod('manual')
        else:
            print 'Invalid method for extent defintion'
            print "Method input is either 'auto' or 'manual'"
            sys.exit(1)
        
        print 'Starting area calculations'
        
        #Get pixel polygons
        if self._method is 'auto':
            if self._pxpoly is None:
                self.calcExtents(color, verify)
            xyz = []
            area = []
            for p in self._pxpoly:
                pts, a = self.calcArea(p)
                xyz.append(pts)
                area.append(a)
        
        elif self._method is 'manual':
            if self._pxpoly is None:
                self.manualExtents()
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
        xyz = []   
        area = []                               #Create outputs

        for p in pxpoly:                        
            allxyz = self._camEnv.invproject(p)
            xyz.append(allxyz)                  #Project image coordinates
           
        rpoly = self.ogrPolyN(xyz)              #Create polygons
           
        for r in rpoly:
            area.append(r.Area())               #Determine area of each polygon
           
        return xyz, area  
   
    
    def ogrPolyN(self, xyz):
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
        
        
#--------------------------   PLOTTING FUNCTIONS  -----------------------------
#Plotting functions for visualising the pixel areas on an image (plotPX) and 
#the real world areas on a DEM (plotXYZ). Use the number variable to define 
#which image in the sequence you want to plot along with its corresponding
#areas (defined as a integer e.g. number=1). The plot will be saved to a 
#specific filepath if defined in the dest variable.
#e.g. C:/python_workspace/pytrx/results/plot_output.jpg
#A simple for loop can be used to plot pixel/real areas in all images.
        
    def plotPX(self, number, dest=None, crop=False):
        '''Return image overlayed with pixel extent polygons for a given image 
        number.'''
        #Call corrected/uncorrected image
        if self._calibFlag is True:
            img=self._imageSet[number].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                self._camEnv.getDistortCoeffsCv2())      
        else:
            img=self._imageSet[number].getImageArray() 
                      
        #Create image plotting window
        fig=mp.gcf()
        fig.canvas.set_window_title('Image extent output '+str(number))
        imgplot = mp.imshow(img)        
        imgplot.set_cmap('gray')
        mp.axis('off')
        
        if crop is True:
            if self._pxplot is not None:
                mp.axis([self._pxplot[0],self._pxplot[1],self._pxplot[2],
                         self._pxplot[3]])
        
        polys = self._pxpoly[number]

        for p in polys:
#            pts = np.vstack(p).squeeze()
            x=[]
            y=[]
            for xy in p:
                x.append(xy[0])
                y.append(xy[1])            
            mp.plot(x,y,'w-')                

        if dest != None:
            mp.savefig(dest + 'extentimg_' + str(number) + '.jpg') 
            
#        mp.show()  
        mp.close()
        
        
    def plotXYZ(self, number, dest=None, dem=True, show=True):
        '''Plot xyz points of real polygons for a given image number'''                       
        #Get xyz points for polygons in a given image
        xyz = self._realpoly[number]
                
        #Prepare DEM
        if dem is True:
            demobj=self._camEnv.getDEM()
            demextent=demobj.getExtent()
            dem=demobj.getZ()
       
            #Get camera position (from the CamEnv class)
            print self._camEnv._camloc
            post = self._camEnv._camloc            
        
            #Plot DEM 
            fig=mp.gcf()
            fig.canvas.set_window_title('Area output '+str(number))
            mp.locator_params(axis = 'x', nbins=8)
            mp.tick_params(axis='both', which='major', labelsize=10)
            mp.imshow(dem, origin='lower', extent=demextent, cmap='gray')
            mp.scatter(post[0], post[1], c='g')
        
        #Extract xy data from poly pts
        count=1                
        for shp in xyz: 
            xl=[]
            yl=[]
            for pt in shp:
                xl.append(pt[0])
                yl.append(pt[1])
            lab = 'Area ' + str(count)
            mp.plot(xl, yl, c=np.random.rand(3,1), linestyle='-', label=lab)
            count=count+1
        
        mp.legend()
        mp.suptitle('Projected extents', fontsize=14)
        
        if dest != None:
            mp.savefig(dest + 'area_' + str(number) + '.jpg')        

        if show is True:
            mp.show()

        mp.close()


    def setPXExt(self,xmin,xmax,ymin,ymax):
        '''Set plotting extent (makes it easier for defining colour ranges and
        verifying areas).'''
        self._pxplot = [xmin,xmax,ymin,ymax]


#------------------------------------------------------------------------------

class Length(TimeLapse):
    '''Class for point and line measurements.'''             
    def __init__(self, imageList, CamEnv, maxMaskPath=None, image0=0, calibFlag=False):
        TimeLapse.__init__(self, imageList, CamEnv, None, None, image0, calibFlag)
        
        self._maximg = None
        self._colourrange = None
        self._threshold = None
        self._method = None
        self._enhance = None
        
        self._pxpts = None
        self._pxline = None
        self._realpts = None
        self._realline = None
                
        if maxMaskPath==None:
            self._maxMaskPath=None
        else:
            self._maxMaskPath=maxMaskPath
            self.setMaxMask()


#-----------------------   IMAGE MASKING FUNCTIONS   --------------------------
#This collection of functions deal with masking the image for automated
#detection of target areas. By masking the image, it makes it easier to 
#distinguish an area of interest. If a .jpg mask has been included in the Area
#class initialisation, then that mask will be used. If not, the mask is set 
#using the image in the sequence with the largest area of interest. This can 
#be set using the setMaxImg function (if it is not set then it is automatically
#set to the first image of the sequence). The mask is manually defined using 
#this image.
    
    def setMaxImg(self, number):
        '''Set image with largest extent. This image is used for mask definition 
        and setting enhancement parameters. If maximg is undefined, it is 
        automatically set to the first image in the sequence.'''
        img_len = self.getLength()        
        if number >= img_len:
            print 'Please choose a maximum image number less than ' + str(img_len)
            sys.exit(1)
        self._maximg = number
        
    
    def getMaxImgData(self):
        '''Get data for the image with the largest extent.'''       
        #If MaxImg is not set, then it is automatically assigned to the first
        #image in the sequence
        if self._maximg is None:
            self.setMaxImg(0)
        return self.getImageNo(self._maximg)

    
    def setMaxMask(self):
        ''' Set mask for tracking areal extent using the image with the 
        largest extent function. Click around the target extent using the left
        click on a mouse. Right click will delete the previous point. Press
        Enter when you have completed the mask that you wish to use - this will
        save the points. Then exit the window to continue the program.'''
        #If a mask has not been set in the initialisation of the Extents class
        #then the mask will be manually defined using the assigned maxImg
        self._mask = readMask(self.getMaxImgData(), self._maxMaskPath)

        
    def maskImg(self, img):
        '''Mask images using the largest extent mask (boolean object). Unlike 
        the masking function in the TimeLapse class, the boolean mask is used
        to reassign overlapping image pixels to zero. The masking function in 
        the TimeLapse class uses a numpy masking object (numpy.ma).'''
        #Get mask from TimeLapse class
        if self._mask is None:
            self.setMaxMask()
            
        #Mask the glacier
        booleanMask = np.array(self._mask, dtype=bool)
        booleanMask = np.invert(booleanMask)
        
        #Copy properties of img
#        img2 = np.copy(img)
        
        #Mask extent image with boolean array
        np.where(booleanMask, 0, img) #fit arrays to each other
        img[booleanMask] = 0 #mask image with boolean mask object

        return img
        
#---------------------   IMAGE MODIFICATION FUNCTIONS   -----------------------
#The following functions are used in the automated extent detection to enhance 
#an image in order to better distinguish the target extent. This enhancement 
#largely involves changing the brightness and contrast of an image using phi 
#and theta variables. These can either be set directly using the setEnhance
#function, or can be previewed using the seeEnhance function.
       
    def enhanceImg (self, img):
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
        #If enhance parameters are undefined, apply a given enhancement setting
        if self._enhance is None:
            self.setEnhance('light', 50, 20)                    
        
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
            print 'Invalid diff variable' 
            sys.exit(1)
            
        return img1


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
        #Get maximum extent image and plot
        img=self.getMaxImgData()
        fig,ax = mp.subplots()
        mp.subplots_adjust(left=0.25, bottom=0.25)
        ax = mp.subplot(111)
        ax.imshow(img)
        
        diff = 'light'  
                
        #Inititalise sliders for phi and theta
        axcolor = 'lightgoldenrodyellow'
        axphi  = mp.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        axtheta = mp.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
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
        resetax = mp.axes([0.8, 0.025, 0.1, 0.04])
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
                
        mp.show()
        mp.close()
        
        print sphi.val
        print stheta.val
        self.setEnhance('light', sphi.val, stheta.val)
        
        
#----------------------    MANUAL TERMINUS DEFINITION   -----------------------
#Functions for manual and automated detection of lines in the image plane. The
#majority of the heavy-lifting is done by either the calcLines or 
#manualLines functions, which define lines through an image sequence.
#These two functions also directly feed into the Length class, defining the
#pxpts and pxline objects in the class which can be subsequently used in
#the transformation functions, plotting functions and exporting functions.
#The other functions are associated with the line detection method 
#(setMethod, getMethod), threshold for the number of lines detected 
#(setThreshold, getThreshold), and the pixel colourrange that is used to define
#target lines (setColourrange, getColourrange, defineColourrange and getRBG).       

    def manualTermini(self):
        '''Get manual pixel line from a series of images. Return the 
        line pixel coordinates and pixel length.'''                 
        #Set up output dataset
        pts=[]        
        lines=[]
        count=1
        
        #Define line in all images        
        for i in range(self._imageN):
            img = self.getImageNo(i)
            pt,length = self.manualTerminus(img)
            print 'Img%i line length: %d px' % (count, length.Length())
            print 'Line contains %i points\n' % (length.GetPointCount())
            pts.append(pt)
            lines.append(length)
            count=count+1
    
        #Return all line coordinates and length
        self._pxpts = pts
        self._pxline = lines        
        return self._pxpts, self._pxline
        

    def manualTerminus(self, img):
        '''Manually define a line by clicking in the target image.'''
         #Manual interaction to define line
        fig=mp.gcf()
        fig.canvas.set_window_title('Define line. Press enter to record points.')
        mp.imshow(img, origin='upper',cmap='gray')        
        pts = mp.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
        print 'you clicked:', pts
        mp.show()
        mp.close()
        
        #Create OGR line object
        line = self.ogrLine(pts)
        
        #Re-format point coordinates
        pts = np.squeeze(pts)

        return pts, line


    def ogrLine(self, pts):
        '''Create OGR line from a set of pts.'''              
        line = ogr.Geometry(ogr.wkbLineString)   
        for p in pts:
            line.AddPoint(p[0],p[1])
        
        return line



    def seeCanny(self):
        '''Enhance image using an interactive plot. WARNING: this function will
        significantly slow down your computer. Only use if your computer can
        handle it.'''
        # this function is needed for the createTrackbar step downstream
        def nothing(x):
            pass
        
        # read the experimental image
        img = cv2.imread('C:/Users/s0824923/Local Documents/python_workspace/pytrx/messi.jpg', 0)
        
        # create trackbar for canny edge detection threshold changes
        cv2.namedWindow('canny')
        
        # add ON/OFF switch to "canny"
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'canny', 0, 1, nothing)
        
        # add lower and upper threshold slidebars to "canny"
        cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
        cv2.createTrackbar('upper', 'canny', 0, 255, nothing)
        
        # Infinite loop until we hit the escape key on keyboard
        while(1):
        
            # get current positions of four trackbars
            lower = cv2.getTrackbarPos('lower', 'canny')
            upper = cv2.getTrackbarPos('upper', 'canny')
            s = cv2.getTrackbarPos(switch, 'canny')
        
            if s == 0:
                edges = img
            else:
                edges = cv2.Canny(img, lower, upper)
        
            # display images
            cv2.imshow('original', img)
            cv2.imshow('canny', edges)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:   # hit escape to quit
                break
        
        cv2.destroyAllWindows()


#----------------------   AUTOMATED TERMINUS DEFINITION   ---------------------
        
    def autoTermini(self):
        img = self.getImageNo(0)
#        img=source.copy()
#        img=self.maskImg(img)

#        img = cv2.equalizeHist(img)
#        img = cv2.bilateralFilter(img,16,75,75)
#        img = cv2.Canny(img, 100, 200)
        
#        f, ((ax1, ax2),(ax3, ax4))=mp.subplots(2,2,sharex='col',sharey='row')
#        ax1.imshow(img,cmap='gray')     
#        ax1.set_title('Original')      
#        
#        ax2.imshow(img1,cmap='gray')     
#        ax2.set_title('Equalize')       
#
#        ax3.imshow(img2,cmap='gray')
#        ax3.set_title('Bilateral filter')
#        
#        ax4.imshow(img3,cmap='gray')
#        ax4.set_title('Canny detection')
#        mp.show()

#        img = cv2.bilateralFilter(img,25,75,75)
#        
#        self.setEnhance('light',3,10)
#        img = self.enhanceImg(img)
#
#        mp.imshow(img)
#        mp.show()
        
        thresh=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.filterSpeckles(thresh, 1, 30, 2)
        mp.imshow(thresh, cmap='gray')
        mp.show()
        
        conimg, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=len)
        maxc = contours[-3:] 

        px_im = Img.new('L', (5184,3456), 'black')
        px_im = np.array(px_im)        
        cv2.drawContours(px_im, contours, -1, (255,255,255), 4)
        mp.imshow(px_im, cmap='gray')
        mp.show()        

     
    def autoTerm(self):
        i0 = self.getImageNo(0)
        i1 = self.getImageNo(12)
        
        p0,line = self.manualTerminus(i0)
        p0 = np.float32(p0)
        
        lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))                

        p1,st,err =cv2.calcOpticalFlowPyrLK(i0,i1,p0,None,**lk_params)
        p0r,st,err=cv2.calcOpticalFlowPyrLK(i1,i0,p1,None,**lk_params)
        
        # Find the distance and discard point greater than a pixel away         
        dist = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = dist < 1
        
        # Find the SNR and append this, signal and noise to error list
        new_pts0 = []
        new_pts1 = []
        new_pts0r = []
        error = []
        
        for pts0, pts1, pts0r, d, val in itertools.izip(p0, p1, p0r, dist, good):
            if val:
                p0 = [pts0[0], pts0[1]]
                p1 = [pts1[0], pts1[1]]
                p0r = [pts0r[0], pts0r[1]]
                new_pts0.append(p0)
                new_pts1.append(p1)
                new_pts0r.append(p0r)
                
                xd = p1[0] - p0[0]
                yd = p1[1] - p0[1]
                length = math.sqrt((xd*xd)+(yd*yd))
                if d == 0:
                    d = 0.0000000001
                snr = length/d
                ls = [length, d, snr]
                error.append(ls)

        x1=[]
        y1=[]        
        for pt in new_pts1:
            x1.append(pt[0])
            y1.append(pt[1])
        
        x2=[]
        y2=[]
        for pt in new_pts0r:
            x2.append(pt[0])
            y2.append(pt[1])
        
        x3=[]
        y3=[]
        for pt in new_pts0:
            x3.append(pt[0])
            y3.append(pt[1])
            
        mp.imshow(i1)
        mp.plot(x1,y1,'r')
        mp.plot(x2,y2,'g')
        mp.plot(x3,y3,'y')
        mp.show()     


#---------------------    AUTOMATIC TERMINUS TRACKING   -----------------------
    def trackTerminus(self, plot=False):
        # Feature tracking setup
        lk_params = dict( winSize  = (25,25),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        #Get first image in sequence
        self.setImage0(0)
        im0 = self.getImage0Data()
              
        #Manually define terminus in first image
        pts=[]
        p0 = []
        errors = []
        
        p0, l0 = self.manualTerminus(im0)              
        p0 = np.float32(p0).reshape((p0.shape[0],1,2))              
        pts.append(p0)
        errors.append(0)
        
        for i in range(self._imageN):
            #Get image pair
            im0 = self.getImageNo(i)
            imN = self.getImageNo(i+1)
            
            #Track points
            p1, status1, error1  = cv2.calcOpticalFlowPyrLK(im0, imN, p0, None, **lk_params)

            #Plot points
            if plot is True:
                mp.imshow(im0,cmap='gray')             
                for i,j in zip(p0,p1):
                    mp.scatter([pt[0] for pt in i], [pt[1] for pt in i], s=40, color='y', marker='+')
                    mp.scatter([pt[0] for pt in j], [pt[1] for pt in j], s=40, color='g', marker='+')
                mp.show()
                
            #Append points
            pts.append(p1)
            errors.append(error1)
            p0=p1
            
        return pts, errors
            
#-------------------------   PROJECTION FUNCTIONS   ---------------------------
#Functions for transforming pixel line information into real world lines and
#coordinates. The function calcLengths will calculate real world lines and 
#coordinates for all pixel lines detected from an image sequence. Use the 
#calcLengths function to also set up the real world line coordinates object
#(self._realpts) and real world line length object (self._realline). 
#The inversion is performed by the invproject function in the CamEnv, which is 
#called in the calcXYZ function in this section (see CamEnv.py for more 
#information on the georectification method).
   
    def calcLengths(self):
        '''Get real world lines from an image set. Calculates the line 
        coordinates and length of each given set of pixel points.'''
        #Get pixel points if not already defined
        if self._pxpts is None:
            self.manualTermini()
        
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


    def calcLength(self, px):
        '''Get real world line from px points defined in one image.'''                    
        rpts = []                     #Create outputs
        xyz = self.calcXYZ(px)
        rpts.append(xyz)              #Project image coordinates
        rpts = np.squeeze(rpts)        
        rline = self.ogrLine(rpts)    #Create polygons
           
        return rpts, rline  

        
    def calcXYZ(self, px):
        '''Transform points on image plane to real world coordinates using 
        info from the CamEnv object.'''                         
        xyz = self._CamEnv.invproject(px)
        return xyz
    
 
#--------------------------   PLOTTING FUNCTIONS   ----------------------------
#Plotting functions for visualising the pixel lines on an image (plotPX) and 
#the real world lines on a DEM (plotXYZ). Use the number variable to define 
#which image in the sequence you want to plot along with its corresponding
#lines (defined as a integer e.g. number=1). The plot will be saved to a 
#specific filepath if defined in the dest variable.
#e.g. C:/python_workspace/pytrx/results/plot_output.jpg
#A simple for loop can be used to plot pixel/real areas in all images.
        
    def plotPX(self, number, dest=None):
        '''Return image overlayed with pixel lines for a given image 
        number.'''
        #Get requested image in grayscale 
        img = self.getImageNo(number)
                      
        #Create image plotting window
        fig=mp.gcf()
        fig.canvas.set_window_title('Image extent output '+str(number))
        imgplot = mp.imshow(img)        
        imgplot.set_cmap('gray')
        mp.axis('off')
    
        #Get xy pixel coordinates and plot on figure        
        line = self._pxpts[number]        
        x=[]
        y=[]
        for xy in line:
            x.append(xy[0])
            y.append(xy[1])            
        mp.plot(x,y,'w-')                
        
        #Save figure if destination is defined
        if dest != None:
            mp.savefig(dest + 'extentimg_' + str(number) + '.jpg') 
            
        mp.show()  
        mp.close()
        
        
    def plotXYZ(self, number, dest=None):
        '''Plot xyz points of real lines for a given image number'''                       
        #Get xyz points for lines in a given image
        line = self._realpts[number]
        
        #Get xy data from line pts
        xl=[]
        yl=[]        
        for pt in line:
            xl.append(pt[0])
            yl.append(pt[1])
                
        #Prepare DEM
        demobj=self._CamEnv.getDEM()
        demextent=demobj.getExtent()
        dem=demobj.getZ()
       
        #Get camera position (getCameraPose is from the CamEnv class)
        post, pose = self._CamEnv.getCameraPose()             
        
        #Plot DEM 
        fig=mp.gcf()
        fig.canvas.set_window_title('Line output '+str(number))
        mp.xlim(demextent[0],demextent[1])
        mp.ylim(demextent[2],demextent[3])
#        mp.xlim(449000, 451000)
#        mp.ylim(8757500, 8758500)
        mp.locator_params(axis = 'x', nbins=8)
        mp.tick_params(axis='both', which='major', labelsize=10)
        mp.imshow(dem, origin='lower', extent=demextent, cmap='gray')
        
        #Plot line points and camera position on to DEM image        
        mp.plot(xl, yl, 'y-')
        mp.scatter(post[0], post[1], c='g')
        mp.suptitle('Projected extents', fontsize=14)
        
        #Save figure if destination is defined
        if dest != None:
            mp.savefig(dest + 'line_' + str(number) + '.jpg')        
        
        mp.show()
        mp.close()


#---------------------------   IMPORT FUNCTIONS   -----------------------------
#Functions for importing pixel length data and real world data from text file.
#The functions importData, importXYZ and importPX read data directly into the 
#Area class from a specified fileDirectory (i.e. folder, NOT file). Coordinate 
#and area text files must be named correctly - 'line_pxcoords.txt', 
#'line_pxlength.txt', 'line_realcoords.txt', and 'line_reallength.txt'.
    
    def importData(self, fileDirectory):
        '''Get xyz and px data from text files and import into Length class.
        Inputs:
        -fileDirectory: Path to the folder where the four text files are. The
         text file containing the pixel coordinates must be named 
         'line_pxcoords.txt' and 'line_pxlength.txt'. The text file containing 
         real world polygon areas must be named 'line_realcoords.txt' and 
         'line_reallength.txt'. Files will not be recognised if they are not 
         named correctly.'''
        rline, rlength = self.importXYZ(fileDirectory)
        pxline, pxlength = self.importPX(fileDirectory)
        return rline, rlength, pxline, pxlength
         
         
    def importXYZ(self, fileDirectory):
        '''Get xyz line and length data from text files and import into Length 
        class.
        Inputs:
        -fileDirectory: Path to the folder where the two text files are. The
         text file containing the line coordinates must be named
         'line_realcoords.txt' and the text file containing the line lengths
         must be names 'line_reallength.txt'. Files will not be recognised if 
         they are not named correctly.'''
        #Import polygon coordinates from text file
        target1 = fileDirectory + 'line_realcoords.txt'
        xyz = self.coordFromTXT(target1, xyz=True)
        
        #Create OGR line object
        ogrline=[]
        for line in xyz: 
            print line
            length = self.ogrLine(line)
            ogrline.append(length)
        
        #Import data into Area class
        self._realpts = xyz
        self._realline = ogrline
        
        return self._realpts, self._realline
   
    
    def importPX(self, fileDirectory):
        '''Get px line and length data from multiple text files and import
        into Length class.
        Inputs:
        -fileDirectory: Path to the folder where two text files are containing
         the pixel coordinates and the pixel extents.
         The text files must be in the same folder and named 'line_pxcoords' 
         and 'line_pxlength'. Files will not be recognised if they are not 
         named correctly.'''
        #Import px polygon coordinates from text file
        target1 = fileDirectory + 'line_pxcoords.txt'
        xy = self.coordFromTXT(target1, xyz=False)
        
        #Create OGR line object
        ogrline = []
        for line in xy:        
            length = self.ogrLine(line)
            ogrline.append(length)
        
        #Import data into Area class
        self._pxpts = xy
        self._pxline = ogrline
        
        return self._pxpts, self._pxline

        
    def coordFromTXT(self, filename, xyz=True):
        '''Import XYZ pts data from text file. Return list of arrays with 
        xyz pts.'''
        #Read file and detect number of images based on number of lines
        f=file(filename,'r')      
        alllines=[]
        for line in f.readlines():
            if len(line) >= 6:
                alllines.append(line)  #Read lines in file             
        print 'Detected coordinates from ' + str(len(alllines)) + ' images'
        f.close()
        
        allcoords=[]  
        
        #Extract strings from lines         
        for line in alllines:
            vals = line.split('\t')
            
            coords=[]
            raw=[]
                          
            #Extract coordinate values from strings
            for v in vals:
                try:
                    a=float(v)
                    raw.append(a)
                except ValueError:
                    pass
                
            #Restructure coordinates based on whether 2 or 3 dimensions
            if xyz is True:
                dim = 3
                struc = len(raw)/dim
            elif xyz is False:
                dim = 2
                struc = len(raw)/dim
                
            coords = np.array(raw).reshape(struc, dim)
            allcoords.append(coords)
        
        #Return xyz as list of arrays        
        return allcoords        



#----------------------------------  END   ------------------------------------
if __name__ == "__main__":

    #Main file directory
    fileDirectory = 'C:/Users/s0824923/Local Documents/python_workspace/pytrx/'
    
    #Directory to related files
    cam2data = fileDirectory + 'Data/GCPdata/CameraEnvironmentData_cam2_2014.txt'
    cam2mask = fileDirectory + 'Data/GCPdata/masks/c2_2014_amask.JPG'
    cam2imgs = fileDirectory + 'Data/Images/Area/cam2_2014_plume/demo/*.JPG'
    
    #Define data output directory
    destination = fileDirectory + 'Results/cam2/'
    
    #Define camera environment object
    cam2 = CamEnv(cam2data)
    
    #Define area object where our data can be processed
    plume = Area(cam2imgs, cam2, cam2mask)
    

#------------------------   Manually define areas   ---------------------------
    
    #OPTION 1. ONLY USE OPTION 1, 2 OR 3.
    #Calculate real areas by manually defining them on the image (point and click)
    rpoly, rarea = plume.calcAreas('manual')
    
    
#--------------------   Automatically calculate areas   -----------------------
    
    ##OPTION 2. ONLY USE OPTION 1, 2 OR 3.
    ##Automatically calculate areas by first enhancing the images to excentuate 
    ##target areas and then defining the area by a colour range
    #
    ##Set image that contains the biggest target area (i.e. the biggest plume extent)
    #plume.setMaxImg(33)
    #
    ##Set image enhancement parameters
    #plume.setEnhance('light', 50, 20)
    #
    ##Show example of image enhancement
    #im=plume.getMaxImgData()
    #im=plume.maskImg(im)  
    #im=plume.enhanceImg(im)
    #plt.imshow(im)
    #plt.show()
    #
    ##Set colour range that areas will be detected in
    #plume.setColourrange(10, 1)
    #
    ##Set number of areas that will be extracted from detection
    #plume.setThreshold(1)
    #
    ##Calculate real areas
    #rpoly, rarea = plume.calcAreas(method='auto')
    
    
#------------------------   Import existing data   ----------------------------
    
    ##OPTION 3. ONLY USE OPTION 1, 2 OR 3.
    ##Import data. If you already have pixel polygon shapes, then use this to
    ##directly import shapes for further processing (e.g. georectification, plotting)
    
    #print 'Importing data...'
    #target = fileDirectory + 'Results/cam2/run01/'
    #pxpolys, pxareas = plume.importPX(target)
    
    
    #----------------------------   Export data   ---------------------------------
    
    #Write data to text file (this can be imported into other software such as Excel)
    plume.writeData(destination)
    
    #Export areal polygons as shape files
    geodata = destination + 'shp_proj/'
    proj = 32633
#    plume.exportSHP(geodata, proj)
    
    print plume.getFileList()
    #----------------------------   Show results   --------------------------------
    
    #Plot and save all extent and area images
    length=len(rpoly)
    for i in range(length):         #This loops through all the areas measured
        plume.plotPX(i)             #Plot pixel area onto image
        plume.plotXYZ(i)            #Plot real area onto DEM
    
    
    print 'Finished'
    
    
    #--------------------------------   END   -------------------------------------