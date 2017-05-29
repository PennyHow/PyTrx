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
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import cv2
import sys
from PIL import Image as Img
from pylab import array, uint8
from osgeo import ogr, osr
import os
import itertools
import math

from FileHandler import readMask
from Images import TimeLapse
from CamEnv import CamEnv


#------------------------------------------------------------------------------

class Area(TimeLapse):
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
    loadall:
    timingMethod:
    quiet:'''
    
    #Initialisation of Area class object          
    def __init__(self, imageList, cameraenv, method='auto', calibFlag=True, 
                 maxMaskPath=None, maxim=0, band='L', quiet=2, loadall=False, 
                 timingMethod='EXIF'):
                     
        #Initialise and inherit from the TimeLapse class object
        TimeLapse.__init__(self, imageList, cameraenv, None, None, 0, band,
                           quiet, loadall, timingMethod)

        #Optional commentary        
        if self._quiet>0:
            print '\nAREA DETECTION COMMENCING'
            
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
        #Create outputs
        xyz = []   
        area = []                               

        #Project image coordinates
        for p in pxpoly:                        
            allxyz = self._camEnv.invproject(p)
            xyz.append(allxyz)                  
        
        #Create polygons
        rpoly = self._ogrPolyN(xyz)              
        
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
            maximn=self._imageSet[self._maximg].getImagePath()
              
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
            imn=self._imageSet[i].getImagePath()
        
            img2 = np.copy(img1)
            if self._mask is not None:
                img2 = self._maskImg(img2)
            if self._enhance is not None:
                img2 = self.enhanceImg(img2)
            if color is True:
                self.defineColourrange(img2, imn)
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
            imn = self._imageSet[i].getImagePath()
            
            #Verify polygons
            img2 = np.copy(img1)
            
            if 1:             
                if self._quiet>0:                
                    print 'VERIFYING DETECTED AREAS'
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
            imn = self._imageSet[i].getImagePath()
            
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
            print 'Extent: ', pxextent, 'px (out of ', pxcount, 'px) \n'
        
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
   
    
    def _ogrPolyN(self, xyz):
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
            imn=self._imageSet[i].getImagePath()
            
            #Define line
            pt,length = self.manualLinePX(img, imn)
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
        fig=plt.gcf()
        fig.canvas.set_window_title('Image extent output '+str(number))
        imgplot = plt.imshow(img)        
        imgplot.set_cmap('gray')
        plt.axis('off')
    
        #Get xy pixel coordinates and plot on figure        
        line = self._pxpts[number]        
        x=[]
        y=[]
        for xy in line:
            x.append(xy[0])
            y.append(xy[1])            
        plt.plot(x,y,'w-')                
        
        #Save figure if destination is defined
        if dest != None:
            plt.savefig(dest + 'extentimg_' + str(number) + '.jpg') 
            
        plt.show()  
        plt.close()
        
        
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
        fig=plt.gcf()
        fig.canvas.set_window_title('Line output '+str(number))
        plt.xlim(demextent[0],demextent[1])
        plt.ylim(demextent[2],demextent[3])
#        plt.xlim(449000, 451000)
#        plt.ylim(8757500, 8758500)
        plt.locator_params(axis = 'x', nbins=8)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.imshow(dem, origin='lower', extent=demextent, cmap='gray')
        
        #Plot line points and camera position on to DEM image        
        plt.plot(xl, yl, 'y-')
        plt.scatter(post[0], post[1], c='g')
        plt.suptitle('Projected extents', fontsize=14)
        
        #Save figure if destination is defined
        if dest != None:
            plt.savefig(dest + 'line_' + str(number) + '.jpg')        
        
        plt.show()
        plt.close()


##----------------------------------  END   ------------------------------------
#if __name__ == "__main__":
#
#    #Main file directory
#    fileDirectory = 'C:/Users/s0824923/Local Documents/python_workspace/pytrx/'
#    
#    #Directory to related files
#    cam2data = fileDirectory + 'Data/GCPdata/CameraEnvironmentData_cam2_2014.txt'
#    cam2mask = fileDirectory + 'Data/GCPdata/masks/c2_2014_amask.JPG'
#    cam2imgs = fileDirectory + 'Data/Images/Area/cam2_2014_plume/demo/*.JPG'
#    
#    #Define data output directory
#    destination = fileDirectory + 'Results/cam2/'
#    
#    #Define camera environment object
#    cam2 = CamEnv(cam2data)
#    
#    #Define area object where our data can be processed
#    plume = Area(cam2imgs, cam2, cam2mask)
#    
#
##------------------------   Manually define areas   ---------------------------
#    
#    #OPTION 1. ONLY USE OPTION 1, 2 OR 3.
#    #Calculate real areas by manually defining them on the image (point and click)
#    rpoly, rarea = plume.calcAreas('manual')
#    
#    
##--------------------   Automatically calculate areas   -----------------------
#    
#    ##OPTION 2. ONLY USE OPTION 1, 2 OR 3.
#    ##Automatically calculate areas by first enhancing the images to excentuate 
#    ##target areas and then defining the area by a colour range
#    #
#    ##Set image that contains the biggest target area (i.e. the biggest plume extent)
#    #plume.setMaxImg(33)
#    #
#    ##Set image enhancement parameters
#    #plume.setEnhance('light', 50, 20)
#    #
#    ##Show example of image enhancement
#    #im=plume.getMaxImgData()
#    #im=plume.maskImg(im)  
#    #im=plume.enhanceImg(im)
#    #plt.imshow(im)
#    #plt.show()
#    #
#    ##Set colour range that areas will be detected in
#    #plume.setColourrange(10, 1)
#    #
#    ##Set number of areas that will be extracted from detection
#    #plume.setThreshold(1)
#    #
#    ##Calculate real areas
#    #rpoly, rarea = plume.calcAreas(method='auto')
#    
#    
##------------------------   Import existing data   ----------------------------
#    
#    ##OPTION 3. ONLY USE OPTION 1, 2 OR 3.
#    ##Import data. If you already have pixel polygon shapes, then use this to
#    ##directly import shapes for further processing (e.g. georectification, plotting)
#    
#    #print 'Importing data...'
#    #target = fileDirectory + 'Results/cam2/run01/'
#    #pxpolys, pxareas = plume.importPX(target)
#    
#    
#    #----------------------------   Export data   ---------------------------------
#    
#    #Write data to text file (this can be imported into other software such as Excel)
#    plume.writeData(destination)
#    
#    #Export areal polygons as shape files
#    geodata = destination + 'shp_proj/'
#    proj = 32633
##    plume.exportSHP(geodata, proj)
#    
#    print plume.getFileList()
#    #----------------------------   Show results   --------------------------------
#    
#    #Plot and save all extent and area images
#    length=len(rpoly)
#    for i in range(length):         #This loops through all the areas measured
#        plume.plotPX(i)             #Plot pixel area onto image
#        plume.plotXYZ(i)            #Plot real area onto DEM
#    
#    
#    print 'Finished'
#    
#    
#    #--------------------------------   END   -------------------------------------