#PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie
#
#PyTrx is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

"""
The Area module handles the functionality for obtaining areal measurements from 
oblique time-lapse imagery. Specifically, this module contains functions for:
(1) Performing automated and manual detection of areal extents in oblique 
imagery; and (2) Determining real-world surface areas from oblique imagery.                                                                                                                                
"""

#Import packages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2
from PIL import Image
import ogr

#Import PyTrx functions and classes
from PyTrx.FileHandler import readMask
from PyTrx.Images import ImageSequence, enhanceImage
from PyTrx import Velocity
from PyTrx.CamEnv import projectUV, setProjection

#------------------------------------------------------------------------------
class Area(ImageSequence):
    """A class for processing change in area (i.e. a lake or plume) through an 
    image sequence, with methods to calculate extent change in an image plane 
    (px) and real areal change via georectification.
       
    :param imageList: List of images, for the :class:`PyTrx.Images.ImageSequence` object
    :type imageList: str/list            
    :param cameraenv: Camera environment parameters which can be read into the :class:`PyTrx.CamEnv.CamEnv` object as a text file
    :type cameraenv: str           
    :param hmatrix: Homography matrix 
    :type hmatrix: arr
    :param calibFlag: An indicator of whether images are calibrated, for the :class:`PyTrx.Images.ImageSequence` object
    :type calibFlag: bool          
    :param band: String denoting the desired image band, default to 'L' (grayscale)
    :type band: str, optional
    :param equal: Flag denoting whether histogram equalisation is applied to images (histogram equalisation is applied if True). Default to True. 
    :type equal: bool, optional                          
    """
    #Initialisation of Area class object          
    def __init__(self, imageList, cameraenv, hmatrix, calibFlag=True, 
                 band='L', equal=True):
        
        #Initialise and inherit from the ImageSequence object
        ImageSequence.__init__(self, imageList, band, equal) 
        
        #Set up class properties
        self._camEnv = cameraenv
        self._calibFlag = calibFlag
        self._pxplot = None
        self._maximg = 0
        self._mask = None
        self._enhance = None
        
        if hmatrix is not None:
            self._hmatrix=hmatrix
            hmat0=None
            self._hmatrix.insert(0, hmat0)
        else:
            self._hmatrix=None
            
            
    def calcAutoAreas(self, colour=False, verify=False):
        """Detects areas of interest from a sequence of images, and returns 
        pixel and xyz areas. 
        
        :param colour: Flag to denote whether colour range for detection should be defined for each image or only once, default to False
        :type colour: bool, optional 
        :param verify: Flag to denote whether detected polygons should be manually verified by user, default to False
        :type verify: bool, optional           
        :returns: XYZ and UV area information
        :rtype: list
        """               
        print('\n\nCOMMENCING AUTOMATED AREA DETECTION')

        #Get DEM from camera environment
        dem = self._camEnv.getDEM() 

        #Get inverse projection variables through camera info               
        invprojvars = setProjection(dem, self._camEnv._camloc, 
                                    self._camEnv._camDirection, 
                                    self._camEnv._radCorr, 
                                    self._camEnv._tanCorr, 
                                    self._camEnv._focLen, 
                                    self._camEnv._camCen, 
                                    self._camEnv._refImage)
            
        #If user is only defining the color range once
        if colour is False: 
            
            #Define colour range if none is given
            if self._colourrange is None:

                #Get image (either corrected or distorted)
                if self._calibFlag is True:
                    cameraMatrix=self._camEnv.getCamMatrixCV2()
                    distortP=self._camEnv.getDistortCoeffsCV2()
                    setting=self._imageSet[self._maximg].getImageCorr(cameraMatrix, 
                                                               distortP)
                else:
                    setting = self._imageSet[self._maximg].getImageArray() 
    
                #Get image name
                setimn=self._imageSet[self._maximg].getImageName()
                  
                #Get mask and mask image if present
                if self._mask is not None:                       
                    booleanMask = np.array(self._mask, dtype=bool)
                    booleanMask = np.invert(booleanMask)
                    
                    #Mask extent image with boolean array
                    np.where(booleanMask, 0, setting) #Fit arrays to each other
                    setting[booleanMask] = 0 #Mask image with boolean mask object       
                       
                #Enhance image if enhancement parameters given
                if self._enhance is not None:
                    setting = enhanceImage(setting, self._enhance[0], 
                                           self._enhance[1], self._enhance[2])
                
                #Define colour range
                defineColourrange(setting, setimn, pxplot=self._pxplot)    
            
        #Set up output datasets
        area=[]
                       
        #Cycle through image sequence (numbered from 0)
        for i in range(self.getLength()):
            
            #Get corrected/distorted image
            if self._calibFlag is True:
                cameraMatrix=self._camEnv.getCamMatrixCV2()
                distortP=self._camEnv.getDistortCoeffsCV2()
                img1 = self._imageSet[i].getImageCorr(cameraMatrix, 
                                                      distortP)
            else:
                img1=self._imageSet[i].getImageArray()

            #Get image name
            imn=self._imageSet[i].getImageName()
               
            #Make a copy of the image array
            img2 = np.copy(img1)
            
            #Mask image if mask is present
            if self._mask is not None:
                booleanMask = np.array(self._mask, dtype=bool)
                booleanMask = np.invert(booleanMask)
                
                #Mask extent image with boolean array
                np.where(booleanMask, 0, img2) #Fit arrays to each other
                img2[booleanMask] = 0 #Mask image with boolean mask object
            
            #Enhance image if enhancement parameters are present
            if self._enhance is not None:
                img2 = enhanceImage(img2, self._enhance[0], self._enhance[1],
                                    self._enhance[2])
            
            #Define colour range if required
            if colour is True:
                defineColourrange(img2, imn, pxplot=self._pxplot)
            
            #Calculate extent
            if self._hmatrix is not None:
                out = calcAutoArea(img2, imn, self._colourrange, 
                                   self._hmatrix[i], self._threshold, 
                                   invprojvars)  
            else:
                out = calcAutoArea(img2, imn, self._colourrange, None, 
                                   self._threshold, invprojvars)  
            
            area.append(out)

            #Clear memory
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
        
        #Verify areas if flag is true
        if verify is True:
            area = self.verifyAreas(area, invprojvars)

        #Return all xy coordinates and pixel extents                 
        return area


    def calcManualAreas(self):
        """Manually define areas of interest in a sequence of images. User 
        input is facilitated through an interactive plot to click around the 
        area of interest
        
        :returns: XYZ and UV area information
        :rtype: list
        """               
        '\n\nCOMMENCING MANUAL AREA DETECTION'            
        #Set up output dataset
        area=[]

        #Get DEM from camera environment
        dem = self._camEnv.getDEM() 

        #Get inverse projection variables through camera info               
        invprojvars = setProjection(dem, self._camEnv._camloc, 
                                    self._camEnv._camDirection, 
                                    self._camEnv._radCorr, 
                                    self._camEnv._tanCorr, 
                                    self._camEnv._focLen, 
                                    self._camEnv._camCen, 
                                    self._camEnv._refImage)
                
        #Cycle through images        
        for i in (range(self.getLength())):
            
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                   self._camEnv.getDistortCoeffsCV2())      
            else:
                img=self._imageSet[i].getImageArray()          

            #Get image name
            imn=self._imageSet[i].getImageName()
            
            #Manually define extent and append
            if self._hmatrix is not None:
                polys = calcManualArea(img, imn, self._hmatrix[i], 
                                       self._pxplot, invprojvars) 
            else:
                polys = calcManualArea(img, imn, None, self._pxplot, 
                                       invprojvars)                 
            area.append(polys)
            
            #Clear memory
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
    
        #Return all extents, all cropped images and corresponding image names       
        return area
    
        
    def verifyAreas(self, areas, invprojvars):
        """Method to manually verify all polygons in images. Plots sequential
        images with detected polygons and the user manually verifies them by 
        clicking them.
        
        :param area: XYZ and UV area information
        :type area: list 
        :param invprojvars: Inverse projection variables [X,Y,Z,uv0]
        :type invprojvars: list
        :param verified: Verified XYZ and UV area information
        :type verified: list 
        """
        #Create output
        verified = []
        
        #Get UV point coordinates
        uvpts=[item[1][1] for item in areas]
                
        #Verify pixel polygons in each image        
        for i in range(len(uvpts)):
            
            #Call corrected/uncorrected image
            if self._calibFlag is True:
                img1=self._imageSet[i].getImageCorr(self._camEnv.getCamMatrixCV2(), 
                                                    self._camEnv.getDistortCoeffsCV2())      
            else:
                img1=self._imageSet[i].getImageArray()            
            
            #Get image name
            imn=self._imageSet[i].getImageName()
            
            #Verify polygons
            img2 = np.copy(img1)
            
            if 1:                            
                print('\nVerifying detected areas from ' + str(imn))
                
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
                for a in uvpts[i]:
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
            plt.close()
            
            #Append all verified extents
            vpx=[]
            vpx=verf               
            
            #Get areas of verified extents
            h = img2.shape[0]
            w = img2.shape[1]
            px_im = Image.new('L', (w,h), 'black')
            px_im = np.array(px_im) 
            cv2.drawContours(px_im, vpx, -1, (255,255,255), 4)
            for p in vpx:
                cv2.fillConvexPoly(px_im, p, color=(255,255,255))           
            output = Image.fromarray(px_im)

            pixels = output.getdata()
            values = []    
            for px in pixels:
                if px > 0:
                    values.append(px)
            pxext = len(values)        
            print('Total verified extent: ' + str(pxext))  

            #Get xyz coordinates with inverse projection
            if invprojvars is not None:
                vxyzpts=[]
                vxyzarea=[]
                for i in vpx:
                    #Inverse project points 
                    proj=projectUV(i, invprojvars)           
                    vxyzpts.append(proj)
                    ogrpol = getOGRArea(proj)                   
                    vxyzarea.append(ogrpol.GetArea())
                    
                print('Total verified area: ' + str(sum(vxyzarea)) + ' m')            

            verified.append([[pxext, vpx],[vxyzarea, vxyzpts]])                    
            
            #Clear memory            
            self._imageSet[i].clearImage()
            self._imageSet[i].clearImageArray()
        
        #Rewrite verified area data
        return verified
        

    def setMax(self, maxMaskPath, maxim):
        """Set image in sequence which pictures the maximum extent of the area
        of interest.
        
        :param maxMaskPath: File path to mask with maximum extent
        :type maxMaskPath: str
        :param maxim: Image with maximum extent
        :type maxim: arr
        """
        #Calibrate image if calibration flag is true
        if self._calibFlag is True:
            cameraMatrix=self._camEnv.getCamMatrixCV2()
            distortP=self._camEnv.getDistortCoeffsCV2()
            maxi = self._imageSet[maxim].getImageCorr(cameraMatrix, 
                                                               distortP)
        else:
            maxi = self._imageSet[maxim].getImageArray()
            
        #Define mask on image with maximum areal extent
        self._mask = readMask(maxi, maxMaskPath)
        
        #Retain image sequence number for image with maximum extent
        self._maximg = maxim


    def setPXExt(self,xmin,xmax,ymin,ymax):
        """Set plotting extent. Setting the plot extent will make it easier to 
        define colour ranges and verify areas.
        
        :param xmin: X-axis minimum value.
        :type xmin: int
        :param xmax: X-axis maximum value.
        :type xmax: int
        :param ymin: Y-axis minimum value.
        :type ymin: int
        :param ymax: Y-axis maximum value.
        :type ymax: int
        """
        self._pxplot = [xmin,xmax,ymin,ymax]


    def setThreshold(self, number):
        """Set threshold for number of polgons kept from an image.
        
        :param number: Number denoting the number of detected polygons that will be retained
        :type number: int 
        """
        self._threshold = number
                                

    def setColourrange(self, upper, lower):
        """Manually define the RBG colour range that will be used to filter
        the image/images.
        
        :param upper: Upper value of colour range
        :type upper: int
        :param lower: Lower value of colour range
        :type lower: int
        """    
        print('\nColour range defined from given values:')
        print('Upper RBG boundary: ', upper)
        print('Lower RBG boundary: ', lower)
        
        #Assign colour range
        self._colourrange = [upper, lower]
        

    def setEnhance(self, diff, phi, theta):
        """Set image enhancement parameters. Change brightness and contrast of 
        image using phi and theta variables. Change phi and theta values 
        accordingly. See enhanceImg function for detailed explanation of the 
        parameters.
        
        :param diff: Inputted as either 'light or 'dark', signifying the intensity of the image pixels. 'light' increases the intensity such that dark pixels become much brighter and bright pixels become slightly brighter. 'dark' decreases the intensity such that dark pixels become much darker and bright pixels become slightly darker.
        :type diff: str
        :param phi: Defines the intensity of all pixel values
        :type phi: int
        :param theta: Defines the number of "colours" in the image, e.g. 3 signifies that all the pixels will be grouped into one of three pixel values
        :type theta: int               .
        """
        self._enhance = diff, phi, theta
 

#------------------------------------------------------------------------------   

def calcAutoArea(img, imn, colourrange, hmatrix=None, threshold=None, 
                 invprojvars=None):
    """Detects areas of interest from a given image, and returns pixel and xyz 
    areas along with polygon coordinates. Detection is performed from the image 
    using a predefined RBG colour range. The colour range is then used to 
    extract pixels within that range using the OpenCV function inRange. If a 
    threshold has been set (using the setThreshold function) then only nth 
    polygons will be retained. XYZ areas and polygon coordinates are only 
    calculated when a set of inverse projection variables are provided.
    
    :param img: Image array
    :type img: arr
    :param imn: Image name
    :type imn: str
    :param colourrange: RBG colour range for areas to be detected from
    :type colourrange: list
    :param hmatrix: Homography matrix, default to None
    :type hmatrix: arr
    :param threshold: Threshold number of detected areas to retain, default to None
    :type threshold: int, optional
    :param invprojvars: Inverse projection variables [X,Y,Z,uv0], default to None
    :type invprojvars: list, optional
    :returns: Four list items containing 1) the sum of total detected areas (xyz), 2) XYZ coordinates of detected areas, 3) Sum of total detected areas (px), and 4) UV coordinates of detected areas
    :rtype: list
    """                       
    #Get upper and lower RBG boundaries from colour range
    upper_boundary = colourrange[0]
    lower_boundary = colourrange[1]

    #Transform RBG range to array    
    upper_boundary = np.array(upper_boundary, dtype='uint8')
    lower_boundary = np.array(lower_boundary, dtype='uint8')

    #Extract extent based on RBG range
    mask = cv2.inRange(img, lower_boundary, upper_boundary)

#    #Speckle filter to remove noise - needs fixing
#    mask = cv2.filterSpeckles(mask, 1, 30, 2)

    #Polygonize extents using OpenCV findContours function        
    polyimg, line, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)
    
    print('\nDetected ' + str(len(line)) + ' regions in ' + str(imn))
    
    #Append all polygons from the polys list that have more than 
    #a given number of points     
    rawpx = []
    for c in line:
        if len(c) >= 40:
            rawpx.append(c)
    
    #If threshold has been set, only keep the nth longest polygons
    if threshold is not None:
        if len(rawpx) >= threshold:
            rawpx.sort(key=len)
            rawpx = rawpx[-(threshold):]        

    print('Kept ' + str(len(rawpx)) + ' regions')
    
    #Calculate homography-corrected pts if desired
    if hmatrix is not None:
        print('Correcting for camera motion')
        pxpts=[]
        for i in rawpx:
            corr = Velocity.apply_persp_homographyPts(i, hmatrix, inverse=True)
            pxpts.append(corr)
    else:
        pxpts=rawpx
                
        
    #Calculate areas
    pxextent=[]
    for p in range(len(pxpts)): 

        try:        
            #Create geometry
            pxpoly=getOGRArea(pxpts[p].squeeze())
            
            #Calculate area of polygon area
            pxextent.append(pxpoly.Area())
        
        #Create zero object if no polygon has been recorded 
        except:
            pxextent = 0
        
    print ('Total extent: ' + str(sum(pxextent)) + ' px (out of ' 
            + str(img.shape[0]*img.shape[1]) + ' px)')  
    
    #Get xyz coordinates with inverse projection
    if invprojvars is not None:
        xyzpts=[]
        xyzarea=[]
        
        for i in pxpts:           
            #Inverse project points 
            proj=projectUV(i, invprojvars)           
            xyzpts.append(proj)
            
            #Get areas for xyz polygons
            ogrpol = getOGRArea(proj)                   
            xyzarea.append(ogrpol.GetArea())
            
        print('Total area: ' + str(sum(xyzarea)) + ' m')
                
        #Return XYZ and pixel areas
        return [[xyzarea, xyzpts], [pxextent, pxpts]]
    
    else:
        #Return pixel areas only
        return [[None, None], [pxextent, pxpts]]
        

def calcManualArea(img, imn, hmatrix=None, pxplot=None, invprojvars=None):
    """Manually define an area in a given image. User input is facilitated
    through an interactive plot to click around the area of interest. XYZ areas
    are calculated if a set of inverse projection variables are given.
    
    :param img: Image array
    :type img: arr
    :param imn: Image name
    :type imn: str
    :param hmatrix: Homography matrix, default to None
    :type hmatrix: arr    
    :param pxplot: Plotting extent for manual area definition, default to None
    :type pxplot: list, optional
    :param invprojvars: Inverse projection variables [X,Y,Z,uv0], default to None
    :type invprojvars: list, optional
    :returns: Four list items containing 1) the sum of total detected areas (xyz), 2) XYZ coordinates of detected areas, 3) Sum of total detected areas (px), and 4) UV coordinates of detected areas
    :rtype: list
    """   
    #Initialise figure window and plot image
    fig=plt.gcf()
    fig.canvas.set_window_title(imn + ': Click around region. Press enter '
                                'to record points.')
    plt.imshow(img, origin='upper', cmap='gray')
    
    #Set plotting extent if required
    if pxplot is not None:
        plt.axis([pxplot[0],pxplot[1],
                  pxplot[2],pxplot[3]]) 
    
    #Manual input of points from clicking on plot using pyplot.ginput
    rawpx = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, 
                       mouse_pop=3, mouse_stop=2)
    print('\n' + imn + ': you clicked ' + str(len(rawpx)) + ' points')
    
    #Show plot
    plt.show()
    plt.close()

    #Convert coordinates to array
    pxpts=[]
    for i in rawpx:
        pxpts.append([[i[0],i[1]]])
    pxpts.append([[rawpx[0][0],rawpx[0][1]]])
    pxpts=np.asarray(pxpts)
    
    #Calculate homography-corrected pts if desired
    if hmatrix is not None:
        print('Correcting for camera motion')
        pxpts = Velocity.apply_persp_homographyPts(pxpts, hmatrix, 
                                                   inverse=True)
        
    #Create polygon if area has been recorded   
    try:    
        #Create geometry
        pxpoly=getOGRArea(pxpts.squeeze())
        
        #Calculate area of polygon area
        pxextent = pxpoly.Area()
    
    #Create zero object if no polygon has been recorded 
    except:
        pxextent = 0
    
    print('Total extent: ' + str(pxextent) + ' px (out of ' + 
          str(img.shape[0]*img.shape[1]) + ' px)')    
    
    #Convert pts list to array
    pxpts = np.array(pxpts)           
    pxpts = np.squeeze(pxpts)
    

    if invprojvars is not None:
        #Get xyz coordinates with inverse projection
        xyzpts=projectUV(pxpts, invprojvars) 
        
        #Calculate area of xyz polygon
        xyzarea = getOGRArea(xyzpts)                   
        xyzarea=xyzarea.GetArea()
        
        #Return XYZ and pixel areas
        print('Total area: ' + str(xyzarea) + ' m')
        return [[[xyzarea], [xyzpts]], [[pxextent], [pxpts]]]

    #Return pixel areas only    
    else:
        return [[None, None], [pxextent, pxpts]]          


def defineColourrange(img, imn, pxplot=None):
    """Define colour range manually by clicking on the lightest and 
    darkest regions of the target extent that will be defined. Plot interaction 
    information: Left click to select, right click to undo selection, close the 
    image window to continue, and the window automatically times out after two 
    clicks.
    
    :param img: Image array
    :type img: arr
    :param imn: Image name
    :type imn: str
    :param pxplot: Plotting extent for manual area definition, default to None
    :type pxplot: list, optional
    :returns: List containing the upper and lower boundary for pixel detection
    :rtype: list
    """
    #Initialise figure window
    fig=plt.gcf()
    fig.canvas.set_window_title(imn + ': Click lightest colour and darkest' 
                                ' colour')
    
    #Plot image
    plt.imshow(img, origin='upper')
    
    #Define plotting extent if required
    if pxplot is not None:
        plt.axis([pxplot[0],pxplot[1],pxplot[2],pxplot[3]])

    #Manually interact to select lightest and darkest part of the region            
    colours = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, 
                        mouse_pop=3, mouse_stop=2)
    
    print('\n' + imn + ': you clicked ' + colours)
    
    #Show plot
    plt.show()
    plt.close()
    
    #Get pixel intensity value for pt1       
    col1_rbg = img[colours[0][1],colours[0][0]]
    if col1_rbg == 0:
        col1_rbg=1

    #Get pixel intensity value for pt2        
    col2_rbg = img[colours[1][1],colours[1][0]]
    if col2_rbg == 0:
        col2_rbg=1
        
    #Assign RBG range based on value of the chosen RBG values
    if col1_rbg > col2_rbg:
        upper_boundary = col1_rbg
        lower_boundary = col2_rbg
    else:
        upper_boundary = col2_rbg
        lower_boundary = col1_rbg
    
    print('\nColour range found from manual selection')
    print('Upper RBG boundary: ' + str(upper_boundary))
    print('Lower RBG boundary: ' + str(lower_boundary))

    #Return RBG range
    return [upper_boundary, lower_boundary]
   
    
def getOGRArea(pts):
    """Get real world OGR polygons (.shp) from xyz poly pts with real world 
    points which are compatible with mapping software (e.g. ArcGIS).

    :param pts: UV/XYZ coordinates of a given area shape
    :type pts: arr 
    :returns: List of OGR geometry polygons
    :rtype: list                           
    """                      
    #Create geometries from uv/xyz coordinates using ogr                     
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in pts:
        if np.isnan(p[0]) == False:
            if len(p)==2:
                ring.AddPoint(int(p[0]),int(p[1]))
            else:                  
                ring.AddPoint(p[0],p[1],p[2])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly
