#PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie
#
#PyTrx is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

"""
The Line module handles the functionality for obtaining line measurements from 
oblique time-lapse imagery. Specifically, this module contains functions for:
(1) Performing manual detection of lines in oblique imagery; and (2) 
Determining real-world distances from oblique imagery.
"""

#Import packages
import matplotlib.pyplot as plt
import numpy as np
import ogr

#Import PyTrx functions and classes
from PyTrx import Velocity
from PyTrx.Images import ImageSequence
from PyTrx.CamEnv import projectUV, setProjection

#------------------------------------------------------------------------------

class Line(ImageSequence):
    """A class for handling lines/distances (e.g. glacier terminus position)
    through an image sequence, with methods to manually define pixel lines in 
    the image plane and georectify them to generate real-world coordinates and 
    distances. The Line class object primarily inherits from the Area class.
    
    :param imageList: List of images to be inputted into the :class:`PyTrx.Images.ImageSequence` object
    :type imageList: str/list            
    :param cameraenv: Camera environment parameters which can be read into the :class:`PyTrx.CamEnv.CamEnv` object as a text file
    :type cameraenv: str 
    :param hmatrix: Homography matrix
    :type hmatrix: arr          
    :param calibFlag: An indicator of whether images are calibrated, for the :class:`PyTrx.Images.ImageSequence` object, default to True
    :type calibFlag: bool, optional         .          
    :param band: String denoting the desired image band, default to 'L' (grayscale)
    :type band: str, optional           .
    :param equal: Flag denoting whether histogram equalisation is applied to images (histogram equalisation is applied if True). Default to True.
    :type equal: bool, optional              
    """     
    #Object initialisation        
    def __init__(self, imageList, cameraenv, hmatrix, calibFlag=True, band='L', 
                 equal=True):
        '''Line object initialisation'''
        #Initialise and inherit from the ImageSequence object        
        ImageSequence.__init__(self, imageList, band, equal)

        #Set camera environment and calibration flag
        self._camEnv=cameraenv
        self._calibFlag=calibFlag
        
        if hmatrix is not None:
            self._hmatrix=hmatrix
            hmat0=None
            self._hmatrix.insert(0, hmat0)
        else:
            self._hmatrix=None
        
        
    def calcManualLines(self):
        """Method to manually define pixel lines from an image sequence. The 
        lines are manually defined by the user on an image plot. Returns the 
        line pixel coordinates and pixel length.
        
        :returns: XYZ and UV line lengths and coordinates
        :rtype: list
        """
        print('\n\nCOMMENCING LINE DETECTION')                        
            
        #Set up output dataset
        lines=[]        

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
        
        #Cycle through image pairs (numbered from 0)
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
            
            #Define line data
            if self._hmatrix is not None:
                out = calcManualLine(img1, imn, self._hmatrix[i], invprojvars)
            else:
                out = calcManualLine(img1, imn, None, invprojvars)
                
            #Append to list
            lines.append(out)
        
        #Return pixel point coordinates and lines
        return lines


#------------------------------------------------------------------------------

def calcManualLine(img, imn, hmatrix=None, invprojvars=None):
    """Manually define a line in a given image to produce XYZ and UV line 
    length and corresponding coordinates. Lines are defined through user input 
    by clicking in the interactive image plot. This primarily operates via the 
    pyplot.ginput function which allows users to define coordinates through 
    plot interaction. If inverse projection variables are given, XYZ lines
    and coordinates are also calculated.
    
    :param img: Image array for plotting.
    :type img: arr
    :param imn: Image name
    :type imn: str
    :param hmatrix: Homography matrix, default to None
    :type hmatrix: arr, optional
    :param invprojvars: Inverse projection variables [X,Y,Z,uv0], default to None
    :type invprojvars: list, optional    
    :returns: Four list elements containing: line length in xyz (list), xyz coordinates of lines (list), line length in pixels (list), and uvcoordinates of lines (list)
    :rtype: list
    """
    #Initialise figure window
    fig=plt.gcf()
    fig.canvas.set_window_title(imn + ': Define line. ' 
                                'Press enter to record points.')
    
    #Plot image
    plt.imshow(img, origin='upper',cmap='gray')        
    rawpx = plt.ginput(n=0, timeout=0, show_clicks=True, 
                       mouse_add=1, mouse_pop=3, mouse_stop=2)            
    print('\nYou clicked ' + str(len(rawpx)) + ' points in image ' + str(imn))
    
    #Show plot
    plt.show()
    plt.close()

    #Convert coordinates to array
    pxpts=[]
    for i in rawpx:
        pxpts.append([[i[0],i[1]]])
    pxpts=np.asarray(pxpts)
    
    #Calculate homography-corrected pts if desired
    if hmatrix is not None:
        print('Correcting for camera motion')
        pxpts = Velocity.apply_persp_homographyPts(pxpts, hmatrix, 
                                                   inverse=True)
        
    #Re-format pixel point coordinates
    pxpts = np.squeeze(pxpts)
        
    #Create OGR pixl line object and extract length
    pxline = getOGRLine(pxpts)
    print('Line contains ' + str(pxline.GetPointCount()) + ' points')  
    pxline = pxline.Length()                 
    print('Line length: ' + str(pxline) + ' px')
    
             
    if invprojvars is not None:  
        #Get xyz coordinates with inverse projection           
        xyzpts = projectUV(pxpts, invprojvars)
            
        #Create ogr line object
        xyzline = getOGRLine(xyzpts)
        xyzline = xyzline.Length()
            
        print('Line length: ' + str(xyzline) + ' m')
        
        return [[xyzline, xyzpts], [pxline, pxpts]]
    
    else:
        #Return pixel coordinates only
        return [[None, None], [pxline, pxpts]]


def getOGRLine(pts):
    """Function to construct an OGR line from a set of uv coordinates.
    
    :param pts: A series of uv coordinates denoting a line
    :type pts: arr
    :returns: A line object (ogr.Geometry) constructed from the input coordinates
    :rtype: ogr.Geometry
    """
    #Initially construct geometry object             
    line = ogr.Geometry(ogr.wkbLineString)
    
    #Append points to geometry object
    for p in pts:
        if np.isnan(p[0]) == False: 
            if len(p)==2:
                line.AddPoint(p[0],p[1])
            else:
                line.AddPoint(p[0],p[1],p[2])
            
    #Return geometry line
    return line 
        

#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   
