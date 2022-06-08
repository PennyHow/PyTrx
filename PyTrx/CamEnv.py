#PyTrx (c) is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

"""
The Camera Environment module contains the object-constructors and functions 
for: (1) Representing a camera model in three-dimensional space; and (2) 
Effective translation of measurements in an XY image plane to XYZ real-world 
coordinates. The projection and inverse transformation functions are based on 
those available in the ImGRAFT toolbox for Matlab. Translations from
ImGRAFT are noted in related script comments.              
"""

#Import PyTrx packages
try:
    from Utilities import plotGCPs, plotCalib, plotResiduals, plotPrincipalPoint
    from FileHandler import readImg, readGCPs, readMatrixDistortion 
    from DEM import ExplicitRaster,load_DEM,voxelviewshed
    from Images import CamImage
except:
    from PyTrx.Utilities import plotGCPs, plotCalib, plotResiduals, plotPrincipalPoint
    from PyTrx.FileHandler import readImg, readGCPs, readMatrixDistortion 
    from PyTrx.DEM import ExplicitRaster,load_DEM,voxelviewshed
    from PyTrx.Images import CamImage
    
#Import other packages
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
import numpy as np
import cv2, glob, unittest

#------------------------------------------------------------------------------

class GCPs():    
    """A class representing the geography of the camera scene. Contains
    ground control points, as the world and image points, the DEM data and 
    extent, and the image the ground control points correspond to, as an 
    Image object
    
    Attributes
    ----------
    _dem : list
      DEM file path
    _gcpImage : arr
      Image that GCPs were defined in
    _gcpxyz : list
      XYZ positions of GCPs
    _gcpuv : list
      UV positions of GCPs 
    """
    def __init__(self, dem, GCPpath, imagePath):
        """"Initialise the GCP object  

        Parameters
        ----------            
        dem : str 
          The file path of the ASCII DEM
        GCPpath : str 
          The file path of the GCP text file, with a header line, and tab 
          delimited x, y, z world coordinates and u, v image coordinates on 
          each line
        imagePath : str 
          The file path of the image the GCP positions correspond to
        """ 
        #DEM handling
        self._dem = dem
       
        #Get image from CamImage object
        if imagePath!=None:
            self._gcpImage=CamImage(imagePath)
        
        #Get GCP data using the readGCP function in FileHandler
        if GCPpath!=None:
            world, image = readGCPs(GCPpath)
            self._gcpxyz = world
            self._gcpuv = image                

        
    def getGCPs(self):
        """Return the world and image GCPs"""      
        return self._gcpxyz, self._gcpuv

        
    def getDEM(self):
        """Return the dem object."""       
        return self._dem

    
    def getImage(self):
        """Return the GCP reference image."""        
        return self._gcpImage

                
#------------------------------------------------------------------------------        
   
class CamCalib(object):
    """This base class models a standard camera calibration matrix as per 
    OpenCV, MatLab and ImGRAFT. The class uses a standard pinhole camera model, 
    drawing on the functions within OpenCV. A scene view is formed by 
    projecting 3D points into the image plane using a perspective 
    transformation.         
    The camera intrinsic matrix is defined as a 3 x 3 array: [fx,0,0][s,fy,0]
    [cx,cy,1], where fx and fy is the camera focal length (in pixel units) and 
    cx and cy as the location of the image centre (in pixels too), s is the 
    skew, and cx and cy are the image dimensions in pixels.        
    In addition, the radial distortion and tangential distortion are 
    represented as a series of coefficients. These distortions are introduced 
    by discrepancies in the camera lens and between the lens and the camera 
    sensor: 1) Radial Distortion Coefficients: k ([k1,k2,k3,k4,k5,k6]), between 
    2 and 6 coefficients needed; and 2) Tangential Distortion Coefficients: 
    p ([p1,p2])
    The object can be initiated directly either as a list of three elements for 
    each of the intrinsic, tangential and radial arrays, or by referencing a 
    file (.mat or .txt) containing the calibration data in a pre-designated 
    format
    
    Attributes
    ----------
    _intrMat : arr
      Intrinsic camera matrix
    _intrMat : arr
      OpenCV-compatiable intrinsic camera matrix
    _tanCorr : list
      Tangential lens correction coefficients
    _radCorr : list
      Radial lens correction coefficients
    _calibErr : int
      Calibration residual error
    _focLen : list
      Camera focal length (px)
    _camCen : list
      Camera principal point
    """    
    def __init__(self, *args): 
        """Initialise the camera calibration object 
        
        Parameters
        ----------
        *args : list/str
          Either a calibration text file, a series of calibration text files, a 
          list of raw parameters, or a set of calibration images (along with 
          calibration chessboard dimensions) 
        """ 
        failed=False 
            
        #Read calibration from file
        if isinstance(args[0],str):
            print('\nAttempting to read camera calibs from a single file')
            args=readMatrixDistortion(args[0])
            args=self.checkMatrix(args)
            if args==None:
                failed=True
            else:
                self._intrMat=args[0]
                self._tanCorr=args[1]
                self._radCorr=args[2]
                self._calibErr=None
            

        #Read calibration from several files                      
        elif isinstance(args[0],list):                 
            if args[0][0][-4:] == '.txt':
                print('\nAttempting to read camera calibs from average over ' 
                      'several files')
                intrMat=[]
                tanCorr=[]
                radCorr=[]               
                for item in args[0]:
                    if isinstance(item,str):
                        arg=readMatrixDistortion(item)
                        arg=self.checkMatrix(arg)
                        if arg==None:
                            failed=True
                            break
                        else:
                            intrMat.append(arg[0])
                            tanCorr.append(arg[1])
                            radCorr.append(arg[2])
                    else:
                        failed=True

                self._intrMat = sum(intrMat)/len(intrMat)
                self._tanCorr = sum(tanCorr)/len(tanCorr)
                self._radCorr = sum(radCorr)/len(radCorr)
                self._calibErr=None
                
            #Calculate calibration from images                    
            elif args[0][0][-4:] == '.JPG' or '.PNG':
                print ('\nAttempting to calculate camera calibs from input'
                        + ' images')
                calibimgs=[]
                for i in sorted(glob.glob(args[0][0])):
                    calibimgs.append(i)
                    
                arg, err = calibrateImages(calibimgs,[int(args[0][1]),
                                           int(args[0][2])])
                arg = self.checkMatrix(arg)
                
                if arg==None:
                    failed=True
                else:
                    self._intrMat=arg[0]
                    self._tanCorr=arg[1]
                    self._radCorr=arg[2]
                    self._calibErr=err
            else:
                failed=True
        
        #Define calibration from raw input
        elif isinstance(args[0],tuple):
            print ('\nAttempting to make camera calibs from raw data '
                   + 'sequences')
            
            arg = self.checkMatrix([args[0][0],args[0][1],args[0][2]]) 
            
            self._intrMat=arg[0]
            self._tanCorr=arg[1]
            self._radCorr=arg[2]
            self._calibErr=None 
            
        else:
            failed=True
                        
            
        if failed:
            print('\nError creating camera calibration object' +
                  '\nPlease check calibration specification or files')
            return None
            
        self._focLen=[self._intrMat[0,0], self._intrMat[1,1]]       
        self._camCen=[self._intrMat[2,0], self._intrMat[2,1]] 
        self._intrMatCV2=None
                
            
    def getCalibdata(self):
        """Return camera matrix, and tangential and radial distortion 
        coefficients"""
        return self._intrMat, self._tanCorr, self._radCorr

        
    def getCamMatrix(self):
        """Return camera matrix"""
        return self._intrMat

    
    def getDistortCoeffsCV2(self):
        """Return radial and tangential distortion coefficients"""
        #Returns certain number of values depending on number of coefficients
        #inputted  
        if len(self._radCorr)==2:
            return np.append(self._radCorr[0:2], self._tanCorr)
        else:
            return np.append(np.append(self._radCorr[0:2], self._tanCorr),
                             self._radCorr[2:])

        
    def getCamMatrixCV2(self):
        """Return camera matrix in a structure that is compatible with 
        subsequent photogrammetric processing using OpenCV"""
        if self._intrMatCV2 is None:
            
            # Transpose if 0's are not in correct places
            if (self._intrMat[2,0]!=0 and self._intrMat[2,1]!=0 and 
                self._intrMat[0,2]==0 and self._intrMat[1,2]==0):
                self._intrMatCV2 = self._intrMat.transpose()
            else:
                self._intrMatCV2=self._intrMat[:]
                
            # Set 0's and 1's in the correct locations
            it=np.array([[0,1],[1,0],[2,0],[2,1]])                 
            for i in range(4):
                x = it[i,0]
                y = it[i,1]
                self._intrMatCV2[x,y]=0.        
            self._intrMatCV2[2,2]=1. 
    
        return self._intrMatCV2

        
    def reportCalibData(self):
        """Self reporter for Camera Calibration object data"""
        print('\nDATA FROM CAMERA CALIBRATION OBJECT')
        print('Intrinsic Matrix:')
        for row in self._intrMat:
            print(str(row[0]) + str(row[1]) + str(row[2]))
        print('\nTangential Correction:')
        print(str(self._tanCorr))
        print('\nRadial Correction:')
        print(str(self._radCorr))
        print('\nFocal Length:')
        print(str(self._focLen))
        print('\nCamera Centre:')
        print(str(self._camCen))
        if self._calibErr != None:
            print('\nCalibration Error:')
            print(str(self._calibErr))


    def checkMatrix(self, matrix):
        """Function to support the calibrate function. Checks and converts the 
        intrinsic matrix to the correct format for calibration with opencv.
        
        Parameters
        ----------
        matrix : arr 
          Intrinsic camera matrix

        Returns
        -------
        list
          The object's intrinsic matrix (checked), tangential distortion and 
          radial distortion information
        """  
        ###This is moved over from readfile. Need to check calibration matrices
        if matrix==None:
            return None
                
        #Check matrix
        intrMat=matrix[0]
        
        #Check tangential distortion coefficients
        tanDis=np.zeros(2)
        td = np.array(matrix[1])
        tanDis[:td.size] = td
        
        #Check radial distortion coefficients
        radDis=np.zeros(6)
        rd = np.array(matrix[2]) 
        radDis[:rd.size] = rd
           
        return intrMat, tanDis, radDis

                          
#------------------------------------------------------------------------------
                          
class CamEnv(CamCalib):    
    """A class to represent the camera object, containing the intrinsic
    matrix, distortion parameters and camera pose (position and direction).    
    Also inherits from the :class:`PyTrx.CamEnv.CamCalib` object, representing 
    the intrinsic camera information.
    This object can be initialised either through an environment file (and
    passed to the initialiser as a filepath), or with the set intput parameters
    
    Attributes
    ----------
    _name : str 
      The reference name for the camera
    _GCPpath : str 
      The file path of the GCPs, for the GCPs object
    _DEMpath : str 
      The file path for the DEM, for the GCPs object  
    _DEMdensify : int
      DEM densification factor
    _DEM : PyTrx.DEM.ExplicitRaster object
      DEM object
    _invProjVars : list
      Inverse projection variables
    _imagePath : str 
      The file path for the GCP reference image, for the GCPs object
    _calibPath : str
      The file path for the calibration file. This can be either as a .mat 
      Matlab file or a text file. The text file should be of the following tab 
      delimited format: RadialDistortion [k1 k2 k3...k7], 
      TangentialDistortion [p1 p2], IntrinsicMatrix [x y z][x y z][x y z] End
    _camLoc : list 
      The x,y,z coordinates of the camera location, as a list 
    _camDirection : list 
      The yaw, pitch and roll of the camera, as a list
    """   
    def __init__(self, envFile):
        """Initialise Camera Environment object
        
        Parameters
        ----------
        envFile : str/list
          Filepath to environment file, or list containing camera name, GCPs
          filepath, DEM filepath, reference image filepath, camera calibration 
          filepath, DEM densification factor, camera coordinates, and camera 
          pose (YPR)
          """ 
        print('\nINITIALISING CAMERA ENVIRONMENT')
 
        #Read camera environment from text file        
        if isinstance(envFile, str):
            #Read parameters from the environment file 
            params = self.dataFromFile(envFile)
    
            #Exit programme if file is invalid
            if params==False:
                print('\nUnable to define camera environment')
                pass
            
            #Extract input files from camera environment file 
            else:
                (name, GCPpath, DEMpath, imagePath, 
                 calibPath, coords, ypr, DEMdensify) = params           

        #Read camera environment from files as input variables
        elif isinstance(envFile, list):
            name = envFile[0]
            GCPpath = envFile[1]
            DEMpath = envFile[2]
            imagePath = envFile[3]
            calibPath = envFile[4]
            coords = envFile[5]
            ypr = envFile[6]
            DEMdensify = envFile[7]

        else:
            print('\nInvalid camera environment data type')
            pass
            
        #Set up object parameters
        self._name = name
        self._camloc = np.array(coords)
        self._DEMpath = DEMpath        
        self._DEMdensify = DEMdensify
        self._GCPpath = GCPpath
        self._imagePath = imagePath
        self._refImage = CamImage(imagePath) 

        #Set yaw, pitch and roll to 0 if no information is given        
        if ypr is None:
            self._camDirection = np.array([0,0,0])
        else:
            self._camDirection =  np.array(ypr)

        #Initialise CamCalib object for calibration information        
        self._calibPath=calibPath
        CamCalib.__init__(self,calibPath)                
                
        #Leave DEM and inverse projection variables empty to begin with
        self._DEM = None
        self._invProjVars = None
      
        #Initialise GCPs object for GCP and DEM information
        if (self._GCPpath!=None and self._imagePath!=None):
            print('\nCreating GCP environment')
            self._gcp=GCPs(self._DEM, self._GCPpath, self._imagePath)        
        
       
    def dataFromFile(self, filename):
        """Read CamEnv data from .txt file containing keywords and filepaths
        to associated data.
        
        Parameters
        ----------
        filename : str 
          Environment file path
        
        Returns
        -------
        list 
          Camera environment information (name, GCP filepath, DEM filepath, 
          image filepath, calibration file path, camera coordinates, camera 
          pose (ypr) and DEM densification factor)
        """
        #Define keywords to search for in file        
        self.key_labels={'name':'camera_environment_name',
                         'GCPpath':'gcp_path',
                         'DEMpath':'dem_path',
                         'imagePath':'image_path',
                         'calibPath':'calibration_path',
                         'coords':'camera_location',
                         'ypr':'yaw_pitch_roll',
                         'DEMdensify':'dem_densification'}
        key_lines=dict(self.key_labels)
        for key in key_lines:
            key_lines.update({key:None})
        
        #Extract all lines in the specification file                       
#        f=filename.open() 
        f=open(filename)
        lines=f.readlines()
        f.close()
        
        #Search for keywords and identify which line they are in       
        for i in range(len(lines)):
            stripped=lines[i].split("#")[0].strip().lower().replace(" ","")
            for key in self.key_labels:
                if self.key_labels[key]==stripped:
                    key_lines.update({key:i})
        
        #Define CamEnv name if information is present in .txt file
        lineNo=key_lines["name"]
        if lineNo!=None:
            name = self.__getFileDataLine__(lines,lineNo)
        else:
            print('\nName not supplied in: ' + str(filename))              
            return False

        #Define GCPpath if information is present in .txt file
        lineNo=key_lines["GCPpath"]
        if lineNo!=None:
            GCPpath = self.__getFileDataLine__(lines,lineNo)
        else:
            print('\nGCPpath not supplied in: ' + str(filename))              
            GCPpath=None
            
        #Define DEMpath if information is present in .txt file
        lineNo=key_lines["DEMpath"]
        if lineNo!=None:
            DEMpath = self.__getFileDataLine__(lines,lineNo)
        else:
            print('\nDEMpath not supplied in: ' + str(filename))              
            return False
            
        #Define imagePath if information is present in .txt file
        lineNo=key_lines["imagePath"]
        if lineNo!=None:
            imagePath = self.__getFileDataLine__(lines,lineNo)
        else:
            print('\nimagePath not supplied in: ' + str(filename))              
            return False 

        #Define DEM densification specifications (DEMdensify)          
        lineNo=key_lines["DEMdensify"]
        if lineNo!=None:
            DEMdensify = self.__getFileDataLine__(lines,lineNo)
            DEMdensify = int(DEMdensify)
        else:
            print('\nDem densification level not supplied in: ' 
                  + str(filename))  
            print('Setting to 1 (No densification)')
            DEMdensify=1

        #Define calibPath if information is present in .txt file
        lineNo=key_lines["calibPath"]
        if lineNo!=None:
            calibPath = self.__getFileDataLine__(lines,lineNo)            
            fields = calibPath.strip('[]').split(',')
            calibPath = []
            for f in fields:
                calibPath.append(f)
            if len(calibPath) == 1:
                calibPath = calibPath[0]              
        else:
            print('\ncalibPath not supplied in: ' + str(filename))             
            return False   

        #Define camera location coordinates (coords)
        lineNo=key_lines["coords"]
        if lineNo!=None:
            coords = self.__getFileDataLine__(lines,lineNo)
            fields = coords.strip('[]').split()    
            coords = []
            for f in fields:
                coords.append(float(f)) 
        else:
            print('\nCoordinates not supplied in: ' + str(filename))              
            return False 

        #Define yaw, pitch, roll if information is present in .txt file
        lineNo=key_lines["ypr"]
        if lineNo!=None:
            ypr = self.__getFileDataLine__(lines,lineNo)           
            fields = ypr.strip('[]').split()
            ypr = []
            for f in fields:
                ypr.append(float(f)) 
        else:
            print('\nYPR not supplied in: ' + str(filename))              
            return False
           
        return name,GCPpath,DEMpath,imagePath,calibPath,coords,ypr,DEMdensify


    def optimiseCamEnv(self, optimise, optmethod='trf', show=False):
        """Optimise projection variables in the camera environment. The precise 
        parameters to optimise are defined by the optimise variable
        
        Parameters
        ----------
        optimise : str 
          Parameters to optimise - 'YPR' (optimise camera pose only), 'EXT' 
          (optimise external camera parameters), 'INT' (optimise internal 
          camera parameters), or 'ALL' (optimise all projection parameters)
        optmethod : str, optional 
          Optimisation method (default='trf')
        show : bool, optional
          Flag to denote if optimisation output should be plotted 
          (default=False)
        """
        #Get GCPs
        xyz, uv = self._gcp.getGCPs()

        #Get camera environment parameters
        projvars = [self._camloc, self._camDirection, self._radCorr, 
                    self._tanCorr, self._focLen, self._camCen, self._refImage]
        
        opt_projvars = optimiseCamera(optimise, projvars, xyz, uv, 
                                      optmethod, show)
        
        self._camloc = opt_projvars[0] 
        self._camDirection = opt_projvars[1] 
        self._radCorr = opt_projvars[2] 
        self._tanCorr = opt_projvars[3]
        self._focLen = opt_projvars[4] 
        self._camCen = opt_projvars[5] 
        self._refImage = opt_projvars[6]

    
    def __getFileDataLine__(self, lines, lineNo):
        """Return a data line from the Camera Environment specification file
        
        Parameters
        ----------
        lines : str 
          Line string
        lineNo : int 
          Line number
          
        Returns
        -------
        str
          Data line
        """
        return lines[lineNo+1].split('#')[0].strip()

        
    def getRefImageSize(self):
        """Return the dimensions of the reference image"""
        return self._refImage.getImageSize()
      
        
    def getDEM(self):
        """Return DEM as PyTrx.DEM.ExplicitRaster object"""
        if self._DEM is None:
            dem = load_DEM(self._DEMpath)
            if self._DEMdensify>1:
                dem=dem.densify(self._DEMdensify)
            self._DEM=dem
            return self._DEM
        
        else:
            return self._DEM


    def showGCPs(self):
        """Plot GCPs in image plane and DEM scene"""
        xyz, uv = self._gcp.getGCPs()               #Get GCP positions
        dem = self.getDEM()                         #Get DEM        
        refimage=self._refImage
        img = refimage.getImageArray()              #Get image array
        imn = refimage.getImageName()               #Get image name

        #Plot GCPs
        plotGCPs([xyz,uv], img, imn, dem, self._camloc, extent=None)            


    def showPrincipalPoint(self):
        """Plot Principal Point on reference image"""
        refimage=self._refImage
        img = refimage.getImageArray()              #Get image array
        imn = refimage.getImageName()               #Get image name
        
        #Plot principal point 
        plotPrincipalPoint(self._camCen, img, imn)


    def showCalib(self):
        """Plot corrected and uncorrected reference image"""
        refimage=self._refImage
        img = refimage.getImageArray()              #Get image array
        imn = refimage.getImageName()               #Get image name        
        matrix = self.getCamMatrixCV2()             #Get camera matrix
        distort = self.getDistortCoeffsCV2()        #Get distortion parameters

        #Plot calibrated image
        plotCalib(matrix, distort, img, imn)       


    def showResiduals(self):
        """Show positions of xyz GCPs and projected GCPs, and residual 
        differences between their positions. This can be used as a measure of
        a error in the georectification of measurements"""        
        xyz, uv = self._gcp.getGCPs()               #Get GCPs
        dem = self.getDEM()                         #Get DEM

        #Set inverse projection parameters
        invprojvars = setProjection(dem, self._camloc, self._camDirection, 
                                    self._radCorr, self._tanCorr, self._focLen, 
                                    self._camCen, self._refImage)
        
        #Compute residuals
        computeResidualsXYZ(invprojvars, xyz, uv, dem)


    def reportCamData(self):
        """Reporter for testing that the relevant data has been successfully 
        imported. Testing for camera Environment name, camera location (xyz),
        reference image, DEM, DEM densification, GCPs, yaw pitch roll, camera 
        matrix, and distortion coefficients""" 
        #Camera name and location
        print('\nCAMERA ENVIRONMENT REPORT')
        print('Camera Environment name: ' + str(self._name)) 
        print('Camera Location [X,Y,Z]: ' + str(self._camloc))
        
        #Reference image
        print('\nReference image used for baseline homography ' + 
              'and/or GCP control:')
        print(str(self._imagePath))
        
        #DEM and densification        
        print('\nDEM file used for projection:')
        print(str(self._DEMpath))
        if self._DEMdensify==1:
            print('DEM is used at original resolution')
        else:
            print('DEM is resampled at '+ str(self._DEMdensify) + 
                  ' times resolution')
        
        #GCPs        
        if self._GCPpath!=None:
            print('\nGCP file used to define camera pose:')
            print(str(self._GCPpath))
        else:
            print('No GCP file defined')
         
        #Yaw, pitch, roll
        if self._camDirection is None:
            print('\nCamera pose assumed unset (zero values)')
        else:
            print('\nCamera pose set as [Yaw,Pitch,Roll]: ')
            print(str(self._camDirection))

        #Camera calibration (matrix and distortion coefficients)
        if isinstance(self._calibPath[0],list):
            if self._calibPath[0][0][-4:] == '.txt':
                print('\nCalibration calculated from multiple files:')
                print(str(self._calibPath))

        elif isinstance(self._calibPath[0],str):
            if self._calibPath[0][-4:] == '.txt':
                print('\nCalibration calculated from single file:')
                print(str(self._calibPath))
                
            elif self._calibPath[0][0][-4:] == '.JPG' or '.PNG':
                print('\nCalibration calculated from raw images:')                    
                print(str(self._calibPath))
                                         
        elif isinstance(self._calibPath[0],np.array):   
            print('\nCalibration calculated from raw data:')
            print(str(self._calibPath))
        
        else:
            print('\nCalibration undefined')
             
        #Report raster DEM details from the DEM class
        if isinstance(self._DEM,ExplicitRaster):
            print('\nDEM set:')
            self._DEM.reportDEM()

        #Report calibration parameters from CamCalib class
        self.reportCalibData()


def calibrateImages(imageFiles, xy, refine=None):
    """Function for calibrating a camera from a set of input calibration
    images. Calibration is performed using OpenCV's chessboard calibration 
    functions. Input images (imageFile) need to be of a chessboard with 
    regular dimensions and a known number of corner features (xy).   
    Please note that OpenCV's calibrateCamera function is incompatible 
    between different versions of OpenCV. Included here is the function
    for version 3. Please see OpenCV's documentation for older versions.
     
    Parameters
    ----------
    imageFiles : list 
      List of image file names
    xy : list
      Chessboard corner dimensions [rows, columns]          
    refine : int, optional 
      OpenCV camera model refinement method - cv2.CALIB_FIX_PRINCIPAL_POINT 
      (fix principal point), cv2.CALIB_FIX_ASPECT_RATIO (Fix aspect ratio), 
      cv2.CALIB_FIX_FOCAL_LENGTH (Fix focal length), cv2.CALIB_FIX_INTRINSIC 
      (Fix camera model), cv2.CALIB_FIX_K1...6 (Fix radial coefficient 1-6), 
      cv2.CALIB_FIX_TANGENT_DIST (Fix tangential coefficients), 
      cv2.CALIB_USE_INTRINSIC_GUESS (Use initial intrinsic values), 
      cv2.CALIB_ZERO_TANGENT_DIST (Set tangential distortion coefficients to 
      zero), cv2.CALIB_RATIONAL_MODEL (Calculate radial distortion coefficients 
      k4, k5, and k6) (default=None)
    
    Returns
    -------
    arr/list 
      A list containing the camera intrinsic matrix (arr), and tangential (arr) 
      and radial distortion coefficents (arr), and the Camera calibration error 
      (int)
    """   
    #Define shape of array
    objp = np.zeros((xy[0]*xy[1],3), np.float32)           
    objp[:,:2] = np.mgrid[0:xy[1],0:xy[0]].T.reshape(-1,2) 

    #Array to store object pts and img pts from all images
    objpoints = []                                   
    imgpoints = []                                   
    
    #Set image counter for loop
    imageCount = 0
    
    #Loop to determine if each image contains a chessboard pattern and 
    #store corner values if it does
    for fname in imageFiles:
        
        #Read file as an image using OpenCV
        img = cv2.imread(fname)   

        #Change RGB values to grayscale             
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
        
        #Find chessboard corners in image
        patternFound, corners = cv2.findChessboardCorners(gray,
                                                          (xy[1],xy[0]),
                                                          None)
        
        #Cycle through images, print if chessboard corners have been found 
        #for each image
        imageCount += 1
        print(str(imageCount) + ': ' + str(patternFound) + ' ' + fname)
        
        #If found, append object points to objp array
        if patternFound == True:
            objpoints.append(objp)
            
            #Determine chessboard corners to subpixel accuracy
            #Inputs: winSize specified 11x11, zeroZone is nothing (-1,-1), 
            #opencv criteria
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                             (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,
                             30,0.001))
                             
            imgpoints.append(corners)
            
            #Draw and display corners
            cv2.drawChessboardCorners(img,(xy[1],xy[0]),corners,
                                      patternFound)

    #Calculate initial camera matrix and distortion
    err,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   gray.shape[::-1],
                                                   None,
                                                   5)
    
    #Optimise camera matrix and distortion using fixed principal point
    if refine != None:
        err,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       mtx,
                                                       5,
                                                       flags=refine)                                                                   

    #Change matrix structure for compatibility with PyTrx
    mtx = np.array([mtx[0][0],mtx[0][1],0,
                   0,mtx[1][1],0,
                   mtx[0][2],mtx[1][2],1]).reshape(3,3)

    
    #Restructure distortion parameters for compatibility with PyTrx
    rad = np.array([dist[0],dist[1],dist[4], 0.0, 0.0, 0.0]).reshape(6)
    tan = np.array(dist[2:4]).reshape(2)
    
    #Return matrix, radial distortion and tangential distortion parameters
    return [mtx, tan, rad], err
        

def constructDEM(dempath, densefactor):
    """Construct DEM from a given file path and densification factor
    
    Parameters
    ----------
    dempath : str
      DEM filepath
    densefactor : int 
      Densification factor
    
    Returns
    -------
    dem : PyTrx.DEM.ExplicitRaster
      DEM object
    """
    #Prepare DEM from file
    dem=load_DEM(dempath)
        
    #DEM densification
    if densefactor>1:
        dem=dem.densify(densefactor)
            
    return dem

            
def setProjection(dem, camloc, camdir, radial, tangen, foclen, camcen, refimg,
                  viewshed=True):
    """Set the inverse projection variables.
    
    Parameters
    ----------
    dem : PyTrx.DEM.ExplicitRaster
      DEM object
    camloc : arr 
      Camera location (X,Y,Z)
    camdir : arr 
      Camera pose [yaw, pitch, roll]             
    radial : arr 
      Radial distortion coefficients
    tangen : arr 
      Tangential distortion coefficients
    foclen : arr 
      Camera focal length
    camcen : arr 
      Camera principal point
    refimg : arr 
      Reference image (function only uses the image dimensions)
    viewshed : bool 
      Flag to denote if viewshed from camera should be determined before 
      projection
    
    Returns
    -------
    invProjVars : list
      Inverse projection coefficients [X,Y,Z,uv0]
    """             
    print('\nSetting inverse projection coefficients')

    if isinstance(dem, list):
        dem=constructDEM(dem[0], dem[1])
        X=dem.getData(0)
        Y=dem.getData(1)
        Z=dem.getData(2)        
    else:    
        X=dem.getData(0)
        Y=dem.getData(1)
        Z=dem.getData(2)
    
    #Define visible extent of the DEM from the location of the camera
    if viewshed is True:
        visible=voxelviewshed(dem, camloc)
        XYZ=np.column_stack([X[visible[:]],Y[visible[:]],Z[visible[:]]])
    else:
        X=np.ravel(X)
        Y=np.ravel(Y)
        Z=np.ravel(Z)
        XYZ=np.column_stack([X,Y,Z])
        
    #Snap image plane to DEM extent
    uv0,dummy,inframe=projectXYZ(camloc, camdir, radial, tangen, foclen, 
                                 camcen, refimg, XYZ)
    uv0=np.column_stack([uv0,XYZ])
    uv0=uv0[inframe,:]

    #Assign real-world XYZ coordinates to image pixel coordinates         
    X=uv0[:,2]
    Y=uv0[:,3]
    Z=uv0[:,4]
    uv0=uv0[:,0:2]
    
    #Set inverse projection variables
    print('\nInverse projection coefficients defined')
    invProjVars=[X,Y,Z,uv0]             
    return invProjVars
            

def projectXYZ(camloc, camdirection, radial, tangen, foclen, camcen, refimg, 
               xyz):
    """Project the xyz world coordinates into the corresponding image 
    coordinates (uv). This is primarily executed using the ImGRAFT projection 
    function found in camera.m: uv,depth,inframe=cam.project(xyz)
    
    Parameters
    ----------
    camloc : arr 
      Camera location [X,Y,Z]
    camdirection : arr
      Camera pose (yaw, pitch, roll)
    radial : arr
      Radial distortion coefficients
    tangen : arr 
      Tangential distortion coefficients             
    foclen : arr 
      Camera focal length              
    camcen : arr 
      Camera principal point
    refimg : arr 
      Reference image (function only uses the image dimensions)
    xyz : arr 
      world coordinates            

    Returns
    -------
    uv : arr
      Pixel coordinates in image
    depth : int
      View depth
    inframe : arr
      Boolean vector containing whether each projected 3D point is inside the 
      frame       
    """  
    #This was in ImGRAFT/Matlab to transpose the input array if it's 
    #ordered differently 
    #if size(xyz,2)>3                                                 (MAT)
    #   xyz=xyz';                                                     (MAT)
    #end                                                              (MAT)
    #xyz=bsxfun(@minus,xyz,cam.xyz);                                  (MAT)
    ###need to check xyz is an array of the correct size
    ###this does element-wise subtraction on the array columns
    
    #Get camera location
    xyz=xyz-camloc
    
    #Get camera rotation matrix from pose 
    if camdirection.ndim == 1:
        Rprime=np.transpose(getRotation(camdirection))
    
    #Assign Rprime if camdirection variable is already a rotation matrix
    elif camdirection.ndim == 2:
        Rprime = camdirection
    
    #Multiply matrix
    xyz=np.dot(xyz,Rprime)
    
    #ImGRAFT/Matlab equiv to below command: 
    #xy=bsxfun(@rdivide,xyz(:,1:2),xyz(:,3))                          (MAT)
    xy=xyz[:,0:2]/xyz[:,2:3]
                
    if False:
        #Transposed from ImGRAFT
        r2=np.sum(xy*xy,1)                
        r2[r2>4]=4
        
        #Transposed from ImGRAFT
        if not np.allclose(radial[2:6], [0., 0., 0., 0.]):
            a=(1.+radial[0]*r2+radial[1]*r2*r2+radial[2]*r2*r2*r2)
            a=a/(1.+radial[3]*r2+radial[4]*r2*r2+radial[5]*r2*r2*r2)
        else:
            a=(1.+radial[0]*r2+radial[1]*r2*r2+radial[2]*r2*r2*r2)

        xty=xy[:,0]*xy[:,1]            
        pt1=a*xy[:,0]+2*tangen[0]*xty+tangen[1]*(r2+2*xy[:,0]*xy[:,0])
        pt2=a*xy[:,1]+2*tangen[0]*xty+tangen[1]*(r2+2*xy[:,1]*xy[:,1])            
        xy=np.column_stack((pt1,pt2))

    #ImGRAFT/Matlab version of code below: 
    #uv=[cam.f[1]*xy(:,1)+cam.c(1), cam.f(2)*xy(:,2)+cam.c(2)];       (MAT)
    uv=np.empty([xy.shape[0],xy.shape[1]])
               
    for i in range(xy.shape[0]):
        uv[i,0]=foclen[0] * xy[i,0] + camcen[0]
        uv[i,1]=foclen[1] * xy[i,1] + camcen[1]
 
    for i in range(xy.shape[0]):
        if xyz[i,2]<=0:
            uv[i,0]=np.nan
            uv[i,1]=np.nan

    depth=xyz[:,2]
    
    #Create empty array representing the image
    inframe=np.zeros(xy.shape[0],dtype=bool)

    #Get size of reference image
    if isinstance(refimg, str):
        ims=readImg(refimg)
        ims=ims.shape
    elif isinstance(refimg, np.ndarray):
        ims=refimg.shape
    else:
        ims=refimg.getImageSize()
    
    for i in range(xy.shape[0]):
        inframe[i]=(depth[i]>0)&(uv[i,0]>=1)&(uv[i,1]>=1)
        inframe[i]=inframe[i]&(uv[i,0]<=ims[1])&(uv[i,1]<=ims[0])
    
    return uv,depth,inframe

 
def projectUV(uv, invprojvars):  
    """Inverse project image coordinates (uv) to xyz world coordinates
    using inverse projection variables (set using setProjection function).         
    This function is primarily adopted from the ImGRAFT projection function 
    found in camera.m: uv,depth,inframe=cam.project(xyz)
    
    Parameters
    ----------
    uv : arr 
      Pixel coordinates in image
    invprojvars : list 
      Inverse projection variables [X,Y,Z,uv0]

    Returns
    -------
    xyz : arr      
      World coordinates of inputted pixel coordinates
    """                 
    #Create empty numpy array
    xyz=np.zeros([uv.shape[0],3])
    xyz[::]=float('NaN')
    
    #Get XYZ real world coordinates and corresponding uv coordinates
    X=invprojvars[0]
    Y=invprojvars[1]
    Z=invprojvars[2]
    uv0=invprojvars[3]
    
    #Snap uv and xyz grids together
    xi=interpolate.griddata(uv0, X, uv, method='linear')
    yi=interpolate.griddata(uv0, Y, uv, method='linear')
    zi=interpolate.griddata(uv0, Z, uv, method='linear')
    
    #Return xyz grids                
    xyz=np.column_stack([xi,yi,zi])       
    return xyz


def optimiseCamera(optimise, projvars, GCPxyz, GCPuv, optmethod='trf', 
                   show=False):
    """Optimise camera parameters using the pixel differences between a set
    of image GCPs and projected XYZ GCPs. The optimisation routine adopts the
    least_square function in scipy's optimize tools, using either the Trust 
    Region Reflective algorithm, the dogleg algorithm or the 
    Levenberg-Marquardt algorithm to refine a set group of projection 
    parameters - camera pose only, the internal camera parameters (i.e. radial 
    distortion, tangential distortion, focal length, principal point), the 
    external camera parameters (i.e. camera location, camera pose), or all 
    projection parameters (i.e. camera location, camera pose, radial 
    distortion, tangential distortion, focal length, principal point).    
    The Trust Region Reflective algorithm is generally a robust method, ideal 
    for solving many variables (default). The Dogleg algorithm is ideal for 
    solving few variables. The Levenberg-Margquardt algorithm is the most 
    efficient method, ideal for solving few variables.
    Pixel differences between a set of image GCPs and projected XYZ GCPs are
    calculated and refined within the optimisation function, performing 
    iterations until an optimum solution is reached. A new set of optimised 
    projection parameters are returned
    
    Parameters
    ----------
    optimise : str 
      Flag denoting which variables will be optimised: YPR (camera pose only), 
      INT (internal camera parameters), EXT (external camera parameters), 
      LOC (all parameters except camera location), or ALL (all projection 
      parameters)            
    projvars : list 
      Projection parameters [camera location, camera pose, radial distortion, 
      tangential distortion, focal length, principal point, reference image]
    GCPuv : arr 
      UV positions for GCPs, as shape (m, 2)
    optmethod : str, optional
      Optimisation method: 'trf' (Trust Region Reflective algorithm), 
      'dogbox' (dogleg algorithm), or 'lm' (Levenberg-Marquardt algorithm) 
      (default='trf')
    show : bool 
      Flag denoting whether plot of residuals should be shown

    Returns
    -------
    projvars1 : list                                          .                                 
      A list containing the optimised projection parameters. If optimisation 
      fails then None is returned   
    """   
    #Get projectiion parameters from projvars
    camloc, campose, radcorr, tancorr, focal, camcen, refimg = projvars

    #Compute GCP residuals with original camera info
    stable = [camloc, campose, radcorr, tancorr, focal, camcen]    
    res0 = computeResidualsUV(None, stable, GCPxyz, GCPuv, refimg, 
                            optimise=None)   
    GCPxyz_proj0,depth,inframe = projectXYZ(camloc, campose, radcorr, tancorr, 
                                           focal, camcen, refimg, GCPxyz)
    
    #Get variables for optimising    
    if optimise=='YPR':
        params = campose
        stable = [camloc, radcorr, tancorr, focal, camcen]
        print('Commencing optimisation of YPR')        
    elif optimise=='INT':
        params = np.concatenate((radcorr.flatten(), tancorr.flatten(), 
                                 np.array(focal), np.array(camcen)))
        stable = [camloc, campose]
        print('Commencing optimisation of internal camera parameters')    
    elif optimise == 'EXT':
        params = np.concatenate((camloc, campose))
        stable = [radcorr, tancorr, focal, camcen]
        print('Commencing optimisation of external camera parameters')
    else:
        optimise='ALL'
        params = np.concatenate((camloc, campose, radcorr.flatten(), 
                                 tancorr.flatten(), np.array(focal), 
                                 np.array(camcen)))               
        stable=None
        print('Commencing optimisation of all projection parameters')
     
    #Optimise, passing through the computeResiduals function for iterating
    out = optimize.least_squares(computeResidualsUV, params, method=optmethod, 
                                 verbose=2, max_nfev=5000, 
                                 args=(stable, GCPxyz, GCPuv, refimg, optimise))  

    #If optimisation was sucessful
    if out.success is True:
        print('Optimisation successful')
        
        #Retrieve optimised parameters
        if optimise=='YPR':
            campose = out.x
        elif optimise == 'INT':
            radcorr = list(out.x[0:3])
            tancorr = list(out.x[3:5])
            focal = list(out.x[5:7])
            camcen = list(out.x[7:9])        
        elif optimise == 'EXT':
            camloc = out.x[0:3]
            campose = out.x[3:6]             
        else:
            camloc = out.x[0:3]
            campose = out.x[3:6]
            radcorr = list(out.x[6:9])
            tancorr = list(out.x[9:11])
            focal = list(out.x[11:13])
            camcen = list(out.x[13:15]) 
        
        #Calculate projected GCPs with new projection parameters
        GCPxyz_proj1, depth, inframe = projectXYZ(camloc, campose, radcorr, 
                                                  tancorr, focal, camcen, 
                                                  refimg, GCPxyz)   
     
        #Calculate new residuals
        res1=[]
        for i in range(len(GCPxyz_proj1)):
            res1.append(np.sqrt((GCPxyz_proj1[i][0]-GCPuv[i][0])*
                               (GCPxyz_proj1[i][0]-GCPuv[i][0])+
                               (GCPxyz_proj1[i][1]-GCPuv[i][1])*
                               (GCPxyz_proj1[i][1]-GCPuv[i][1])))
        print('Original px residuals (average): ' + str(np.nanmean(res0)))        
        print('Optimised px residuals (average): ' + str(np.nanmean(res1)))
        
        #Compile new projection parameter list
        projvars1 = [camloc, campose, radcorr, tancorr, focal, camcen, 
                     refimg]
        
        #If plotting flag is set to True
        if show == True:
            
            #Get reference image
            if isinstance(refimg, str):
                refimg=readImg(refimg)
                ims=refimg.shape
            elif isinstance(refimg, np.ndarray):
                ims=refimg.shape
            else:
                ims=refimg.getImageSize()
                refimg=refimg.getImageArray()

            
            #Plot GCPs using Utilities.plotResiduals function 
            plotResiduals(refimg, ims, GCPuv, GCPxyz_proj0, GCPxyz_proj1)
        
        #Return new projection parameters list        
        return projvars1

    #If optimisation failed, print statement and return none
    else:
        print('Optimisation failed')
        return None     
    

def getRotation(camDirection):
    """Calculates camera rotation matrix calculated from view direction
    
    Parameters
    ----------
    camDirection : arr 
      Camera pose (yaw,pitch,roll)

    Returns
    -------
    value : arr            
      Rotation matrix as array
    """

    C = np.cos(camDirection) 
    S = np.sin(camDirection)
                
    p=[S[2]*S[1]*C[0]-C[2]*S[0] , S[2]*S[1]*S[0] + C[2]*C[0] , S[2]*C[1]]
    q=[C[2]*S[1]*C[0] + S[2]*S[0], C[2]*S[1]*S[0] - S[2]*C[0],C[2]*C[1]]
    r=[C[1]*C[0] , C[1]*S[0] , -S[1]]
        
    value = np.array([p,q,r])
    value[0:2,:]=-value[0:2,:]

    return value


def computeResidualsXYZ(invprojvars, GCPxyz, GCPuv, dem):
    """Function for computing the pixel difference between GCP image points
    and GCP projected XYZ points. This function is used in the optimisation 
    function (optimiseCamera), with parameters for optimising defined in the
    first variable and stable parameters defined in the second. If no 
    optimisable parameters are given and the optimise flag is set to None then 
    residuals are computed for the original parameters (i.e. no optimisation)
    
    Parameters
    ----------
    params : arr 
      Optimisable parameters, given as a 1-D array of shape (m, )
    stable : list 
      Stable parameters that will not be optimised
    GCPxyz : arr
      GCPs in scene space (x,y,z)
    GCPuv : arr 
      GCPs in image space (u,v)
    refimg : str/arr/PyTrx.Images.CamImage 
      Reference image, given as a CamImage object, file path string, or image 
      array
    optimise : str 
      Flag denoting which variables will be optimised: YPR (camera pose only), 
      INT (internal camera parameters), EXT (external camera parameters), LOC 
      (all parameters except camera location), or ALL (all projection 
      parameters)
    
    Returns
    -------
    residual : arr
      Array denoting pixel difference between UV and projected XYZ position of 
      each GCP
    """
    GCPxyz_proj = projectUV(GCPuv, invprojvars)  
        
    #Compute residuals using pythag theorem (i.e. pixel difference between pts)
    residual=[]
    for i in range(len(GCPxyz_proj)):
        residual.append(np.sqrt((GCPxyz_proj[i][0]-GCPxyz[i][0])**2 + 
                                (GCPxyz_proj[i][1]-GCPxyz[i][1])**2))  
    residual = np.array(residual)    

    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title('Average residual difference: ' + 
                                str(np.nanmean(residual)) + ' m')
          
    #Plot DEM and set cmap
    demextent = dem.getExtent()
    demz = dem.getZ()  
    implot = ax1.imshow(demz, origin='lower', extent=demextent)
    implot.set_cmap('gray')
    ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
    
    #Plot UV GCPs
    ax1.scatter(GCPxyz[:,0], GCPxyz[:,1], color='red', marker='+', 
                label='XYZ')
    
    #Plot projected XYZ GCPs
    ax1.scatter(GCPxyz_proj[:,0], GCPxyz_proj[:,1], color='blue', 
                marker='+', label='Projected UV')
    
    #Add legend and show plot
    ax1.legend()
    plt.show() 
        
    #Return all residuals
    return residual
   

def computeResidualsUV(params, stable, GCPxyz, GCPuv, refimg, 
                       optimise='YPR'):
    """Function for computing the pixel difference between GCP image points
    and GCP projected XYZ points. This function is used in the optimisation 
    function (optimiseCamera), with parameters for optimising defined in the
    first variable and stable parameters defined in the second. If no 
    optimisable parameters are given and the optimise flag is set to None then 
    residuals are computed for the original parameters (i.e. no optimisation)
    
    Parameters
    ----------
    params : arr 
      Optimisable parameters, given as a 1-D array of shape (m, )
    stable : list 
      Stable parameters that will not be optimised
    GCPxyz : arr 
      GCPs in scene space (x,y,z)
    GCPuv : arr 
      GCPs in image space (u,v)
    refimg : str/arr/PyTrx.Images.CamImage 
      Reference image, given as a CamImage object, file path string, or image 
      array
    optimise : str 
      Flag denoting which variables will be optimised: YPR (camera pose only), 
      INT (internal camera parameters), EXT (external camera parameters), LOC 
      (all parameters except camera location), or ALL (all projection 
      parameters)

    Returns
    -------
    residual : arr
      Pixel difference between UV and projected XYZ position of each GCP
    """   
    #Assign optimisable and stable parameters depending on optimise flag
    if optimise == 'YPR':
        campose = params      
        camloc, radcorr, tancorr, focal, camcen = stable
            
    elif optimise == 'INT':
        radcorr = params[1:3]
        tancorr = params[3:5]
        focal = params[5:7]
        camcen = params[7:9]
        camloc, campose = stable 
    
    elif optimise == 'EXT':
        camloc = params[0:3]
        campose = params[3:6]
        radcorr, tancorr, focal, camcen = stable 

    elif optimise == 'ALL':
        camloc = params[0:3]
        campose = params[3:6]
        radcorr = params[6:9]
        tancorr = params[9:11]
        focal = params[11:13]
        camcen = params[13:15]
        
    else:       
        camloc, campose, radcorr, tancorr, focal, camcen = stable
    
    #Project XYZ points to UV space       
    GCPxyz_proj,depth,inframe = projectXYZ(camloc, campose, radcorr, tancorr, 
                                           focal, camcen, refimg, GCPxyz)
    
    #Compute residuals using pythag theorem (i.e. pixel difference between pts)
    residual=[]
    for i in range(len(GCPxyz_proj)):
        residual.append(np.sqrt((GCPxyz_proj[i][0]-GCPuv[i][0])**2 + 
                                (GCPxyz_proj[i][1]-GCPuv[i][1])**2))  
    residual = np.array(residual)

    #Return all residuals
    return residual
 
#------------------------------------------------------------------------------

class TestCamEnv(unittest.TestCase):
    def test_getRotation(self):
        actual = getRotation(np.array([0, 0, 0]))
        expected = np.array([-0.,-1.,-0.,-0.,-0.,-1.,1.,0.,-0.], 
                            dtype=np.float64).reshape(3,3)
        self.assertIn(actual, expected)
        
if __name__ == "__main__":   
    unittest.main()       


#------------------------------------------------------------------------------   
