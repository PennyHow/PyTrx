
#PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie
#
#PyTrx is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

"""
Th FileHandler module contains all the functions called by a PyTrx object to 
load and export data.
"""

#Import packages
from PIL import Image, ImageDraw
import numpy as np
import operator
import matplotlib.pyplot as plt
import scipy.io as sio
from osgeo import ogr,osr
import os
from functools import reduce

#------------------------------------------------------------------------------   

def readMask(img, writeMask=None):
    """Function to create a mask for point seeding using PIL to rasterize 
    polygon. The mask is manually defined by the user using the pyplot ginput 
    function. This subsequently returns the manually defined area as a .jpg 
    mask. The writeMask file path is used to either open the existing mask at 
    that path or to write the generated mask to this path.
    
    :param img: Image to define mask in
    :type img: arr
    :param writeMask: File destination that mask output is written to, default to None
    :type writeMask: str, optional
    :returns: Array defining the image mask
    :rtype: arr      
    """
    #Check if a mask already exists, if not enter digitising
    if writeMask!=None:
        try:
            myMask = Image.open(writeMask)
            myMask = np.array(myMask)
            print('\nImage mask loaded')
            return myMask
        except:
            print('\nMask file not found. Proceeding to manually digitise...')

    #Plot mask manually on the selected image
    fig=plt.gcf()
    fig.canvas.set_window_title('Click to create mask. Press enter to record' 
                                ' points.')
    imgplot = plt.imshow(img, origin='upper')
    imgplot.set_cmap('gray')
    x1 = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, 
                    mouse_stop=2)
    print('\n' + str(len(x1)) + ' points seeded')
    plt.show()
    plt.close()
    
    #Close shape
    x1.append(x1[0])
    
    #Generate polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in x1:
        ring.AddPoint(p[0],p[1])
    p=x1[0]
    ring.AddPoint(p[0],p[1])    
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
     
    #Rasterize polygon using PIL
    height = img.shape[0]
    width = img.shape[1]
    img1 = Image.new('L', (width,height), 0)
    draw=ImageDraw.Draw(img1)
    draw.polygon(x1, outline=1, fill=1)
    myMask=np.array(img1)
    
    #Write to .jpg file    
    if writeMask!=None:
        print('\nMask plotted: ' + str(writeMask))
        try:
            img1.save(writeMask, quality=75)
        except:
            print('\nFailed to write file: ' + str(writeMask))
        
    return myMask  


def readCalib(fileName, paramList):
    """Function to find camera calibrations from a file given a list or 
    Matlab file containing the required parameters. Returns the parameters as a
    dictionary object. Compatible file structures: 1) .txt file 
    ("RadialDistortion [k1,k2,k3...k8], TangentialDistortion [p1,p2],
    IntrinsicMatrix [fx 0. 0.][s fy 0.][cx cy 1] End"); 2) .mat file
    (Camera calibration file output from the Matlab Camera Calibration App 
    (available in the Computer Vision Systems toolbox).
    
    :param fileName: File directory for calibration file
    :type fileName: str    
    :param paramList: List of strings denoting keywords to look for in calibration file
    :type paramList: list       
    :returns: Calibration parameters denoted by keywords
    :rtype: list    
    """    
    #Load as text file if txt format
    if fileName[-3:] == 'txt':            
        #Open file
        try:
            myFile=open(fileName,'r')
        except:
            print('\nProblem opening calibration text file: ' + str(fileName))
            print('No calibration parameters successfully read')
            return None
        
        #Read lines in file
        lines = []
        for line in myFile.readlines():
            lines.append(line[0:-1]) 
                
        #Set up a dictionary and input the calibration parameters 
        calib = {}
        for p in paramList:
            dataLines= lineSearch(lines, p)
            data = returnData(lines, dataLines)
            param = {}
            param[p] = data
            calib.update(param)        
        return calib
    
    #Load as matlab file if mat format
    elif fileName[-3:] == 'mat':            
        #Get the Matlab file as a dictionary
        mat = sio.loadmat(fileName)
           
        # Get desired keys from the dictionary
        calib = dict([(i, mat[i]) for i in paramList if i in mat])
        return calib
   
    #Incompatible file types     
    else:
        print ('\nYou have specified an incorrect calibration file type'
               '\nAcceptable file types are .txt and .mat.')


def lineSearch(lineList, search):
    """Function to supplement the readCalib function. Given an input parameter 
    to search within the file, this will return the line numbers of the data.
    
    :param lineList: List of strings within a file line
    :type lineList: list        
    :param search: Target keyword to search for
    :type search: str
    :returns:  Line numbers with keyword match
    :rtype: list      
    """
    #Find line matching the parameter
    for i, line in enumerate(lineList):
        if search in line: 
            if len(search) == len(lineList[i]):
                match = i
    
    #Find lines containing the associated data        
    dataLines=[]
    atEnd=False
    match += 1
    while not(atEnd):
        if lineList[match][0].isupper() or lineList[match][0] == '_':
            atEnd=True
        else: 
            dataLines.append(match)
            match += 1
            
    return dataLines


def returnData(lines, data):
    """Function to supplement the importCalibration function. Given the line 
    numbers of the parameter data (the ouput of the lineSearch function), this 
    will return the data.
    
    :param lines: Given line numbers to extract data from
    :type lines: list
    :param data: Raw line data
    :type data: list        
    :returns: Extracted data
    :rtype: arr
    """ 
    #Create empty list for output data
    D=[]
    
    for x in data:
                
        # Find the fields on each line
        fields = lines[x].replace('[','').replace(']','')
        fields=fields.split()
        
        d = []
        
        #Float all values
        for f in fields:
            d.append(float(f))
                            
        # Add the fields to the data list
        D.append(d)

    if len(D) > 1:
        D = np.array(D)
    
    return D


def readMatrixDistortion(path):
    """Function to support the calibrate function. Returns the 
    intrinsic matrix and distortion parameters required for calibration from a
    given file.
    
    :param path: Directory of calibration file
    :type path: str
    :returns: Three camera matrix parameters: 1) the intrinsic matrix as a 3x3 array, including focal length, principal point, and skew (arr); 2) Tangential distortion values (p1, p2) (arr); and 3) Radial distortion values (k1, k2... k6)
    :rtype: arr
    """   
    #Find the calibration parameters from the list
    ls = ['IntrinsicMatrix', 'RadialDistortion', 'TangentialDistortion']
    calibDict = readCalib(path, ls)
    
    #Stop if there is an issue in reading calibration file    
    if calibDict==None:
        return None   

    #Set calibration parameters in the correct format    
    intrMat = np.array(calibDict["IntrinsicMatrix"]) #fx,fy,s,cx,cy
    radDis = np.array(calibDict["RadialDistortion"]) # k1-k6
    tanDis = np.array(calibDict["TangentialDistortion"]) # p1,p2
   
    return intrMat, tanDis, radDis


def checkMatrix(matrix):
    """Function to support the calibrate function. Checks and converts the 
    intrinsic matrix to the correct format for calibration with OpenCV.
    
    :param matrix: Inputted matrix for checking
    :type matrix: arr
    :returns: Validated matrix
    :rtype: arr          
    """  
    # Transpose if zeros in matrix are not in correct places
    if matrix[2,0]!=0 and matrix[2,1]!=0 and matrix[0,2]==0 and matrix[1,2]==0:
        mat = matrix.transpose()
    else:
        mat = matrix
     
    # Set 0's and 1's in the correct locations
    it=np.array([0,1,1,0,2,0,2,1])     
    it.shape=(4,2)
    for i in range(4):
        x = it[i,0]
        y = it[i,1]
        mat[x,y]=0        
    mat[2,2]=1 

    return mat


def readImg(path, band='L', equal=True):
    """Function to prepare an image by opening, equalising, converting to 
    either grayscale or a specified band, then returning a copy.

    :param path: Image file path directory
    :type path: str
    :param band: Desired band output - 'R': red band; 'B': blue band; 'G': green band; 'L': grayscale (default).
    :type band: str 
    :param equal: Flag to denote if histogram equalisation should be applied, defaults to True
    :type equal: bool    
    :returns: Image array
    :rtype: arr
    """  
    # Open image file
    band=band.upper()
    im=Image.open(path)
    
    #Equalise histogram
    if equal is True:
        h = im.convert("L").histogram()
        lut = []
    
        for b in range(0, len(h), 256):
    
            #Step size
            step = reduce(operator.add, h[b:b+256]) / 255
    
            #Create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i+b]
        
        #Convert to grayscale
        gray = im.point(lut*im.layers)    
    else:
        gray=im
    
    #Split bands if R, B or G is specified in inputs    
    if band=='R':
        gray,g,b=gray.split()
    elif band=='G':
        r,gray,b=gray.split() 
    elif band=='B':
        r,g,gray=gray.split() 
    else:
        gray = gray.convert('L')
    
    #Copy and return image    
    bw = np.array(gray).copy()
    
    return bw


def readGCPs(fileName):
    """Function to read ground control points from a .txt file. The data in the
    file is referenced to under a header line. Data is appended by skipping the
    header line and finding the world and image coordinates from each line.
    
    :param fileName: File path directory for GCP file
    :type fileName: str
    :returns: Two arrays containing the GCPs in xyz coordinates, and the GCPs in image coordinates
    :rtype: array    
    """   
    # Open the file and read the first line (i.e. the header line)
    myFile=open(fileName,'r')  
    myFile.readline()

    #Iterate through lines and append to lists
    gcps= []
    count=2
    for line in myFile.readlines():  
        items=line.split('\t')
        gcp = []
        
        #Append values if they are valid
        for i in items:
            try:
                gcp.append(float(i)) 
            except ValueError:
                pass
        
        #Append points if there are 5 corresponding values
        if len(gcp)==5:    
            gcps.append(gcp)
        else:
            print ('\nGCP file line ' + str(count) + ' not passed forward')
        
        #Update counter        
        count=count+1
                  
    #Split the list into world and image coordinates
    gcparray = np.array(gcps)
    world, image = np.hsplit(gcparray,np.array([3]))
           
    return world, image

    
def writeCalibFile(intrMat, tanDis, radDis, fname):
    """Write camera calibration data to .txt file, including camera matrix, and 
    radial and tangential distortion parameters. 

    :param intrMat: Intrinsic camera matrix
    :type intrMat: arr 
    :param tanDis: Tangential distortion parameters
    :type tanDis: arr 
    :param radDis: Radial distortion parameters
    :type radDis: arr           
    :param fname: Directory to write file to
    :type fname: str            
    """         
    #Write camera calibration variables to text file
    f=open(fname,'w')                            
    f.write('RadialDistortion' + '\n' + str(radDis) + '\n' +
            'TangentialDistortion' + '\n' + str(tanDis) + '\n' +
            'IntrinsicMatrix' + '\n' + str(intrMat) + '\n' +
            'End')
            

def writeVeloFile(xyzvel, uvvel, homog, imn, fname):
    """Function to write all velocity data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following
    information: Image pair 1 name, Image pair 2 name, Average xyz velocity,
    Number of features tracked, Average pixel velocity, Homography residual 
    mean error (RMS), and Signal-to-noise ratio.
         
    :param xyzvel: XYZ velocities
    :type xyzvel: list
    :param uvvel: Pixel velocities
    :type uvvel: list
    :param homog: Homography [mtx, im0pts, im1pts, ptserr, homogerr]
    :type homog: list 
    :param imn: List of image names
    :param imn: list
    :param fname: Filename for output file. File destination can also specified
    :type fname: str
    """        
    #Initialise file writing
    f=open(fname,'w')

    #Assign first image in pair
    fn1=imn[0]
    
    #Define column headers
    header=('Image 0, Image 1, Average xyz velocity, Features tracked, '
            'Average px velocity, Homography RMS Error, SNR')    
    f.write(header + '\n')

    #Iterate through timeLapse object
    for i in range(len(xyzvel)):
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        fn1=imn[i+1]
               
        #Get velocity data                     
        xyz = xyzvel[i]
        uv = uvvel[i]
        
        #Calculate average xyz and px velocities
        xyzvelav = sum(xyz)/len(xyz)
        uvvelav = sum(uv)/len(uv)
        
        #Write average xyzvel, number of features tracked, and uvvel
        f.write(str(fn0) + ','+ str(fn1) + ',' + str(xyzvelav) + ',' 
                + str(len(xyz)) + ',' + str(uvvelav) + ',')
                     
        #Get homography information if desired
        if homog is not None:
            hpt0 = homog[i][1][0]
            hpt1 = homog[i][1][1]
            hpt1corr = homog[i][1][2]
            herr = homog[i][3]
    
            #Get xyz homography errors                      
            xd=herr[1][0]
            yd=herr[1][1]
            
            #Get uv point positions                
            psx=hpt0[:,0,0]
            psy=hpt0[:,0,1]
            
            if hpt1corr is not None:
                pfx=hpt1corr[:,0,0]
                pfy=hpt1corr[:,0,1] 
            else:                   
                pfx=hpt1[:,0,0]
                pfy=hpt1[:,0,1]    
                
            #Determine uv point position difference
            pdx=pfx-psx
            pdy=pfy-psy
            
            #Calculate homography and homography error            
            homogdist=np.sqrt(pdx*pdx+pdy*pdy)
            errdist=np.sqrt(xd*xd+yd*yd)
            
            #Calculate mean homography and mean error 
            meanerrdist=np.mean(errdist)
            
            #Write pixel velocity, mean homog error, and SNR (pxvel/err)
            f.write(str(meanerrdist) + ','  + str(meanerrdist/uvvelav))

        else:
            #Write pixel velocity and NaN homography information
            f.write('NaN, NaN')
                         
        #Break line in output file
        f.write('\n')
            
    print('\nVelocity file written:' + str(fname))        
        
   
def writeHomogFile(homog, imn, fname):
    """Function to write all homography data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following 
    information: Image pair 1 name, Image pair 2 name, Homography matrix (i.e. 
    all values in the 3x3 matrix), Number of features tracked, X mean 
    displacement, Y mean displacement, X standard deviation, Y standard 
    deviation, Mean error magnitude, Mean homographic displacement, and 
    Signal-to-noise ratio.
    
    :param homog: Homography [mtx, im0pts, im1pts, ptserr, homogerr]
    :type homog: list
    :param imn: List of image names
    :type imn: list 
    :param fname: Directory for file to be written to
    :type fname: str 
    """
    #Initialise file writing
    f=open(fname,'w')
    
    #Write active directory to file
    fn1=imn[0]
    
    #Define column headers
    header=('Image 0,Image 1,"Homography Matrix[0,0]","[0,1]","[0,2]",'
            '"[1,0]","[1,1]","[1,2]","[2,0]","[2,1]","[2,2]",Features Tracked,'
            'xmean,ymean,xsd,ysd,Mean Error Magnitude,'
            'Mean Homographic displacement,Homography SNR')  
    
    f.write(header+'\n')

    #Iterate through timeLapse object
    for i in range(len(homog)):
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        fn1=imn[i+1]

        #Write image file names to file        
        f.write(fn0 + ',' + fn1 + ',')
        
        #Get homography info
        hmatrix = homog[i][0]               #Homography matrix
        hpt0 = homog[i][1][0]               #Seeded pts in im0
        hpt1 = homog[i][1][1]               #Tracked pts in im1
        hpt1corr = homog[i][1][2]           #Corrected pts im1
        herr = homog[i][3]                  #Homography error
        
        #Define output homography matrix
        if hmatrix is not None:
            hmatrix.shape=(9)
            for val in hmatrix:
                f.write(str(val) + ',')
                
        #Get xyz homography errors
        xd = herr[1][0]
        yd = herr[1][1] 
        
        #Get uv point positions
        psx = hpt0[:,0,0]
        psy = hpt0[:,0,1]
        
        if hpt1corr is None:
            pfx = hpt1[:,0,0]
            pfy = hpt1[:,0,1]
        else:
            pfx = hpt1corr[:,0,0]
            pfy = hpt1corr[:,0,1]
        
        #Determine uv point position difference
        pdx=pfx-psx
        pdy=pfy-psy
        
        #Calculate signal-to-noise ratio            
        errdist=np.sqrt(xd*xd+yd*yd)
        homogdist=np.sqrt(pdx*pdx+pdy*pdy)
        sn=errdist/homogdist
        
        #Determine number of points tracked in homography calculations
        f.write(str(len(hpt0)) + ',')
        
        #Define output homography matrix errors
        for val in herr[0]:
            f.write(str(val) + ',')
        
        #Compile all data for output file
        f.write(str(np.mean(errdist)) + ',' + str(np.mean(homogdist)) + ',' + 
                str(np.mean(sn)) + '\n')
        
    print('\nHomography file written' + str(fname))


def writeAreaFile(pxareas, xyzareas, imn, destination):
    """Write UV and XYZ areas to csv file.

    :param pxarea: Pixel extents
    :type pxarea: list
    :param xyzarea: XYZ areas
    :type xyzarea: list
    :param imn: Image names
    :type imn: list
    :param destination: File directory where csv file will be written to
    :type destination: str
    """ 
    #Create file location
    f = open(destination, 'w')
    
    #Write data headers
    f.write('Image,UV area,XYZ area \n')    
     
    #Write image name, pixel area, and xyz area data     
    for i in range(len(imn)):
        if xyzareas is None:
            f.write(str(imn[i]) + ',' + str(sum(pxareas[i])) + '\n')
        else:
            f.write(str(imn[i]) + ',' + str(sum(pxareas[i])) + ',' 
                    + str(sum(xyzareas[i])) + '\n')
            

def writeAreaCoords(pxpts, xyzpts, imn, pxdestination, xyzdestination):
    """Write UV and XYZ area coordinates to text files. These file types are 
    compatible with the importing tools (importAreaPX, importAreaXYZ).

    :param xyzarea: XYZ areas
    :type xyzarea: list
    :param xyzpts: XYZ coordinates
    :type xyzpts: list
    :param imn: Image names
    :type imn: list
    :param pxdestination: File directory where UV coordinates will be written to
    :type pxdestination: str
    :param xyzdestination: File directory where XYZ coordinates will be written to
    :type xyzdestination: str
    """    
    #Pixel cooridnates of all pixel extents
    if pxpts is not None:       
        f = open(pxdestination, 'w')
        polycount=1        
        for i in range(len(pxpts)):
            f.write('Img ' + str(imn[i]) + '\t')                
            for pol in pxpts[i]:
                f.write('Poly' + str(polycount) + '\t')                    
                for pts in pol:
                    if len(pts)==1:
                        f.write(str(pts[0][0]) + '\t' + str(pts[0][1]) + '\t')
                    elif len(pts)==2:
                        f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')
                polycount = polycount+1
            f.write('\n\n')
            polycount=1

    #XYZ coordinates of polygons
    if xyzpts is not None:                               
        polycount=1  
        f = open(xyzdestination, 'w')
        for i in range(len(xyzpts)):                
            f.write('Img ' + str(imn[i]) + '\t')
            for pol in xyzpts[i]:
                f.write('Poly' + str(polycount) + '\t')
                for pts in pol:
                    f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t' + 
                            str(pts[2]) + '\t')
                polycount=polycount+1
            f.write('\n\n')
            polycount=1
                                        

def writeLineFile(pxline, xyzline, imn, destination):
    """Write UV and XYZ line lengths to csv file.

    :param pxline: Pixel line lengths
    :type pxline: list   
    :param xyzline: XYZ line lengths
    :type xyzline: list
    :param imn: Image names
    :type imn: list
    :param destination: File directory where output will be written to 
    :type destination: str
    """
    #Create file
    f = open(destination, 'w')
    
    #Write file headers
    f.write('Image, UV length, XYZ length \n')            
        
    for i in range(len(pxline)):
        if xyzline is None:
            f.write(str(imn[i]) + ',' + str(pxline[i]) + '\n')
        else:
            f.write(str(imn[i]) + ',' + str(pxline[i]) + ',' + str(xyzline[i]) 
                    + '\n')
    

def writeLineCoords(uvpts, xyzpts, imn, pxdestination, xyzdestination):
    """Write UV and XYZ line coordinates to text file. These file types are 
    compatible with the importing tools (importLinePX, importLineXYZ)
    
    :param uvpts: Pixel coordinates
    :type uvpts: list
    :param xyzpts: XYZ coordinates
    :type xyzpts: list
    :param imn: Image names
    :type imn: list
    :param pxdestination: File directory where UV coordinates will be written to
    :type pxdestination: str
    :param xyzdestination: File directory where XYZ coordinates will be written to
    :type xyzdestination: str    
    """  
    #Pixel line coordinates file generation             
    if uvpts is not None:            
        f = open(pxdestination, 'w')
        for i in range(len(uvpts)):
            f.write('Img ' + str(imn[i]) + '\t')                                   
            for pts in uvpts[i]:
                f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')    
            f.write('\n\n')
            
    #Real line coordinates file generation
    if xyzpts is not None:                   
        f = open(xyzdestination, 'w')
        for i in range(len(xyzpts)):                
            f.write('Img ' + str(imn[i]) + '\t')
            for pts in xyzpts[i]:
                f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t' + str(pts[2]) 
                + '\t')
            f.write('\n\n')
                
            
def writeVeloSHP(xyzvel, xyzerr, xyz0, imn, fileDirectory, projection=None): 
    """Write OGR real velocity points (from ALL images) to file in a .shp
    file type that is compatible with ESRI mapping software.
    
    :param xyzvel: XYZ velocities
    :type xyzvel: list
    :param xyz0: XYZ pt0
    :type xyz0: list
    :param imn: Image name
    :type imn: list
    :param fileDirectory: Destination that shapefiles will be written to 
    :type fileDirectory: str
    :param projection: Coordinate projection that the shapefile will exist in. This can either be an ESPG number (expressed as an integer) or a well-known geographical coordinate system (expressed as a string). Well-known geographical coordinate systems are: 'WGS84', 'WGS72', NAD83' or 'EPSG:n'
    :type projection: int/str
    """ 
    #Make directory if it does not exist
    if not os.path.exists(fileDirectory):
        os.makedirs(fileDirectory)
        
    #Get driver and create shapeData in shp file directory              
    typ = 'ESRI Shapefile'
    driver = ogr.GetDriverByName(typ)
    if driver is None:
        raise IOError('%s Driver not available:\n' % typ)
        pass
        
    for i in range(len(xyzvel)): 
        
        #Get velocity, pt and image name for time step
        vel = xyzvel[i]
        if xyzerr != None:
            err = xyzerr[i]
        pt0 = xyz0[i]            
        im = imn[i] 
        
        #Create file space            
        shp = fileDirectory + str(im) + '_vel.shp'
        if os.path.exists(shp):
            driver.DeleteDataSource(shp)
        ds = driver.CreateDataSource(shp)
        if ds is None:
            print('Could not create file ' + str(shp))
        
        #Set projection
        if type(projection) is int:
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbPoint)
        elif type(projection) is str:
            proj = osr.SpatialReference()
            proj.SetWellKnownGeogCS(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbPoint)
        else:
            layer = ds.CreateLayer(' ', None, ogr.wkbPoint)
   
        #Add attributes to layer
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))     #ID    
        layer.CreateField(ogr.FieldDefn('velocity', ogr.OFTReal))  #Velo
        layer.CreateField(ogr.FieldDefn('error', ogr.OFTReal))  #Velo        
        layer.CreateField(ogr.FieldDefn('snr', ogr.OFTReal))  #Velo 
        
        #Get xy coordinates
        x0 = pt0[:,0]
        y0 = pt0[:,1]
        
        #Create point features with data attributes in layer
        if xyzerr != None:
            for v,e,x,y in zip(vel, err, x0, y0):
                count=1
            
                #Create feature    
                feature = ogr.Feature(layer.GetLayerDefn())
            
                #Create feature attributes    
                feature.SetField('id', count)
                feature.SetField('velocity', v)
                feature.SetField('error', e)
                feature.SetField('snr', e/v)
                
                #Create feature location
                wkt = "POINT(%f %f)" %  (float(x) , float(y))
                point = ogr.CreateGeometryFromWkt(wkt)
                feature.SetGeometry(point)
                layer.CreateFeature(feature)
            
                #Free up data space
                feature.Destroy()                       
                count=count+1
        else:
            for v,x,y in zip(vel, x0, y0):
                count=1
            
                #Create feature    
                feature = ogr.Feature(layer.GetLayerDefn())
            
                #Create feature attributes    
                feature.SetField('id', count)
                feature.SetField('velocity', v)
                
                #Create feature location
                wkt = "POINT(%f %f)" %  (float(x) , float(y))
                point = ogr.CreateGeometryFromWkt(wkt)
                feature.SetGeometry(point)
                layer.CreateFeature(feature)
            
                #Free up data space
                feature.Destroy()                       
                count=count+1
        #Free up data space                          
        ds.Destroy()
        

def writeAreaSHP(xyzpts, imn, fileDirectory, projection=None):
    """Write OGR real polygon areas (from ALL images) to file in a .shp
    file type that is compatible with ESRI mapping software.
    
    :param xyzpts: XYZ coordinates for polygons
    :type xyzpts: list
    :param imn: Image name
    :type imn: list
    :param fileDirectory: Destination that shapefiles will be written to
    :type fileDirectory: str
    :param projection: Coordinate projection that the shapefile will exist in. This can either be an ESPG number (expressed as an integer) or a well-known geographical coordinate system (expressed as a string). Well-known geographical coordinate systems are: 'WGS84', 'WGS72', NAD83' or 'EPSG:n'
    :type projection: int/str
    """
    #Make directory if it does not exist
    if not os.path.exists(fileDirectory):
        os.makedirs(fileDirectory)
        
    #Get driver and create shapeData in shp file directory        
    typ = 'ESRI Shapefile'        
    driver = ogr.GetDriverByName(typ)
    if driver is None:
        raise IOError('%s Driver not available:\n' % typ)
        pass
        
    for i in range(len(xyzpts)): 
        
        #Get polygon coordinates and image name for each time step
        polys = xyzpts[i]          
        im = imn[i] 
                      
        #Create datasource in shapefile
        shp = fileDirectory + str(im) + '_area.shp'
        
        if os.path.exists(shp):
            driver.DeleteDataSource(shp)
        ds = driver.CreateDataSource(shp)
        if ds is None:
            print('Could not create file ' + str(shp))
    
        if type(projection) is int:
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbPolygon)
        elif type(projection) is str:
            proj = osr.SpatialReference()
            proj.SetWellKnownGeogCS(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbPolygon)
        else:
            layer = ds.CreateLayer(' ', None, ogr.wkbPolygon)
    
        #Add attributes
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('area', ogr.OFTReal))
        
        #Get ogr polygons from xyz areal data at given image 
        ogrpolys = []        
        for shape in polys:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for pt in shape:
                if np.isnan(pt[0]) == False:                   
                    ring.AddPoint(pt[0],pt[1],pt[2])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            ogrpolys.append(poly)
            
        polycount=1

        #Create feature            
        for p in ogrpolys:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(p)
            feature.SetField('id', polycount)
            feature.SetField('area', p.Area())
            layer.CreateFeature(feature)
            feature.Destroy()                
            polycount=polycount+1
        
        ds.Destroy()
        
        
def writeLineSHP(xyzpts, imn, fileDirectory, projection=None):
    """Write OGR real line features (from ALL images) to file in a .shp
    file type that is compatible with ESRI mapping software.
    
    :param xyzpts: XYZ coordinates for polygons
    :type xyzpts: list
    :param imn: Image name
    :type imn: list
    :param fileDirectory: Destination that shapefiles will be written to    
    :type fileDirectory: str
    :param projection: Coordinate projection that the shapefile will exist in. This can either be an ESPG number (expressed as an integer) or a well-known geographical coordinate system (expressed as a string). Well-known geographical coordinate systems are: 'WGS84', 'WGS72', NAD83' or 'EPSG:n'
    :type projection: int/str
    """
    #Make directory if it does not exist
    if not os.path.exists(fileDirectory):
        os.makedirs(fileDirectory)
        
    #Get driver and create shapeData in shp file directory        
    typ = 'ESRI Shapefile'        
    driver = ogr.GetDriverByName(typ)
    if driver is None:
        raise IOError('%s Driver not available:\n' % typ)
        pass
       
    for i in range(len(xyzpts)):
        
        #Get polygon coordinates and image name for each time step
        rline = xyzpts[i]  
        im = imn[i]

        #Create datasource in shapefile
        shp = fileDirectory + str(im) + '_line.shp'
        
        if os.path.exists(shp):
            driver.DeleteDataSource(shp)
        ds = driver.CreateDataSource(shp)
        if ds is None:
            print('Could not create file ' + str(shp))
    
        if type(projection) is int:
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbLineString)
        elif type(projection) is str:
            proj = osr.SpatialReference()
            proj.SetWellKnownGeogCS(projection)
            layer = ds.CreateLayer(' ', proj, ogr.wkbLineString)
        else:
            layer = ds.CreateLayer(' ', None, ogr.wkbLineString)
        
        #Add attributes
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('length', ogr.OFTReal))

        lcount=1
        
        line = ogr.Geometry(ogr.wkbLineString)   
        for p in rline:
            line.AddPoint(p[0],p[1])
        
        #Create feature            
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(line)
        feature.SetField('id', lcount)
        feature.SetField('length', line.Length())
        layer.CreateFeature(feature)
        feature.Destroy() 
        lcount=lcount+1
        
        ds.Destroy()


def importAreaData(xyzfile, pxfile):
    """Import xyz and px data from text files.
    
    :param xyzfile: File directory to xyz coordinates
    :type xyzfile: str
    :param pxfile: File directory to uv coordinates
    :type pxfile: str
    :returns: Coordinates and areas of detected areas
    :rtype: list
    """
    #Get real-world coordinates and areas       
    xyz = importAreaFile(xyzfile,3)

    #Get pixel coordinates and areas    
    uv = importAreaFile(pxfile,2)

    #Compile data together
    areas=[]
    for i,j in zip(xyz, uv):
        areas.append([i,j])
        
    #Return all area data   
    return areas


def importLineData(xyzfile, pxfile):
    """Import xyz and px data from text files.

    :param xyzfile: File directory to xyz coordinates
    :type xyzfile: str
    :param pxfile: File directory to uv coordinates
    :type pxfile: str
    :returns: Coordinates and lengths of detected lines
    :rtype: list
    """
    #Get real-world coordinates and distances
    xyz = importLineFile(xyzfile, 3)
    
    #Get pixel coordinates and distances
    uv = importLineFile(pxfile, 2)
    
    #Compile data together
    lines=[]
    for i,j in zip(xyz, uv):
        lines.append([i,j])
        
    #Return all line data
    return lines
   

def importAreaFile(fname, dimension):
    """Import pixel polygon data from text file and compute pixel extents.
      
    :param fname: Path to the text file containing the UV coordinate data
    :type fname: str
    :param dimension: Integer denoting the number of dimensions in coordinates
    :type dimension: int
    :returns: Two lists, containing the UV coordinates for polygons and the pixel areas for polygons
    :rtype: list                
    """
    #Read file and detect number of images based on number of lines
    f=open(fname,'r')      
    alllines=[]
    for line in f.readlines():
        if len(line) >= 6:
            alllines.append(line)  #Read lines in file             
    print('\nDetected coordinates from ' + str(len(alllines)) + ' images')
    f.close() 
    
    #Extract strings from lines         
    areas=[] 
    for line in alllines:        
        coords=[]
        raw=[]
                      
        #Extract coordinate values from strings
        vals = line.split('\t')
        for v in vals:
            try:
                a=float(v)
                raw.append(a)
            except ValueError:
                pass
            
        if dimension==2:
            struc = len(raw)/2
            coords = np.array(raw).reshape(struc, 2)
        elif dimension==3:
            struc = len(raw)/3
            coords = np.array(raw).reshape(struc, 3)
        
        #Create geometries from polgon values using ogr               
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in coords:
            if dimension==2:                   
                ring.AddPoint(pt[0],pt[1])
            elif dimension==3:
                ring.AddPoint(pt[0],pt[1],pt[2])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring) 
        
        #Append coordinates and polygon area
        areas.append([[poly.GetArea()],[coords]])
    
    return areas
     
     
def importLineFile(fname, dimension):
    """Import XYZ line data from text file and compute line lengths.
        
    :param fname: Path to the text file containing the XYZ coordinate data
    :type fname: str
    :param dimension: Number of dimensions in point coordinates i.e. 2 or 3
    :type dimension: int    
    :returns: List containing line coordinates and lengths
    :rtype: list 
    """
    #Read file and detect number of images based on number of lines
    f=open(fname,'r')      
    alllines=[]
    for line in f.readlines():
        if len(line) >= 6:
            alllines.append(line)  #Read lines in file             
    print('\nDetected coordinates from ' + str(len(alllines)) + ' images')
    f.close() 
    
    #Extract strings from lines         
    lines=[] 
    for line in alllines:        
        coords=[]
        raw=[]
                      
        #Extract coordinate values from strings
        vals = line.split('\t')
        for v in vals:
            try:
                a=float(v)
                raw.append(a)
            except ValueError:
                pass
            
        #Restructure coordinates based into 2D or 3D points
        if dimension==2:
            struc = len(raw)/2            
            coords = np.array(raw).reshape(struc, 2)
        elif dimension==3:
            struc = len(raw)/3            
            coords = np.array(raw).reshape(struc, 3)
     
        #Create OGR line object       
        ogrline = ogr.Geometry(ogr.wkbLineString)   
        for p in coords:
            if dimension==2:
                ogrline.AddPoint(p[0],p[1])
            elif dimension==3:
                ogrline.AddPoint(p[0],p[1],p[2])
        lines.append([ogrline.Length(),coords])
    
    return lines
   
    
#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   
