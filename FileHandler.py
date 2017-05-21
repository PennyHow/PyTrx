# -*- coding: utf-8 -*-
'''
This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This module, FileHandler, contains all the functions called by an object to 
load and export data.

Functions available in FileHandler:
readMask:               Function to create a mask for point seeding using PIL 
                        to rasterize polygon. The mask is manually defined by 
                        the user using the pyplot ginput function. This 
                        subsequently returns the manually defined area as a 
                        .jpg mask. The writeMask file path is used to either 
                        open the existing mask at that path or to write the 
                        generated mask to this path.
readCalib:              Function to find camera calibrations from a file given 
                        a list or Matlab file containing the required 
                        parameters. Returns the parameters as a dictionary 
                        object.
lineSearch:             Function to supplement the readCalib function. Given an 
                        input parameter to search within the file, this will 
                        return the line numbers of the data.
returnData:             Function to supplement the importCalibration function. 
                        Given the line numbers of the parameter data (the ouput 
                        of the lineSearch function), this will return the data.
readMatrixDistortion:   Function to support the calibrate function. Returns the 
                        intrinsic matrix and distortion parameters required for 
                        calibration from a given file.
checkMatrix:            Function to support the calibrate function. Checks and 
                        converts the intrinsic matrix to the correct format for 
                        calibration with OpenCV.
readImage:              Function to prepare an image by opening, equalising, #
                        converting to either grayscale or a specified band, 
                        then returning a copy.
readGCPs:               Function to read ground control points from a .txt 
                        file. The data in the file is referenced to under a 
                        header line. Data is appended by skipping the header 
                        line and finding the world and image coordinates from 
                        each line.
readDEM:                Function to read a DEM from an ASCII file by parsing 
                        the header and data lines to return the data as a NumPy 
                        array, the origin and the cell size. The xyz DEM data 
                        is compiled together in the returned item.
readDEMxyz:             Similar function to above but returns xyz DEM data as 
                        separate X, Y and Z numpy arrays.
readDEMmat:             Function to read xyz DEM data from a .mat file and 
                        return the xyz data as separate arrays.
writeTIFF:              Write data to .tif file. A reference to the file's 
                        spatial coordinate system is assigned using GDAL 
                        (compatible with ArcGIS and QGIS). 
writeHomography:        Function to write all homography data from a given 
                        timeLapse sequence to .csv file.
createThumbs:           Function to create thumbnail images from a given image 
                        file directory using the PIL (Python Imaging Library) 
                        toolbox.


@authors: Lynne Addison 
          Nick Hulton (nick.hulton@ed.ac.uk) 
          Penny How (p.how@ed.ac.uk)
'''

#Import packages
from PIL import Image, ImageDraw
import numpy as np
import operator
import matplotlib.pyplot as plt
import scipy.io as sio
from osgeo import ogr,osr
import gdal
import glob
import math

#Import PyTrx modules
from Utilities import filterSparse

#------------------------------------------------------------------------------

def readMask(img, writeMask=None):
    '''Function to create a mask for point seeding using PIL to rasterize 
    polygon. The mask is manually defined by the user using the pyplot ginput 
    function. This subsequently returns the manually defined area as a .jpg 
    mask. 
    The writeMask file path is used to either open the existing mask at that 
    path or to write the generated mask to this path.'''
    #Check if a mask already exists, if not enter digitising
    if writeMask!=None:
        try:
            myMask = Image.open(writeMask)
            myMask = np.array(myMask)
            print ('Mask loaded. It is recommended that you check this against' 
                   'the start and end of the sequence using the self.checkMask()'
                   'function of the TimeLapse object')
            return myMask
        except:
            print 'Image file not found. Proceeding to manually digitise...'

    #Plot mask manually on the selected image
    imgplot = plt.imshow(img, origin='upper')
    imgplot.set_cmap('gray')
    x1 = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    print 'you clicked:', x1
    plt.show()
    
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
    writeMask = writeMask.split('.')[0] + '.jpg'
    print 'Mask plotted: ' + writeMask
    if writeMask!=None:
        try:
            img1.save(writeMask, 'jpeg', quality=75)
        except:
            print 'Failed to write file: ' + writeMask
        
    return myMask  


def readCalib(fileName, paramList):
    '''Function to find camera calibrations from a file given a list or 
    Matlab file containing the required parameters. Returns the parameters as a
    dictionary object.
    Compatible file structures:
    .txt file:   "RadialDistortion [k1,k2,k3...k8] 
                 TangentialDistortion [p1,p2]
                 IntrinsicMatrix [fx 0. 0.][s fy 0.][cx cy 1]
                 End"
    .mat file:   Camera calibration file output from the Matlab Camera 
                 Calibration App (available in the Computer Vision Systems 
                 toolbox).             
    '''    
    #Load as text file if txt format
    if fileName[-3:] == 'txt':       
        #print 'You have loaded a text file to find the parameters: ' 
        #print str(paramList)
        
        #Open file
        try:        
            myFile=open(fileName,'r')
        except:
            print '\nProblem opening calibration text file: ',fileName
            print 'No calibration parameters successfully read'
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
        #print 'You have loaded a matlab file to find the parameters: '
        #print str(paramList)
    
        #Get the Matlab file as a dictionary
        mat = sio.loadmat(fileName)
           
        # Get desired keys from the dictionary
        calib = dict([(i, mat[i]) for i in paramList if i in mat])
        return calib
   
    #Incompatible file types     
    else:
        print ('You have specified an incorrect calibration file type'
               'Acceptable file types are .txt and .mat.')


def lineSearch(lineList, search):
    '''Function to supplement the readCalib function. Given an input parameter 
    to search within the file, this will return the line numbers of the data.
    '''
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
    '''Function to supplement the importCalibration function. Given the line 
    numbers of the parameter data (the ouput of the lineSearch function), this 
    will return the data.'''    
    #Create empty list for output data
    D=[]
    
    for x in data:
        # Find the fields on each line
        lines[x] = lines[x].translate(None, '[]')
        fields = lines[x].split()
        d = []
        for f in fields:
            d.append(float(f))
        # Add the fields to the data list
        D.append(d)

    if len(D) > 1:
        D = np.array(D)
    
    return D


def readMatrixDistortion(path):
    '''Function to support the calibrate function. Returns the 
    intrinsic matrix and distortion parameters required for calibration from a
    given file.'''   
    #Find the calibration parameters from the list
    ls = ['IntrinsicMatrix', 'RadialDistortion', 'TangentialDistortion']
    calibDict = readCalib(path, ls)
    
    #Stop if there is an issue in reading calibration file    
    if calibDict==None:
        return None   
    #print calibDict

    #Set calibration parameters in the correct format    
    intrMat = np.array(calibDict["IntrinsicMatrix"]) #fx,fy,s,cx,cy
    radDis = np.array(calibDict["RadialDistortion"]) # k1-k6
    tanDis = np.array(calibDict["TangentialDistortion"]) # p1,p2
   
    return intrMat, tanDis, radDis


def checkMatrix(matrix):
    '''Function to support the calibrate function. Checks and converts the 
    intrinsic matrix to the correct format for calibration with OpenCV.'''  
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


def readImage(path, band='L'):
    '''Function to prepare an image by opening, equalising, converting to 
    either grayscale or a specified band, then returning a copy.

    Inputs:
    path:       Image file path directory.
    band:       Desired band output
                'R': red band
                'B': blue band
                'G': green band
                'L': grayscale (default).
    '''   
    # Open image file
    band=band.upper()
    im=Image.open(path)
    
    #Equalise histogram
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
    '''Function to read ground control points from a .txt file. The data in the
    file is referenced to under a header line. Data is appended by skipping the
    header line and finding the world and image coordinates from each line.'''    
    # Open the file and read the first line (i.e. the header line)
    myFile=open(fileName,'r')  
    myFile.readline()

    #Iterate through lines and append to lists
    gcps= []
    for line in myFile.readlines():
        items=line.split('\t')
        gcp = []
        for i in items:
            gcp.append(float(i))               
        gcps.append(gcp)
    
    # Split the list into world and image coordinates
    gcparray = np.array(gcps) 
    world, image = np.hsplit(gcparray,np.array([3]))
           
    return world, image

    
def readDEM(fileName):
    '''Function to read a DEM from an ASCII file by parsing the header and data 
    lines to return the data as a NumPy array, the origin and the cell size. 
    The xyz DEM data is compiled together in the returned item.'''    
    # Open the fileName argument with read permissions
    myFile=open(fileName,'r')
    end_header=False
    
    #Set info about coords, no data or cell size in case it is not passed in
    xll=0.
    yll=0.
    nodata=-999.999
    cellsize=1.0
    
    #Search through header to find the keywords and assign to variables
    while (not end_header):
        line=myFile.readline()  
        items=line.split()
        keyword=items[0].lower()
        value=items[1]
        if (keyword=='ncols'):
            ncols=int(value)
        elif (keyword=='nrows'):
            nrows=int(value)
        elif (keyword=='xllcorner'):
            xll=float(value)
        elif (keyword=='yllcorner'):
            yll=float(value)  
        elif (keyword=='nodata_value'):
            nodata=float(value)
        elif (keyword=='cellsize'):
            cellsize=float(value)  
        else:
            end_header=True
           
    if (nrows==None or ncols==None):
        print "Row or Column size not specified for Raster file read"
        return None

    #Get lists of rows
    datarows=[]
    items=line.split()
    row=[]
    for item in items:
        row.append(float(item))  
    datarows.append(row)

    #Split the lists of rows and parse items as floats   
    for line in myFile.readlines():       
        items=line.split()
        row=[]
        for item in items:
            row.append(float(item))
        datarows.append(row)

    #Convert to array
    data=np.array(datarows)

    #Return numpy array, xy origin and cell size.    
    return data, xll, yll, cellsize   
    
    
def readDEMxyz(fileName):
    '''Function to read xyz DEM data from an ASCII file by parsing the header 
    and data lines to return the data as a NumPy array, the origin and the cell 
    size. The xyz DEM data is returned as separate objects.'''    
    #Open the fileName argument with read permissions
    myFile=open(fileName,'r')
    end_header=False
    
    #Set info about coords, no data or cell size incase not passed in
    xll=0.
    yll=0.
    nodata=-999.999
    cellsize=1.0
    
    #Search through header to find the keywords and assign to variables
    while (not end_header):
        line=myFile.readline()  
        items=line.split()
        keyword=items[0].lower()
        value=items[1]
        if (keyword=='ncols'):
            ncols=int(value)
        elif (keyword=='nrows'):
            nrows=int(value)
        elif (keyword=='xllcorner'):
            xll=float(value)
        elif (keyword=='yllcorner'):
            yll=float(value)  
        elif (keyword=='nodata_value'):
            nodata=float(value)
        elif (keyword=='cellsize'):
            cellsize=float(value)  
        else:
            end_header=True
           
    if (nrows==None or ncols==None):
        print "Row or Column size not specified for Raster file read"
        return None

    #Create empty lists for xyz data 
    X=[]
    Y=[]
    Z=[]

    #Get lists of rows    
    items=line.split()
    i=0
    y = (cellsize*nrows)+yll    
    for item in items:
        Z.append(float(item)) 
        X.append(xll + (cellsize*i))
        Y.append(y)
        i=i+1
    
    j = nrows 
    
    #Split the lists of rows and parse items as floats   
    for line in myFile.readlines():
        j=nrows-1
        y=(cellsize*j)+yll         
        items=line.split()
        i=0
        for item in items:
            Z.append(float(item)) 
            X.append(xll + (cellsize*i))
            Y.append(y)
            i=i+1            

    #Convert to array and return xyz data
    Z=np.array(Z)
    Y=np.array(Y)
    X=np.array(X)  
    return [X,Y,Z]  
 
   
def readDEMmat(matfile):
    '''Function to read xyz DEM data from a .mat file and return the xyz data 
    as separate arrays.'''
    #Load data from .mat file
    mat = sio.loadmat(matfile)
    
    #Read data from xyz objects in file
    X=np.ascontiguousarray(mat['X'])
    Y=np.ascontiguousarray(mat['Y'])
    Z=np.ascontiguousarray(mat['Z'])
    
    #Flip data in up/down direction for compatibility
    X = np.flipud(X)
    Y = np.flipud(Y)
    Z = np.flipud(Z)
    return X,Y,Z  
    
    
def writeTIFF(outFileName,OutArray,affineT,EPSGcode=32633,units="SRS_UL_METER",unitconvert=1.0):
    '''Write data to .tif file. A reference to the file's spatial coordinate 
    system is assigned using GDAL (compatible with ArcGIS and QGIS). 
    The input variable affineT should contain the following parameters for 
    affine tranformation: 
        0: X coordinate of the top left corner of the top left pixel
        1: Pixel Width
        2: Rotation (zero for North up)
        3: Y coordinate of the top left corner of the top left pixel
        4: Rotation (zero for North up)
        5: Pixel Height.
    '''
    
    gdal.AllRegister()
    
    #Set up projection and linear units
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)  
    srs.SetLinearUnits(units,unitconvert)
    
    #Assign columns and rows in file
    cols=OutArray.shape[0]
    rows=OutArray.shape[1]
    
    #Get GDAL driver
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, rows, cols, 1, gdal.GDT_UInt16)
    
    #Set up affine projection information between raw raster array and real 
    #world coordinates. Data in array organised from top down
    outdata.SetGeoTransform(affineT)
    
    #Define projection for output
    outdata.SetProjection(srs.ExportToWkt())
    
    #Output single band
    outband = outdata.GetRasterBand(1)
    outband.WriteArray(OutArray)
    
    #Clear memory   
    outdata = None 

    print 'Output tiff file: ',outFileName


def writeVelocityFile(veloset, timeLapse, fname='velocity.csv',span=[0,-1]):
    '''Function to write all velocity data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following
    information:
    Image pair 1 name
    Image pair 2 name
    Average velocity
    Number of features tracked
    Error
    Signal-to-noise ratio
    
    Input variables:
    veloset:            List of xyz and uv points over multiple images 
                        ([xyz,uv], [xyz,uv]...).
    timeLapse:          A TimeLapse object.
    fname:              Filename for output file. File destination can also 
                        specified.
    span:               The range of images within the image sequence to 
                        iterate over. Default set to all images.
    '''
    #Initialise file writing
    f=open(fname,'w')

    #Write active directory to file
    im1=timeLapse.getImageObj(0)
    pathnam=im1.getImagePath().split('\\')
    dirnam=pathnam[0]
    fn1=pathnam[-1]
    f.write(dirnam+'\n')
    
    #Define column headers
    header=('Image 0, Image 1, Average velocity (unfiltered),'
            'Features tracked (unfiltered),Average velocity (filtered),'
            'Features Tracked,Error, SNR')    
    f.write(header+'\n')

    #Iterate through timeLapse object
    for i in range(timeLapse.getLength()-1)[span[0]:span[1]]:
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        im1=timeLapse.getImageObj(i+1)
        
        #Write image file names to file        
        fn1=im1.getImagePath().split('\\')[-1]
        out=fn0+','+fn1
        
        if veloset[i]!=None:

            #Get velocity data          
            xyz, uv = veloset[i]
                
            #Get xyz coordinates from points in image pair
            xyz1 = xyz[0]               #Pts from image pair 1
            xyz2 = xyz[1]               #Pts from image pair 2
                        
            #Get point positions and differences   
            x1=[]
            y1=[]
            x2=[]
            y2=[]
            xdif=[]
            ydif=[]
            for i,j in zip(xyz1,xyz2):
                if math.isnan(i[0]):
                    pass
                else:
                    x1.append(i[0])
                    y1.append(i[1])
                    x2.append(j[0])
                    y2.append(j[1])
                    xdif.append(i[0]-j[0])
                    ydif.append(i[1]-j[1])
        
            #Calculate velocity with Pythagoras' theorem
            speed=[]
            for i,j in zip(xdif, ydif):
                sp = np.sqrt(i*i+j*j)
                speed.append(sp)
            
            #Calculate average unfiltered velocity
            velav = sum(speed)/len(speed)
            
            #Determine number of features (unfiltered) tracked
            numtrack = len(speed)

            #Filter outlier points 
            v_all=np.vstack((x1,y1,x2,y2,speed))
            v_all=v_all.transpose()
            filtered=filterSparse(v_all,numNearest=12,threshold=2,item=4)           
            fspeed=filtered[:,4]
            
            #Calculate average unfiltered velocity
            velfav = sum(fspeed)/len(fspeed)
            
            #Determine number of features (unfiltered) tracked
            numtrackf = len(fspeed)
            
            #Compile all data for output file
            out=out+','+str(velav)+','+str(velfav)+','+str(numtrack)+','+str(numtrackf)
        
        #Write to output file
        f.write(out+'\n')
        
    
def writeHomographyFile(homogset,timeLapse,fname='homography.csv',span=[0,-1]):
    '''Function to write all homography data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following 
    information:
    Image pair 1 name
    Image pair 2 name
    Homography matrix [0,0]
    Homography matrix [0,1]
    Homography matrix [0,2]
    Homography matrix [1,0]
    Homography matrix [1,1]
    Homography matrix [1,2]
    Homography matrix [2,0]
    Homography matrix [2,1]
    Homography matrix [2,2]
    Number of features tracked
    X mean displacement
    Y mean displacement
    X standard deviation
    Y standard deviation
    Mean error magnitude
    Mean homographic displacement
    Magnitude
    Homography signal-to-noise ratio
    '''
    #Initialise file writing
    f=open(fname,'w')
    
    #Write active directory to file
    im1=timeLapse.getImageObj(0)
    pathnam=im1.getImagePath().split('\\')
    dirnam=pathnam[0]
    fn1=pathnam[-1]
    f.write(dirnam+'\n')
    
    #Define column headers
    header=('Image 0, Image 1,"Homography Matrix[0,0]","[0,1]","[0,2]",'
            '"[1,0]","[1,1]","[1,2]","[2,0]","[2,1]","[2,2]",Features Tracked,'
            'xmean,ymean,xsd,ysd,"Mean Error Magnitude",'
            '"Mean Homographic displacement","magnitude",'
            '"Homography S-N ratio"')    
    f.write(header+'\n')

    #Iterate through timeLapse object
    for i in range(timeLapse.getLength()-1)[span[0]:span[1]]:
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        im1=timeLapse.getImageObj(i+1)

        #Write image file names to file        
        fn1=im1.getImagePath().split('\\')[-1]
        out=fn0+','+fn1
        
        if homogset[i]!=None:

            #Get homography data for image pair            
            hgm, points, ptserrors, homogerrs=homogset[i]

            #Get xyz homography errors            
            homogerrors=homogerrs[0]            
            xd=homogerrs[1][0]
            yd=homogerrs[1][1]
            
            #Get uv point positions
            ps=points[0]
            pf=points[1]
            psx=ps[:,0,0]
            psy=ps[:,0,1]
            pfx=pf[:,0,0]
            pfy=pf[:,0,1]
            
            #Determine uv point position difference
            pdx=pfx-psx
            pdy=pfy-psy
            
            #Calculate signal-to-noise ratio            
            errdist=np.sqrt(xd*xd+yd*yd)
            homogdist=np.sqrt(pdx*pdx+pdy*pdy)
            sn=errdist/homogdist
            
            #Calculate mean homography, mean error and mean SNR 
            meanerrdist=np.mean(errdist)
            meanhomogdist=np.mean(homogdist)
            meansn=np.mean(sn)
            
            #Define output homography matrix
            if hgm!=None:
                hgm.shape=(9)
                for val in hgm:
                    out=out+','+str(val)
            
            #Determine number of points tracked in homography calculations
            tracked=len(points[0])
            out=out+','+str(tracked)
            
            #Define output homography matrix errors
            for val in homogerrors:
                    out=out+','+str(val)
            
            #Compile all data for output file
            out=out+','+str(meanerrdist)+','+str(meanhomogdist)+','+str(meansn)
        
        #Write to output file
        f.write(out+'\n')
 

def createThumbs(directory='.'):
    '''Function to create thumbnail images from a given image file directory
    using the PIL (Python Imaging Library) toolbox.
    The directory input variable is a string containing the file directory to a
    given sequence of images (default is set to current work directory).
    '''
    #Get all images in given file directory
    directory=directory+'/*.jpg'
    imageList = glob.glob(directory)

    #Iterate through all images    
    for impath in imageList:
        
        #Create thumbnail using PIL
        im=Image.open(impath)
        im.thumbnail([512,512],Image.ANTIALIAS)
        
        #Save thumbnail to file directory
        out=impath.split('\\')[0]+'/thumb_'+impath.split('\\')[1]
        print 'Saving thumbnail image as',out
        im.save(out, "JPEG")

        
#------------------------------------------------------------------------------
#Testing code. Requires suitable files in ..\Data\Images\Velocity test sets
if __name__ == "__main__":   
    createThumbs('./Data/Images/Velocity/c1_2014')