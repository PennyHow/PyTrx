'''
PYTRX FILEHANDLER MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This module, FileHandler, contains all the functions called by an object to 
load and export data.

Core functions available in FileHandler
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
writeVelocityFile:      Function to write all velocity data from a given 
                        timeLapse sequence to .csv file.
writeHomography:        Function to write all homography data from a given 
                        timeLapse sequence to .csv file.
createThumbs:           Function to create thumbnail images from a given image 
                        file directory using the PIL (Python Imaging Library) 
                        toolbox.
writeAreaFile:          Function to write all area data (if it has been 
                        calculated) to separate .txt files containing the pixel
                        extents, polygon pixel coordinates, xyz polygon areas,
                        cumulative areas of all polygons, and polygon xyz 
                        coordinates. All these output files are compatible with 
                        the importing tools, namely importAreaData.
WriteLineFile:          Function to write all line data (if it has been 
                        calculated) to separate .txt files containing the pixel
                        coordinates of the lines, the pixel line length, the
                        xyz coordinates of the lines, and the real (xyz) line 
                        lengths. All these output files are compatible with the 
                        importing tools, namely importLineData.
writeSHPFile:           Function to write OGR real velocities/areas/lines (from 
                        ALL images) to file in a .shp file type that is 
                        compatible with mapping sofrware such as ArcMap and 
                        QGIS.
importAreaData:         Function to get xyz and px area data from text files 
                        and import it into a specified Measure.Area class 
                        object. This uses the importAreaXYZ and 
                        importAreaPX functions, which are also contained within
                        this FileHandler script. All imported data is held in 
                        the Area class object specified as an input variable. 
                        This can be easily retrieved from the Area class object 
                        itself.
importLineData:         Function to get xyz and px line data from text files 
                        and import into a specified Measure.Line class object.
                        This uses the importLineXYZ and importLinePX functions,
                        which are also contained within this FileHandler 
                        script. All imported data is held in the Line class 
                        object specified as an input variable. This can be 
                        easily retrieved from the Line class object itself.
                        
@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton 
         Lynne Addison
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
import os
import sys

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
            print ('\nMask loaded. It is recommended that you check this ' 
                   'against the start and end of the sequence using the ' 
                   'self.checkMask() function of the TimeLapse object')
            return myMask
        except:
            print '\nMask file not found. Proceeding to manually digitise...'

    #Plot mask manually on the selected image
    fig=plt.gcf()
    fig.canvas.set_window_title('Click to create mask. Press enter to record' 
                                ' points.')
    imgplot = plt.imshow(img, origin='upper')
    imgplot.set_cmap('gray')
    x1 = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, 
                    mouse_stop=2)
    print '\n' + str(len(x1)) + ' points seeded'
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
    print '\nMask plotted: ' + writeMask
    if writeMask!=None:
        try:
            img1.save(writeMask, 'jpeg', quality=75)
        except:
            print '\nFailed to write file: ' + writeMask
        
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
        print ('\nYou have specified an incorrect calibration file type'
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
    will return the data.
    '''    
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
    given file.
    '''   
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
    '''Function to support the calibrate function. Checks and converts the 
    intrinsic matrix to the correct format for calibration with OpenCV.
    '''  
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
    header line and finding the world and image coordinates from each line.
    '''    
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
            print ('\nGCP ERROR: ' + str(len(gcp)) + 
                   '/5 values found in line ' + str(count))
            print 'Values not passed forward. Check GCP file'
        
        #Update counter        
        count=count+1
                  
    #Split the list into world and image coordinates
    gcparray = np.array(gcps)
    world, image = np.hsplit(gcparray,np.array([3]))
           
    return world, image

    
def readDEM(fileName):
    '''Function to read a DEM from an ASCII file by parsing the header and data 
    lines to return the data as a NumPy array, the origin and the cell size. 
    The xyz DEM data is compiled together in the returned item.
    '''    
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
    size. The xyz DEM data is returned as separate objects.
    '''    
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
    as separate arrays.
    '''
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
    
    
def writeTIFF(outFileName, OutArray, affineT, EPSGcode=32633, 
              units="SRS_UL_METER",unitconvert=1.0):
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

    print '\nOutput tiff file: ',outFileName


def writeVelocityFile(velocity, fname='velocity_xyz.csv'):
    '''Function to write all velocity data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following
    information:
        Image pair 1 name
        Image pair 2 name
        Average xyz velocity (unfiltered)
        Number of features tracked (unfiltered)
        Average xyz velocity (filtered)
        Number of features tracks (filtered)
        Average pixel velocity
        Homography residual mean error (RMS)
        Signal-to-noise ratio
    
    Input variables:
    velocity:           Measure.Velocity class object which contains xyz and 
                        uv velocities.
    fname:              Filename for output file. File destination can also 
                        specified.
    '''
    #Initialise file writing
    f=open(fname,'w')

    #Write active directory to file
    im1=velocity.getImageObj(0)
    pathnam=im1.getImagePath().split('\\')
    dirnam=pathnam[0]
    fn1=pathnam[-1]
    f.write(dirnam+'\n')
    
    #Define column headers
    header=('Image 0, Image 1, Average xyz velocity (unfiltered),'
            'Features tracked (unfiltered), Average xyz velocity (filtered),'
            'Features Tracked (filtered), Average px velocity,'
            'Homography RMS Error, SNR')    
    f.write(header + '\n')

    #Iterate through timeLapse object
    for i in range(velocity.getLength()-1):
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        im1=velocity.getImageObj(i+1)
        
        #Write image file names to file        
        fn1=im1.getImagePath().split('\\')[-1]
        out=fn0+','+fn1
        
        
        #Get velocity data                     
        xyz = velocity._xyzvel[i]
        uv = velocity._uvvel[i]

        #Calculate average unfiltered velocity
        xyzvelav = sum(xyz)/len(xyz)
        pxvelav= sum(uv)/len(uv)
                       
        #Determine number of features (unfiltered) tracked
        numtrack = len(xyz)

        #Write unfiltered velocity information
        f.write(out + ',' +str(xyzvelav) + ',' +str(numtrack) + ',')

        pt0 = velocity._xyz0[i]
        pt1 = velocity._xyz1[i]
        x1 = pt0[:,0]
        y1 = pt0[:,1]
        x2 = pt1[:,0]
        y2 = pt1[:,1]        
        
        #Filter outlier points 
        v_all=np.vstack((x1,y1,x2,y2,xyz))
        v_all=v_all.transpose()
        filtered=filterSparse(v_all,numNearest=12,threshold=2,item=4)
            
        #Write filtered velocity information           
        if len(filtered) > 1:
            fspeed=filtered[:,4]
        
            #Calculate average filtered velocity
            velfav = sum(fspeed)/len(fspeed)
        
            #Determine number of features (filtered) tracked
            numtrackf = len(fspeed)
        
            #Compile all data for output file
            f.write((str(velfav) + ',' + str(numtrackf) + ','))
            
                    
        #Get homography information if desired
        if hasattr(velocity, '_homogmatrix'):
            hpt0 = velocity._homogpts0[i]
            hpt1 = velocity._homogpts1[i]
            hpt1corr = velocity._homogpts1corr[i]
            herr = velocity._homogerr[i]
    
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
            
            #Calculate SNR between pixel velocities and error
            snr=meanerrdist/pxvelav          
            
            #Write pixel velocity and homography information
            f.write((str(pxvelav) + ',' + str(meanerrdist) + ','  +
                     str(snr)))
             
            #Break line in output file
            f.write('\n')
            
    print '\nVelocity file written:' + fname        
 
   
def writeHomographyFile(velocity,fname='homography.csv'):
    '''Function to write all homography data from a given timeLapse sequence to 
    .csv file. Data is formatted as sequential columns containing the following 
    information:
    
    Image pair 1 name; image pair 2 name; homography matrix (i.e. all values
    in the 3x3 matrix starting from top left; number of features tracked;
    X mean displacement; Y mean displacement; X standard deviation; Y standard 
    deviation; mean error magnitude; mean homographic displacement; homography 
    signal-to-noise ratio.
    '''
    #Initialise file writing
    f=open(fname,'w')
    
    #Write active directory to file
    im1=velocity.getImageObj(0)
    pathnam=im1.getImagePath().split('\\')
    dirnam=pathnam[0]
    fn1=pathnam[-1]
    f.write(dirnam+'\n')
    
    #Define column headers
    header=('Image 0, Image 1,"Homography Matrix[0,0]","[0,1]","[0,2]",'
            '"[1,0]","[1,1]","[1,2]","[2,0]","[2,1]","[2,2]",Features Tracked,'
            'xmean,ymean,xsd,ysd,"Mean Error Magnitude",'
            '"Mean Homographic displacement","Homography SNR"')    
    f.write(header+'\n')

    #Iterate through timeLapse object
    for i in range(velocity.getLength()-1):
        
        #Re-define image0 for each iteration
        fn0=fn1
        
        #Get image1
        im1=velocity.getImageObj(i+1)

        #Write image file names to file        
        fn1=im1.getImagePath().split('\\')[-1]
        out=fn0+','+fn1
        
        #Get homography information from velocity object
        hmatrix = velocity._homogmatrix[i]                  #Homography matrix
        hpt0 = velocity._homogpts0[i]                       #Seeded pts in im0
        hpt1 = velocity._homogpts1[i]                       #Tracked pts in im1
        hpt1corr = velocity._homogpts1corr[i]               #Corrected pts im1
        herr = velocity._homogerr[i]                        #Homography error
        
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
        
        #Calculate mean homography, mean error and mean SNR 
        meanerrdist=np.mean(errdist)
        meanhomogdist=np.mean(homogdist)
        meansn=np.mean(sn)
        
        #Define output homography matrix
        if hmatrix is not None:
            hmatrix.shape=(9)
            for val in hmatrix:
                out=out+','+str(val)
        
        #Determine number of points tracked in homography calculations
        tracked=len(hpt0)
        out=out+','+str(tracked)
        
        #Define output homography matrix errors
        for val in herr[0]:
            out=out+','+str(val)
        
        #Compile all data for output file
        out = (out+','+str(meanerrdist)+','+str(meanhomogdist)+','
               +str(meansn))
        
        #Write to output file
        f.write(out+'\n') 
        
    print '\nHomography file written' + fname
 

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
        print '\nSaving thumbnail image as',out
        im.save(out, "JPEG")


def writeAreaFile(a, dest):
    '''Write all area data (if it has been calculated) to separate files.
    
    Inputs
    a:                                  Area class object that contains 
                                        detected areas.
    dest:                               Folder directory where output files 
                                        will be written to (NOT a specific 
                                        file).
    
    Outputs
    Cumulative pixel extent:            Written as a single text file 
                                        containing delineated values of total 
                                        pixel area recorded for all images in 
                                        the sequence.
    Polygon pixel coordinates:          Written as a tab delimited text file 
                                        containing polygon xy coordinates.
    Polygon areas:                      Real world area for each polygon in an 
                                        image sequence.
    Cumulative area:                    Written as a single text file 
                                        containing the total area of all the 
                                        polgons from each image.
    Polygon real coordinates:           Written as a tab delimited text file 
                                        containing polygon xyz coordinates
     
    All these file types are compatible with the importing tools 
    (importPX, importXYZ).
    '''  
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    pxextent = a._pxextent
    pxpoly = a._pxpoly
    rarea = a._area
    rpoly = a._realpoly
    
    #Cumulative area of all pixel extents       
    if pxextent is not None:
        target = dest + 'px_sum.txt'
        imgcount=1 
        f = open(target, 'w')            
        for p in pxextent:
            f.write('Img' + str(imgcount) + '_polysum(px)\t')
            f.write(str(p) + '\n')
            imgcount=imgcount+1
        
        #Pixel cooridnates of all pixel extents
        target = dest + 'px_coords.txt'
        imgcount=1
        polycount=1
        f = open(target, 'w')
        for im in pxpoly:
            f.write('Img' + str(imgcount) + '\t')                
            for pol in im:
                f.write('Poly' + str(polycount) + '\t')                    
                for pts in pol:
                    if len(pts)==1:
                        f.write(str(pts[0][0]) + '\t' + str(pts[0][1]) + '\t')
                    elif len(pts)==2:
                        f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')
                polycount = polycount+1
            f.write('\n\n')
            imgcount = imgcount+1
            polycount=1
            
    #Areas and cumulative areas of polygons
    if rarea is not None:
        sumarea=[]
        target = dest + 'area_all.txt'
        imgcount = 1
        f = open(target, 'w')

        for im in rarea:            
            all_areas = sum(im)
            sumarea.append(all_areas)
            
            f.write('Img' + str(imgcount) + '_polyareas\t')                
            for ara in im:
                f.write(str(ara) + '\t')
            f.write('\n')
            imgcount=imgcount+1

        target = dest + 'area_sum.txt'
        f = open(target, 'w')  
        imgcount=1                          
        for s in sumarea:
            f.write('Img' + str(imgcount) + '_totalpolyarea\t')
            f.write(str(s) + '\n')
            imgcount=imgcount+1
             
        #Pt coordinates of polygons
        target = dest + 'area_coords.txt'                   
        imgcount=1
        polycount=1  
        f = open(target, 'w')
        for im in rpoly:                
            f.write('Img' + str(imgcount) + '\t')
            for p in im:
                f.write('Poly' + str(polycount) + '\t')
                for pts in p:
                    f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t' + 
                            str(pts[2]) + '\t')
                polycount=polycount+1
            f.write('\n\n')
            imgcount=imgcount+1
            polycount=1
                    

def writeLineFile(l, dest):
    '''Write all line data (if it has been calculated) to separate files.
    
    Inputs
    l:                                  Length class object containing detected
                                        lines.
    dest:                               Folder directory where output files 
                                        will be written to (NOT a specific 
                                        file).
    
    Outputs
    -Pixel coordinates of lines:        Written as a single text file 
                                        containing line xy coordinates for all 
                                        images in the sequence.
    -Pixel line lengths:                Contains pixel length of each line in 
                                        all images.
    -Real coordinates:                  Written as a tab delimited text file 
                                        containing line xyz coordinates.
    -Real line lengths:                 Real world length of each line in all 
                                        images.
     
    All these file types are compatible with the importing tools 
    (importAreaPX, importAreaXYZ)'''
    
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    #Pixel line coordinates file generation             
    if l._pxpts is not None:            
        target = dest + 'line_pxcoords.txt'
        imgcount=1
        f = open(target, 'w')
        for im in l._pxpts:
            f.write('Img' + str(imgcount) + '\t')                                   
            for pts in im:
                f.write(str(pts[0]) + '\t' + str(pts[1]) + '\t')    
            f.write('\n\n')
            imgcount = imgcount+1

    #Pixel line length file generation            
    if l._pxline is not None:
        target = dest + 'line_pxlength.txt'
        imgcount=1 
        f = open(target, 'w')            
        for p in l._pxline:
            f.write('Img' + str(imgcount) + '_length(px)\t')
            f.write(str(p.Length()) + '\n')
            imgcount=imgcount+1 
    
    #Real line coordinates file generation
    if l._realpts is not None:
        target = dest + 'line_realcoords.txt'                   
        imgcount=1  
        f = open(target, 'w')
        for im in l._realpts:                
            f.write('Img' + str(imgcount) + '\t')
            for pts in im:
                f.write(str(pts[0]) + '\t' + 
                        str(pts[1]) + '\t' + 
                        str(pts[2]) + '\t')
            f.write('\n\n')
            imgcount=imgcount+1
    
    #Real line length file generation            
    if l._realline is not None:
        target = dest + 'line_reallength.txt'
        imgcount = 1
        f = open(target, 'w')
        for p in l._realline:
            f.write('Img' + str(imgcount) + '_length(m)\t')                
            f.write(str(p.Length()) + '\n')
            imgcount=imgcount+1


def writeSHPFile(measure, fileDirectory, projection=None):
    '''Write OGR real polygon areas (from ALL images) to file in a .shp
    file type that is compatible with ESRI mapping software.
    
    Inputs
    a:                          Velocity/Area/Line class object containing 
                                velocity/area/line measurements.
    fileDirectory:              Destination that shapefiles will be written to           
                                e.g. C:/python_workspace/Results/
    projection:                 Coordinate projection that the shapefile will 
                                exist in. This can either be an ESPG number 
                                (expressed as an integer) or a well-known 
                                geographical coordinate system (expressed as a 
                                string). Well-known geographical coordinate 
                                systems are: 'WGS84', 'WGS72', NAD83' or 
                                'EPSG:n'
    ''' 

    if not os.path.exists(fileDirectory):
        os.makedirs(fileDirectory)
        
    #Get driver and create shapeData in shp file directory        
    typ = 'ESRI Shapefile'        
    driver = ogr.GetDriverByName(typ)
    if driver is None:
        raise IOError('%s Driver not available:\n' % typ)
        sys.exit(1)
        

    #Set projection and initialise velocity layer
    if hasattr(measure, '_xyzvel'): 
        print '\nDetected velocity points to write as shapefiles'
        for i in range(measure.getLength()-1): 
            
            #Get velocity, pt and image name for time step
            vel = measure._xyzvel[i]
            pt0 = measure._xyz0[i]            
            imn = measure._imageSet[i].getImagePath().split('\\')[1] 
            
            #Create file space            
            shp = fileDirectory + str(imn) + '_vel.shp'
            if os.path.exists(shp):
                print '\nDeleting pre-existing datasource'
                driver.DeleteDataSource(shp)
            ds = driver.CreateDataSource(shp)
            if ds is None:
                print 'Could not create file %s' %shp
            
            #Set projection
            if type(projection) is int:
                print '\nESPG projection detected'
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbPoint)
            elif type(projection) is str:
                print '\nCoordinate system detected'
                proj = osr.SpatialReference()
                proj.SetWellKnownGeogCS(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbPoint)
            else:
                print '\nProjection for shapefiles not recognised'
                print 'Proceeding shapefile generation without projection'
                layer = ds.CreateLayer(' ', None, ogr.wkbPoint)
       
            #Add attributes to layer
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))     #ID    
            layer.CreateField(ogr.FieldDefn('velocity', ogr.OFTReal))  #Velo
            
            #Get xy coordinates
            x0 = pt0[:,0]
            y0 = pt0[:,1]
            
            #Create point features with data attributes in layer           
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
                
    #Set projection and initialise area layer            
    elif hasattr(measure, '_realpoly'):
        print '\nDetected polygons to write as shapefiles'
        for i in range(measure.getLength()-1): 
            
            #Get polygon coordinates and image name for each time step
            polys = measure._realpoly[i]          
            imn = measure._imageSet[i].getImagePath().split('\\')[1] 
                          
            #Create datasource in shapefile
            shp = fileDirectory + str(imn) + '_area.shp'
            
            if os.path.exists(shp):
                print '\nDeleting pre-existing datasource'
                driver.DeleteDataSource(shp)
            ds = driver.CreateDataSource(shp)
            if ds is None:
                print 'Could not create file %s' %shp
        
            if type(projection) is int:
                print '\nESPG projection detected'
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbPolygon)
            elif type(projection) is str:
                print '\nCoordinate system detected'
                proj = osr.SpatialReference()
                proj.SetWellKnownGeogCS(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbPolygon)
            else:
                print '\nProjection for shapefiles not recognised'
                print 'Proceeding shapefile generation without projection'
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
        
        
    #Set projection and initialise line layer            
    elif hasattr(measure, '_realline'):
        print '\nDetected lines to write as shapefiles'        
        for i in range(measure.getLength()-1):
            
            #Get polygon coordinates and image name for each time step
            line = measure._realpts[i]          
            imn = measure._imageSet[i].getImagePath().split('\\')[1] 

            #Create datasource in shapefile
            shp = fileDirectory + str(imn) + '_line.shp'
            
            if os.path.exists(shp):
                print '\nDeleting pre-existing datasource'
                driver.DeleteDataSource(shp)
            ds = driver.CreateDataSource(shp)
            if ds is None:
                print 'Could not create file %s' %shp
        
            if type(projection) is int:
                print '\nESPG projection detected'
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbLineString)
            elif type(projection) is str:
                print '\nCoordinate system detected'
                proj = osr.SpatialReference()
                proj.SetWellKnownGeogCS(projection)
                layer = ds.CreateLayer(' ', proj, ogr.wkbLineString)
            else:
                print '\nProjection for shapefiles not recognised'
                print 'Proceeding shapefile generation without projection'
                layer = ds.CreateLayer(' ', None, ogr.wkbLineString)
            
            #Add attributes
            layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
            layer.CreateField(ogr.FieldDefn('length', ogr.OFTReal))

            lcount=1
            
#            for shape in polys:
#                line = ogr.Geometry(ogr.wkbLineString)   
#                for p in shape:
#                    line.AddPoint(p[0],p[1])
            
            #Create feature            
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(line)
            feature.SetField('id', lcount)
            feature.SetField('length', line.Length())
            layer.CreateFeature(feature)
            feature.Destroy() 
            lcount=lcount+1
            
            ds.Destroy()
    else:
        print '\nUnrecognised Velocity/Area/Line class object'


def importAreaData(trxclass, fileDirectory):
    '''Get xyz and px data from text files and import into a specified 
    Measure.Area class object.
    
    Inputs
    trxclass:           Area/Line class object that data will be imported to.
    fileDirectory:      Path to the folder where the text files containing data
                        are. Specific files are needed for importing Area data:
                        4 text files needed. Two contain the pixel 
                        and real world polygon coordinates and must be named 
                        'px_coords.txt' and 'area_coords.txt'. The text files 
                        containing the pixel and real world polygon areas must 
                        be named 'px_sum.txt' and 'area_all.txt'. 
            
    Outputs:
    rpoly:              Real xyz coordinates of detected areas.
    rarea:              Real-world surface areas.
    pxpoly:             XY pixel coordinates of detected areas.
    pxarea:             Pixel areas.

    Files from which data is imported will not be recognised if they are not 
    named correctly. If files are located in multiple directories or file names 
    cannot be changed then use the importAreaXYZ and importAreaPX functions.  
    
    All imported data is held in the Area class object specified as an 
    input variable. This can be easily retrieved from the Area class 
    object itself.
    '''
    #Get real-world coordinates and areas    
    target1 = fileDirectory + 'area_coords.txt'
    target2 = fileDirectory + 'area_all.txt'    
    rpoly, rarea = importAreaXYZ(trxclass, target1, target2)

    #Get pixel coordinates and areas    
    target3 = fileDirectory + 'px_coords.txt'
    target4 = fileDirectory + 'px_sum.txt'
    pxpoly, pxarea = importAreaPX(trxclass, target3, target4)

    #Return all area data   
    return rpoly, rarea, pxpoly, pxarea


def importLineData(trxclass, fileDirectory):
    '''Get xyz and px data from text files and import into a specified 
    Measure.Line class object.
    
    Inputs
    trxclass:           Line class object that data will be imported to.
    fileDirectory:      Path to the folder where the text files containing data
                        are. Specific files are needed for importing Line data:
                        2 text files needed. These files contain 
                        the pixel and real-world line coordinates and  must be
                        named 'line_realcoords.txt' and 'line_pxcoords.txt'.
            
    Outputs:
    rline:        Real xyz coordinates of detected lines.
    rlength:      Real-world surface distances.
    pxline:       XY pixel coordinates of detected lines.
    pxlength:     Pixel distances.

    Files from which data is imported will not be recognised if they are not 
    named correctly. If files are located in multiple directories or file names 
    cannot be changed then use the importLineXYZ and importLinePX functions.  
    
    All imported data is held in the Line class object specified as an 
    input variable. This can be easily retrieved from the Line class 
    object itself.
    '''
    #Get real-world coordinates and distances
    target1 = fileDirectory + 'line_realcoords.txt'
    rline, rlength = importLineXYZ(trxclass, target1)
    
    #Get pixel coordinates and distances
    target2 = fileDirectory + 'line_pxcoords.txt'
    pxline, pxlength = importLinePX(trxclass, target2)
    
    #Return all line data
    return rline, rlength, pxline, pxlength
        
        
def importAreaXYZ(areaclass, target1, target2):
    '''Get xyz polygon and area data from text files and import into Areas 
    class.
    Inputs
    a:             Area class object that data will be imported to.     
    target1:       Path to the text file containing the xyz coordinate data. 
    target2:       Path to the text file containing the polygon area data.
    '''
    #Import polygon coordinates from text file
    f=file(target1,'r')                                
    alllines=[]
    for line in f.readlines():
        if len(line) >= 6:
            alllines.append(line)  #Read lines in file             
    print '\nDetected xyz coordinates from ' + str(len(alllines)) + ' images'
    f.close()
    #Set up xyz object
    xyz=[] 
    count=0

    #Extract polygon data as strings
    for line in alllines:
        img=[]
        strpolys=[]            
        temp=line.split('Pol')                                    
        for i in temp:
            if len(i) >=10:
                strpolys.append(i)
        count=count+1
        print 'Detected '+str(len(strpolys))+' polygons in image '+(str(count))                
        
        #Extract polygon values from strings
        for strp in strpolys:
            pts = strp.split('\t')
            coords=[]
            for p in pts:
                try:
                    coords.append(float(p))
                except ValueError:
                    pass
            struc = len(coords)/3
            polygon = np.array(coords).reshape(struc, 3)
            img.append(polygon)
        xyz.append(img)
     
    #Import polygon areas from text file
    f=file(target2,'r')                                
    alllines=[]
    for line in f.readlines():
         alllines.append(line)
    print 'Detected areas from ' + str(len(alllines)) + ' images'
    f.close()
    
    #Extract area values from lines
    areas=[]          
    for line in alllines:
        vals = line.split('\t')
        imgs=[]
        
        #Change area values from strings to floats
        for v in vals:
            polys=[]
            try:
                a = float(v)
                polys.append(a) 
                imgs.append(polys)
                myimgs = [p[0] for p in imgs]                    
            except ValueError:
                myimgs=None
        
        #Compile areas from all images into a list of lists                                           
        areas.append(myimgs) 
        
    #Import data into Area class
    areaclass._realpoly = xyz
    areaclass._area = areas
    
    #Return polygon and area data
    return areaclass._realpoly, areaclass._area
   

def importAreaPX(areaclass, target1, target2):
    '''Get px polygon and extent data from multiple text files and import
    into Areas class.
    Inputs
    a:             Area class object that data will be imported to.     
    target1:       Path to the text file containing the xy coordinate data. 
    target2:       Path to the text file containing the polygon pixel extent 
                   data.              
    '''
    #Import px polygon coordinates from text file
    f=file(target1, 'r')
    alllines=[]
    for line in f.readlines():
        if len(line) >=6:
            alllines.append(line)
    print '\nDetected px coordinates from ' + str(len(alllines)) + ' images'
    f.close()
    
    #Set up xyz object
    xy=[] 
    count=0

    #Extract polygon data as strings
    for line in alllines:
        img=[]
        strpolys=[]           
        temp=line.split('Pol')                                    
        for i in temp:
            if len(i) >=10:
                strpolys.append(i)

        count=count+1
        print ('Detected ' + str(len(strpolys)) + 
               ' polygons in image ' + (str(count)))                
        
        #Extract polygon values from strings
        for strp in strpolys:
            pts = strp.split('\t')
            coords=[]
            for p in pts:
                try:
                    coords.append(float(p))
                except ValueError:
                    pass
            struc = len(coords)/2
            polygon = np.array(coords).reshape(struc, 2)
            img.append(polygon)
        xy.append(img)    
        
    #Import px polygon extents from text file
    f=file(target2,'r')                                
    alllines=[]
    for line in f.readlines():
         alllines.append(line)
    print 'Detected cumulative extents from ' + str(len(alllines)) + ' images'
    f.close()
    
    #Extract area values from lines
    areas=[]
    a=[]          
    for line in alllines:
        vals = line.split('\t')          
        #Change area values from strings to floats
        for v in vals:
            poly=[]
            try:
                poly.append(float(v)) 
            except ValueError:
                pass
        #Compile areas from all images into a list of lists                                           
        a.append(poly) 
        areas = [ext[0] for ext in a]            
    
    #Import data into Area class
    areaclass._pxpoly = xy
    areaclass._pxextent = areas
    
    return areaclass._pxpoly, areaclass._pxextent
     
     
def importLineXYZ(lineclass, filename):
    '''Get xyz line and length data from text files and import into a Line 
    class object.
    Inputs
    l:                      Line class object that data will be imported to.
    filename:               Path to the text file containing the xyz line 
                            coordinates.
    '''
    #Read file and detect number of images based on number of lines
    xyz = _coordFromTXT(filename, xyz=True)
     
    #Create OGR line object
    ogrline=[]
    for line in xyz:        
        length = ogr.Geometry(ogr.wkbLineString)   
        for p in line:
            length.AddPoint(p[0],p[1])
        ogrline.append(length)
    
    #Import data into Area class
    lineclass._realpts = xyz
    lineclass._realline = ogrline
    
    return lineclass._realpts, lineclass._realline
   

def importLinePX(lineclass, filename):
    '''Get px line and length data from multiple text files and import
    into a Line class object.
    Inputs
    l:                      Line class object that data will be imported to.
    filename:               Path to the folder where the text files containing
                            the xy pixel coordinates.
     '''
    #Read file and detect number of images based on number of lines
    xy = _coordFromTXT(filename, xyz=False)
       
    #Create OGR line object
    ogrline = []
    for line in xy:        
        length = ogr.Geometry(ogr.wkbLineString)   
        for p in line:
            length.AddPoint(p[0],p[1])
        ogrline.append(length)
    
    #Import data into Area class
    lineclass._pxpts = xy
    lineclass._pxline = ogrline
    
    return lineclass._pxpts, lineclass._pxline

     
def _coordFromTXT(filename, xyz):
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
    
#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   