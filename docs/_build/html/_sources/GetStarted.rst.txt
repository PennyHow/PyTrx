Getting Started
===============

PyTrx comes with working examples to get started with. These scripts are available in the PyTrx repository 
`here <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples>`_. These examples are for applications in glaciology, which can be adapted and used. We hope these are especially useful for beginners in coding.


Automated detection of supraglacial lakes
-----------------------------------------
In this example, we will derive changes in surface area of supraglacial lakes captured from Kronebreen, Svalbard, for a small subset of the 2014 melt season. This example can be found in `KR_autoarea.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_autoarea.py>`_.

We will automatically detect water on the glacier based on differences in pixel intensity and corrected for image distortion; using images from `Kronebreen camera 3 <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples/images/KR3_2014_subset>`_ and the associated `camera environment <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_KR3_2014.txt>`_.

First, we need to import the PyTrx packages that we are going to use.


.. code-block:: python

   from PyTrx.CamEnv import CamEnv
   from PyTrx.Area import Area
   from PyTrx.Velocity import Homography
   import PyTrx.FileHandler as FileHandler
   from PyTrx.Utilities import plotAreaPx, pltAreaXYZ


We load our camera environment and masks (for feature extraction and image registration), and set the paths to our input images and output folder.


.. code-block:: python
   
   # Define camera environment input file
   camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR3_2014.txt'
   
   # Define feature detection and registration mask files
   camamask = '../Examples/camenv_data/masks/KR3_2014_amask.jpg'
   caminvmask = '../Examples/camenv_data/invmasks/KR3_2014_inv.jpg'
   
   # Define image folder
   camimgs = '../Examples/images/KR3_2014_subset/*.JPG'


Next, we create a CamEnv object using our previously defined camera environment text file which contains information about the camera location and pose, and file paths to our DEM, ground control point positions, camera calibration coefficients, and reference image.       
 
 
.. code-block:: python
   
   # Create camera environment
   cameraenvironment = CamEnv(camdata)


If certain camera environment parameters are unknown or guessed, then PyTrx's optimisation parameters can be used to refine the camera environment and improve the georectification. This refinement is conducted based on the ground control points.

In this case, the camera pose (yaw, pitch, roll - YPR) is unknown, so we will use the optimisation routine the refine the YPR values.


.. code-block:: python
   
   # Set camera optimisation parameters

   optparams = 'YPR'      # Flag to denote which parameters to optimise: 
                          # YPR=camera pose; INT=intrinsic camera model; 
                          # EXT=extrinsic camera model; ALL=all camera 
                          # parameters
                                
   optmethod = 'trf'      # Optimisation method: trf=Trust Region 
                          # Reflective algorithm; dogbox=dogleg algorithm;
                          # lm=Levenberg-Marquardt algorithm

   # Optimise camera                                
   cameraenvironment.optimiseCamEnv(optparams, optmethod, show=True)

In order to make measurements from the images, we need to ensure that motion in the camera platform is corrected for (otherwise we will see jumps in the positions of our detected lakes when the camera platform moves). 

We will use PyTrx's Homography object to track static features in the image and identify camera platform motion. We can subsequently use these movements to create a homography model and correct for this motion.
 
         
.. code-block:: python
   
   # Set homography parameters
   # Homography tracking method - sparse or dense tracking
   hgmethod='sparse'
   
   # Pt seeding parameters (max. pts, quality, min. distance               
   hgseed = [50000, 0.1, 5.0]      
   
   # Tracking parameters (window size, backtracking threshold, min. num of pts)
   hgtrack = [(25,25), 1.0, 4]  


   # Set up Homography object
   homog = Homography(camimgs, cameraenvironment, caminvmask, 
                      calibFlag=True, band='L', equal=True)

   # Calculate homography
   hg = homog.calcHomographies([hgmethod, hgseed, hgtrack])
   
   # Compile homography matrices from output        
   homogmatrix = [item[0] for item in hg]


Now we have our homography model, we can look at detecting lakes in the images. As we want the lake features as polygons, we will use PyTrx's Area object to automatically identify these features. First, we will initialise the object with our images, camera environment object, homography model, and three flags denoting whether the images should be corrected for lens distortion, which pixel band should be used in the detection process (red, green, blue or grayscale), and whether the pixels in the images should be adjusted with histogram equalisation.

Lakes will be identified based on the difference in pixel intensities between the water and adjacent ice. The time-lapse images will also be enhanced to aid in identifying them.


.. code-block:: python

   # Set parameters to initialise Area object
   # Detect with corrected or uncorrected images   
   calibFlag = True           
   
   # Pixel band to carry forward ('R', 'G', 'B' or 'L')
   imband = 'R'               
   
   # Images with histogram equalisation or not
   equal = True               
     
   # Set up Area object
   lakes = Area(camimgs, cameraenvironment, homogmatrix, calibFlag, imband, equal)


We can set a number of detection parameters in our Area object to aid in the automated identification of lakes, including image enhancing, image masking, and setting athreshold for the number of detected polygons that will be retained. 


.. code-block:: python
   
   # Set image enhancement parameters
   diff = 'light'   
   phi = 50     
   theta = 20        
   lakes.setEnhance(diff, phi, theta)

   # Set mask and image number with maximum area of interest 
   maxim = 0                 
   lakes.setMax(camamask,maxim)                   

   # Set polygon threshold (i.e. number of polygons kept)
   threshold = 5             
   lakes.setThreshold(threshold)
   
   
Following this, we will use a pre-defined pixel value range to detect lakes from the images. In this case, pixel values between 1 and 8 will be classified as water. The calcAutoAreas function will then be executed to detect water through all the time-lapse images in our sequence.


.. code-block:: python

   # Set pixel colour range, from which extents will be distinguished
   maxcol = 8                 
   mincol = 1  
   lakes.setColourrange(maxcol, mincol) 


The calcAutoAreas function will then be executed to detect water through all the time-lapse images in our sequence. The colour and verify flags can be toggled for defining the pixel colour range in each image and verifying each identified polygon manually, respectively.


.. code-block:: python

   # Calculate real areas
   areas = lakes.calcAutoAreas(colour=False, verify=False)


Now we have our detected lakes, we can plot them in both the image plane (u,v) and real-world coordinates (x,y,z) to see how they look using the plotting functions in the Utilities module.


.. code-block:: python

   # Retrieve images and distortion parameters for plotting
   imgset=lakes._imageSet                                             
   cameraMatrix=cameraenvironment.getCamMatrixCV2()                   
   distortP=cameraenvironment.getDistortCoeffsCV2()                   

   # Retrieve DEM array for plotting
   dem = cameraenvironment.getDEM() 
   
   # Retrieve uv and xyz coordinates of lakes
   uvpts = [item[1][1] for item in areas]                            
   xyzpts = [item[0][1] for item in areas] 
                              
   # Show image extents and dems 
   for i in range(len(areas)):
       plotAreaPX(uvpts[i], 
                  imgset[i].getImageCorr(cameraMatrix, distortP), 
                  show=True, save=None)  
       plotAreaXYZ(xyzpts[i], dem, show=True, save=None)
    
    
And finally, we can export our identified lakes as both text files and shapefiles using the writing functions in the FileHandler module (we suggest modifying the output file paths to your desired workspace).


.. code-block:: python

   # Get all image names for reference
   imn = lakes.getImageNames()

   # Get pixel and sq m lake areas 
   uvareas = [item[1][0] for item in areas] 
   xyzareas = [item[0][0] for item in areas]  


   # Write areas to text file
   FileHandler.writeAreaFile(uvareas, xyzareas, imn, 'areas.csv')
   
   # Write area coordinates to text file
   FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                               'uvcoords.txt', 'xyzcoords.txt')
   
   # Write lakes to shapefiles with WGS84 projection
   proj = 32633                                                               
   FileHandler.writeAreaSHP(xyzpts, imn, 'shpfiles', proj)   
  

Manual detection of plume footprints
------------------------------------

In this example, we will derive meltwater plume footprints from the front of Kronebreen, Svalbard, for a small subset of the 2014 melt season. This example can be found in `KR_manualarea.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_manualarea.py>`_.

We will manually delineate meltwater plume footprints from corrected time-lapse images to derive surface areas at sea level. In this example, we will use images from `Kronebreen camera 1 <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples/images/KR1_2014_subset>`_ and the `KR1 camera environment data <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_KR1_2014.txt>`_.

First, we need to import the PyTrx packages that we are going to use.


.. code-block:: python

   from PyTrx.CamEnv import CamEnv
   from PyTrx.Area import Area
   from PyTrx.Velocity import Homography
   import PyTrx.FileHandler as FileHandler

And then define the filepaths to our camera information (for creating our camera environment), our image mask (for identifying camera motion), and our time-lapse images.


.. code-block:: python 

  
   # Define camera info filepath
   camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR1_2014.txt'
   
   # Define image mask filepath
   caminvmask = '../Examples/camenv_data/invmasks/KR1_2014_inv.jpg'
   
   # Define folder path with time-lapse images
   camimgs = '../Examples/images/KR1_2014_subset/*.JPG'


Next we need to create our camera environment using PyTrx's CamEnv object. As we do not know the camera pose (yaw, pitch, roll - YPR), we can estimate this using PyTrx's optimisation routines. The optimisation routine uses the difference between the u,v ground control points and the reprojected x,y,z ground control points to adjust and refine the camera model.
  

.. code-block:: python

   # Define camera environment
   cameraenvironment = CamEnv(camdata)

   # Optimise camera YPR
   cameraenvironment.optimiseCamEnv('YPR')


To correct for motion in the camera platform, we will use PyTrx's Homography object (found in the Velocity module) to track static features and identify camera motion. From this motion, the Homography object creates a series of homography matrices (also known as a homography model) to co-register the images to one another.


.. code-block:: python

   # Set up Homography object
   homog = Homography(camimgs, cameraenvironment, 
                      caminvmask, calibFlag=True, 
                      band='L', equal=True)

   # Set homography parameters
   hmethod='sparse'                #Method
   hgmax=50000                     #Max number of seeding pts
   hgqual=0.1                      #Seeding corner quality
   hgmind=5.0                      #Min seeding pt distance
   hgwinsize=(25,25)               #Tracking window size
   hgback=1.0                      #Back-tracking threshold
   hgminf=4                        #Min seeded pts to track
   
   # Calculate homography
   hg = homog.calcHomographies([hmethod, [hgmax, hgqual, hgmind], [hgwinsize, hgback, hgminf]])
   
   # Extract homography model        
   homogmatrix = [item[0] for item in hg] 


Now we can initialise our Area object and manually delineate the plume footprints using the calcManualAreas function. This should bring up a pop-up window for each image, where you can click around each plume footprint and press 'enter' to move to the next.
   

.. code-block:: python

   # Set up Area object
   plumes = Area(camimgs, cameraenvironment, 
                 homogmatrix, calibFlag=True, 
                 imband='R', equal=True)

   # Calculate real areas
   areas = plumes.calcManualAreas()


We will save our manually-delineated plume footprints as area and coordinate text files using the export functions in the FileHandler module.


.. code-block:: python

   # Retrieve plume areas
   uvareas = [item[1][0] for item in areas]   
   xyzareas = [item[0][0] for item in areas]
   
   # Retrieve image names
   imn=plumes.getImageNames()
      
   # Write areas to text file
   FileHandler.writeAreaFile(uvareas, xyzareas, imn, 'areas.csv')
   
   # Retrieve coordinates of plume extents
   xyzpts = [item[0][1] for item in areas]
   uvpts = [item[1][1] for item in areas]   
   
   # Write coordinates to text file
   FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                               'uvcoords.txt', 
                               'xyzcoords.txt')


And we will also export the plume footprints as shapefiles, using the same projection as our inputted DEM. These shapefiles can be used in subsequent analysis and imported into GIS software for viewing.


.. code-block:: python

   # Define projection
   proj = 32633
   
   # Write to shapefile 
   FileHandler.writeAreaSHP(xyzpts, imn, 'shpfiles', proj) 

 
And finally, we can plot the plume footprints onto the time-lapse images for viewing purposes. Here is an example to plot the footprints onto RGB versions of the images, using a workflow using opencv and matplotlib.

   
.. code-block:: python

   # Import packages
   import glob,cv2
   import matplotlib.image as mpimg
   import matplotlib.pyplot as plt
   
   # Get original images in directory
   ims = sorted(glob.glob(camimgs))

   # Get camera correction variables
   cameraMatrix=cameraenvironment.getCamMatrixCV2()
   distortP=cameraenvironment.getDistortCoeffsCV2()
   newMat, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortP, 
                                               (5184,3456),1,(5184,3456))    

   # Get corresponding xy pixel areas and images  
   count=1
   for p,i in zip(uvpts,ims):
       x=[]
       y=[]
       for ps in p[0]:    
           x.append(ps[0])
           y.append(ps[1])
  
       # Read image and undistort 
       im1=mpimg.imread(i)
       im1 = cv2.undistort(im1, cameraMatrix, distortP, 
                           newCameraMatrix=newMat)
       
    # Plot image
    plt.figure(figsize=(20,10))             
    plt.imshow(im1)              
    plt.axis([0,5184,3456,0])  
    plt.xticks([])                          
    plt.yticks([])
    
    # Plot pixel area 
    plt.plot(x,y,'#fff544',linewidth=2)
    
    # Save image to file            
    plt.savefig('plumeplotted' + str(count) + '.JPG', dpi=300)
    plt.show()
    count=count+1
    

Manual detection of glacier terminus profiles
---------------------------------------------

Here, we will delineate glacier terminus profiles (as line features) from a small subset of time-lapse images from Tunabreen, Svalbard, during the 2014 melt season. This example can be found in `TU_manualline.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/TU_manualline.py>`_.

We will manually delineate terminus profiles from corrected time-lapse images to derive a sequence of positions representing glacier retreat. In this example, we will use images from `Tunabreen camera 1 <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples/images/TU1_2015_subset>`_ and the associated `camera environment data <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_TU1_2015.txt>`_.

First, we need to import the PyTrx packages that we are going to use.


.. code-block:: python

   from PyTrx.CamEnv import CamEnv
   from PyTrx.Line import Line
   from PyTrx.Velocity import Homography
   import PyTrx.FileHandler as FileHandler
   from PyTrx.Utilities import plotLinePx, plotLineXYZ
   
   
And define the paths to our camera information, image mask (for tracking static points and correcting for camera platform motion), and time-lapse images.


.. code-block:: python

   # Define data input directories
   camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU1_2015.txt'
   invmask = '../Examples/camenv_data/invmasks/TU1_2015_inv.jpg'  
   camimgs = '../Examples/images/TU1_2015_subset/*.JPG'


Firstly, we can initialise a CamEnv object which represents our camera environment, using our camera information .txt file.

 
.. code-block:: python

   # Create camera environment
   cam = CamEnv(camdata)
   

In this example, the camera pose (yaw, pitch, roll - YPR) is unknown as it is difficult to measure this in the field. We can determine the YPR using PyTrx's optimisation routine.


.. code-block:: python

   # Define what parameters to optimise 
   optflag = 'YPR'              
   
   # Define optimisation method
   optmethod = 'trf'               

   # Optimise camera environment
   cam.optimiseCamEnv(optflag, optmethod, show=False)


To account for motion in the camera platform, we will track static features in the image (in the areas defined by our image mask) using PyTrx's Homography object. Here, we track selected corner features in the image to derive a homography matrix for each image pair.


.. code-block:: python

   # Set homography parameters
   hmethod='sparse'                #Seeding method
   hgwinsize=(25,25)               #Tracking window size
   hgback=1.0                      #Back-tracking threshold
   hgmax=50000                     #Max num of pts to seed
   hgqual=0.1                      #Corner quality for seeding
   hgmind=5.0                      #Min distance between seeded pts
   hgminf=4                        #Min num seeded pts to track

   # Set up Homography object
   homog = Homography(camimgs, cam, invmask, calibFlag=True, band='L', 
                      equal=True)

   # Calculate homography
   hg = homog.calcHomographies([hmethod, [hgmax, hgqual, hgmind], 
                               [hgwinsize, hgback, hgminf]])    
      
   # Extract homography matrices
   homogmatrix = [item[0] for item in hg] 


Now we can manually delineate our terminus profiles from each time-lapse image using the Line object in PyTrx. First, we initialise the object, and then use the calcManualLines() function to start the manual delineations. For each image, an interactive window will open, where you can click points to trace the terminus, and press 'enter' when you are finished to prompt the next image to load.


.. code-block:: python

   # Set up line object
   terminus = Line(camimgs, cam, homogmatrix)


   # Manually define terminus lines
   lines = terminus.calcManualLines()


PyTrx's FileHandler module can be used to export all findings to file. Here, we will write out two files containing line lengths and coordinates, shapefiles for each line geometry, and information about the homography to file.


.. code-block:: python

   # Get image names
   imn=terminus.getImageNames()

   # Get uv and xyz lines
   pxlines = [item[1][0] for item in lines]
   xyzlines = [item[0][0] for item in lines]

   # Write line data to .csv file
   FileHandler.writeLineFile(pxlines, xyzlines, imn, 'lines.csv')

   # Write line coordinates to txt file
   FileHandler.writeLineCoords(pxcoords, xyzcoords, imn, 
                           'uvcoord.txt', 'xyzcoords.txt')

   # Get uv and xyz line coordinates
   pxcoords = [item[1][1] for item in lines]
   xyzcoords = [item[0][1] for item in lines]


   # Write shapefiles from line data
   projection=32633  
   FileHandler.writeLineSHP(xyzcoords, imn, 'shapefiles', projection)

   # Write homography data to .csv file
   FileHandler.writeHomogFile(hg, imn, 'homography.csv')


Lastly, we can view our delineated terminus profiles in both the image and the DEM space using the plotting function in PyTrx's FileHandler module.


.. code-block:: python

   # Get dem array
   dem = cam.getDEM()
   
   # Get image sequence as arrays
   imgset=terminus._imageSet
   
   # Retrieve image correction coefficients
   cameraMatrix=cam.getCamMatrixCV2()
   distortP=cam.getDistortCoeffsCV2()

   # Plot uv lines on image 
   for i in range(len(pxcoords)):

      # Plot lines in image plane and as XYZ lines 
       plotLinePX(pxcoords[i], 
                  imgset[i].getImageCorr(cameraMatrix, distortP), 
                  show=True, 
                  save='uv_'+str(imn[i]))
       # Plot xyz lines on DEM 
       plotLineXYZ(xyzcoords[i], 
                   dem, 
                   show=True,  
                   save='xyz_'+str(imn[i]))
                   
                   
Georectification of glacier calving event point locations
---------------------------------------------------------

Here, we will georectify some pre-defined points that denote the locations of glacier calving events at Tunabreen, Svalbard, captured from high-frequency time-lapse images. One point represents a calving event identified in the image plane, which will be imported and georectified to x,y,z coordinates using the georectification functions in PyTrx. The x,y,z coordinates will then be plotting onto the DEM, and exported to shapefile.  

This example can be found in `TU_ptsgeorectify.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/TU_ptsgeorectify.py>`_, using the `Tunabreen camera 1 environment data file <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_TU1_2015.txt>`_.

First, we need to import the PyTrx functions that we are going to use along with some other packages (for GIS, data manipulation and plotting), and define the file paths to our camera environment information and point data.


.. code-block:: python

   # Import PyTrx CamEnv functions   
   from PyTrx.CamEnv import CamEnv, setProjection, projectUV
   
   # Import other packages to use
   import matplotlib.pyplot as plt
   import osgeo.ogr as ogr
   import osgeo.osr as osr
   import numpy as np
      
   # Define camera environment file path
   tu1camenv='../Examples/camenv_data/camenvs/CameraEnvironmentData_TU1_2015.txt'
   
   # Define calving pt data file path
   tu1calving = '../Examples/results/ptsgeorectify/TU1_calving_xy.csv'
   

Next, we will load our point data (i.e. calving event locations)   


.. code-block:: python

   # Open file
   f=open(tu1calving,'r')                            
   
   # Read header line
   header=f.readline()  
  
   # Create empty variables to populate                                
   time=[]
   region=[]
   style=[]
   tu1_xy=[]

   # Read each line from file
   for line in f.readlines():
      
      # Split line into variables 
      temp=line.split(',')    
      
      # Extract variables
      time.append(float(temp[0].rstrip()))                 
      region.append(temp[1].rstrip())                             
      style.append(temp[2].rstrip())  
      tu1_xy.append([float(temp[3].rstrip()), float(temp[4].rstrip())])        

   print(f'{len(tu1_xy)} locations for calving events detected')
   
   # Change pt coordinate list to array
   tu1_xy = np.array(tu1_xy)


Next, we will create a CamEnv object to hold all the information about our camera. We will initialise the object with our camera environment file, which includes paths to the camera calibration, ground control point positions, reference image and DEM, along with the position of our camera and its pose represented along three axes (yaw, pitch, roll - YPR).


.. code-block:: python

   # Define camera environment
   tu1cam = CamEnv(tu1camenv)
   

Now we have our camera environment, we need to model how the three-dimensional world (represented by the DEM) is translated to the two-dimensional image plane (represented by our reference image). We will use the setProjection function in PyTrx's CamEnv module in order to do this.


.. code-block:: python

   # Get DEM from camera environment
   demobj = tu1cam.getDEM() 

   # Get inverse projection variables through camera info               
   invprojvars = setProjection(demobj, tu1cam._camloc, tu1cam._camDirection, 
                               tu1cam._radCorr, tu1cam._tanCorr, tu1cam._focLen, 
                               tu1cam._camCen, tu1cam._refImage)
        

With our inverse projection model, we can translate the calving event locations defined in the image plane to x,y,z coordinates with the project UV function.

        
 .. code-block:: python

   # Inverse project uv coodinates to xyz coordinates
   tu1_xyz = projectUV(tu1_xy, invprojvars)


To view our reprojected x,y,z points, we can plot them using the plotting functionality in matplotlib. We will plot the points over our DEM.


.. code-block:: python

   # Retrieve DEM extent and elevation array
   demextent = demobj.getExtent()
   dem = demobj.getZ()
   
   # Get camera position (xyz) for plotting
   post = tu1cam._camloc            
   
   # Plot DEM and camera location
   fig,(ax1) = plt.subplots(1, figsize=(15,15))
   fig.canvas.set_window_title('TU1 calving event locations')
   ax1.locator_params(axis = 'x', nbins=8)
   ax1.tick_params(axis='both', which='major', labelsize=0)
   ax1.imshow(dem, origin='lower', extent=demextent, cmap='gray')
   ax1.axis([demextent[0], demextent[1], demextent[2], demextent[3]])
   cloc = ax1.scatter(post[0], post[1], c='g', s=10, label='Camera location')
           
   # Plot calving locations on DEM
   xr = [pt[0] for pt in tu1_xyz]
   yr = [pt[1] for pt in tu1_xyz]
   ax1.scatter(xr, yr, c='r',s=10)   

   # Save and show plot
   plt.savefig('TU1_calving_xyz.JPG', dpi=300) 
   plt.show() 


And finally we will export the inverse projected x,y,z point coordinates to a shapefile using the osgeo modules ogr and osr.


.. code-block:: python

   # Get ESRI shapefile driver            
   driver = ogr.GetDriverByName('ESRI Shapefile' )

   # Create data source
   shp = 'tu1_calving.shp'   
   ds = driver.CreateDataSource(shp)
   if ds is None:
       print(f'Could not create file {shp}')
     
   # Set WGS84 projection
   proj = osr.SpatialReference()
   proj.ImportFromEPSG(32633)

   # Create layer in data source
   layer = ds.CreateLayer('tu1_calving', proj, ogr.wkbPoint)
  
  
   # Add ID and time attributes to layer
   layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))          
   layer.CreateField(ogr.FieldDefn('time', ogr.OFTReal))       
   
   # Add terminus region attribute
   field_region = ogr.FieldDefn('region', ogr.OFTString)        
   field_region.SetWidth(8)    
   layer.CreateField(field_region)                           
   
   # Add calving style attribute
   field_style = ogr.FieldDefn('style', ogr.OFTString)        
   field_style.SetWidth(10)    
   layer.CreateField(field_style)                  
 
  
   # Create point features with data attributes in layer           
   for a,b,c,d in zip(tu1_xyz, time, region, style):
       count=1

       # Create feature    
       feature = ogr.Feature(layer.GetLayerDefn())

       # Write feature attributes      
       feature.SetField('id', count)
       feature.SetField('time', b)
       feature.SetField('region', c) 
       feature.SetField('style', d)         

       # Create feature geometry
       wkt = "POINT(%f %f)" %  (float(a[0]) , float(a[1]))
       point = ogr.CreateGeometryFromWkt(wkt)
       feature.SetGeometry(point)
       
       # Compile feature
       layer.CreateFeature(feature)

       # Close feature
       feature.Destroy()                       
       count=count+1

   # Close layer    
   ds.Destroy()

              
Sparse feature-tracking to derive glacier flow
----------------------------------------------

In this example, we will calculate glacier flow velocities from Kronebreen, Svalbard, using PyTrx's sparse feature-tracking method. The sparse feature-tracking method using corner feature detection to identify coherent features on the glacier surface, and then tracks them between image pairs using Optical Flow. 

We will derive glacier velocities from a subset of time-lapse images from the 2014 melt season, which can be found in the `PyTrx GitHub repository <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples/images/KR2_2014_subset>`_, using the `Kronebreen camera 2 environment data file <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_KR2_2014.txt>`_. This example can be found in `KR_velocity1.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_velocity1.py>`_.

Let's firstly import the PyTrx modules we need.


.. code-block:: python

   from PyTrx.CamEnv import CamEnv
   from PyTrx.Velocity import Velocity, Homography
   from PyTrx.FileHandler import writeHomogFile, writeVeloFile, \
        writeVeloSHP, writeCalibFile
   from PyTrx.Utilities import plotVeloPX, plotVeloXYZ, \
        interpolateHelper, plotInterpolate


And then define the file paths to our camera information, our time-lapse images, and the masks we will use to identify the regions of the image we want to use for deriving glacier flow velocities and tracking static features. 


.. code-block:: python
  
   # Camera environment file path
   camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_KR2_2014.txt'
   
   # Mask for velocity feature-tracking
   camvmask = '../Examples/camenv_data/masks/KR2_2014_vmask.jpg'
   
   # Inverse mask for image registration
   caminvmask = '../Examples/camenv_data/invmasks/KR2_2014_inv.jpg'
   
   # Time-lapse images
   camimgs = '../Examples/images/KR2_2014_subset/*.JPG'


We will construct a CamEnv object using our camera environment file, which will hold all information about the translation of our images to x,y,z space (represented by our DEM). We will optimise our camera environment, using our pre-defined ground control points to refine the model and estimate the camera pose (i.e. yaw, pitch, roll - YPR)


.. code-block:: python

   # Define camera environment
   cameraenvironment = CamEnv(camdata)

   # Optimise camera environment to refine camera pose
   cameraenvironment.optimiseCamEnv('YPR')


We can check our camera environment parameters using a reporter and various plotting functions.


.. code-block:: python

   # Report camera environment parameters
   cameraenvironment.reportCamData()
   
   # Show ground control points
   cameraenvironment.showGCPs()
   
   # Show image principal point
   cameraenvironment.showPrincipalPoint()
   
   # Show ground control point residuals
   cameraenvironment.showResiduals()


Next we will calculate the homography model using PyTrx's Homography object. This represents correction for motion in the camera platform which, if uncorrected, can introduce false motion into our velocity measurements. We can account for this using our homography model to co-register our time-lapse images.


.. code-block:: python

   # Set homography parameters
   hmethod='sparse'                #Method
   hgwinsize=(25,25)               #Tracking window size
   hgback=1.0                      #Back-tracking threshold
   hgmax=50000                     #Maximum number of points to seed
   hgqual=0.1                      #Corner quality for seeding 
   hgmind=5.0                      #Minimum distance between seeded points
   hgminf=4                        #Minimum number of seeded points to track

   # Set up Homography object
   homog = Homography(camimgs, cameraenvironment, caminvmask, calibFlag=True, 
                      band='L', equal=True)

   # Calculate homography
   hgout = homog.calcHomographies([hmethod, [hgmax, hgqual, hgmind], [hgwinsize, 
                                hgback, hgminf]])


Now we can look at measuring the flow of the glacier using the feature-tracking functionality in PyTrx's Velocity object. There are a number of parameters we can set to adjust our tracking conditions

    
.. code-block:: python

   # Set image conditions
   calibration = True 		    # Correct images for distortion?
   iband = 'L'                     # Image band to track with (R/G/B/L)
   eq = True  			    # Images with histogram equalisation?
   
   # Set up Velocity object
   velo=Velocity(camimgs, cameraenvironment, hgout, camvmask, calibFlag=True, 
                 band='L', equal=True) 
                                  
   # Set velocity parameters
   vmethod = 'sparse'              # Method
   vwinsize = (25,25)              # Tracking window size
   bk = 1.0                        # Back-tracking threshold  
   mpt = 50000                     # Maximum number of points to seed
   ql = 0.1                        # Corner quality for seeding
   mdis = 5.0                      # Minimum distance between seeded points
   mfeat = 4                       # Minimum number of seeded points to track

   # Calculate glacier flow velocity
   velocities = velo.calcVelocities([vmethod, [mpt, ql, mdis], [vwinsize, bk, 
                                    mfeat]])                                   

To export our results, we can write out our intrinsic camera matrix (which can be useful when you have optimised the intrinsic camera parameters of the camera environment) and calculated homography using the exporting functions in PyTrx's FileHandler module.


.. code-block:: python
  
   # Write out camera calibration info to .txt file
   matrix, tancorr, radcorr = cameraenvironment.getCalibdata()
   writeCalibFile(matrix, tancorr, radcorr, 'KR2_2014_1.txt')
      
   # Write homography data to .csv file
   imn = velo.getImageNames()
   writeHomogFile(hgout, imn, 'homography.csv')


And then we can export our calculated velocities to .csv file and .shp shapefiles for plotting and further analysis 


.. code-block:: python

   # Fetch uv and xyz velocities 
   xyzvel=[item[0][0] for item in velocities]
   uvvel=[item[1][0] for item in velocities]
   
   # Write out velocity data to .csv file
   writeVeloFile(xyzvel, uvvel, hgout, imn, 'velo_output.csv') 

   # Fetch xyz pt coordinates and tracking errors
   xyz0=[item[0][1] for item in velocities]
   xyzerr=[item[0][3] for item in velocities]

   # Write points to shp file with EPSG:32633 projection
   proj = 32633                            
   writeVeloSHP(xyzvel, xyzerr, xyz0, imn, 'shpfiles', proj)   


If we want to view the results, we can retrieve all of our tracked points (in both the images and x,y,z coordinates) and plot them over the top of our images and DEM.

  
.. code-block:: python

   # Get calibration coefficients for plotting corrected images
   cameraMatrix=cameraenvironment.getCamMatrixCV2()
   distortP=cameraenvironment.getDistortCoeffsCV2() 

   # Get images for overlaying uv pts
   imgset=velo._imageSet
      
   # Get DEM array for overlaying xyz pts
   dem=cameraenvironment.getDEM()
   
   # Get uv seeded and tracked point positions
   uv0=[item[1][1] for item in velocities] 
   uv1corr=[item[1][3] for item in velocities]
      
   # Get xyz seeded and tracked point positions
   xyz0=[item[0][1] for item in velocities]
   xyz1=[item[0][2] for item in velocities]

   # Cycle through data from image pairs   
   for i in range(len(imn)-1):
 
       # Get image name and print
       print('\nVisualising data for ' + str(imn[i]))

       # Plot uv velocity points on image plane   
       print('Plotting image plane output')
       plotVeloPX(uvvel[i], uv0[i], uv1corr[i], 
                  imgset[i].getImageCorr(cameraMatrix, distortP), 
                  show=True, save='uv_'+imn[i])


       # Plot xyz velocity points on dem  
       print('Plotting XYZ output')
       plotVeloXYZ(xyzvel[i], xyz0[i], xyz1[i], 
                   dem, show=True, save='xyz_'+imn[i])
    
                
       # Plot interpolation map with linear interpolation
       print('Plotting interpolation map')
       grid, pointsextent = interpolateHelper(xyzvel[i], xyz0[i], xyz1[i], 'linear')
       plotInterpolate(grid, pointsextent, dem, show=True, 
                       save='interp_'+imn[i])                        


Additionally, we can export our velocities as gridded ASCII files. These files are recognised by many mapping software, such as ArcGIS and QGIS, and can be imported to create raster surfaces.


.. code-block:: python

   # import numpy for grid operations
   import numpy as np

   # Cycle through velocity data from image pairs   
   for i in range(velo.getLength()-1): 
    
       # Change all the nans to -999.999 and flip the y axis
       grid[np.isnan(grid)] = -999.999     
       grid = np.flipud(grid)  
    
       # Open new file with write permissions
       imn=velo._imageSet[i].getImageName()
       afile = open(imn + '_interpmap.txt','w')
    
    # Make a list for each raster header variable, with the label and value
    col = ['ncols', str(grid.shape[1])]
    row = ['nrows', str(grid.shape[0])]
    x = ['xllcorner', str(pointsextent[0])]
    y = ['yllcorner', str(pointsextent[2])]
    cell = ['cellsize', str(10.)]
    nd = ['NODATA_value', str(-999.999)]
    
    # Write each header line on a new line of the file
    header = [col,row,x,y,cell,nd]       
    for i in header:
        afile.write(' '.join(i) + '\n')
    
    # Iterate through each row and column value
    for i in range(grid.shape[0]): 
        for j in range(grid.shape[1]):
            
            # Write each data value to the row, separated by spaces
            afile.write(str(grid[i,j]) + ' ')
            
        # New line at end of row
        afile.write('\n')
    
    # Close file
    afile.close() 


Dense feature-tracking to derive glacier flow
---------------------------------------------

PyTrx's dense feature-tracking utilises traditional cross-correlation template matching to track a regular grid of points between an image pair. In this example, we calculate glacier flow velocities from Kronebreen, Svalbard, using a similar workflow to the sparse feature-tracking workflow shown previously. 

This workflow can be found in `KR_velocity2.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_velocity2.py>`_, and a merged workflow using both sparse and dense feature-tracking can be found in `KR_velocity2.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_velocity2.py>`_. The main difference in the merged workflow is that velocities are processed using the stand-alone functions provided in PyTrx, rather than handled by PyTrx's class objects. This provides the user with a script that is more flexible and adaptable.

The main difference in the dense feature-tracking workflow (compared to the sparse workflow) is in the input variables to the Velocity object's calcVelocities function. When the tracking method is set to 'dense' then the following variables can be defined -- grid spacing, template and search window size, template matching method, threshold correlation, and minimum number of tracked points.

.. code-block:: python 

   # Set up Velocity object
   velo=Velocity(camimgs, cameraenvironment, hgout, camvmask, calibFlag=True, 
                 band='L', equal=True) 

   # Set velocity tracking parameters
   vmethod = 'dense'                   # Method
   vgrid = [50,50]                     # Dense matching grid distance
   vtemplate = 10                      # Template size
   vsearch = 50                        # Search window size
   vmethod = 'cv2.TM_CCORR_NORMED'     # Method for template matching
   vthres = 0.8                        # Threshold average template correlation
   vminf = 5                           # Minimum number of tracked points
   
   # Calculate dense velocities
   velocities = velo.calcVelocities([vmethod, vgrid, [vmethod, vtemplate, vsearch, 
                                    vthres, vminf]])   
                                 
