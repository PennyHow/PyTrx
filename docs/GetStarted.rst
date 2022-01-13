Get Started
===========

PyTrx comes with working examples to get started with. These scripts are available in the PyTrx repository 
`here <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples>`_. These examples are for applications in glaciology, which can be adapted and used. We hope these are especially useful for beginners in coding.


Automated detection of supraglacial lakes
-----------------------------------------
In this example, we will derived changes in surface area of supraglacial lakes captured from Kronebreen, Svalbard, for a small subset of the 2014 melt season. This example can be found in `KR_autoarea.py <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/KR_autoarea.py>`_.

We will automatically detect water on the glacier based on differences in pixel intensity and corrected for image distortion; using images from `Kronebreen camera 3 <https://github.com/PennyHow/PyTrx/tree/master/PyTrx/Examples/images/KR3_2014_subset>`_ and the associated `camera environment data <https://github.com/PennyHow/PyTrx/blob/master/PyTrx/Examples/camenv_data/camenvs/CameraEnvironmentData_KR3_2014.txt>`_.

First, we need to import os (for file checking and creation) and the PyTrx packages that we are going to use.


.. code-block:: python

   import os
   
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

   # Define data output directory
   destination = '../Examples/results/KR_autoarea/'
   if not os.path.exists(destination):
       os.makedirs(destination)


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
   maxim = 0                 t 
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
    
And finally, we can export our identified lakes as both text files and shapefiles using the writing functions in the FileHandler module.


.. code-block:: python

   # Get all image names for reference
   imn = lakes.getImageNames()

   # Get pixel and sq m lake areas 
   uvareas = [item[1][0] for item in areas] 
   xyzareas = [item[0][0] for item in areas]  


   # Write areas to text file
   FileHandler.writeAreaFile(uvareas, xyzareas, imn, destination+'areas.csv')
   
   # Write area coordinates to text file
   FileHandler.writeAreaCoords(uvpts, xyzpts, imn, 
                               destination+'uvcoords.txt', 
                               destination+'xyzcoords.txt')
   
   # Write lakes to shapefiles with WGS84 projection
   proj = 32633                                                               
   FileHandler.writeAreaSHP(xyzpts, imn, destination+'shpfiles/', proj)   
  

Manual detection of plume footprints
------------------------------------

Manual detection of meltwater plume extents, *KR_manualarea.py*

Example driver for calculating meltwater plume surface extent at Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs manual detection of meltwater plumes through sequential images of the glacier to derive surface areas which have been corrected for image distortion. Images are imported from those found in the 'KR1_2014_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_KR1_2014.txt'


Manual detection of terminus profiles
-------------------------------------

Manual detection of glacier terminus profiles, *TU_manualline.py*

Example driver for calculating terminus profiles (as line features) at Tunabreen, Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. This script performs manual detection of terminus position through sequential images of the glacier to derive line profiles which have been corrected for image distortion. Images are imported from those found in the 'TU2_2015_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_TU2_2014.txt'


Georectification of glacier calving event point locations
---------------------------------------------------------

Georectification of calving event point locations, *TU_ptsgeorectify.py*

Example driver which demonstrates the capabilities of the georectification functions provided in PyTrx (which are based upon those available in ImGRAFT). Pre-defined points are imported which denote calving events at Tunabreen, Svalbard, that have been distinguished in the image plane. These are subsequently projected to xyz locations using the georectification functions in PyTrx. The xyz locations are plotted onto the DEM, with the colour of each point denoting the style of calving in that particular instance. The xyz locations are finally exported as a text file (.txt) and as a shape file (.shp).

Sparse feature-tracking
-----------------------

Glacier velocities derived through feature-tracking of sparse points, *KR_velocity1.py*

Example driver for deriving sparse velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.


Dense feature-tracking
----------------------

Glacier velocities derived through feature-tracking of dense grid, *KR_velocity2.py*

Example driver for deriving dense velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.


Sparse and dense feature-tracking
----------------------------------

Alternative script for glacier velocity feature-tracking with both the sparse and dense methods, *KR_velocity3.py*

Extended example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. This script produces the same outputs as *KR_velocity1.py* and *KR_velocity2.py*. The difference is that velocities are processed using the stand-alone functions provided in PyTrx, rather than handled by PyTrx's class objects. This provides the user with a script that is more flexible and adaptable.



