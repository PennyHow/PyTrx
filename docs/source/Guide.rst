PyTrx Guide 
===============

Detailed documentation is included in the 8 scripts that make up PyTrx. Each script contains classes and functions for handling each aspect needed for photogrammetric processing.

For beginners in programming, it is advised to look at the example applications provided in the `GitHub repository <https://github.com/PennyHow/PyTrx>`_ and adapt them accordingly for your own use. For experienced programmers, get stuck in. Feel free to contact us if you run into major problems or have constructive comments that will help us further PyTrx and its capabilities. We will not respond to minor troubleshooting or unconstructive comments.


CamEnv
------

Handles the associated data with the camera environment. The GCPs class handles the Ground Control Points (GCPs) and their correspondence to the associated DEM and CamImage object. The CamCalib class handles information concerning the camera calibration, i.e. the intrinsic camera matrix and lens distortion coefficients. This class contains functionality for reading in calibration files from .txt and .mat formats.
The CamEnv compiles all the information about the camera environment from the GCPs and CamCalib classes, and also contains information about the camera object (pose and location). This is also where georectification functionality is held, with functions for projection and inverse projection. The class is initialised using a .txt file containing file path directories to all the associated data files.


DEM
---

Handles the DEM data. This currently supports .mat and .tif file types. The ExplicitRaster class represents a DEM as a numeric raster with explicit XY cell referencing in each grid cell. The class includes functions for densification, calculating viewsheds, and incorporates unbound functions that import a DEM file from .mat and .tif formats.


FileHandler
-----------

This module contains a set of functions for reading in data from files (such as image data and calibration information) and writing out data.


Images
------

Handles the image data, and the image sequence. The CamImage class holds information about a singular image and contains functionality for importing image data from file and passing specific image bands forward for subsequent processing. The ImageSequence class holds information about an image sequence, i.e. a collection of CamImage objects, from which specific images and image pairs can be called.


Velocity
--------

Calculates velocities and homography. This can either be achieved through the Velocity class for processing velocities and homography through a series of images, or using the functions provided within the script for processing velocities and homography between an image pair.


Area
----

Automated and manual detection of surface areas from imagery (e.g. supraglacial lakes, meltwater plume surface extent). This can either be achieved through the Area class for defining areas of interest through a series of images, or using the functions provided within the script for defining areas of interest in a single image.


Line
----

Manual detection of line features from imagery (e.g. glacier terminus position). This can either be achieved through the Line class for defining line features through a series of images, or using the functions provided within the script for defining line features in a single image.


Utilities
---------

This module contains a set of functions for plotting and interpolating data.
