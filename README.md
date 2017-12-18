# PyTrx v1.1
PyTrx is an object-oriented toolbox created for the purpose of calculating real-world measurements from oblique images and time-lapse image series. Its primary purpose is to obtain velocities, surface areas, and volume losses from imagery of glacial environments.<br>

These scripts have not yet been published, but PyTrx has been used for data processing in the following publications: <br>
How et al. (2017) Rapidly changing subglacial hydrological pathways at a tidewater glacier revealed through simultaneous observations of water pressure, supraglacial lakes, meltwater plumes and surface velocities. <i>The Cryosphere</i> 11, 2691-2710, <a href="https://doi.org/10.5194/tc-11-2691-2017">doi:10.5194/tc-11-2691-2017</a><br>
Please use this citation if you are using PyTrx in publishing articles.

Each script contains classes and functions for handling each aspect needed for photogrammetric processing:<br>

<b><u>CamEnv.py</b></u><br>
Handles the associated data with the camera environment.<br>
The <b>GCPs</b> class handles the Ground Control Points (GCPs) and their correspondence to the associated DEM and CamImage object.<br>
The <b>CamCalib</b> class handles information concerning the camera calibration, i.e. the intrinsic camera matrix and lens distortion coefficients. This class contains functionality for reading in calibration files from .txt and .mat formats.<br>
The <b>CamEnv</b> compiles all the information about the camera environment from the GCPs and CamCalib classes, and also contains information about the camera object (YPR and location). This is also where georectification functionality is held, with functions for projection and inverse projection. The class is initialised using a .txt file containing file path directories to all the associated data files.<br>

<b><u>DEM.py</b></u><br>
Handles the DEM data. This currently supports .mat and .tif file types.<br>
The <b>ExplicitRaster</b> class represents a DEM as a numeric raster with explicit XY cell referencing in each grid cell. The class includes functions for densification, calculating viewsheds, and incorporates unbound functions that import a DEM file from .mat and .tif formats.<br>

<b><u>FileHandler.py</b></u><br>
This module contains a set of functions for reading in data from files (such as image data and calibration information) and writing out data (currently this is limited to just writing out homography data, but soon velocities and velocity maps will be incorporated into this).<br>

<b><u>Images.py</b></u><br>
Handles the image data and image sequence functionality.<br> 
The <b>CamImage</b> class holds information about a singular image and contains functionality for importing image data from file and passing specific image bands forward for subsequent processing.<br>
The <b>ImageSequence</b> class holds information about an image sequence, i.e. a collection of CamImage objects, from which specific images and image pairs can be called.<br>


<b><u>Measure.py</b></u><br>
Contains classes for calculating homography and velocities, and measuring surface areas and distances from oblique imagery. This module has not yet been fully incorporated into the most up-to-date version of PyTrx.<br>
The <b>Velocity</b> class enables processing with an ImageSequence object. Camera homography and velocities are derived and held within this class.<br>
The <b>Area</b> class performs automated and manual detection of surface areas from imagery and georectifies polygons to real-world coordinates.<br>
The <b>Line</b> class performs manual detection of lines from imagery (e.g. glacier terminus position) and georectifies lines to real-world coordinates.
<br>

<b><u>Utilities.py</b></u><br>
This module contains a set of functions for plotting and interpolating data.


