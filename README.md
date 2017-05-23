# PyTrx
PyTrx is an object-oriented toolbox created for the purpose of calculating real-world measurements from oblique images and time-lapse image series. Its primary purpose is to obtain velocities, surface areas, and volume losses from imagery of glacial environments.<br>

These scripts have not yet been published, but PyTrx has been used for data processing in the following publications: <br>
How et al. (In Review) Rapidly-changing subglacial hydrology pathways at a tidewater glacier revealed through simultaneous observations of water pressure, supraglacial lakes, meltwater plumes and surface velocities. <i>The Cryosphere Discussions</i>, <a href="http://www.the-cryosphere-discuss.net/tc-2017-74/">doi:10.5194/tc-2017/74</a><br>
Please use these citations if you are using PyTrx in publishing articles.

Each script contains classes and functions for handling each aspect needed for photogrammetric processing:<br>

<u>CamEnv.py</u><br>
Handles the associated data with the camera environment.<br>
The <b>GCPs</b> class handles the Ground Control Points (GCPs) and their correspondence to the associated DEM and CamImage object.<br>
The <b>CamCalib</b> class handles information concerning the camera calibration, i.e. the intrinsic camera matrix and lens distortion coefficients. This class contains functionality for reading in calibration files from .txt and .mat formats.<br>
The <b>CamEnv</b> compiles all the information about the camera environment from the GCPs and CamCalib classes, and also contains information about the camera object (YPR and location). This is also where georectification functionality is held, with functions for projection and inverse projection. The class is initialised using a .txt file containing file path directories to all the associated data files.<br>

<u>DEM.py</u><br>
Handles the DEM data. This currently supports .mat and .tif file types.<br>
The <b>ExplicitRaster</b> class represents a DEM as a numeric raster with explicit XY cell referencing in each grid cell. The class includes functions for densification, calculating viewsheds, and incorporates unbound functions that import a DEM file from .mat and .tif formats.<br>

<u>FileHandler.py</u><br>
This module contains a set of functions for reading in data from files (such as image data and calibration information) and writing out data (currently this is limited to just writing out homography data, but soon velocities and velocity maps will be incorporated into this).<br>

<u>Images.py</u><br>
Handles the image data, the image sequence, and homography/feature-tracking functionality.<br> 
The <b>CamImage</b> class holds information about a singular image and contains functionality for importing image data from file and passing specific image bands forward for subsequent processing.<br>
The <b>ImageSequence</b> class holds information about an image sequence, i.e. a collection of CamImage objects, from which specific images and image pairs can be called.<br>
The <b>TimeLapse</b> class enables processing with an ImageSequence object. Camera homography and velocities are derived and held within this class.<br>

<u>Measure.py</u><br>
Contains classes for measuring surface areas and distances from oblique imagery. This module has not yet been fully incorporated into the most up-to-date version of PyTrx.<br>
The <b>Area</b> class performs automated and manual detection of surface areas from imagery and georectifies polygons to real-world coordinates.<br>
The <b>Length</b> class performs manual detection of lines from imagery (e.g. glacier terminus position) and georectifies lines to real-world coordinates.
<br>

<u>Utilities.py</u><br>
This module contains a set of functions for plotting data.


