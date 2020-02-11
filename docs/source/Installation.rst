Quickstart
==========

PyTrx set-up
------------

PyTrx has been coded with Python 3 and has been tested on Linux and Windows operating systems (it should also work on Apple operating systems too, it just hasn't been tested). PyTrx was originally written using a Linux operating system, so the inputted file path structures given in the example scripts may differ between operating systems and it is therefore advised to check file path structures before running these.

PyTrx v1.1 can either be downloaded directly from the `GitHub repository <https://github.com/PennyHow/PyTrx>`_, or installed through one of two package managers (conda or pip). If installing with a package manager, **we recommend using conda**.


Downloading PyTrx from GitHub
-----------------------------

PyTrx can be downloaded directly through the 'clone or download' icon on `PyTrx's GitHub repository <https://github.com/PennyHow/PyTrx>`_. To use PyTrx, you will need a working distribution of Python and the following key packages, which PyTrx strongly depends on:

* OpenCV (v3 and above): `<https://opencv.org>`_

* GDAL (v2 and above): `<https://gisinternals.com>`_

* Pillow (PIL) (v5 and above): `<https://pythonware.com>`_

Be aware that these dependencies may not necessarily be installed with your distribution of Python (e.g. PythonXY, Anaconda), so you may have to install them separately. The .yml environment file provided in the GitHub repository can be used to `set up an environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ that holds all of the necessary Python packages to run PyTrx. 

PyTrx has been tried and tested with the following dependency version configuration: *OpenCV=3.4.2*, *GDAL=2.3.2*, and *PIL=5.3*. PyTrx also needs other packages, which are commonly included with distributions of Python: *datetime*, *glob*, *imghdr*, *math*, *Matplotlib*, *NumPy*, *operator*, *os*, *pathlib*, *PyLab*, *SciPy*, *struct*, and *sys*. Compatibility with all newer versions of these packages are highly likely.


Installing PyTrx through pip
----------------------------

PyTrx is available through pip and can be installed with the following simple command:

.. code-block:: python

   pip install pytrx

**WARNING** There are difficulties with the GDAL package on pip, meaning that GDAL could not be declared explicitly as a PyTrx dependency. Please ensure that GDAL is installed separately if installing PyTrx through pip.

To check that PyTrx is working, open a Python console or IDE such as Spyder, type 'import PyTrx' and hit enter, followed by 'help(PyTrx)'. If PyTrx is working correctly, this should print PyTrx's metadata, including PyTrx's license, a brief description of the toolset, and its structure. If this does not work and throws up an error, it is likely that the package dependencies are invalid so reconfigure them and then try again. Now you are all set up to use PyTrx.

Be aware that the PyTrx example scripts are not included with the conda distribution of PyTrx given the size of the example dataset files. If you wish to use/adapt them, feel free to download them from the `PyTrx GitHub repository <https://github.com/PennyHow/PyTrx>`_ and use them with your installed version of PyTrx.
 

PyTrx Structure 
---------------

Detailed documentation is included in the scripts that make up PyTrx. Each script contains classes and functions for handling each aspect needed for photogrammetric processing.

For beginners in programming, it is advised to look at the example applications provided and adapt them accordingly for your own use. For experienced programmers, get stuck in. Feel free to contact us if you run into major problems or have constructive comments that will help us further PyTrx and its capabilities. We will not respond to minor troubleshooting or unconstructive comments.


CamEnv.py
*********

Handles the associated data with the camera environment. The GCPs class handles the Ground Control Points (GCPs) and their correspondence to the associated DEM and CamImage object. The CamCalib class handles information concerning the camera calibration, i.e. the intrinsic camera matrix and lens distortion coefficients. This class contains functionality for reading in calibration files from .txt and .mat formats.
The CamEnv compiles all the information about the camera environment from the GCPs and CamCalib classes, and also contains information about the camera object (pose and location). This is also where georectification functionality is held, with functions for projection and inverse projection. The class is initialised using a .txt file containing file path directories to all the associated data files.


DEM.py
******

Handles the DEM data. This currently supports .mat and .tif file types. The ExplicitRaster class represents a DEM as a numeric raster with explicit XY cell referencing in each grid cell. The class includes functions for densification, calculating viewsheds, and incorporates unbound functions that import a DEM file from .mat and .tif formats.


FileHandler.py
**************

This module contains a set of functions for reading in data from files (such as image data and calibration information) and writing out data.


Images.py
*********

Handles the image data, and the image sequence. The CamImage class holds information about a singular image and contains functionality for importing image data from file and passing specific image bands forward for subsequent processing. The ImageSequence class holds information about an image sequence, i.e. a collection of CamImage objects, from which specific images and image pairs can be called.


Velocity.py
***********

Calculates velocities and homography. This can either be achieved through the Velocity class for processing velocities and homography through a series of images, or using the functions provided within the script for processing velocities and homography between an image pair.


Area.py
*******

Automated and manual detection of surface areas from imagery (e.g. supraglacial lakes, meltwater plume surface extent). This can either be achieved through the Area class for defining areas of interest through a series of images, or using the functions provided within the script for defining areas of interest in a single image.


Line.py
*******

Manual detection of line features from imagery (e.g. glacier terminus position). This can either be achieved through the Line class for defining line features through a series of images, or using the functions provided within the script for defining line features in a single image.


Utilities.py
************

This module contains a set of functions for plotting and interpolating data.
