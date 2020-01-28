PyTrx Examples
=============

PyTrx comes with working examples, where the toolset has been used to generate glacial datasets. These examples can be adapted and used, and we hope these are especially useful for beginners in coding and glacial photogrammetry.


Automated area detection
------------------------

Automated detection of supraglacial lakes, *driver_autoarea.py*
 
Example driver for deriving changes in surface area of supraglacial lakes captured from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Regions of interest are automatically detected based on differences in pixel intensity and corrected for image distortion. Previously defined areas can also be imported from file (this can be changed by commenting and uncommenting commands in the 'Calculate areas' section). This script uses images from those found in the 'KR5_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR5_2014.txt.'


Manual area detection
---------------------

Manual detection of meltwater plume extents, *driver_manualarea.py*

Example driver for calculating meltwater plume surface extent at Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs manual detection of meltwater plumes through sequential images of the glacier to derive surface areas which have been corrected for image distortion. Images are imported from those found in the 'KR1_2014_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_KR1_2014.txt'


Manual line detection 
---------------------

Manual detection of glacier terminus profiles, *driver_manualline.py*

Example driver for calculating terminus profiles (as line features) at Tunabreen, Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. This script performs manual detection of terminus position through sequential images of the glacier to derive line profiles which have been corrected for image distortion. Images are imported from those found in the 'TU2_2015_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_TU2_2014.txt'


Point georectification
----------------------

Georectification of calving event point locations, *driver_ptsgeorectify.py*

Example driver which demonstrates the capabilities of the georectification functions provided in PyTrx (which are based upon those available in ImGRAFT). Pre-defined points are imported which denote calving events at Tunabreen, Svalbard, that have been distinguished in the image plane. These are subsequently projected to xyz locations using the georectification functions in PyTrx. The xyz locations are plotted onto the DEM, with the colour of each point denoting the style of calving in that particular instance. The xyz locations are finally exported as a text file (.txt) and as a shape file (.shp).


Sparse feature-tracking
-----------------------

Glacier velocities derived through feature-tracking of sparse points, *driver_velocity1.py*

Example driver for deriving sparse velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.


Dense feature-tracking
----------------------

Glacier velocities derived through feature-tracking of dense grid, *driver_velocity2.py*

Example driver for deriving dense velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.


Sparse and dense feature-tracking
----------------------------------

Alternative script for glacier velocity feature-tracking with both the sparse and dense methods, *driver_velocity3.py*

Extended example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. This script produces the same outputs as *driver_velocity1.py* and *driver_velocity2.py*. The difference is that velocities are processed using the stand-alone functions provided in PyTrx, rather than handled by PyTrx's class objects. This provides the user with a script that is more flexible and adaptable.



