# Example drivers for PyTrx
This folder contains example applications of PyTrx. Specifically it contains script drivers and the associated data with these examples. These can easily be adapted to different datasets and applications.<br>

Example image sets distributed with PyTrx were collected as part of <a href="https://www.researchinsvalbard.no/project/7037">CRIOS</a> (Calving Rates and Impact On Sea level), and are used here with permission. Other datasets are included with each of these example image sets, including camera location and pose (yaw, pitch, roll), calibration models, ground control points (GCPs), and Digital Elevation Models (DEMs). The available DEMs have been modified and manipulated from their original form, which are derived from TanDEM-X and the Norwegian Geodetic Survey.<br>

<hr>

<b>driver_autoarea.py</b>
<br>Example driver for deriving changes in surface area of supraglacial lakes captured from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Regions of interest are automatically detected based on differences in pixel intensity and corrected for image distortion. Previously defined areas can also be imported from file (this can be changed by commenting and uncommenting commands in the "Calculate areas" section). This script uses images from those found in the 'KR5_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR5_2014.txt.'

<b>driver_manualarea.py</b>
<br>Example driver for calculating meltwater plume surface extent at Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs manual detection of meltwater plumes through sequential images of the glacier to derive surface areas which have been corrected for image distortion. Images are imported from those found in the 'KR1_2014_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_KR1_2014.txt'

<b>driver_manualline.py</b>
<br>Example driver for calculating terminus profiles (as line features) at Tunabreen, Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. This script performs manual detection of terminus position through sequential images of the glacier to derive line profiles which have been corrected for image distortion. Images are imported from those found in the 'TU2_2015_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_TU2_2014.txt'

<b>driver_ptsgeorectify.py</b>
<br>Example driver which demonstrates the capabilities of the georectification functions provided in PyTrx (which are based upon those available in ImGRAFT). Pre-defined points are imported which denote calving events at Tunabreen, Svalbard, that have been distinguished in the image plane. These are subsequently projected to xyz locations using the georectification functions in PyTrx. The xyz locations are plotted onto the DEM, with the colour of each point denoting the style of calving in that particular instance. The xyz locations are finally exported as a text file (.txt) and as a shape file (.shp).

<b>driver_velocity.py</b>
<br>Example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image
registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.
