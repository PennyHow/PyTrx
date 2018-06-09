# Example applications of PyTrx
This folder contains example applications of PyTrx. Specifically it contains script drivers and the associated data with these examples. These can easily be adapted to different datasets and applications.<br>

<hr>

Example image sets distributed with PyTrx were collected as part of <a href="https://www.researchinsvalbard.no/project/7037">CRIOS</a> (Calving Rates and Impact On Sea level), and are used here with permission. Other data are included with each of these example image sets, including camera location and pose (yaw, pitch, roll), Digital Elevation Models (DEMs), calibration models, and ground control points (GCPs). <br>

The DEM of the Kongsfjorden area distributed here orginates from a freely available dataset provided by the <a href="https://geodata.npolar.no/">Norwegian Geodetic Survey</a> (data product 'S0 Terrengmodell'), licensed under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0) license</a>:<br>

Norwegian Polar Institute (2014). Terrengmodell Svalbard (S0 Terrengmodell) [Data set]. Norwegian Polar Institute. https://doi.org/10.21334/npolar.2014.dce53a47

and distributed here in a modified and manipulated form with permission. 

The DEM of the Tempelfjorden area is freely available from <a href="">ArcticDEM</a>, and distributed here in a modified and manipulated form. The DEM was created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736. Refer to the readme document in the Examples folder of this repository for more information on the DEMs provided and distributed with PyTrx.

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
