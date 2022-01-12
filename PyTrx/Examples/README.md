# Example applications of PyTrx
This folder contains example applications of PyTrx. Specifically it contains script drivers and the associated data with these examples. These can easily be adapted to different datasets and applications. A selection of these examples, along with others, are presented in the PyTrx methods paper:

<h3>How et al. (2020) PyTrx: a Python-based monoscopic terrestrial photogrammetry toolset for glaciology. <i>Frontiers in Earth Science</i> 8:21, <a href="https://dx.doi.org/10.3389/feart.2020.00021">doi:10.3389/feart.2020.00021</a></h3>

<hr>

<h3>Data provided with PyTrx</h3>

<b>*Image sets*</b><br>
Example image sets distributed with PyTrx were collected as part of <a href="https://www.researchinsvalbard.no/project/7037">CRIOS</a> (Calving Rates and Impact On Sea level), and are used here with permission. <br>

<b>*Digital Elevation Models (DEMs)*</b><br>
<b>*1. Kongsfjorden DEMs*</b><br>
The DEM of the Kongsfjorden area provided as an example dataset for PyTrx originates from the freely available DEM dataset provided by the <a href="https://geodata.npolar.no/">Norwegian Polar Institute</a>, data product 'S0 Terrengmodell - Delmodell_5m_2009_13822_33 (GeoTIFF)'. This data is licensed under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0) license</a>:<br>

Norwegian Polar Institute (2014). Terrengmodell Svalbard (S0 Terrengmodell) [Data set]. Norwegian Polar Institute. <a href="https://doi.org/10.21334/npolar.2014.dce53a47">doi:10.21334/npolar.2014.dce53a47</a><br>

The two DEMs distributed with PyTrx for the Kongsfjorden region are 'KR_demsmooth.tif' and 'KR_demzero.tif', which have been modified and manipulated from the original NPI data. In both cases, the scene has been clipped to the area of interest, downgraded to 20 metre resolution, and smoothed using a linear interpolation method. The latter of these DEMs has been manipulated in order to better represent the terminus position of Kronebreen in 2014 (the time at which the images were taken) and project meltwater plumes to a flat, homogeneous surface at sea level. <br>

<b>*2. Tempelfjorden DEM*</b><br>
The DEM of the Tempelfjorden area provided as an example dataset for PyTrx originates from <a href="https://www.pgc.umn.edu/data/arcticdem/">ArcticDEM</a>, Scene ID: WV01_20130714_1020010 (July 14, 2013). <a href="https://www.pgc.umn.edu/guides/arcticdem/additional-information/">There is no license for the ArcticDEM data and it can be used and distributed freely</a>. The DEM was created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736. <br>

The DEM distributed with PyTrx of the Tempelfjorden region is called 'TU_demzero.tif', which has been modified and manipulated from the original ArcticDEM data. The scene has been clipped to the area of interest, downgraded to 20 metre resolution, and all low-lying elevations (< 150 m) have been transformed to 0 m a.s.l. in order to project point locations and line profiles to a flat, homogeneous surface at sea level.

<hr>

<h3>Example driver scripts provided with PyTrx</h3>

<b>*Automated detection of supraglacial lakes (KR_autoarea.py)*</b>
<br>Example driver for deriving changes in surface area of supraglacial lakes captured from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Regions of interest are automatically detected based on differences in pixel intensity and corrected for image distortion. Previously defined areas can also be imported from file (this can be changed by commenting and uncommenting commands in the 'Calculate areas' section). This script uses images from those found in the 'KR5_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR5_2014.txt.'<br>

<b>*Manual detection of meltwater plume extents (KR_manualarea.py)*</b>
<br>Example driver for calculating meltwater plume surface extent at Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs manual detection of meltwater plumes through sequential images of the glacier to derive surface areas which have been corrected for image distortion. Images are imported from those found in the 'KR1_2014_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_KR1_2014.txt'<br>

<b>*Manual detection of terminus profiles (TU_manualline.py)*</b>
<br>Example driver for calculating terminus profiles (as line features) at Tunabreen, Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. This script performs manual detection of terminus position through sequential images of the glacier to derive line profiles which have been corrected for image distortion. Images are imported from those found in the 'TU2_2015_subset' folder, and the camera environment associated with the text file 'CameraEnvironmentData_TU2_2014.txt'<br>

<b>*Georectification of calving event point locations (TU_ptsgeorectify.py)*</b>
<br>Example driver which demonstrates the capabilities of the georectification functions provided in PyTrx (which are based upon those available in ImGRAFT). Pre-defined points are imported which denote calving events at Tunabreen, Svalbard, that have been distinguished in the image plane. These are subsequently projected to xyz locations using the georectification functions in PyTrx. The xyz locations are plotted onto the DEM, with the colour of each point denoting the style of calving in that particular instance. The xyz locations are finally exported as a text file (.txt) and as a shape file (.shp).<br>

<b>*Automated detection of snow extent (QAS_autoarea.py)*</b>
<br>This driver calculates snow/bare ice extent at Qasigiannguit Glacier, West 
Greenland, for a small subset of the 2020/21 melt season. Specifically, this examples performs automated detection of snow regions through images of the glacier to derive surface areas which have been corrected for image distortion.<br>

<b>*Automated detection of snow extent (QAS_manualarea.py)*</b>
<br>This driver calculates snow/bare ice extent at Qasigiannguit Glacier, West 
Greenland, for a small subset of the 2020/21 melt season. Specifically, this examples performs manual detection of snow regions through images of the glacier to derive surface areas which have been corrected for image distortion.<br>

<b>*Glacier velocity sparse feature-tracking (KR_velocity1.py)*</b>
<br>Example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.<br>

<b>*Glacier velocity dense feature-tracking (KR_velocity2.py)*</b>
<br>Example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. Specifically this script performs feature-tracking through sequential daily images of the glacier to derive surface velocities (spatial average, individual point displacements and interpolated velocity maps) which have been corrected for image distortion and motion in the camera platform (i.e. image registration). This script uses images from those found in the 'KR2_2014_subset' folder, and camera environment data associated with the text file 'CameraEnvironmentData_KR2_2014.txt'.<br>

<b>*Alternative script for glacier velocity feature-tracking (KR_velocity3.py)*</b>
<br>Extended example driver for deriving velocities from Kronebreen, Svalbard, for a small subset of the 2014 melt season. This script produces the same outputs as driver_velocity.py. The difference is that velocities are processed using the stand-alone functions provided in PyTrx, rather than handled by PyTrx's class objects. This provides the user with a script that is more flexible and adaptable. <br>
