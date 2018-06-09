# Camera Environment data for PyTrx examples
This folder contains the camera environment data needed in order to run the examples provided with PyTrx.<br>

<b>*Camera calibration files (calib)*</b><br>
Matlab Camera Calibration toolbox. <br>

<b>*Camera Environment text files (camenv)*</b><br>
Text files for mapping the data to construct the camera environment. <br>

<b>*Digital elevation models (dem)*</b><br>
DEMs for georectification. <br>

<b>*1. Kongsfjorden DEMs*</b><br>
The DEM of the Kongsfjorden area provided as an example dataset for PyTrx orginates from the freely available DEM dataset provided by the <a href="https://geodata.npolar.no/">Norwegian Polar Institute</a>, data product 'S0 Terrengmodell - Delmodell_5m_2009_13822_33 (GeoTIFF)'. This data is licensed under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0) license</a>:<br>

Norwegian Polar Institute (2014). Terrengmodell Svalbard (S0 Terrengmodell) [Data set]. Norwegian Polar Institute. <a href="https://doi.org/10.21334/npolar.2014.dce53a47">doi:10.21334/npolar.2014.dce53a47</a><br>

The two DEMs distributed with PyTrx for the Kongsfjorden region are 'KR_demsmooth.tif' and 'KR_demzero.mat', which have been modified and manipulated from the original NPI data. In both cases, the scene has been clipped to the area of interest, downgraded to 20 metre resolution, and smoothed using a linear interpolation method. The latter of these DEMs has been manipulated in order to better represent the terminus position of Kronebreen in 2014 (the time at which the images were taken) and project meltwater plumes to a flat, homogeneous surface at sea level. <br>

<b>*2. Tempelfjorden DEM*</b><br>
The DEM of the Tempelfjorden area provided as an example dataset for PyTrx originates from <a href="">ArcticDEM</a>, Scene ID: WV01_20130714_1020010 (July 14, 2013). <a href="https://www.pgc.umn.edu/guides/arcticdem/additional-information/">There is no license for the ArcticDEM data and it can be used and distributed freely</a>. The DEM was created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736. <br>

The DEM distributed with PyTrx of the Tempelfjorden region is called 'TU_demzero.tif', which has been modified and manipulated from the original ArcticDEM data. The scene has been clipped to the area of interest, downgraded to 20 metre resolution, and all low-lying elevations (< 150 m) have been transformed to 0 m a.s.l. in order to project point locations and line profiles to a flat, homogeneous surface at sea level. <br>

<b>*Ground control points (gcps)*</b><br>
Ground control points corresponding to image positions and real-world coordinates. <br>

<b>*Image registration masks (invmasks)*</b><br>
Masks for tracking static features in order to perform image registration. <br>

<b>*Feature-tracking masks (masks)*</b><br>
Masks for tracking moving features in order to perform feature-tracking and derive velocities. <br>

<b>*Reference Images (refimages)*</b><br>
Reference images from which image-based GCPs are derived from. <br>
