# Results from PyTrx examples 
This folder contains the outputs of the example applications of PyTrx. Each folder contains the data and image outputs of each of the driver scripts. A selection of these outputs, along with others, are presented in the PyTrx methods paper:

<h3>How et al. (2020) PyTrx: a Python-based monoscopic terrestrial photogrammetry toolset for glaciology. <i>Frontiers in Earth Science</i> 8:21, <a href="https://dx.doi.org/10.3389/feart.2020.00021">doi:10.3389/feart.2020.00021</a></h3>

<hr>

<h3>Automated detection of supraglacial lakes (KR_autoarea)</h3>
Outputs from the driver script for deriving changes in surface area of supraglacial lakes at Kronebreen (KR_autoarea.py). These outputs include: <br>
1. Output images of the detected lakes (overlaid onto the oblique imagery and the input DEM) <br>
2. Shapefiles (.shp) of the detected lakes <br>
3. Text files containing information about the xyz area and pixel area of each individual lake (area_all.txt and px_all.txt), the xyz and pixel coordinates of each individual lake (area_coords.txt and px_coords.txt), and the cumulative xyz area and pixel area of all the detected lakes in each scene (area_sum.txt and px_sum.txt) <br>

<hr>

<h3>Manual detection of meltwater plume extents (KR_manualarea)</h3>
Outputs from the driver script for calculating meltwater plume surface extent at Kronebreen (KR_manualarea.py). These outputs include: <br>
1. Output images of the defined plume extents (overlaid onto the oblique imagery and the input DEM) <br>
2. Colour output images of the defined plume extents <br> 
3. Shapefiles (.shp) of the defined plume extents <br>
4. Text files containing information about the xyz area and pixel area of each plume (area_all.txt and px_all.txt), the xyz and pixel coordinates of each plume (area_coords.txt and px_coords.txt), and the cumulative xyz area and pixel area of all the detected plumes in each scene (area_sum.txt and px_sum.txt)

<hr>

<h3>Manual detection of terminus profiles (TU_manualline)</h3>
Outputs from the driver script for calculating terminus profiles (as line features) at Tunabreen (TU_manualline.py). These outputs include: <br>
1. Output images of the defined termini (overlaid onto the oblique imagery and the input DEM) <br>
2. Shapefiles (.shp) of the defined termini <br>
3. Text files containing information about the xyz and pixel coordinates of each defined termini (line_realcoords.txt and line_pxcoords.txt), and the xyz and pixel length of each line (line_reallength.txt and line_pxlength.txt)

<hr>

<h3>Georectification of calving event point locations (TU_ptsgeorectify)</h3>
Outputs from the driver script for georectifying calving event point locations at Tunabreen (TU_ptsgeorectify.py). These outputs include: <br>
1. Shapefile (.shp) of the georectified points <br> 
2. An output image of the georectified points overlaid onto the DEM <br>
3. Text files containing the xyz locations of all the georectified points <br>
In addition, this folder contains the original pixel locations of each calving event (TU1_calving_xy.csv).

<hr>

<h3>Glacier velocity feature-tracking (KR_velocity1 and KR_velocity2)</h3>
Outputs from the driver scripts for deriving surface glacier velocities at Kronebreen, Svalbard (KR_velocity1.py, KR_velocity2.py, and KR_velocity3.py). These outputs include: <br>
1. ASCII files of each interpolated velocity map, which can be imported into most mapping software as a raster grid <br>
2. Output images of the tracked feature points on the oblique time-lapse image, the georectified points on the DEM, and the interpolated velocity maps <br> 
3. Shapefiles (.shp) of the georectified feature points (including velocity information) <br>
4. CSV file containing information about the image registration and camera homography - image 1, image 2, the 3 by 3 homography matrix, number of static features successfully tracked, mean x displacement, mean y displacement, standard deviation x displacement, standard deviation y displacement, mean error, mean homography displacement, mean homography signal-to-noise ratio (SNR) <br>
5. CSV file containing information about the measured velocities - image 1, image 2, average xyz velocity, number of features successfully tracked, average pixel velocity, average homography error, signal-to-noise ratio (the signal being velocity and the noise being average homography error)
