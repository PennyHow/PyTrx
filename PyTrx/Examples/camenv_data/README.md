# Camera Environment data for PyTrx examples
This folder contains the camera environment data needed to run the examples provided with PyTrx. <br>

<hr>

<h3>Camera calibration files (calib)</h3>
Text and image files for determining information about the camera matrix and lens parameters, which are needed to correct images for distortion. The text files contain the camera matrix and lens distortion parameters already calculated - four separate calibrations were performed for first of the Kronebreen cameras (KR1), and one was calculated for the Tunabreen camera (TU1). Where defined (in the camera environment text file), PyTrx will take an average of multiple calibration files. Calibration images from the second and third Kronebreen cameras (KR2 and KR3) are provided in the two folders of images, which are used to calculate the camera matrix and lens distortion parameters within PyTrx and its calibration capabilities. This can either be defined in the camera environment text file, or performed in situ (see one of the extended example scripts for an example).

<hr>

<h3>Camera Environment text files (camenv)</h3>
Text files for defining data (camera location and pose) and mapping the relevant data paths (GCPs DEM, reference image, calibration file), which are needed to construct the camera environment. 

<hr>

<h3>Digital elevation models (dem)</h3>
DEMs for defining the camera environment and performing georectification. 
<hr>

<h3>Ground control points (gcps)</h3>
Text files containing the coordinates for the ground control points, corresponding to image positions and real-world coordinates. The text files are  specifically formatted as follows: X, Y, Z (real-world coordinates), X, Y (pixel coordinates).

<hr>

<h3>Image registration masks (invmasks)</h3>
The boolean image masks that are needed for tracking static features and performing image registration. <br>

<hr>

<h3>Feature-tracking masks (masks)</h3>
The boolean image masks for masking areas of interest, used subsequently for either tracking moving features and performing feature-tracking (e.g. glacier surface velocities), or for detecting areal features (e.g. supraglacial lakes).

<hr>

<h3>Reference Images (refimages)</h3>
The reference images from which the image-based GCPs are derived. <br>
