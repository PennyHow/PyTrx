# Example applications of PyTrx

This folder contains example applications of PyTrx. Specifically it contains script drivers and the associated data with these examples. These can easily be adapted to different datasets and applications. A selection of these examples, along with others, are presented in the PyTrx methods paper:

<h3>How et al. (2020) PyTrx: a Python-based monoscopic terrestrial photogrammetry toolset for glaciology. <i>Frontiers in Earth Science</i> 8:21, <a href="https://dx.doi.org/10.3389/feart.2020.00021">doi:10.3389/feart.2020.00021</a></h3>

Please see the <a href="https://pytrx.readthedocs.io/en/latest/GetStarted.html">readthedocs page</a> for a comprehensive walkthrough of a selection of these examples.

## Data provided with PyTrx

### Image sets

Example image sets from Svalbard glaciers were collected as part of <a href="https://www.researchinsvalbard.no/project/7037">CRIOS</a> (Calving Rates and Impact On Sea level), and are used here with permission. <br>

Example image sets from Qasigiannguit glacier are used courtesy of <a href="https://www.asiaq-greenlandsurvey.gl/frontpage/">Asiaq Greenland Survey</a> as part of Messerli et al. (In Review), through GlacioBasis Nuuk under the <a href="https://g-e-m.dk/">GEM (Greenland Ecosystem Monitoring) programme</a>.<br>

### Digital Elevation Models (DEMs)

<b>*1. Kongsfjorden DEMs*</b><br>
The DEM of the Kongsfjorden area provided as an example dataset for PyTrx originates from the freely available DEM dataset provided by the <a href="https://geodata.npolar.no/">Norwegian Polar Institute</a>, data product 'S0 Terrengmodell - Delmodell_5m_2009_13822_33 (GeoTIFF)'. This data is licensed under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0) license</a>:<br>

Norwegian Polar Institute (2014). Terrengmodell Svalbard (S0 Terrengmodell) [Data set]. Norwegian Polar Institute. <a href="https://doi.org/10.21334/npolar.2014.dce53a47">doi:10.21334/npolar.2014.dce53a47</a><br>

The two DEMs distributed with PyTrx for the Kongsfjorden region are 'KR_demsmooth.tif' and 'KR_demzero.tif', which have been modified and manipulated from the original NPI data. In both cases, the scene has been clipped to the area of interest, downgraded to 20 metre resolution, and smoothed using a linear interpolation method. The latter of these DEMs has been manipulated in order to better represent the terminus position of Kronebreen in 2014 (the time at which the images were taken) and project meltwater plumes to a flat, homogeneous surface at sea level. <br>

<b>*2. Tempelfjorden DEM*</b><br>
The DEM of the Tempelfjorden area provided as an example dataset for PyTrx originates from <a href="https://www.pgc.umn.edu/data/arcticdem/">ArcticDEM</a>, Scene ID: WV01_20130714_1020010 (July 14, 2013). <a href="https://www.pgc.umn.edu/guides/arcticdem/additional-information/">There is no license for the ArcticDEM data and it can be used and distributed freely</a>. The DEM was created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736. <br>

The DEM distributed with PyTrx of the Tempelfjorden region is called 'TU_demzero.tif', which has been modified and manipulated from the original ArcticDEM data. The scene has been clipped to the area of interest, downgraded to 20 metre resolution, and all low-lying elevations (< 150 m) have been transformed to 0 m a.s.l. in order to project point locations and line profiles to a flat, homogeneous surface at sea level. <br>

<b>*3. Qasigiannguit glacier*</b><br>

The DEM of Qasigiannguit glacier is provided courtesy of <a href="https://www.asiaq-greenlandsurvey.gl/frontpage/">Asiaq Greenland Survey</a>. The DEM was acquired from UAV surveys in September 2020, as part of GlacioBasis Nuuk under the <a href="https://g-e-m.dk/">GEM (Greenland Ecosystem Monitoring) programme.</a> The version provided here has been downgraded to 20 metre resolution.<br>

<hr>
