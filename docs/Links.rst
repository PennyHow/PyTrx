Links and Acknowledgements
==========================

PyTrx citations
---------------

We are happy for others to use and adapt PyTrx for their own processing needs. If used, please cite the following key publication and Digital Object Identifier:

**How et al. (2020) PyTrx: a Python-based monoscopic terrestrial photogrammetry toolset for glaciology. Frontiers in Earth Science, accepted, doi:10.3389/feart.2020.00021**

PyTrx has been used in the following publications. In addition to the publication above, please cite any that are applicable where possible:

* How et al. (2019) Calving controlled by melt-undercutting: detailed mechanisms revealed through time-lapse observations. *Annals of Glaciology* 60 (78), 20-31, `doi:10.1017/aog.2018.28 <https://doi.org.10.1017/aog.2018.28>`_

* How (2018) Dynamical change at tidewater glaciers examined using time-lapse photogrammetry. PhD thesis, University of Edinburgh, UK, `<https://hdl.handle.net/1842/31103>`_

* How et al. (2017) Rapidly changing subglacial hydrological pathways at a tidewater glacier revealed through simultaneous observations of water pressure, supraglacial lakes, meltwater plumes and surface velocities. *The Cryosphere* 11, 2691-2710, `doi:10.5194/tc-11-2691-2017 <https://doi.org.10.5194/tc-11-2691-2017>`_

* Addison (2015) PyTrx: feature tracking software for automated production of glacier velocity. MSc thesis, University of Edinburgh, UK, `<https://hdl.handle.net/1842/11794>`_


Permissions
-----------

The DEM of the Kongsfjorden area provided as an example dataset for PyTrx originates from the freely available DEM dataset provided by the Norwegian Polar Institute, data product 'S0 Terrengmodell - Delmodell_5m_2009_13822_33 (GeoTIFF)'. This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license:

Norwegian Polar Institute (2014). Terrengmodell Svalbard (S0 Terrengmodell) [Data set]. Norwegian Polar Institute. `doi:10.21334/npolar.2014.dce53a47 <https://doi.org/10.21334/npolar.2014.dce53a47>`_

The two DEMs distributed with PyTrx for the Kongsfjorden region are *KR_demsmooth.tif* and *KR_demzero.tif*, which have been modified and manipulated from the original NPI data. In both cases, the scene has been clipped to the area of interest, downgraded to 20 metre resolution, and smoothed using a linear interpolation method. The latter of these DEMs has been manipulated in order to better represent the terminus position of Kronebreen in 2014 (the time at which the images were taken) and project meltwater plumes to a flat, homogeneous surface at sea level.

The DEM of the Tempelfjorden area provided as an example dataset for PyTrx originates from ArcticDEM, Scene ID: WV01_20130714_1020010 (July 14, 2013). There is no license for the ArcticDEM data and it can be used and distributed freely. The DEM was created from DigitalGlobe, Inc., imagery and funded under National Science Foundation awards 1043681, 1559691, and 1542736. 

The DEM distributed with PyTrx of the Tempelfjorden region is called *TU_demzero.tif*, which has been modified and manipulated from the original ArcticDEM data. The scene has been clipped to the area of interest, downgraded to 20 metre resolution, and all low-lying elevations (< 150 m) have been transformed to 0 m a.s.l. in order to project point locations and line profiles to a flat, homogeneous surface at sea level.

Acknowledgements
----------------

This work would not have been possible without the CRIOS (Calving Rates and Impact on Sea Level) project, who own the example image sets distributed with PyTrx.

Parts of the georectification functions in the PyTrx toolbox were inspired and translated from `ImGRAFT <http://imgraft.glaciology.net/>`_, a photogrammetry toolbox for Matlab (`Messerli and Grinsted, 2015 <https://www.geosci-instrum-method-data-syst.net/4/23/2015/gi-4-23-2015.html>`_). Where possible, ImGRAFT has been credited for in the corresponding PyTrx scripts (primarily some passages in the CamEnv.py script) and cited in relevant PyTrx publications.


Links
-----

There are other useful software available for terrestrial photogrammetry in glaciology:

* `Pointcatcher <http://www.lancaster.ac.uk/staff/jamesm/software/pointcatcher.htm>`_: Matlab-based GUI toolbox for feature-tracking and georectification

* `ImGRAFT <http://imgraft.glaciology.net/>`_: Matlab toolbox for feature-tracking and georectification

* `CIAS <http://www.mn.uio.no/geo/english/research/projects/icemass/cias/>`_: IDL gui for feature-tracking

* `PRACTISE <https://www.geosci-model-dev.net/9/307/2016/>`_: Matlab toolbox for georectification

