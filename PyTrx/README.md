# PyTrx
<a href='https://pytrx.readthedocs.io/en/latest/?badge=latest'> <img src='https://readthedocs.org/projects/pytrx/badge/?version=latest' alt='Documentation status' /></a> <a href="https://badge.fury.io/py/pytrx"><img src="https://badge.fury.io/py/pytrx.svg" alt="PyPI status" height="18"></a><br>

PyTrx (short for 'Python Tracking') is a Python object-oriented toolbox created for the purpose of calculating real-world measurements from oblique images and time-lapse image series. Its primary purpose is to obtain velocities, surface areas, and distances from imagery of glacial environments. <br>

<hr>

<h3>PyTrx citations</h3>

We are happy for others to use and adapt PyTrx for their own processing needs. If used, please cite the following key publication and Digital Object Identifier:<br>

<h3>How et al. (2020) PyTrx: a Python-based monoscopic terrestrial photogrammetry toolset for glaciology. <i>Frontiers in Earth Science</i> 8:21, <a href="https://dx.doi.org/10.3389/feart.2020.00021">doi:10.3389/feart.2020.00021</a></h3><br>
  
PyTrx has been used in the following publications. In addition to the publication above, please cite any that are applicable where possible:<br>

<b>*PyTrx used for georectification of glacier calving event locations*</b><br>
How et al. (2019) Calving controlled by melt-undercutting: detailed mechanisms revealed through time-lapse observations. <i>Annals of Glaciology</i> 60 (78), 20-31, <a href="https://dx.doi.org/10.1017/aog.2018.28">doi:10.1017/aog.2018.28</a><br>

<b>*PhD thesis by Penelope How, for which PyTrx was developed primarily*</b><br>
How (2018) Dynamical change at tidewater glaciers examined using time-lapse photogrammetry. PhD thesis, University of Edinburgh, UK, <a href="https://hdl.handle.net/1842/31103">https://hdl.handle.net/1842/31103</a><br>

<b>*PyTrx used for detection of supraglacial lakes and meltwater plumes*</b><br>
How et al. (2017) Rapidly changing subglacial hydrological pathways at a tidewater glacier revealed through simultaneous observations of water pressure, supraglacial lakes, meltwater plumes and surface velocities. <i>The Cryosphere</i> 11, 2691-2710, <a href="https://doi.org/10.5194/tc-11-2691-2017">doi:10.5194/tc-11-2691-2017</a><br>

<b>*MSc thesis by Lynne Buie, where PyTrx was created in its earliest form*</b><br>
Addison (2015) PyTrx: feature tracking software for automated production of glacier velocity. MSc thesis, University of Edinburgh, UK, <a href="https://hdl.handle.net/1842/11794">https://hdl.handle.net/1842/11794</a><br>

<hr>

<h3>Installation</h3>

The PyTrx installation has been tested on Linux and Windows operating systems (it should also work on Apple operating systems too, it just hasn't been tested). PyTrx is primarily available through pip:

```bash
pip install pytrx
```
Be warned that there are difficulties with the GDAL package on pip, meaning that gdal could not be declared explicitly as a PyTrx dependency in the pip package compiling. Please ensure that gdal is installed separately if installing PyTrx through pip. You should be able to create a new environment, install GDAL and the other dependencies with conda, and then install PyTrx with pip.

```bash
conda create --name pytrx python=3.7
conda install gdal opencv pillow scipy matplotlib spyder
pip install pytrx
```strawberry jam swiss roll cake recipe

Be aware that the PyTrx example scripts in this repository are not included with the pip distribution of PyTrx, given the size of the example dataset files. Either download these separately, or create a new conda environment (using the <a href="https://github.com/PennyHow/PyTrx/blob/master/environment.yml">.yml environment file</a> provided) and clone the PyTrx GitHub repository:

```bash
conda env create --file environment.yml
git clone https://github.com/PennyHow/PyTrx.git
```

See our <a href="https://pytrx.readthedocs.io/en/latest/Installation.html">readthedocs page on setting up PyTrx</a> for more details.

<hr>

<h3>Permissions and acknowledgements</h3>

PyTrx was initially developed and released as part of the  <a href="https://www.researchinsvalbard.no/project/20000000-0000-0000-0000-000000007037/project-info"> CRIOS (Calving Rates and Impact on Sea Level</a> project. PyTrx's continued development and maintenance is funded by an <a href="https://eo4society.esa.int/projects/griml/">ESA Living Planet Fellowship</a>.<br><br>

Parts of the <b>georectification functions</b> in the PyTrx toolbox were inspired and translated from <a href="http://imgraft.glaciology.net/">ImGRAFT</a>, a photogrammetry toolbox for Matlab (<a href="https://www.geosci-instrum-method-data-syst.net/4/23/2015/gi-4-23-2015.pdf">Messerli and Grinsted, 2015</a>). Where possible, ImGRAFT has been credited for in the corresponding PyTrx scripts (primarily some passages in the CamEnv.py script) and cited in relevant PyTrx publications. <br><br>

See PyTrx's <a href="https://pytrx.readthedocs.io/en/latest/Links.html">readthedocs</a> for full acknowledgements and permissions.

<hr>
 
<h3>Links</h3>

There are other useful software available for terrestrial photogrammetry in glaciology: <br>

<a href="http://www.lancaster.ac.uk/staff/jamesm/software/pointcatcher.htm">Pointcatcher</a> - Matlab-based GUI toolbox for feature-tracking and georectification <br>
<a href="http://imgraft.glaciology.net/">ImGRAFT</a> - Matlab toolbox for feature-tracking and georectification <br>
<a href="https://tu-dresden.de/bu/umwelt/geo/ipf/photogrammetrie/forschung/forschungsprojekte/emt">EMT (Environmental Motion Tracking)</a> - GUI toolbox for feature-tracking and georectification <br>
<a href="http://www.mn.uio.no/geo/english/research/projects/icemass/cias/">CIAS</a> - IDL gui for feature-tracking <br>
<a href="https://www.geosci-model-dev.net/9/307/2016/">PRACTISE</a> - Matlab toolbox for georectification

<hr>

<h3>Copyright</h3>

PyTrx is licensed under a <a href="https://choosealicense.com/licenses/mit/">MIT License</a>.
