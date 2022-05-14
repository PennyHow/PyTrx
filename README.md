# Proglacial river stage derived from orthorectified time-lapse camera images, Inglefield Land, Northwest Greenland

Datasets and processing scripts for Goldstein et al. Proglacial river stage derived from orthorectified time-lapse camera images, Inglefield Land, Northwest Greenland


## Datasets

The following datasets are provided:

+ Time-lapse camera information, including camera calibration variables and ground control point positions

+ TLS DEM of the area of interest

+ Time-lapse images of the river from 2019 and 2020

+ River stage datasets from 2019, 2020 and 2021

+ Water level data in pixel and projected positions

+ Production copies of Figures 4, 5, 6 and 7 (which are outputs of the Python scripts)


## Processing scripts

The Python scripts held here include:

1. **waterline_detection.py**, for detecting water level from time-lapse images using a semi-automated Canny Edge detection approach

2. **waterline_orthorectification.py**, for orthorectifying image-derived water levels and comparing these to river stage data

3. **bubbler_data_example.py**, an example script for importing and plotting river stage data

4. **canny_edge_example.py**, an example script for using Canndy Edge detection to identify water level from a selection of time-lapse images 


### Python set-up

To run the provided Python scripts, we recommend using conda to set up a Python environment and then cloning this repository. Dependencies can be installed using conda and pip.
 
```bash
conda create --name pytrx3 python=3.8
conda activate pytrx3

conda install gdal opencv matplotlib pandas pillow scipy scikit-image
pip install pytrx 
```

Or you can create a Python environment with all the installed dependencies using the environment file provided in this repository. For troubleshooting, see the set-up guide in the [PyTrx repo](https://github.com/PennyHow/PyTrx) and [readthedocs installation readme](https://pytrx.readthedocs.io/en/latest/Installation.html#). 

```bash
conda env create --file environment.yml
conda activate pytrx3
```

The repository can then be cloned to a local workspace using git.

```bash
git clone https://github.com/sethnavon/InglefieldData
```
