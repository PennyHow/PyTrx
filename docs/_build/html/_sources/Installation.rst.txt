Installation
============

PyTrx set-up
------------

PyTrx has been coded with Python 3 and has been tested on Linux and Windows operating systems (it should also work on Apple operating systems too, it just hasn't been tested). PyTrx was originally written using a Linux operating system, so the inputted file path structures given in the example scripts may differ between operating systems and it is therefore advised to check file path structures before running these.

PyTrx can either be downloaded directly from the `GitHub repository <https://github.com/PennyHow/PyTrx>`_, or installed PyPI package manager (pip).


Installing PyTrx through pip
----------------------------

PyTrx is available through pip and can be installed with the following simple command:


.. code-block:: bash

   pip install pytrx


Be warned that there are difficulties with the GDAL package on pip, meaning that gdal could not be declared explicitly as a PyTrx dependency in the pip package compiling. Please ensure that gdal is installed separately if installing PyTrx through pip. You should be able to create a new environment, install GDAL and the other dependencies with conda, and then install PyTrx with pip.


.. code-block:: bash
   
   conda create --name pytrx python=3.7 
   
   conda install gdal opencv pillow scipy matplotlib spyder
   
   pip install pytrx


If you still run into problems then we suggest creating a new conda environment from the `.yml environment file <https://github.com/PennyHow/PyTrx/blob/master/environment.yml>`_ provided in the PyTrx repository, as described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. This includes a fresh install of PyTrx.


.. code-block:: bash

   conda env create --file environment.yml
   

To check that PyTrx is working, some simple unit tests can be run from the command line.

.. code-block:: bash

   python -m unittest PyTrx.Area PyTrx.CamEnv PyTrx.DEM PyTrx.Line PyTrx.Velocity 

Additionally, open a Python console or IDE such as Spyder, and try to import PyTrx, PyTrx's help guide, and one of PyTrx's modules as a test.

.. code-block:: python

   import PyTrx
   
   help(PyTrx)
   
   from PyTrx import Area
   
If PyTrx is working correctly, the unit tests should run successfully and the help statement should print PyTrx's metadata, including PyTrx's license, a brief description of the toolset, and its structure. If this does not work and throws up an error, it is likely that the package dependencies are invalid so reconfigure them and then try again.


Cloning PyTrx from GitHub
--------------------------

PyTrx can be cloned from `PyTrx's GitHub repository <https://github.com/PennyHow/PyTrx>`_, or using git. 


.. code-block:: bash

   git clone https://github.com/PennyHow/PyTrx.git


The repository can be installed in your python environment by navigating to the top level of the repository and using pip to install the local package.


.. code-block:: bash
   
   python -m pip install -e .


Or you can use PyTrx in a python environment with the installed dependencies. We recommend installing PyTrx's dependencies with conda.


.. code-block:: bash
   
   conda install gdal opencv pillow scipy matplotlib spyder
