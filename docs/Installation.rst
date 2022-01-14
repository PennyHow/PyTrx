Installation
============

PyTrx set-up
------------

PyTrx has been coded with Python 3 and has been tested on Linux and Windows operating systems (it should also work on Apple operating systems too, it just hasn't been tested). PyTrx was originally written using a Linux operating system, so the inputted file path structures given in the example scripts may differ between operating systems and it is therefore advised to check file path structures before running these.

PyTrx can either be downloaded directly from the `GitHub repository <https://github.com/PennyHow/PyTrx>`_, or installed PyPI package manager (pip).


Cloning PyTrx from GitHub
--------------------------

PyTrx can be downloaded directly through the 'clone or download' icon on `PyTrx's GitHub repository <https://github.com/PennyHow/PyTrx>`_. To use PyTrx, you will need a working distribution of Python and the following key packages, which PyTrx strongly depends on:

* OpenCV (v3 and above): `<https://opencv.org>`_

* GDAL (v2 and above): `<https://gisinternals.com>`_

* Pillow (PIL) (v5 and above): `<https://pythonware.com>`_

Be aware that these dependencies may not necessarily be installed with your distribution of Python (e.g. PythonXY, Anaconda), so you may have to install them separately. The `.yml environment file <https://github.com/PennyHow/PyTrx/blob/master/environment.yml>`_ provided in the GitHub repository can be used to `set up an environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ that holds all of the necessary Python packages to run PyTrx. 

PyTrx has been tried and tested with the following dependency version configuration: *OpenCV=3.4.2*, *GDAL=2.3.2*, and *PIL=5.3*. PyTrx also needs other packages, which are commonly included with distributions of Python: *datetime*, *glob*, *imghdr*, *math*, *Matplotlib*, *NumPy*, *operator*, *os*, *pathlib*, *PyLab*, *SciPy*, *struct*, and *sys*. Compatibility with all newer versions of these packages are highly likely.


Installing PyTrx through pip
----------------------------

PyTrx is available through pip and can be installed with the following simple command:


.. code-block:: bash

   pip install pytrx


Be warned that there are difficulties with the GDAL package on pip, meaning that GDAL could not be declared explicitly as a PyTrx dependency in the pip package compiling. Please ensure that GDAL is installed separately if installing PyTrx through pip.

If you still run into problems then we suggest creating a new conda environment with the .yml environment file <https://github.com/PennyHow/PyTrx/blob/master/environment.yml>`_ provided in the PyTrx repository, which contains all of PyTrx's dependencies. Then the PyTrx pip package can be installed afterwards in this fresh environment.


.. code-block:: bash

   conda env create --file environment.yml
   
   pip install pytrx
   

To check that PyTrx is working, open a Python console or IDE such as Spyder, type 'import PyTrx' and hit enter, followed by 'help(PyTrx)'. If PyTrx is working correctly, this should print PyTrx's metadata, including PyTrx's license, a brief description of the toolset, and its structure. If this does not work and throws up an error, it is likely that the package dependencies are invalid so reconfigure them and then try again. Now you are all set up to use PyTrx.

