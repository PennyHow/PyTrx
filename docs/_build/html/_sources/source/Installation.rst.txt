Quickstart
==========

Installation
------------

PyTrx v1.1 can be installed through pip with the following simple command:

.. code-block:: python

   pip install pytrx

Or through conda:

.. code-block:: python

   conda install pytrx

The most recent version of PyTrx can also be downloaded through the `GitHub repository <https://github.com/PennyHow/PyTrx>`_ (under the 'master' branch). The .yml environment file provided in the PyTrx GitHub repository contains an environment suitable for set-up in a Linux or Windows operating system. 


Requirements
------------

PyTrx was originally written using a Linux operating system. Inputted file path structures may differ between operating systems and it is therefore advised to check file path structures before running scripts.

PyTrx has been coded with Python 3 and has the following key dependencies:

* OpenCV (v3.4.2): `<https://opencv.org>`_

* GDAL (v2.3.2): `<https://gisinternals.com>`_

* Pillow (PIL) (v5.3.0): `<https://pythonware.com>`_

If installed with pip or conda, these dependencies will be accounted for and installed if necessary. If building yourself, these packages may not be installed with distributions of Python (e.g. PythonXY, Anaconda), so you may have to download them from the given links or with a package installer. It is important to download the package versions specified as we cannot guarantee that all others are compatible with PyTrx. PyTrx also needs other packages, which are commonly included with distributions of Python. Compatibility with all versions of these packages are highly likely: *datetime*, *glob*, *imghdr*, *math*, *Matplotlib*, *NumPy*, *operator*, *os*, *pathlib*, *PyLab*, *SciPy*, *struct*, and *sys*.
