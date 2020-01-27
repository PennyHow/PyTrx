'''
PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie

PyTrx is licensed under a MIT License.

You should have received a copy of the license along with this
work. If not, see <https://choosealicense.com/licenses/mit/>.


PYTRX SETUP FILE
This file is needed for the PyTrx package initialisation.
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytrx-phow", 
    version="1.1.0",
    author="Penelope How",
    author_email="pennyruthhow@gmail.com",
    description="An object-oriented toolset for calculating velocities, surface areas and distances from oblique imagery of glacial environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PennyHow/PyTrx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)