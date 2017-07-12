# -*- coding: utf-8 -*-
'''
Created on Fri Jun 24 10:07:59 2016

@author: Penny How
@contact: p.how@ed.ac.uk


Driver for length calculator of Tunabreen terminus from camera 2 (2015)
'''

#hours=[6,12]

#Import packages
import sys
import os

#Import PyTrx packages
sys.path.append('../')
from Measure import Line
from CamEnv import CamEnv


#---------------------------   Initialisation   -------------------------------

#Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU2_2015.txt'   
cammask = '../Examples/camenv_data/masks/TU2_2015_lmask.JPG'
camimgs = '../Examples/images/TU2_2015_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/TU2_manuallines/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)


#--------------------------   Calculate areas   -------------------------------

#Set up area object
terminus = Line(camimgs, cam)

#Manually plot termini
pxpts, pxline = terminus.calcManualLinesXYZ()


#----------------------------   Export data   ---------------------------------

##Write area data to txt file
#terminus.writeData(destination)

#geodata = './Results/cam1/'+day+'/shpterm/'
geodata = destination +'shapefiles/'
if not os.path.exists(geodata):
    os.makedirs(geodata)

proj = 32633
terminus.exportSHP(geodata, hours, proj)


#----------------------------   Show results   --------------------------------

##Plot and save all extent and area images
length=len(pxline)
for i in range(length):
    terminus.plotPX(i, hours,destination)
#   terminus.plotXYZ(i)


print 'Finished'
