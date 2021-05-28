# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:12:47 2021

@author: sethn
"""

from PyTrx.Images import CamImage
import os

directory = os.getcwd()

ingle_calibimg = directory + '/Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.JPG'

calibimg = CamImage(ingle_calibimg, 'l')

calibimg.getImage()

calibimg.imageGood()

calibimg._checkImage(ingle_calibimg)

calibimg.getImageType()
