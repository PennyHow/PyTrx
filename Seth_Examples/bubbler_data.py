# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:03:05 2021

@author: sethn
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pathlib
from datetime import datetime



dateparse = lambda x:datetime.strptime(x, '%m/%d/%Y %H:%M:%S')

# parse_dates={'datetime': ['Dates', 'Times']}, date_parser=dateparse

bubdf = pd.read_csv('C:/Users/sethn/Documents/Inglefield/envs/pytrx/Inglefield Data/ING modified data.csv')

year2019 = bubdf.iloc[0:5365, :]
year2020 = bubdf.iloc[5366:, :]


year2019.plot.scatter('datetime', 'ING Stage DCP-raw')
plt.gca().invert_yaxis()
year2020.plot.scatter('datetime', 'ING Stage DCP-raw')
