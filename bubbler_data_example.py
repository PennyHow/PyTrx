"""
Example script for importing and simple plotting of river stage data

@author: Seth Goldstein
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Read from file
year2019 = pd.read_csv('river_stage_data/modified_2019.csv', delimiter='\t')
year2020 = pd.read_csv('river_stage_data/modified_2020.csv')

# Construct datetime series
year2019['datetime'] = [datetime.strptime(x, '%m/%d/%Y %H:%M:%S') 
                        for x in list(year2019['Dates'] + ' ' + year2019['Times'])]
year2020['datetime'] = [datetime.strptime(x, '%m/%d/%Y %H:%M:%S') 
                        for x in list(year2020['Dates'] + ' ' + year2020['Times'])]

# Plot
year2019.plot.scatter('datetime', 'ING Stage DCP-raw')
plt.gca().invert_yaxis()
year2020.plot.scatter('datetime', 'ING Stage DCP-raw')
