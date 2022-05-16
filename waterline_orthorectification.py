"""
Script for orthorectifying and plotting water stage positions, including 
production of Figures 5, 6 and 7.

@author: Seth Goldstein
"""

#Import packages
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.dates import DateFormatter
import matplotlib.image as mpimg
import pandas as pd

from PyTrx.CamEnv import GCPs, CamEnv, setProjection, projectUV, projectXYZ
from PyTrx.Images import CamImage


# =============================================================================
# Define camera projection
# =============================================================================

ingleCamEnv = 'cam_data/camenv.txt'
ingle_calibimg = 'cam_data/images/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.jpg'
ingle_img = CamImage(ingle_calibimg)

# Define camera environment
ingleCam = CamEnv(ingleCamEnv)

# Get DEM from camera environment
dem = ingleCam.getDEM()

# Get GCPs
inglefield_gcps = 'cam_data/gcps.txt'
gcps = GCPs(dem, inglefield_gcps, ingle_calibimg)

# Define image plane
xy = np.arange(-350,350).reshape(2, 350)
xy = np.swapaxes(xy, 0, 1)
xy[:,0].fill(50)

# Report calibration data
ingleCam.reportCalibData()

# Optimise camera environment
ingleCam.optimiseCamEnv('YPR', 'trf', show=False)

#Get inverse projection variables through camera info            
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, 
                            ingleCam._focLen, ingleCam._camCen, 
                            ingleCam._refImage, viewshed=False)

#Inverse project image coordinates using function from CamEnv object                       
ingle_xyz = projectUV(xy, invprojvars)

# Export xyz locations to csv file
df = pd.DataFrame(ingle_xyz)
df.columns= ['x', 'y', 'z']


# =============================================================================
# Calculate projection uncertainty from GCPs
# =============================================================================

# Get GCPs and reference image
worldgcp, imgcp = ingleCam._gcp.getGCPs()      

# Project gcps
worldgcp_proj = projectUV(imgcp, invprojvars)
imgcp_proj = projectXYZ(ingleCam._camloc, ingleCam._camDirection, 
                        ingleCam._radCorr, ingleCam._tanCorr, ingleCam._focLen, 
                        ingleCam._camCen, ingleCam._refImage, worldgcp)
imgcp_proj=imgcp_proj[0]

# Compute XYZ residual error
residual_xyz=[]
for i in range(len(worldgcp_proj)):
    residual_xyz.append(np.sqrt((worldgcp_proj[i][0]-worldgcp[i][0])**2 + 
                            (worldgcp_proj[i][1]-worldgcp[i][1])**2))  
residual_xyz = np.nanmean(np.array(residual_xyz))
print(f'XYZ residual: {residual_xyz} metres')

# Clip DEM to extent
demex=dem.getExtent()
xscale=dem.getCols()/(demex[1]-demex[0])
yscale=dem.getRows()/(demex[3]-demex[2])
xdmin=(demex[0]-demex[0])*xscale
xdmax=((demex[1]-demex[0])*xscale)+1
ydmin=(demex[2]-demex[2])*yscale
ydmax=((demex[3]-demex[2])*yscale)+1
demred=dem.subset(xdmin,xdmax,ydmin,ydmax)
lims = demred.getExtent() 

# Get DEM z values for plotting and change nan values
demred=demred.getZ()
demred[demred==-9999]=np.nan

# Set plotting styles
psize=100
pmark='+'
org_col='#F3F01F'
proj_col='#18A55F'
cam_col='#1FA6F3'

# Initialise figure  
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
  
# Plot image
img = mpimg.imread(ingle_calibimg)
ax1.axis([0,img.shape[1],
          img.shape[0],0])
ax1.imshow(img, origin='lower', cmap='gray', aspect="auto")

# Plot image points  
ax1.scatter(imgcp[:,0], imgcp[:,1], s=psize, marker=pmark, color=org_col)
ax1.scatter(imgcp_proj[:,0], imgcp_proj[:,1], s=psize, marker=pmark, 
            color=proj_col)

# Plot DEM 
ax2.locator_params(axis = 'x', nbins=8)
ax2.axis([lims[0],lims[1],lims[2],lims[3]])
a = ax2.imshow(demred, origin='lower', 
               extent=[lims[0],lims[1],lims[2],lims[3]], 
               cmap='magma', aspect="auto")

# Plot world points
ax2.scatter(worldgcp[:,0], worldgcp[:,1], s=psize, marker=pmark, color=org_col, 
            label='Measured GCP')
ax2.scatter(worldgcp_proj[:,0], worldgcp_proj[:,1], s=psize, marker=pmark, 
            color=proj_col, label='Projected GCP')
ax2.scatter(ingleCam._camloc[0], ingleCam._camloc[1], s=40, marker='x', 
            color=cam_col, label='Camera location')

# Add/remove annotations
for ax in [ax1, ax2]:
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
ax1.text(1230, 55, 'a', c='#000000', fontsize=30, fontweight='medium')
ax2.text(500382, 8724420, 'b', c='#000000', fontsize=30, fontweight='medium')  

# Add legends
cbar=plt.colorbar(a, orientation='horizontal', fraction=0.15, pad=0.04)
cbar.set_label(label='Elevation (m a.s.l.)', size=11)
cbar.ax.tick_params(labelsize=11) 
ax2.legend(loc=4, fontsize=11, framealpha=1)
  
# Save and show
fig.tight_layout()  
plt.savefig('figures/figure5.jpg', dpi=300)  
plt.show()


# =============================================================================
# Orthorectify all images 
# =============================================================================

def constructSeries(lines, lines_filtered, proj):
     '''Construct projected stage series from dataframe'''
     xyz_df = pd.DataFrame(columns = ['datetime', 'lineloc', 'x', 'y', 'z'])
     for index1, row1 in lines_filtered.iterrows():
         for index2, row2 in df.iterrows():
             if row1['lineloc'] == index2:
                 xyz_df = xyz_df.append({'datetime': index1, 
                                                'lineloc': row1['lineloc'], 
                                                'x': row2['x'], 
                                                'y': row2['y'], 
                                                'z': row2['z'] }, 
                                                ignore_index = True)
     slope, intercept, r_value, p_value, std_err = stats.linregress(xyz_df['lineloc'], 
                                                                    xyz_df['z'])
     projectdf_z = lines[['lineloc', 'stage_filtered']]
     projectdf_z['z']= np.nan
     projectdf_z['underwater'] = np.nan
     for index, row in projectdf_z.iterrows():
         if row['lineloc'] <= 300:
             z_row = row['lineloc'].astype(int)
             df_row = df.iloc[z_row]
             row['z'] = df_row['z']
             row['underwater'] = 1
         elif row['lineloc'] > 300:
             row['z'] = slope * row['lineloc'] + intercept
             row['underwater'] = 0  
     projectdf_z['z_normalized'] = projectdf_z['z']-projectdf_z['z'].mean()   
     projectdf_z['nice_datetimes']= projectdf_z.index.strftime("%b %d")
     return projectdf_z
 
# Construct 2019 series and export
mergedf2019 = pd.read_csv('results/manual_detection_results2019.csv', 
                          parse_dates=['key_0'], index_col='key_0')
mergedf2019_filtered = mergedf2019.loc[mergedf2019['lineloc'] <= 300]
projectdf_z2019 = constructSeries(mergedf2019, mergedf2019_filtered, df)
projectdf_z2019.to_csv('results/projectedresults2019.csv')

# Construct 2020 series and export
mergedf2020 = pd.read_csv('results/manual_detection_results2020.csv', 
                          parse_dates=['key_0'], index_col='key_0')
mergedf2020_filtered = mergedf2020.loc[mergedf2020['lineloc'] <= 300]
projectdf_z2020 = constructSeries(mergedf2020, mergedf2020_filtered, df)
projectdf_z2020.to_csv('results/projectedresults2020.csv')

# Construct 2021 series and export
mergedf2021 = pd.read_csv('results/manual_detection_results2021.csv', 
                          parse_dates=['key_0'], index_col='key_0')
mergedf2021_filtered = mergedf2021.loc[mergedf2021['lineloc'] <= 300]
projectdf_z2021 = constructSeries(mergedf2021, mergedf2021_filtered, df)
projectdf_z2021.to_csv('results/projectedresults2021.csv')


# =============================================================================
# Calculate linear regression stats
# =============================================================================

def getStats(df, year):
    '''Generate statistics from series'''
    df_na = df.dropna()
    sl, intr, rval, pval, stderr = stats.linregress(df_na['stage_filtered'], 
                                                    df_na['z_normalized'])
    print(f'\n\n{year} linear regression values (3 hours)')
    print(f'r squared: {rval}') 
    print(f'std error: {stderr}')
    
    below_water = df_na[df_na['underwater']==0]
    sl, intr, rval, pval, stderr = stats.linregress(below_water['stage_filtered'], 
                                                    below_water['z_normalized'])
    print(f'\nExtrapolated {year} linear regression values (3 hours)')
    print(f'r squared: {rval} \nstd error: {stderr}')
    
    above_water = df_na[df_na['underwater']==1]
    sl, intr, rval, pval, stderr = stats.linregress(above_water['stage_filtered'],
                                                    above_water['z_normalized'])
    print(f'\nOrthorectifed {year} linear regression values (3 hours)')
    print(f'r squared: {rval} \nstd error: {stderr}')
    
    df['bubbler_resampled'] = df['stage_filtered'].resample('1D').mean()
    df['z_resampled'] = df['z_normalized'].resample('1D').mean()
    df_na = df.dropna()
    
    sl, intr, rval, pval, stderr = stats.linregress(df_na['bubbler_resampled'],
                                                    df_na['z_resampled'])
    print(f'\n{year} linear regression values (days)') 
    print(f'r squared: {rval} \nstd error: {stderr}')

# Get stats for each time-series year
getStats(projectdf_z2019, '2019')
getStats(projectdf_z2020, '2020')   
getStats(projectdf_z2021, '2021') 


# =============================================================================
# Plot time-series for all years 
# =============================================================================

def pltTimeSeries(df):
    '''Plot yearly record as time-series'''
    fig, ax = plt.subplots(2, constrained_layout = True, sharex = 'col')
    ax[0].plot(df.index, df['z_normalized'])
    ax[0].set_ylabel('Water Level (m)')
    ax[0].grid(linestyle='dashed')
    ax[0].set_ylim(-4, 4)
    ax[1].plot(df.index, df['stage_filtered'])
    ax[1].set_ylabel('Filtered Stage (m)')
    ax[1].grid(linestyle='dashed')
    ax[1].set_ylim(-4, 4)
    ax[0].tick_params(axis = 'x', labelrotation = 45)
    ax[1].tick_params(axis = 'x', labelrotation = 45)
    plt.show()    

# Plot each year as time-series
pltTimeSeries(projectdf_z2019)  
pltTimeSeries(projectdf_z2020)
pltTimeSeries(projectdf_z2021)


# =============================================================================
# Plot tranducer-camera stage scatter comparison 
# =============================================================================

def pltScatter(ax, var1, var2, pos1, pos2, label=None):
    '''Plot two variables as scatter subplot'''
    ax[pos1,pos2].scatter(var1, var2, facecolors='none', edgecolors='black')
    ax[pos1,pos2].grid(linestyle= 'dashed')
    ax[pos1,pos2].set_ylim(bottom = -0.5, top =4.5)
    ax[pos1,pos2].set_xlim(left = -0.5, right =4.5)
    # ax[0,0].set_xlabel('Filtered Stage (m)')
    # ax[0,0].set_ylabel('Water level (m)')
    if label is not None:
        ax[0,0].set_title('2019')
 
# Prime plotting space  
fig, ax = plt.subplots(nrows=2, ncols=3, sharex = 'col', sharey= 'row')

# Plot all filtered stages as scatter subplots
pltScatter(ax, projectdf_z2019['stage_filtered'], 
           projectdf_z2019['z_normalized'], 0, 0, '2019')
pltScatter(ax, projectdf_z2020['stage_filtered'], 
           projectdf_z2020['z_normalized'], 0, 1, '2020')
pltScatter(ax, projectdf_z2021['stage_filtered'], 
           projectdf_z2021['z_normalized'], 0, 2, '2021')

# Plot all resampled stages as scatter subplots
pltScatter(ax, projectdf_z2019['bubbler_resampled'], 
           projectdf_z2019['z_resampled'], 1, 0)
pltScatter(ax, projectdf_z2020['bubbler_resampled'], 
           projectdf_z2020['z_resampled'], 1, 1)
pltScatter(ax, projectdf_z2021['bubbler_resampled'], 
           projectdf_z2021['z_resampled'], 1, 2)

# Add annotations
fig.text(0.5, 0.02, 'Pressure Transducer Stage (m)', ha='center')
fig.text(0.04, 0.5, 'Camera Derived Stage (m)', va='center', rotation='vertical')

# Save and show plot
fig.savefig('figures/figure6.png', dpi = 600)
plt.show()


# =============================================================================
# Plot three-year time-series 
# =============================================================================

def pltSeries(ax, df, dt1, dt2, pos, dt_form):
    '''Plot yearly record as subplot time-series'''
    ax[pos].plot(df.index, df['z_normalized'], color='black', 
                label='Camera Derived Stage')
    ax[pos].plot(df.index, df['stage_filtered'], color='#1f77b4', 
                label='Pressure Transducer Stage')
    ax[pos].grid(linestyle='dashed')
    ax[pos].set_ylim(-5, 5)
    ax[pos].set_xlim([dt1, dt2])
    ax[pos].xaxis.set_major_formatter(dt_form)
    ax[pos].tick_params(axis='both', which='major', direction='out', 
                        labelsize=10, width=2)

# Prime plotting space and set date formatting
fig, ax = plt.subplots(3, constrained_layout = True)
date_form = DateFormatter("%b %d")

# Plot each time-series as subplot
pltSeries(ax, projectdf_z2019, datetime.date(2019, 6, 8), 
          datetime.date(2019, 9, 8), 0, date_form)
pltSeries(ax, projectdf_z2020, datetime.date(2020, 6, 8), 
          datetime.date(2020, 9, 8), 1, date_form)
pltSeries(ax, projectdf_z2021, datetime.date(2021, 6, 8), 
          datetime.date(2021, 9, 8), 2, date_form)
fig.legend(bbox_to_anchor=(0.72,0))

# Save and show plot
fig.savefig('figures/figure7.png', dpi = 600, bbox_inches='tight')
plt.show()


# =============================================================================
print('Finished')