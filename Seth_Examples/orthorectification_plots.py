#Import packages
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.append('../')
from CamEnv import CamEnv, setProjection, projectUV, projectXYZ


#--------------------------   Create environment   ----------------------------

# Define inputs
ingleCamEnv = 'cam_env/Inglefield_CAM_camenv.txt'
ingle_calibimg = 'Inglefield_Data/INGLEFIELD_CAM/2019/INGLEFIELD_CAM_StarDot1_20190712_030000.jpg'

# Define camera environment
ingleCam = CamEnv(ingleCamEnv)

# Optimise camera environment
ingleCam.optimiseCamEnv('YPR', 'trf', show=False)

# Report camera data and show corrected image
ingleCam.reportCamData()


#----------------------   Get DEM and image for plotting  ---------------------

# Get DEM and DEM extent 
dem = ingleCam.getDEM() 
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


#------------------------------   Get GCPs   ----------------------------------

# Get GCPs and reference image
worldgcp, imgcp = ingleCam._gcp.getGCPs()      

# Get projected GCPs
invprojvars = setProjection(dem, ingleCam._camloc, ingleCam._camDirection, 
                            ingleCam._radCorr, ingleCam._tanCorr, 
                            ingleCam._focLen, ingleCam._camCen, 
                            ingleCam._refImage, viewshed=False)

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


#-------------------------   Pretty GCPs plot   -------------------------------

# Set point styles
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
plt.savefig('results/gcp_plot_pretty.jpg', dpi=300)  
plt.show()

