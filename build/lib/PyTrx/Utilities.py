#PyTrx (c) by Penelope How, Nick Hulton, Lynne Buie
#
#PyTrx is licensed under a MIT License.
#
#You should have received a copy of the license along with this
#work. If not, see <https://choosealicense.com/licenses/mit/>.

'''
The Utilities module contains stand-alone functions needed for simple 
plotting and interpolation. These merely serve as examples and it is highly 
encouraged to adapt these functions for visualising datasets.
'''

#Import packages
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

#------------------------------------------------------------------------------  

def plotGCPs(gcps, img, imn, dem, camloc, extent=None):
    '''Function to show the ground control points, on the image and the DEM.
    
    :param gcps: GCPs
    :type gcps: arr
    :param img: Image array
    :type img: arr
    :param imn: Image name 
    :type imn: str 
    :param dem: :class:`PyTrx.DEM.ExplicitRaster` object 
    :type dem: arr
    :param extent: DEM extent indicator, default to None
    :type extent: list, optional
    :returns: A figure with the plotted uv and xyz GCPs
    '''       
    #Get GCPs      
    worldgcp=gcps[0]
    imgcp = gcps[1]
    
    #Get DEM and DEM extent (if specified)
    demex=dem.getExtent()
    xscale=dem.getCols()/(demex[1]-demex[0])
    yscale=dem.getRows()/(demex[3]-demex[2])
    
    if extent is not None:
        xmin=extent[0]
        xmax=extent[1]
        ymin=extent[2]
        ymax=extent[3]
            
        xdmin=(xmin-demex[0])*xscale
        xdmax=((xmax-demex[0])*xscale)+1
        ydmin=(ymin-demex[2])*yscale
        ydmax=((ymax-demex[2])*yscale)+1
        demred=dem.subset(xdmin,xdmax,ydmin,ydmax)
        lims = demred.getExtent()
        
    else:
        xdmin=(demex[0]-demex[0])*xscale
        xdmax=((demex[1]-demex[0])*xscale)+1
        ydmin=(demex[2]-demex[2])*yscale
        ydmax=((demex[3]-demex[2])*yscale)+1
        demred=dem.subset(xdmin,xdmax,ydmin,ydmax)
        lims = demred.getExtent() 
    
    #Get DEM z values for plotting
    demred=demred.getZ()
    
    #Plot image points    
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.canvas.set_window_title('GCP locations of '+str(imn))
    ax1.axis([0,img.shape[1],
              img.shape[0],0])  
    ax1.imshow(img, origin='lower', cmap='gray')
    ax1.scatter(imgcp[:,0], imgcp[:,1], color='red')
    
    #Plot world points
    ax2.locator_params(axis = 'x', nbins=8)
    ax2.axis([lims[0],lims[1],lims[2],lims[3]])
    ax2.imshow(demred, origin='lower', 
               extent=[lims[0],lims[1],lims[2],lims[3]], cmap='gray')
    ax2.scatter(worldgcp[:,0], worldgcp[:,1], color='red')
    ax2.scatter(camloc[0], camloc[1], color='blue')
    plt.show()
#    plt.close()
 
    
def plotPrincipalPoint(camcen, img, imn):
    """Function to show the principal point on the image, along with the 
    GCPs.

    :param camcen: Principal point coordinates
    :type camcen: list
    :param img: Image array
    :type img: arr
    :param imn: Image name 
    :type imn: str 
    :returns: A figure with the prinicipal point plotted onto the image
    """
    #Get the camera centre from the intrinsic matrix 
    ppx = camcen[0] 
    ppy = camcen[1]       
     
    #Plot image points
    fig, (ax1) = plt.subplots(1)
    fig.canvas.set_window_title('Principal Point of '+str(imn))
    ax1.axis([0,img.shape[1],
              img.shape[0],0])        

    ax1.imshow(img, origin='lower', cmap='gray')
    ax1.scatter(ppx, ppy, color='yellow', s=100)
    ax1.axhline(y=ppy)
    ax1.axvline(x=ppx)
    plt.show() 
#    plt.close()
    
 
def plotCalib(matrix, distortion, img, imn):
    """Function to show camera calibration. Two images are plotted, the 
    first with the original input image and the second with the calibrated
    image. This calibrated image is corrected for distortion using the 
    distortion parameters held in the :class:`PyTrx.CamEnv.CamCalib` object.
    
    :param matrix: Camera matrix
    :type matrix: arr
    :param distortion: Distortion cofficients
    :type distortion: arr
    :param img: Image array
    :type img: arr
    :param imn: Image name 
    :type imn: str 
    :returns: A figure of an uncorred image and corrected image
    """
    #Calculate optimal camera matrix 
    h = int(img.shape[0])
    w = int(img.shape[1])
    newMat, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, 
                                                (w,h), 1, (w,h))

    #Correct image for distortion                                                
    corr_image = cv2.undistort(img, matrix, 
                               distortion, newCameraMatrix=newMat)
   
    
    #Plot uncorrected and corrected images                         
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.canvas.set_window_title('Calibration output of '+str(imn))
    implot1 = ax1.imshow(img)
    implot1.set_cmap('gray')    
    ax1.axis([0,w,h,0])
    implot2 = ax2.imshow(corr_image)        
    implot2.set_cmap('gray')    
    ax2.axis([0,w,h,0])    
    plt.show()
#    plt.close()


def plotResiduals(img, ims, gcp1, gcp2, gcp3):
    """Function to plot sets of points to show offsets. This is 
    commonly used for inspecting differences between image GCPs and projected 
    GCPs, e.g. within the optimiseCamera function.
        
    :param img: Image array
    :type img: arr
    :param ims: Image dimension (height, width)
    :type ims: list
    :param gcp1: Array with uv positions of image gcps
    :type gcp1: arr
    :param gcp2: Array with initial uv positions of projected gcps
    :type gcp2: arr
    :param gcp3: Array with optimised uv positions of projected gcps
    :type gcp3: arr    
    :returns: A figure of an image, plotted with uv gcps, initial projected gcps, and optimised gcps    
    """ 
    #Plot image                
    fig, (ax1) = plt.subplots(1)
    fig.canvas.set_window_title('Average residual difference: ' + 
                                str(np.nanmean(gcp3-gcp2)) + ' px')
    ax1.axis([0,ims[1],ims[0],0])
    ax1.imshow(img, cmap='gray')
    
    #Plot UV GCPs
    ax1.scatter(gcp1[:,0], gcp1[:,1], color='red', marker='+', 
                label='UV')
    
    #Plot projected XYZ GCPs
    ax1.scatter(gcp2[:,0], gcp2[:,1], color='green', 
                marker='+', label='Projected XYZ (original)')

    #Plot optimised XYZ GCPs if given
    ax1.scatter(gcp3[:,0], gcp3[:,1], color='blue', 
                marker='+', label='Projected XYZ (optimised)')
    
    #Add legend and show plot
    ax1.legend()
    plt.show()        
        
    
def plotAreaPX(uv, img, show=True, save=None):
    """Plot figure with image overlayed with pixel area features. 
              
    :param uv: Input uv coordinates for plotting over image
    :type uv: arr          
    :param img: Image array
    :type img: arr
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted area measurements overlaid onto a given image 
    """          
    #Get image size
    imsz = img.shape
          
    #Create image plotting window
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    
    #Plot image
    implot = ax1.imshow(img)        
    implot.set_cmap('gray')    
    ax1.axis([0,imsz[1],imsz[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('UV output: ' + save)
    else:
        fig.canvas.set_window_title('UV output')
    
    #Extract xy data from features               
    for shp in uv: 
        xl=[]
        yl=[]
        for pt in shp:
            if len(pt)==1:
                xl.append(pt[0][0])
                yl.append(pt[0][1])
            elif len(pt)==2:
                xl.append(pt[0])
                yl.append(pt[1])
            else:
                print('Unrecognised point structure for plotting')
                pass
            
        ax1.plot(xl, yl, c='#FFFF33', linestyle='-') 
    
    #Save figure
    if save != None:
        plt.savefig(save, dpi=300) 
    
    #Show figure    
    if show is True:
        plt.show()    


def plotLinePX(uv, img, show=True, save=None):
    """Plot figure with image overlayed with pixel line features.
    
    :param uv: Input uv coordinates for plotting over image
    :type uv: arr          
    :param img: Image array
    :type img: arr
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted line measurements overlaid onto a given image    
    """         
    #Get image size
    imsz = img.shape
          
    #Create image plotting window
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    
    #Plot image
    implot = ax1.imshow(img)        
    implot.set_cmap('gray')    
    ax1.axis([0,imsz[1],imsz[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('UV output: ' + save)
    else:
        fig.canvas.set_window_title('UV output')
                   
    ax1.plot(uv[:,0], uv[:,1], c='#FFFF33', linestyle='-') 
    
    #Save figure
    if save != None:
        plt.savefig(save, dpi=300) 
    
    #Show figure    
    if show is True:
        plt.show()    
    
 
def plotVeloPX(uvvel, uv0, uv1, img, show=True, save=None):
    """Plot figure with image overlayed with pixel velocities. UV data are
    depicted as the uv point in img0 and the corresponding pixel velocity as a 
    proportional arrow (computed using the arrowplot function).
       
    :param uvvel: Input pixel velocities
    :type uvvel: arr        
    :param uv0: Coordinates (u,v) for points in first image
    :type uv0: arr          
    :param uv1: Coordinates (u,v) for points in second image
    :type uv1: arr          
    :param img: Image array
    :type img: arr
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted point velocities overlaid onto a given image    
    """           
    #Get image size
    imsz = img.shape
          
    #Create image plotting window
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    
    #Plot image
    implot = ax1.imshow(img)        
    implot.set_cmap('gray')    
    ax1.axis([0,imsz[1],imsz[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])  
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('UV output: ' + save)
    else:
        fig.canvas.set_window_title('UV output')
        
    #Plot xy positions onto images
    uvplt = ax1.scatter(uv0[:,0,0], uv0[:,0,1], c=uvvel, s=30, vmin=0,
                        vmax=max(uvvel), cmap=plt.get_cmap("gist_ncar"))
    plt.colorbar(uvplt, ax=ax1)

    #Plot arrows
    xar,yar=arrowplot(uv0[:,0,0], uv0[:,0,1], uv1[:,0,0], uv1[:,0,1], 
                      scale=5.0, headangle=15)
    ax1.plot(xar,yar,color='black',linewidth=1)
       
    #Save figure
    if save != None:
        plt.savefig(save, dpi=300) 
    
    #Show figure    
    if show is True:
        plt.show()    
    
       
def plotAreaXYZ(xyz, dem, show=True, save=None):    
    """Plot figure with image overlayed with xyz coordinates representing 
    either areas or line features.
    
    :param xyz: Input xyz coordinates for plotting
    :type xyz: arr     
    :param dem: Underlying DEM for plotting over
    :type dem: :class:`PyTrx.DEM.ExplicitRaster`
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted areas overlaid onto a given DEM    
    """                           
    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('XYZ output: ' + save)
    else:
        fig.canvas.set_window_title('XYZ output')        
        
    #Prepare DEM for plotting
    if dem is not None:
        demextent = dem.getExtent()
        demz = dem.getZ()           
    
        #Plot DEM and set cmap
        implot = ax1.imshow(demz, origin='lower', extent=demextent)
        implot.set_cmap('gray')
        ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
                           
    #Extract xy data from features               
    for shp in xyz: 
        xl=[]
        yl=[]
        for pt in shp:
            xl.append(pt[0])
            yl.append(pt[1])
        ax1.plot(xl, yl, c='#FFFF33', linestyle='-')        
    
    #Save figure
    if save != None:
        plt.savefig(save)
    
    #Show figure
    if show is True:
        plt.show()         


def plotLineXYZ(xyz, dem, show=True, save=None):    
    """Plot figure with image overlayed with xyz coordinates representing 
    either areas or line features.
    
    :param xyz: Input xyz coordinates for plotting
    :type xyz: arr  
    :param dem: Underlying DEM for plotting over
    :type dem: :class:`PyTrx.DEM.ExplicitRaster`
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted lines overlaid onto a given DEM    
    """                                 
    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('XYZ output: ' + save)
    else:
        fig.canvas.set_window_title('XYZ output')        
        
    #Prepare DEM for plotting
    if dem is not None:
        demextent = dem.getExtent()
        demz = dem.getZ()           
    
        #Plot DEM and set cmap
        implot = ax1.imshow(demz, origin='lower', extent=demextent)
        implot.set_cmap('gray')
        ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
                           
    ax1.plot(xyz[:,0], xyz[:,1], c='#FFFF33', linestyle='-')        
    
    #Save figure
    if save != None:
        plt.savefig(save)
    
    #Show figure
    if show is True:
        plt.show()
        

def plotVeloXYZ(xyzvel, xyz0, xyz1, dem, show=True, save=None):
    """Plot figure with image overlayed with xyz velocities. XYZ data are
    depicted as the xyz point in img0 and the corresponding velocity as a 
    proportional arrow (computed using the arrowplot function).
    
    :param xyzvel: Input xyz velocities 
    :type xyzvel: arr           
    :param xyz0: Coordinates (x,y) for points in first image
    :type xyz0: arr
    :param xyz1: Coordinates (x,y) for points in second image 
    :type xyz1: arr
    :param dem: Underlying DEM for plotting over
    :type dem: :class:`PyTrx.DEM.ExplicitRaster`
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with plotted points denoting velocities, overlaid onto a given DEM                          
    """  
    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Set window name
    if save != None:
        fig.canvas.set_window_title('XYZ output: ' + save)
    else:
        fig.canvas.set_window_title('XYZ output')
        
    #Prepare DEM for plotting
    if dem is not None:
        demextent = dem.getExtent()
        demz = dem.getZ()           
    
        #Plot DEM and set cmap
        implot = ax1.imshow(demz, origin='lower', extent=demextent)
        implot.set_cmap('gray')   
                                      
    #Scatter plot velocity points from img0                 
    xyzplt = ax1.scatter(xyz0[:,0], xyz0[:,1], c=xyzvel, s=30, 
                         cmap=plt.get_cmap('gist_ncar'), 
                         vmin=0, vmax=max(xyzvel))  

    #Plot vector arrows denoting direction                             
    xar,yar=arrowplot(xyz0[:,0],xyz0[:,1],xyz1[:,0],xyz1[:,1],scale=5.0,
                      headangle=15)
    ax1.plot(xar,yar,color='black', linewidth=1)

    #Plot color bar
    plt.colorbar(xyzplt, ax=ax1)
                
    #Save figure
    if save != None:
        plt.savefig(save, dpi=300)
     
    #Show figure
    if show is True:
        plt.show()
        

def plotInterpolate(grid, pointextent, dem, show=True, save=None):
    """Function to plot the results of the velocity interpolation process for 
    a particular image pair.
    
    :param grid: Numpy grid. It is recommended that this is constructed using the interpolateHelper function
    :type grid: arr
    :param pointextent: Grid extent
    :type pointextent: list
    :param dem: Underlying DEM for plotting over
    :type dem: :class:`PyTrx.DEM.ExplicitRaster`
    :param show: Flag to denote whether the figure is shown, defatuls to True
    :type show: bool, optional 
    :param save: Destination file to save figure to, defaults to None
    :type save: str, optional
    :returns: A figure with interpolated velocity results overlaid onto a given DEM
    """  
    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    ax1.set_xticks([])
    ax1.set_yticks([])    

    #Set name of window
    if save != None:
        fig.canvas.set_window_title('XYZ interpolate: ' + save)
    else:
        fig.canvas.set_window_title('XYZ interpolate')        
        
    #Prepare DEM for plotting
    if dem != None:
        demextent = dem.getExtent()
        demz = dem.getZ()           
    
        #Plot DEM and set cmap
        implot = ax1.imshow(demz, origin='lower', extent=demextent)
        implot.set_cmap('gray')
        ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
                    
    #Plot interpolated grid with colour bar legend 
    interp = ax1.imshow(grid, origin='lower', cmap=plt.get_cmap("gist_ncar"), 
                        extent=pointextent, alpha=0.5)                
    plt.colorbar(interp, ax=ax1)
    
    #Save if variable not none
    if save != None:
        plt.savefig(save + 'interp.jpg', dpi=300)  
            
    #Show plot
    if show is True:
        plt.show()
    

def interpolateHelper(xyzvel, xyz0, xyz1, method='linear'):
    """Function to interpolate a point dataset. This uses functions of 
    the SciPy package to set up a grid (grid) and then interpolate using a
    linear interpolation method (griddata). Methods are those compatible with 
    SciPy's interpolate.griddata function: 'nearest', 'cubic' and 'linear'.
    
    :param xxyzvel: Input xyz velocities
    :type xyzvel: arr
    :param xyz0: Coordinates (x,y) for points in first image
    :type xyz0: arr
    :param xyz1: Coordinates (x,y) for points in second image
    :type xyz1: arr
    :param method: Interpolation method, defaults to 'linear'
    :type method: str, optional
    :returns: An interpolated grid of points and grid extent
    :rtype: list
    """            
    #Create empty lists for xyz information without NaNs
    velo=[]  
    x1=[]
    x2=[]
    y1=[]
    y2=[] 
                                     
    #Remove NaN values from velocities and points                   
    for v,sx,ex,sy,ey in zip(xyzvel, xyz0[:,0], xyz1[:,0], xyz0[:,1], 
                             xyz1[:,1]):                          
        if np.isnan(v)==False:
            velo.append(v)                            #xyz velocities
            x1.append(sx)                                #pt0 x values
            x2.append(ex)                                #pt1 x values
            y1.append(sy)                                #pt0 y values
            y2.append(ey)                                #pt1 y values
        elif np.isnan(v)==True:
            print('\nNaN value removed for interpolation')
                              
    #Bound point positions in array for grid construction
    newpts=np.array([x1,y1]).T  
       
    #Define gridsize
    gridsize=10.
    
    #Define grid using point extent
    minx=divmod(min(x1),gridsize)[0]*gridsize
    miny=divmod(min(y1),gridsize)[0]*gridsize
    maxx=(divmod(max(x2),gridsize)[0]+1)*gridsize
    maxy=(divmod(max(y2),gridsize)[0]+1)*gridsize
    pointsextent=[minx,maxx,miny,maxy]   
         
    #Generate buffer around grid
    incrsx=((maxx-minx)/gridsize)+1
    incrsy=((maxy-miny)/gridsize)+1

    #Construct grid dimensions
    grid_y,grid_x = np.mgrid[miny:maxy:complex(incrsy),
                             minx:maxx:complex(incrsx)]
    
    #Interpolate the velocity points to the grid
    grid = griddata(newpts, np.float64(velo), (grid_x, grid_y), 
                    method=method)
     
#    DEVELOPMENT TO BE COMPLETED: snr grid
#    error = griddata(newpts, np.float64(snrs), (grid_x, grid_y), 
#                     method=method)      
                
    return grid, pointsextent 
           
    
def arrowplot(xst, yst, xend, yend, scale=1.0, headangle=15, headscale=0.2):    
    """Plot arrows to denote the direction and magnitude of the displacement. 
    Direction is indicated by the bearing of the arrow, and the magnitude is 
    indicated by the length of the arrow.
    
    :param x0: X coordinates for pt0
    :type x0: arr
    :param y0: Y coordinates for pt0
    :type y0: arr
    :param x1: X coordinates for pt1
    :type x1: arr
    :param y1: Y coordinates for pt1
    :type y1: arr
    :param scale: Arrow scale, defaults to 1.0
    :type scale: int, optional
    :param headangle: Plotting angle, defaults to 15
    :type headangle: int, optional
    :param headscale: Arrow head scale, defaults to 0.2
    :type headscale: int, optional
    :returns: Arrow plots as two arrays denoting x and y coordinates 
    :rtype: arr
    """    
    #Define plotting angle
    angle=math.pi*headangle/180.

    xs=[]
    ys=[]
    
    #Iterate through xy point data
    for i in range(len(xst)):
        #Get xy start position
        x1=xst[i]
        y1=yst[i]
        
        #Get xy end position
        x2=xend[i]
        y2=yend[i]
        
        #Calculate point displacement
        xd=x2-x1
        yd=y2-y1
        dist=math.sqrt(xd*xd+yd*yd)
        
        #Determine plotting angle
        sinTheta = xd / dist
        cosTheta = yd / dist
        sinTheta=min(sinTheta,1.0)
        cosTheta=min(cosTheta,1.0)
        sinTheta=max(sinTheta,-1.0)
        cosTheta=max(cosTheta,-1.0)
        aSinTheta = math.asin(sinTheta)
        
        #Determine length based on a given scale
        x2=x1+(xd*scale)
        y2=y1+(yd*scale)
        hs=dist*headscale

        #These conditions give an angle between 0 and 2 Pi radians
        #You should test them to make sure they are correct
        if (sinTheta >= 0.0 and cosTheta >= 0.0):
           theta = aSinTheta
        elif (cosTheta < 0.0):
           theta = math.pi - aSinTheta
        else:
           theta = (2.0 * math.pi + aSinTheta)
           
        theta=theta+math.pi
        x3=x2+hs*math.sin(theta+angle)
        x4=x2+hs*math.sin(theta-angle)
        y3=y2+hs*math.cos(theta+angle)
        y4=y2+hs*math.cos(theta-angle)
        
        #Append arrow plotting information
        xs.append(x1)
        ys.append(y1)
        xs.append(x2)
        ys.append(y2)
        xs.append(x3)
        ys.append(y3)
        xs.append(float('NaN'))
        ys.append(float('NaN'))
        xs.append(x2)
        ys.append(y2)
        xs.append(x4)
        ys.append(y4)
        xs.append(float('NaN'))
        ys.append(float('NaN'))
    
    #Return xy arrow plotting information
    return xs,ys   

       
#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   
