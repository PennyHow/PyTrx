'''
PYTRX UTILITIES MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This module, Utilities, contains functions needed for simple plotting and 
interpolation. These merely serve as examples and it is highly encouraged to 
adapt these functions for visualising datasets.

Functions available in Utilities
plotInterpolate:                Function to plot the results of the 
                                interpolation process for a particular 
                                timestep.
plotPX:                         Return image overlayed with pixel velocities/
                                areas/lines for a given image number. 
                                Measurement is distinguished automatically from 
                                the given Measure class object (i.e. Velocity, 
                                Area, or Line).
plotXYZ:                        Plot xyz points of real velocities/areas/lines 
                                for a given image number. Measurement is 
                                distinguished automatically from the given 
                                Measure class object (i.e. Velocity, Area, or 
                                Line).
interpolateHelper:              Function to interpolate a point dataset. This 
                                uses functions of the SciPy package to set up a 
                                grid (grid) and then interpolate using a linear 
                                interpolation method (griddata).
arrowplot:                      Function to plot arrows to denote the direction 
                                and magnitude of the displacement. Direction is 
                                indicated by the bearing of the arrow, and the
                                magnitude is indicated by the length of the 
                                arrow.
             
@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton 
         Lynne Addison
'''

#Import packages
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import griddata
import sys

#------------------------------------------------------------------------------  
    
def plotInterpolate(a, number, grid, pointextent, show=True, save=None, 
                    crop=None):
    '''Function to plot the results of the velocity interpolation process for 
    a particular timestep.

    Inputs
    a:              A Velocity object.
    number:         Sequence number.
    grid:           Numpy grid. It is recommended that this is constructed 
                    using the interpolateHelper function.
    pointextent:    Grid extent.
    show:           Flag to denote whether the figure is shown.
    save:           Destination file to save figure to.
    crop:           Crop output if desired ([x1,x2,y1,y2]).
    ''' 
    #Get image name
    imn=a._imageSet[number].getImagePath().split('\\')[1]   

    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': XYZ interpolate')
    ax1.set_xticks([])
    ax1.set_yticks([])
        
    #Prepare DEM
    demobj=a._camEnv.getDEM()
    demextent=demobj.getExtent()
    dem=demobj.getZ()
   
    #Get camera position (from the CamEnv class)
    post = a._camEnv._camloc            
    
    #Plot DEM and set cmap
    implot = ax1.imshow(dem, origin='lower', extent=demextent)
    implot.set_cmap('gray')
    ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
        
    #Plot camera location
    ax1.scatter(post[0], post[1], c='g')
     
    #Crop extent
    if crop != None:
        ax1.axis([crop[0], crop[1], crop[2], crop[3]])
    
    #Plot grid onto DEM
    interp = ax1.imshow(grid, 
                        origin='lower', 
                        cmap=plt.get_cmap("gist_ncar"), 
                        extent=pointextent, 
                        alpha=0.5) #alpha=1
               
    plt.colorbar(interp, ax=ax1)
    
    #Save if flag is true
    if save != None:
        plt.savefig(save, dpi=300)
        
    #Show plot
    if show is True:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
    
    plt.close()


def plotPX(a, number, dest=None, crop=None, show=True):
    '''Return image overlayed with pixel velocities/areas/lines for a given 
    image number. Measurement is distinguished automatically from the given
    Measure class object (i.e. Velocity, Area, or Line).
    
    Inputs
    a:              An Area/Line/Velocity object.
    number:         Sequence number.
    dest:           Destination to save plot to.
    crop:           Crop output if desired ([x1,x2,y1,y2]).
    show:           Flag to denote whether the figure is shown.
    '''
    
    #Call corrected/uncorrected image
    if a._calibFlag is True:
        img=a._imageSet[number].getImageCorr(a._camEnv.getCamMatrixCV2(), 
                                            a._camEnv.getDistortCoeffsCv2())      
    else:
        img=a._imageSet[number].getImageArray() 
    
    #Get image name
    imn=a._imageSet[number].getImagePath().split('\\')[1] 
    
    #Get image size
    imsz = a._imageSet[number].getImageSize()
          
    #Create image plotting window
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': UV output ')
    implot = ax1.imshow(img)        
    implot.set_cmap('gray')    
    ax1.axis([0,imsz[1],imsz[0],0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #Crop image if desired
    if crop != None:
        ax1.axis([crop[0],crop[1],crop[2],crop[3]])


    #Plot pixel polygon areas
    if hasattr(a, '_pxpoly'):
        polys = a._pxpoly[number]           #Get polygon
        for p in polys:
            x=[]
            y=[]
            for xy in p:
                if len(xy)==1:
                    x.append(xy[0][0])      #Get X coordinates
                    y.append(xy[0][1])      #Get Y coordinates
                elif len(xy)==2:
                    x.append(xy[0])
                    y.append(xy[1])
            ax1.plot(x,y,'w-')              #Plot polygon

    
    #Plot pixel lines        
    elif hasattr(a, '_pxpts'):
        line = a._pxpts[number]             #Get line        
        x=[]
        y=[]
        for xy in line:
            x.append(xy[0])                 #Get X coordinates
            y.append(xy[1])                 #Get Y coordinates
        ax1.plot(x,y,'w-')                  #Plot line


    #Plot velocity points
    elif hasattr(a, '_uvvel'):
        velocity = a._uvvel[number]         #Get Velocities
        pt0 = a._uv0[number]                #Get point positions from im0
        
        if a._uv1corr[number] is None:           
            pt1 = a._uv1[number]
        else:
            pt1 = a._uv1corr[number]        #Get point positions from im1
                
        pt0x=pt0[:,0,0]                     #pt0 x values
        pt0y=pt0[:,0,1]                     #pt0 y values
        pt1x=pt1[:,0,0]                     #pt1 x values
        pt1y=pt1[:,0,1]                     #pt1 y values
        
        #Plot xy positions onto images
        uvplt = ax1.scatter(pt0x, pt0y, c=velocity, s=50, vmin=0,
                            vmax=max(velocity), cmap=plt.get_cmap("gist_ncar"))
        plt.colorbar(uvplt, ax=ax1)

        #Plot arrows
        xar,yar=arrowplot(pt0x, pt0y, pt1x, pt1y, scale=5.0,headangle=15)
        ax1.plot(xar,yar,color='black')
  
    else:
        print '\nUnrecognised Area/Line class object'
        sys.exit(1)
     
    #Save figure
    if dest != None:
        plt.savefig(dest + 'uv_' + imn, dpi=300) 
    
    #Show figure    
    if show is True:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()    
    
    #Close figure                        
    plt.close()
    
    #Clear memory
    a._imageSet[number].clearAll()  
    
    
def plotXYZ(a, number, dest=None, crop=None, show=True, dem=True):
    '''Plot xyz points of real velocities/areas/lines for a given image number.
    Measurement is distinguished automatically from the given Measure class 
    object (i.e. Velocity, Area, or Line).
    
    Inputs
    a:              An Area/Line/Velocity object.
    number:         Sequence number.
    dest:           Destination to save plot to.
    crop:           Crop output if desired ([x1,x2,y1,y2]).
    show:           Flag to denote whether the figure is shown.
    dem:            Flag to denote whether DEM is shown or not.
    '''                           

    #Get image name
    imn=a._imageSet[number].getImagePath().split('\\')[1]   

    #Set-up plot
    fig, (ax1) = plt.subplots(1, figsize=(20,10))
    fig.canvas.set_window_title(imn + ': XYZ output')
    ax1.set_xticks([])
    ax1.set_yticks([])
        
    #Prepare DEM if desired
    if dem is True:
        demobj=a._camEnv.getDEM()
        demextent=demobj.getExtent()
        dem=demobj.getZ()
   
        #Get camera position (from the CamEnv class)
        post = a._camEnv._camloc            
    
        #Plot DEM and set cmap
        implot = ax1.imshow(dem, origin='lower', extent=demextent)
        implot.set_cmap('gray')
        ax1.axis([demextent[0], demextent[1],demextent[2], demextent[3]])
        
        #Plot camera location
        ax1.scatter(post[0], post[1], c='g')
     
    #Crop extent
    if crop != None:
        ax1.axis([crop[0], crop[1], crop[2], crop[3]])
                
    #If object has area attributes
    if hasattr(a, '_realpoly'):

        #Get xyz points for polygons in a given image
        xyz = a._realpoly[number]
        
        #Extract xy data from poly pts
        count=1                
        for shp in xyz: 
            xl=[]
            yl=[]
            for pt in shp:
                xl.append(pt[0])
                yl.append(pt[1])
            lab = 'Area ' + str(count)
            ax1.plot(xl, yl, c=np.random.rand(3,1), linestyle='-', label=lab)
            count=count+1
        
        ax1.legend()
           
    #If object has line attributes            
    elif hasattr(a, '_realpts'):
        
        #Get xyz points for lines in a given image
        line = a._realpts[number]
        
        #Get xy data from line pts
        xl=[]
        yl=[]        
        for pt in line:
            xl.append(pt[0])
            yl.append(pt[1])
        
        #Plot line points and camera position on to DEM image        
        ax1.plot(xl, yl, 'y-')
        
    #If object has velocity attributes            
    elif hasattr(a, '_xyzvel'):
        
        #Get xyz points and velocities 
        xyzvelo = a._xyzvel[number]               #xyz velocity
        xyzstart = a._xyz0[number]                #xyz pt0 position
        xyzend = a._xyz1[number]                  #xyz pt1 position
           
        #Get xyz positions from image0 and image1
        xyz_xs = xyzstart[:,0]                      #pt0 x values
        xyz_xe = xyzend[:,0]                        #pt1 x values
        xyz_ys = xyzstart[:,1]                      #pt0 y values
        xyz_ye = xyzend[:,1]                        #pt1 y values
                                  
        #Scatter plot velocity                 
        xyzplt = ax1.scatter(xyz_xs, xyz_ys, c=xyzvelo, s=50, 
                             cmap=plt.get_cmap('gist_ncar'), 
                             vmin=0, vmax=max(xyzvelo))  

        #Plot vector arrows denoting direction                             
        xar,yar=arrowplot(xyz_xs,xyz_ys,xyz_xe,xyz_ye,scale=5.0,headangle=15)
        ax1.plot(xar,yar,color='black')

        #Plot color bar
        plt.colorbar(xyzplt, ax=ax1)
            
    else:
        print '\nUnrecognised Area/Line class object'
        sys.exit(1)
    
    #Save figure
    if dest != None:
        plt.savefig(dest  + 'xyz_' + imn, dpi=300)
     
    #Show figure
    if show is True:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
 
    #Close plot   
    plt.close()


def interpolateHelper(a, number, method='linear'):
    '''Function to interpolate a point dataset. This uses functions of 
    the SciPy package to set up a grid (grid) and then interpolate using a
    linear interpolation method (griddata).
    Methods are those compatible with SciPy's interpolate.griddata function: 
    "nearest", "cubic" and "linear"
    
    Inputs
    a:              A Velocity object.
    number:         Sequence number.
    method:         Interpolation method ("nearest"/"cubic"/"linear").
    filt:           Flag to denote if points should be filtered or not.
    
    Output
    grid:           Interpolated grid. 
    pointsextent:   Grid extent. 
    '''   
    
    #Get xyz points and velocities 
    xyzvelo = a._xyzvel[number]                          #xyz velocity
    xyzstart = a._xyz0[number]                           #xyz pt0 position
    xyzend = a._xyz1[number]                             #xyz pt1 position
       
    #Get xyz positions from image0 and image1
    x1 = xyzstart[:,0]                                   #pt0 x values
    x2 = xyzend[:,0]                                     #pt1 x values
    y1 = xyzstart[:,1]                                   #pt0 y values
    y2 = xyzend[:,1]                                     #pt1 y values 

    for v,w,x,y,z in zip(xyzvelo,x1,x2,y1,y2):
        if np.isnan(v) is True:
            print v,w,x,y,z
        elif np.isnan(w) is True:
            print v,w,x,y,z
        elif np.isnan(x) is True:
            print v,w,x,y,z
        elif np.isnan(y) is True:
            print v,w,x,y,z
        elif np.isnan(z) is True:
            print v,w,x,y,z

            
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
    grid = griddata(newpts, np.float64(xyzvelo), (grid_x, grid_y), 
                    method=method)
     
#    DEVELOPMENT TO BE COMPLETED: snr grid
#    error = griddata(newpts, np.float64(snrs), (grid_x, grid_y), 
#                     method=method)      
                
    return grid, pointsextent 
           
    
def arrowplot(xst,yst,xend,yend,scale=1.0,headangle=15,headscale=0.2):    
    '''Function to plot arrows to denote the direction and magnitude of the
    displacement. Direction is indicated by the bearing of the arrow, and the
    magnitude is indicated by the length of the arrow.
    
    Inputs
    xst:            x coordinates for pt0.
    yst:            y coordinates for pt0.
    xend:           x coordinates for pt1.
    yend:           y coordinates for pt1.
    scale:          Arrow scale.
    headangle:      Plotting angle.
    headscale:      Arrow head scale.
    
    Outputs
    xs              x coordinates for arrowplot.
    ys:             y coordinates for arrowplot.
    '''
    
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


##Development: sparse filtering using KD Tree. Currently works for data arrays
##where values have a large range, but bug in function means it doesn't work
##for values with a small range. Either alternative needs to be found or snr
##evaluation is purely used as a filtering technique.

#def filterSparse(data,numNearest=12,threshold=2,item=2):
#    '''A function to remove noise from a sparse dataset using a filtering
#    method. This removes points if they are within a specified value 
#    (threshold) from the mean of a specified number of nearest neighbour 
#    points (numNearest). The item field identifies which column of the 
#    array holds the field to be filtered on.
#    
#    This function works best if called iteratively, as more than one point 
#    may be anomolous compared to neighbouring ranges.
#    
#    Inputs
#    data:              Input data to filter.
#    numNearest:     Sample window to assess filtering.
#    threshold:      Threshold range of neighbouring values.
#    item:           Number of neighbouring values to retrieve for assessment.
#    
#    Output
#    goodset:        Array of filtered data. 
#    ''' 
#    
#    #Get XY point data
#    XY=data[:,0:2]
#
#    #Set up KD tree with XY point data
#    tree=spatial.KDTree(XY, 50)
#    
#    goodset=[]
#    for point in XY:        
#        #Get n nearest neighbouring points
#        d,k=tree.query(point, numNearest)
#       
#        #Get the mean and standard deviation for these points
#        stdev=np.std(data[k[1:],item])
#        m=np.mean(data[k[1:],item])
#       
#        #Get the data value for neighbouring points
#        value=data[k[0],item]
#       
#        #Append point to goodset if is within threshold range of neighbours
#        if (value>(m-(stdev*threshold)) and value<(m+(stdev*threshold))):
#            goodset.append(data[k[0]])
#        
#    return np.array(goodset)

       
#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   