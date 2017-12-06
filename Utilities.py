'''
PYTRX UTILITIES MODULE

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This module, Utilities, contains functions needed for filtering, interpolation,
and plotting data.

Functions available in Utilities:
plotTrackRose:                  Function for circular plotting of tracked 
                                points between two images.
colored_bar:                    Function to plot bars as difference colours.
filterSparse:                   A function to remove noise from a sparse 
                                dataset using a filtering method. This removes 
                                points if they are within a specified value 
                                (threshold) from the mean of a specified number 
                                of nearest neighbour points (numNearest). The 
                                item field identifies which column of the array 
                                holds the field to be filtered on.
filterDensity:                  A function to remove noise from a sparse 
                                dataset using a filtering method. This removes 
                                points if they are within a specified value 
                                (threshold) and an absolute threshold 
                                (absthres) from the mean of a specified number 
                                of nearest neighbour points (numNearest). The 
                                item field identifies which column of the array 
                                holds the field to be filtered on.
arrowplot:                      Function to plot arrows to denote the direction 
                                and magnitude of the displacement. Direction is 
                                indicated by the bearing of the arrow, and the
                                magnitude is indicated by the length of the 
                                arrow.
plotVelocity:                   Produce assorted velocity plots from a set of 
                                velocity outputs.
interpolateHelper:              Function to interpolate a point dataset. This 
                                uses functions of the SciPy package to set up a 
                                grid (grid) and then interpolate using a linear 
                                interpolation method (griddata).
plotInterpolate:                Function to plot the results of the 
                                interpolation process for a particular 
                                timestep.
             
@authors: Nick Hulton (nick.hulton@ed.ac.uk)
          Penny How (p.how@ed.ac.uk)
'''

#Import packages
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import math
from scipy import spatial
from scipy.interpolate import griddata

#------------------------------------------------------------------------------

def plotTrackRose(pts0, pts1, xcen=None,ycen=None):
    '''Function for circular plotting of tracked points between two images. 
    Input variables p0 and p1 represent the start and end of tracks, which can
    be centred on a given location (xcen, ycen).'''
    #Make empty lists for distance and bearing
    dist=[]
    bearing=[]
    
    #Iterate through corresponding points lists
    for p0, p1 in itertools.izip(pts0, pts1):
        
        #Pythagoras' theorem to determine distance
        xd=p1[0]-p0[0]
        yd=p1[1]-p0[1]
        d=math.sqrt(xd*xd+yd*yd)
        dist.append(d)
        
        #Calculate bearing from displacement
        math.atan2(yd, xd)
        b=math.atan2(yd, xd)
        if b<0:
            b=math.pi*2.+b
        bearing.append(b)
       
        #Plot bearing on a circular graph
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        x = np.radians(np.arange(0, 360, 10))
        y = np.random.random(x.size)
        z= np.ones(x.size)
        #z = np.random.random(y.size)
        cmap = plt.get_cmap('cool')
        coll = colored_bar(x, y, z, ax=ax, width=np.radians(10), cmap=cmap)
        fig.colorbar(coll)
        ax.set_yticks([0.5, 1.0])
        plt.show()   
        ###NOT COMPLETE. DOESN'T USE DIST AND BEARING VARIABLES. Currently just
        ###plots random numbers


def colored_bar(left, height, z=None, width=0.8, bottom=0, ax=None, **kwargs):
    '''Function to plot bars as difference colours.'''    
    #Initiate plotting area
    if ax is None:
        ax = plt.gca()
    
    #Convert to 1-D arrays and iterate
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    
    #Construct coloured patches
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x,y), w, h))
    coll = PatchCollection(rects, array=z, **kwargs)
    ax.add_collection(coll)
    ax.autoscale()
    
    return coll
    ###NOT TESTED  
   

def filterSparse(data,numNearest=12,threshold=2,item=2):
    '''A function to remove noise from a sparse dataset using a filtering
    method. This removes points if they are within a specified value 
    (threshold) from the mean of a specified number of nearest neighbour 
    points (numNearest). The item field identifies which column of the 
    array holds the field to be filtered on.
    
    This function works best if called iteratively, as more than one point 
    may be anomolous compared to neighbouring ranges.''' 
    #Get XY point data
    XY=data[:,0:2]

    #Set up KD tree with XY point data
    tree=spatial.KDTree(XY, 50)
    
    goodset=[]
    for point in XY:        
        #Get n nearest neighbouring points
        d,k=tree.query(point, numNearest)
       
        #Get the mean and standard deviation for these points
        stdev=np.std(data[k[1:],item])
        m=np.mean(data[k[1:],item])
       
        #Get the data value for neighbouring points
        value=data[k[0],item]
       
        #Append point to goodset if is within threshold range of neighbours
        if (value>(m-(stdev*threshold)) and value<(m+(stdev*threshold))):
            goodset.append(data[k[0]])
        
    return np.array(goodset)
    
    
def filterDensity(self,data,numNearest=5,threshold=10.,absthres=float("inf")):
    '''A function to remove noise from a sparse dataset using a filtering
    method. This removes points if they are within a specified value 
    (threshold) and an absolute threshold (absthres) from the mean of a 
    specified number of nearest neighbour points (numNearest). The item field 
    identifies which column of the array holds the field to be filtered on.
    
    This function works best if called iteratively, as more than one point 
    may be anomolous compared to neighbouring ranges.''' 
    # Get XY point data
    XY=data[:,0:2]

    #Set up KD tree with XY point data    
    tree=spatial.KDTree(XY)
    
    nearestd=[]
    for point in XY:        
        #Get n nearest neighbouring points
        d,k=tree.query(point, numNearest)
        
        #Calculate mean value of neighbouring points
        nearestd.append(np.mean(d[1:]))
        
    meand=np.mean(nearestd)
            
    goodset=[]
    for point in XY:
        #Get n nearest neighbouring points
        d,k=tree.query(point, numNearest)
        
        #Calculate mean value of neighbouring points
        locmean=np.mean(d[1:])

        #Append point if it is within a given threshold based on neighbouring
        #points
        if (locmean<meand*threshold and locmean<absthres):
            goodset.append(data[k[0]])
        
    return np.array(goodset) 
    
    
def arrowplot(xst,yst,xend,yend,scale=1.0,headangle=15,headscale=0.2):    
    '''Function to plot arrows to denote the direction and magnitude of the
    displacement. Direction is indicated by the bearing of the arrow, and the
    magnitude is indicated by the length of the arrow.'''    
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

    
def plotVelocity(outputV, camim0, camim1, camenv, demred, lims, save=None, 
                 plotcams=True, plotcombined=True, plotspeed=True, 
                 plotmaps=True): 
    '''Produce assorted velocity plots from a set of velocity outputs.
    Plotting:
    plot cams:          Plot points (original, tracked and back-tracked) from 
                        cameras onto oblique image pair.
    plotcombined:       Plot points from camera view and from dem view.
    plotspeed:          Plot filtered speed and direction from dem view. 
    plotmaps:           Plot speed and direction onto dem view.
    '''
   
    #Get point sets from image       
    start=outputV[1][0]
    end=outputV[1][1]
    back=outputV[1][2]

    #Get x positions from image0, image1 and back-track
    xs=start[:,0,0]
    xe=end[:,0,0]
    xb=back[:,0,0]
    
    #Get y positions from image0, image1 and back-track
    ys=start[:,0,1]
    ye=end[:,0,1]
    yb=back[:,0,1]

    #Get xyz velocity direction
    Xs=outputV[0][0][:,0]
    Ys=outputV[0][0][:,1]
    Xd=outputV[0][1][:,0]
    Yd=outputV[0][1][:,1]

    #Get XYZ positions and speeds
    XYZs=outputV[0][0]
    XYZd=outputV[0][1]    
    xd=XYZs[:,0]-XYZd[:,0]
    yd=XYZs[:,1]-XYZd[:,1]

    #Get image arrays 
    im0=camim0.getImageArray()
    im1=camim1.getImageArray()
    
    #Get image names from CamImage object
    imn0=camim0.getImagePath().split('\\')[1]
    imn1=camim1.getImagePath().split('\\')[1]
    
    #Get image size from CamImage object  
    ilims=camim0.getImageSize()    
    ilims=[0,ilims[1],ilims[0],0]
    
    #Get corrected images
    cameraMatrix=camenv.getCamMatrixCV2()
    distortP=camenv.getDistortCoeffsCv2() 
    im0_dist=camim0.getImageCorr(cameraMatrix, distortP)
#    im1_dist=camim1.getImageCorr(cameraMatrix, distortP)
    
    
    #Plot points (original, tracked and back-tracked) from cameras        
    if plotcams is True:        
        #Plot image0 and image1
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        imgplot1 = ax1.matshow(im0)
        ax1.set_title(str(imn0))
        imgplot2 = ax2.matshow(im1)
        ax2.set_title(str(imn1))
        
        #Set colour maps
        imgplot1.set_cmap('gray')
        imgplot2.set_cmap('gray')
        
        #Plot xy positions onto images
        ax1.scatter(xs,ys,color='red')
        ax1.scatter(xe,ye,color='green')
        ax1.scatter(xb,yb,color='magenta')
        ax2.scatter(xs,ys,color='red')
        ax2.scatter(xe,ye,color='green')
        ax2.scatter(xb,yb,color='magenta')
        
#        #For arrow plotting
#        xar,yar=arrowplot(xen,yen,xst,yst,scale=5.0,headangle=15)
#        ax1.plot(xar,yar,color='orange')
#        ax2.plot(xar,yar,color='orange')
#        xar,yar=arrowplot(xen,yen,xhom,yhom,scale=5.0,headangle=15)
#        ax1.plot(xar,yar,color='black')
#        ax2.plot(xar,yar,color='black')
        
        #Make figure full-screen
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        if save != None:
            plt.savefig(save)
        plt.show()

    
    #Plot points from camera view and from dem view
    if plotcombined is True:        
        #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
        f, (ax1, ax2) = plt.subplots(1, 2)
        
        imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)
        #imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
                  
        #Scatter plot speed and direction                    
        ax2.scatter(Xs,Ys)        
        xar,yar=arrowplot(Xs,Ys,Xd,Yd,scale=5.0,headangle=15)
    
        ax2.plot(xar,yar,color='orange')     
        
        imgplot1 = ax1.imshow(im0_dist,extent=ilims)
        ax1.set_title(imn0)
                   
        imgplot1.set_cmap('gray')
    
        ax1.scatter(xs,ys,color='red')
        ax1.scatter(xe,ye,color='green')
        ax1.scatter(xb,yb,color='magenta')
        xar,yar=arrowplot(xs,ys,xb,yb,scale=5.0,headangle=15)
        ax1.plot(xar,yar,color='black')
        
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        if save != None:
            plt.savefig(save)
            
        plt.show()
  
  
    #Plot speed and direction from dem view   
    if plotspeed is True:
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        
        imgplot1 = ax1.imshow(demred, origin='lower', extent=lims)
        imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
        
        XYZs=outputV[0][0]
        XYZd=outputV[0][1]
        
        xd=XYZs[:,0]-XYZd[:,0]
        yd=XYZs[:,1]-XYZd[:,1]
        speed=np.sqrt(xd*xd+yd*yd)
        v_all=np.vstack((XYZs[:,0],XYZs[:,1],XYZd[:,0],XYZd[:,1],speed))       
        v_all=v_all.transpose()
        
        filtered=filterSparse(v_all,numNearest=12,threshold=2,item=4)
        
        ax1.scatter(filtered[:,0],filtered[:,1])
        
        Xs=filtered[:,0]
        Ys=filtered[:,1]
        Xd=filtered[:,2]
        Yd=filtered[:,3]
        
        ax2.scatter(Xs,Ys)
        
        xar,yar=arrowplot(Xs,Ys,Xd,Yd,scale=5.0,headangle=15)
    
        ax2.plot(xar,yar,color='orange')
        
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        if save != None:
            plt.savefig(save)
            
        plt.show()
    
    #Plot speed and direction onto dem view
    if plotmaps is True:
        
        f, (ax1, ax2) = plt.subplots(1, 2)
        
        imgplot1 = ax1.imshow(demred, origin='lower', extent=lims)
        imgplot1 = ax2.imshow(demred, origin='lower', extent=lims)           
        
        XYZs=outputV[0][0]
        XYZd=outputV[0][1]
        
                
        ax1.scatter(XYZs[:,0],XYZs[:,1])
        ax2.scatter(XYZd[:,0],XYZd[:,1])
        
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        if save != None:
            plt.savefig(save)
            
        plt.show()
  
    
def interpolateHelper(xyz1, xyz2, method='linear', filt=True):
    '''Function to interpolate a point dataset. This uses functions of 
    the SciPy package to set up a grid (grid) and then interpolate using a
    linear interpolation method (griddata).
    Methods are those compatible with SciPy's interpolate.griddata function: 
    "nearest", "cubic" and "linear"
    '''         
    #Get point positions and differences   
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    xdif=[]
    ydif=[]
    for i,j in zip(xyz1,xyz2):
        if math.isnan(i[0]):
            pass
        else:
            x1.append(i[0])
            y1.append(i[1])
            x2.append(j[0])
            y2.append(j[1])
            xdif.append(i[0]-j[0])
            ydif.append(i[1]-j[1])
    
    #Calculate velocity with Pythagoras' theorem
    speed=[]
    for i,j in zip(xdif, ydif):
        sp = np.sqrt(i*i+j*j)
        speed.append(sp)
    
    #Filter points if flag is true
    if filt is True:
        #Compile point data and speed 
        v_all=np.vstack((x1,y1,x2,y2,speed))
        v_all=v_all.transpose()
        
        #Filter points and extract xy positions and speed
        filtered=filterSparse(v_all,numNearest=12,threshold=2,item=4)
        x1=filtered[:,0]
        y1=filtered[:,1]
        x2=filtered[:,2]
        y2=filtered[:,3]            
        speed=filtered[:,4]           
            
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
     
     ###NEED TO SORT OUT THIS PART OF THE CODE. Doesn't work currently, but 
     ###seems like an important step   
#    #Find the new point, with the adjusted origin
#    newx = [(x-pointsextent[0]) for x in xs]
#    newy = [(y-pointsextent[2]) for y in ys]
#    newmaxx = math.floor(max(newx))+1
#    newmaxy = math.floor(max(newy))+1
#    #newpts = np.array([newx, newy]).T    
#    newpts=data[:,0:2]
    
    #Generate buffer around grid
    incrsx=((maxx-minx)/gridsize)+1
    incrsy=((maxy-miny)/gridsize)+1

    #Construct grid dimensions
    grid_y,grid_x = np.mgrid[miny:maxy:complex(incrsy),
                             minx:maxx:complex(incrsx)]
    
    #Interpolate the velocity and error points to the grid
    ###WORK OUT HOW TO INCORPORATE SNR GRID
    grid = griddata(newpts, np.float64(speed), (grid_x, grid_y), method=method)
#    error = griddata(newpts, np.float64(snrs), (grid_x, grid_y), method=method)      
                
    return grid, pointsextent          
    

def plotInterpolate(dem, lims, grid, pointextent, show=True, save=None):
    '''Function to plot the results of the interpolation process for 
    a particular timestep.

    Inputs:
    dem:            A DEM object.
    lims:           DEM extent (x1,x2,y1,y2).
    grid:           Numpy grid. It is recommended that this is constructed 
                    using the interpolateHelper function.
    pointextent:    Grid extent.
    save:           Flag to denote whether the figure is saved or not.
    '''   
    #Plot DEM
    plt.figure()
    plt.xlim(lims[0],lims[1])
    plt.ylim(lims[2],lims[3])
    plt.locator_params(axis = 'x', nbins=8)
    plt.tick_params(axis='both', which='major', labelsize=10)
    img = plt.imshow(dem, origin='lower', extent=lims)
    img.set_cmap('gray') 
    
    #Plot grid onto DEM
    plt.imshow(grid, 
               origin='lower', 
               cmap=plt.get_cmap("gist_ncar"), 
               extent=pointextent, 
               alpha=0.5) #alpha=1
               
    col=plt.colorbar()
#    col.set_clim(0,4)
    
    #Save if flag is true
    if save != None:
        plt.savefig(save, bbox_inches='tight')
        
    #Show plot
    if show is True:
        plt.show()
    
    plt.close()


def plotPX(a, number, dest=None, crop=False, show=True):
    '''Return image overlayed with pixel extent polygons for a given image 
    number.'''
    
    #Call corrected/uncorrected image
    if a._calibFlag is True:
        img=a._imageSet[number].getImageCorr(a._camEnv.getCamMatrixCV2(), 
                                            a._camEnv.getDistortCoeffsCv2())      
    else:
        img=a._imageSet[number].getImageArray() 
    
    #Get image name
    imn=a._imageSet[number].getImagePath().split('\\')[1]    
          
    #Create image plotting window
    fig=plt.gcf()
    fig.canvas.set_window_title(imn + ': Image extent output ')
    imgplot = plt.imshow(img)        
    imgplot.set_cmap('gray')
    plt.axis('off')
    
    if crop is True:
        if a._pxplot is not None:
            plt.axis([a._pxplot[0],a._pxplot[1],a._pxplot[2],
                     a._pxplot[3]])

    if hasattr(a, '_pxpoly'):
        polys = a._pxpoly[number]
        for p in polys:
            x=[]
            y=[]
            for xy in p:
                if len(xy)==1:
                    x.append(xy[0][0])
                    y.append(xy[0][1])
                elif len(xy)==2:
                    x.append(xy[0])
                    y.append(xy[1])
            plt.plot(x,y,'w-')
            
    elif hasattr(a, '_pxpts'):
        line = a._pxpts[number]        
        x=[]
        y=[]
        for xy in line:
            x.append(xy[0])
            y.append(xy[1])            
        plt.plot(x,y,'w-')
    
    else:
        print '\nUnrecognised Area/Line class object'
        
    if dest != None:
        plt.savefig(dest + 'extentimg_' + imn) 
        
    if show is True:
        plt.show()    
                            
    plt.close()
    
    
def plotXYZ(a, number, dest=None, dem=True, show=True):
    '''Plot xyz points of real polygons for a given image number'''                           
    #Get image name
    imn=a._imageSet[number].getImagePath().split('\\')[1]   
        
    #Prepare DEM
    if dem is True:
        demobj=a._camEnv.getDEM()
        demextent=demobj.getExtent()
        dem=demobj.getZ()
   
        #Get camera position (from the CamEnv class)
        post = a._camEnv._camloc            
    
        #Plot DEM 
        fig=plt.gcf()
        fig.canvas.set_window_title(imn + ': Area output')
        plt.locator_params(axis = 'x', nbins=8)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.imshow(dem, origin='lower', extent=demextent, cmap='gray')
        plt.xlim(demextent[0], demextent[1])
        plt.ylim(demextent[2], demextent[3])
        plt.scatter(post[0], post[1], c='g')
   
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
            plt.plot(xl, yl, c=np.random.rand(3,1), linestyle='-', label=lab)
            count=count+1
        
        plt.legend()
        plt.suptitle('Projected extents', fontsize=14)
        
        if dest != None:
            plt.savefig(dest + 'area_' + imn) 
            
        if show is True:
            plt.show() 
            
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
        plt.plot(xl, yl, 'y-')
        plt.suptitle('Projected line', fontsize=14)
        
        if dest != None:
            plt.savefig(dest + 'realline_' + imn) 
            
        if show is True:
            plt.show() 
    else:
        print '\nUnrecognised Area/Line class object'
        pass

    plt.close()
    
    
#------------------------------------------------------------------------------

#if __name__ == "__main__":   
#    print '\nProgram finished'

#------------------------------------------------------------------------------   