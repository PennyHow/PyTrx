'''
PYTRX EXAMPLE POINT GEORECTIFICATION DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver imports pre-defined points which denote calving events at the 
terminus of Tunabreen, and projects them to xyz locations using the
georectification functions in PyTrx. These locations are plotted onto the DEM,
with the colour of each point denoting the style of calving in that particular
instance. The xyz locations are exported subsequently as a text file (.txt) 
and as a shape file (.shp).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton (nick.hulton@ed.ac.uk)
'''

#Import packages
import numpy as np
import sys
import os
import osgeo.ogr as ogr
import osgeo.osr as osr
import matplotlib.pyplot as plt

              
#Import PyTrx packages
sys.path.append('../')
from CamEnv import CamEnv


#-----------------------------   Map data files   -----------------------------

#Define data inputs
tu2camenv='../Examples/camenv_data/camenvs/CameraEnvironmentData_TU2_2015.txt'
tu2calving = '../Examples/results/ptsgeorectify/TU2_calving_xy.csv'

#Define data output directory
destination = '../Examples/results/ptsgeorectify/'
if not os.path.exists(destination):
    os.makedirs(destination)

   
#----------------------   Import calving data from file   ---------------------

#Import individual calving event data
f=file(tu2calving,'r')                              #Read file
header=f.readline()                                 #Read first line
h=header.split(',')                                 #Split first line
for name in h:
    print '\nReading ' + str(name) + ' from file'                      

#Read all lines in file
alllines=[]
for line in f.readlines():
    alllines.append(line)
    
#Create data variables
time=[]
region=[]
style=[]
loc_x=[]
loc_y=[]
i=0

#Extract data from line
for line in alllines:
    i=i+1
    temp=line.split(',')    
    
    time.append(float(temp[0].rstrip()))            #Calving event time      
    region.append(temp[1].rstrip())                 #Calving event region             
    style.append(temp[2].rstrip())                  #Calving event style
    loc_x.append(float(temp[3].rstrip()))           #Calving event location x
    loc_y.append(float(temp[4].rstrip()))           #Calving event location y

      
#--------------------------   Georectify points   -----------------------------
        
#Compile xy locations
tu2_xy=[] 
for a,b in zip(loc_x,loc_y):
    tu2_xy.append([a,b])
print '\n\n' + str(len(tu2_xy)) + ' locations for calving events detected'
tu2_xy=np.array(tu2_xy)


#Define camera environment
tu2cam = CamEnv(tu2camenv)                              


#Inverse project image coordinates using function from CamEnv object                       
tu2_xyz = tu2cam.invproject(tu2_xy)
print '\n\n' + str(len(tu2_xyz)) + ' locations for calving events georectified'


#-----------------------   Plot xyz location on DEM   -------------------------

print '\n\nPLOTTING XYZ CALVING LOCATIONS'


#Boolean flags (True/False)
save=True                          #Save plot?
show=True                          #Show plot?                          

        
#Retrieve DEM from CamEnv object
demobj=tu2cam.getDEM()
demextent=demobj.getExtent()
dem=demobj.getZ()

   
#Get camera position (xyz) from CamEnv object
post = tu2cam._camloc            
 
   
#Plot DEM 
fig,(ax1) = plt.subplots(1, figsize=(15,15))
fig.canvas.set_window_title('TU2 calving event locations')
ax1.locator_params(axis = 'x', nbins=8)
ax1.tick_params(axis='both', which='major', labelsize=0)
ax1.imshow(dem, origin='lower', extent=demextent, cmap='gray')
ax1.axis([demextent[0], demextent[1], demextent[2], demextent[3]])
cloc = ax1.scatter(post[0], post[1], c='g', s=100, label='Camera location')
     
      
#Plot calving locations on DEM
xr=[]
yr=[]               
for pt in tu2_xyz: 
    xr.append(pt[0])                                #Separate x values
    yr.append(pt[1])                                #Separate y values


#Group xyz locations by calving style
nanx=[]
nany=[]
waterx=[]
watery=[]
icex=[]
icey=[]
stackx=[]
stacky=[]
sheetx=[]
sheety=[]
subx=[]
suby=[]
for i in range(len(xr)):
    if style[i]=="NaN":        
        nanx.append(xr[i])
        nany.append(yr[i])                  #Append xyz of NaN style
    elif style[i]=="waterline":       
        waterx.append(xr[i])
        watery.append(yr[i])                #Append xyz of waterline style
    elif style[i]=="icefall":
        icex.append(xr[i])
        icey.append(yr[i])                  #Append xyz of icefall style
    elif style[i]=="stack":
        stackx.append(xr[i])
        stacky.append(yr[i])                #Append xyz of stack style
    elif style[i]=="sheet":
        sheetx.append(xr[i])
        sheety.append(yr[i])                #Append xyz of sheet style
    elif style[i]=="subaqueous":
        subx.append(xr[i])
        suby.append(yr[i])                  #Append xyz of subaqueous style
    else:
        print '\nUnrecognised calving style'
        pass


print '\nUnclassified events: ' + str(len(nanx))
print '\nWaterline events: ' + str(len(waterx))
print '\nIce fall events: ' + str(len(icex))
print '\nStack collapses: ' + str(len(stackx))
print '\nSheet collapses: ' + str(len(sheetx)) 
print '\nSubaqueous events: ' + str(len(subx))


#Plot calving event locations by calving style (denoted by colour)
p6=ax1.scatter(nanx, nany, c='k', s=100, 
               label='Unknown', alpha=1.0) 
p1=ax1.scatter(waterx, watery, c='#00c5ff',s=100, 
               label='Waterline', alpha=1.0)
p2=ax1.scatter(icex, icey, c='#00ff00', s=100, 
               label='Ice-fall', alpha=1.0)
p3=ax1.scatter(stackx, stacky, c='#e60000', s=100, 
               label='Stack collapse', alpha=1.0) 
p4=ax1.scatter(sheetx, sheety, c='#ffaa00', s=100, 
               label='Sheet collapse', alpha=1.0)
p5=ax1.scatter(subx, suby, c='#8400a8', s=100, 
               label='Subaqueous', alpha=1.0)


#Add legend to plot     
ax1.legend(handles=[p2,p4,p3,p1,p5,p6,cloc], 
           bbox_to_anchor=(0., 1.02, 1., .102), ncol=7,
           loc='lower center', scatterpoints=1, fontsize=12)                      
    

#Save plot if flag is True
if save is True:
    plt.savefig(destination + 'TU2_calving_xyz.JPG', dpi=300) 


#Show plot if flag is True    
if show is True:
    plt.show() 


plt.close()
    
        
#------------------   Export xyz locations as .txt file   ---------------------

print '\n\nSAVING TEXT FILE'


#Write xyz coordinates to .txt file
target1 = destination + 'TU2_calving_xyz.txt'
f = open(target1, 'w')
f.write('x' + '\t' + 'y' + '\t' + 'z' + '\n')
for i in tu2_xyz:
    f.write(str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')                                  
f.close()


#------------------   Export xyz locations as .shp file   ---------------------

print '\n\nSAVING SHAPE FILE'


#Get ESRI shapefile driver     
typ = 'ESRI Shapefile'        
driver = ogr.GetDriverByName(typ)
if driver is None:
    raise IOError('%s Driver not available:\n' % typ)


#Create data source
shp = destination + 'tu2_calving.shp'   
if os.path.exists(shp):
    driver.DeleteDataSource(shp)
ds = driver.CreateDataSource(shp)
if ds is None:
    print 'Could not create file %s' %shp
 
       
#Set WGS84 projection
proj = osr.SpatialReference()
proj.ImportFromEPSG(32633)


#Create layer in data source
layer = ds.CreateLayer('tu2_calving', proj, ogr.wkbPoint)
  
  
#Add attributes to layer
layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      #ID    
layer.CreateField(ogr.FieldDefn('time', ogr.OFTReal))       #Time
field_region = ogr.FieldDefn('region', ogr.OFTString)        
field_region.SetWidth(8)    
layer.CreateField(field_region)                             #Calving region
field_style = ogr.FieldDefn('style', ogr.OFTString)        
field_style.SetWidth(10)    
layer.CreateField(field_style)                              #Calving size    
 
  
#Create point features with data attributes in layer           
for a,b,c,d in zip(tu2_xyz, time, region, style):
    count=1

    #Create feature    
    feature = ogr.Feature(layer.GetLayerDefn())

    #Create feature attributes    
    feature.SetField('id', count)
    feature.SetField('time', b)
    feature.SetField('region', c) 
    feature.SetField('style', d)         

    #Create feature location
    wkt = "POINT(%f %f)" %  (float(a[0]) , float(a[1]))
    point = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(point)
    layer.CreateFeature(feature)

    #Free up data space
    feature.Destroy()                       
    count=count+1

#Free up data space    
ds.Destroy()


#------------------------------------------------------------------------------

print '\n\nFinished'