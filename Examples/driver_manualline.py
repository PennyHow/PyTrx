'''
PYTRX EXAMPLE MANUAL LINE DRIVER

This script is part of PyTrx, an object-oriented programme created for the 
purpose of calculating real-world measurements from oblique images and 
time-lapse image series.

This driver calculates terminus profiles (as line features) at Tunabreen, 
Svalbard, for a small subset of the 2015 melt season using modules in PyTrx. 
Specifically this script performs manual detection of terminus position through 
sequential images of the glacier to derive line profiles which have been 
corrected for image distortion. Change between terminus profiles (as an area or
a distance) are calculated using functions which have been written outside of
PyTrx.

Previously defined lines can also be imported from text or shape file (this 
can be changed by commenting and uncommenting commands in the "Calculate lines" 
section of this script).

@author: Penny How (p.how@ed.ac.uk)
         Nick Hulton
'''

#Import packages
import sys
import os
import itertools
import matplotlib
import matplotlib.pyplot as mp
import numpy as np
from osgeo import ogr
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import glob

#Import PyTrx packages
sys.path.append('../')
from Measure import Line
from CamEnv import CamEnv
from FileHandler import writeLineFile, writeSHPFile, importLineData
from Utilities import plotPX, plotXYZ

#Import Area change functions
sys.path.append('../Other/')
from PointPlotter import PointPlotter
from Points import Point2D
from Polylines import Polyline,Segment
  

#---------------------------   Initialisation   -------------------------------

#Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU2_2015.txt'   
camimgs = '../Examples/images/TU2_2015_subset/*.JPG'

#Define data output directory
destination = '../Examples/results/manualline/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)

#Set up line object
terminus = Line(camimgs, cam)


#-----------------------   Calculate/import lines   ---------------------------

#Choose action "plot", "importtxt" or "importshp". Plot proceeds with the 
#manual  definition of terminus lines, importtxt imports line data from text 
#files, and  importshp imports line data from shape file (.shp)
action = 'importtxt'      


#Manually define lines from imagery
if action == 'plot':
    rline, rlength = terminus.calcManualLinesXYZ()
    pxlength = terminus._pxline
    pxline = terminus._pxpts


#Import line data from text files   
elif action == 'importtxt':
    #Import lines to terminus object
    rline, rlength, pxline, pxlength = importLineData(terminus, destination)


#Import line data from shape files (only imports real line data, not pixel)
elif action == 'importshp':
    shpfiles = destination + 'shapefiles/*.SHP'
    xyz_line=[]    
    xyz_corr=[]
    xyz_len=[]
   
    #Get shape object from files
    for i in glob.glob(shpfiles):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(i, 0)
        layer = dataSource.GetLayer()
        
        #Append line length and xyz coordinates to lists           
        for feature in layer:
            geom = feature.GetGeometryRef()
            xyz_line.append(geom)
            xyz_len.append(geom.Length())
            ptpos = geom.Centroid().ExportToWkt()
            pt2=[ptpos]
            pt2 = ptpos.split('(')
            pt3 = pt2[1].split(')')
            pt3 = pt3[0].split(' ')
            xyz_corr.append(pt3)
        
    #Append data to Line object
    terminus._realpts = xyz_corr
    terminus._realline = xyz_line


#Program will terminate if an invalid string is inputted as the action variable        
else:
    print 'Invalid action. Please re-define.'
    sys.exit(1)
    
#----------------------------   Export data   ---------------------------------

#Change flags to write text and shape files
write = True
shp = True

#Write line data to txt file
if write is True:   
    writeLineFile(terminus, destination)

#Write shapefiles from line data
if shp is True:   
    target1 = destination + 'shapefiles/'
    if not os.path.exists(target1):
        os.makedirs(target1)    
    proj = 32633
    writeSHPFile(terminus, target1, proj)


#----------------------------   Show results   --------------------------------

#Generate destination location
target2 = destination + 'outputimgs/'
if not os.path.exists(target2):
    os.makedirs(target2)

#Plot and save all extent and area images
length=len(pxline)
for i in range(len(pxline)):
    plotPX(terminus, i, target2, crop=None, show=False)
    plotXYZ(terminus, i, target2, crop=None, show=False, dem=True)
    
    
#-------------------   Functions for volume calculations   --------------------
#This section contains functions for calculating areal change. The driver for
#this is in the subsequent section

#Polyline constructor to represent terminus lines
def polyline_reader(lines, idchar=None, dateTime=None):

    polyl=[]
    count=0
    
    for l in lines:
        if len(l)!=0:
            linecoords=[]
            for pt in l:
                linecoords.append(Point2D(float(pt[0]),float(pt[1])))
            ply=Polyline(linecoords)            
            if idchar is not None:
                idply=idchar[count]
            else:
                idply=None            
            if dateTime!=None:
                dateply=dateTime[count]
            else:
                dateply=None                
            ply.setID([idply,dateply])
            polyl.append(ply)
            count=count+1            
        else:
            pass
   
    return polyl
 
#Sort segment geometries   
def sortGeometry(polyline,segment):    
    x=segment.pointToRight(polyline.getStart())
    if x > 0:
        segment.reverse()

#Connect step line to terminus polyline if they do not meet        
def extendPolylineToMeet(polyline,segs):    
    seg0=segs[0]
    segn=segs[-1]    
    p0f=False
    pnf=False
    
    for i in range(polyline.size()-1):
        segpoly=polyline.getSegment(i)
        p0=seg0.intersectPoint(segpoly)
        if p0==None:
            print 'p0 none case',i,polyline.size()
            print seg0.getStart().getCoords()
            print seg0.getEnd().getCoords()
            print segpoly.getStart().getCoords()
            print segpoly.getEnd().getCoords()
             
        pn=segn.intersectPoint(segpoly)
        
        if i==0:
            p0e=p0
        if i==polyline.size()-2:
            pne=pn

        ua=p0.getUa()
        ub=p0.getUb()
        if (ua>=-0.0000000000001 and ua<=1.0000000000001 and 
            ub>=-0.0000000000001 and ub<=1.0000000000001):
            p0f=p0

        ua=pn.getUa()
        ub=pn.getUb()
        if (ua>=-0.0000000000001 and ua<=1.0000000000001 and 
            ub>=-0.0000000000001 and ub<=1.0000000000001):
            pnf=pn

    #Extend segment at start        
    if not p0f:
        print ('Warning: Extending start of  line since no overlap with '
               'perpendicular on baseline \n')
        polyline.insertStartPoint(p0e)
        p0f=True
    
    #Extend segment at end
    if not pnf:
        print ('Warning: Extending end of line since no overlap with '
               'perpendicular on baseline \n')
        polyline.insertEndPoint(pne)
        pnf=True
        
    return p0f,pnf

#Create areas from polygons of baseline-stepline-terminusline    
def extractPolys(polyline,segs,startflag=False,endflag=False):   
    polyseg=0
    base0=segs[0].getStart().getCoords()

    notfound=True
    while notfound:
        intsOnPoly0=segs[0].intersectPoint(polyline.getSegment(polyseg))
        ub=intsOnPoly0.getUb()
        
        if ub>=-0.00000000001 and ub<=1.00000000001:
            notfound=False
        else:
            polyseg=polyseg+1            
    intsOnPoly0=intsOnPoly0.getCoords()

    pSeg=polyline.getSegment(polyseg)
    allpolys=[]
    areas=[]
    for seg in segs[1:]:        
        notfound=True
        base1=seg.getStart().getCoords()
        polypoints=[]        
        startokay=polyseg>0
        
        while notfound:            
            testintersect=seg.intersectPoint(pSeg)
            ub=testintersect.getUb()
            if ub>=-0.0000000000001 and ub<=1.0000000000001:
                notfound=False
                polypoints.append(testintersect.getCoords())
            else:
                addedP=pSeg.getEnd()
                polyseg=polyseg+1
                pSeg=polyline.getSegment(polyseg)
                polypoints.append(addedP.getCoords())
        endokay=polyseg<polyline.size()-1
        
        plist=[base0,intsOnPoly0]+polypoints+[base1,base0]
        allpolys.append(plist)
        
        try:
                ring = ogr.Geometry(ogr.wkbLinearRing)   
                for p in plist:
                    ring.AddPoint(p[0],p[1])
                ring.AddPoint(p[0],p[1])
                pxpoly = ogr.Geometry(ogr.wkbPolygon)
                pxpoly.AddGeometry(ring)
                pxextent = pxpoly.Area()

        except:
                pxextent = float('Nan')        
                
        if (startflag and not startokay) or (endflag and not endokay):
            pxextent = float('Nan')
        
        areas.append(pxextent)        
        
        base0=base1
        intsOnPoly0=polypoints[-1]

    return allpolys,areas

#Convert polygons to OGR polygon objects    
def calcAreas(polys):
    areas=[]
    for poly in polys:
        try:
                ring = ogr.Geometry(ogr.wkbLinearRing)   
                for p in poly:
                    ring.AddPoint(p[0],p[1])
                ring.AddPoint(p[0],p[1])
                pxpoly = ogr.Geometry(ogr.wkbPolygon)
                pxpoly.AddGeometry(ring)
                pxextent = pxpoly.Area()
        except:
                pxextent = float('Nan')
    areas.append(pxextent)
    
    return areas

#Plot baseline, steplines and terminus profiles   
def doPlots(baseline, line, polys, p0, pn, segs, show=False, lims=None):
    pp=PointPlotter()
    fig, ax = mp.subplots()
    if lims!=None:
        xlo,xhi,ylo,yhi=lims
        ax.set_xlim(xlo,xhi)
        ax.set_ylim(ylo,yhi)      
    pp.plotPoint(p0,'blue')
    pp.plotPoint(pn,'magenta')        
    pp.plotPolylines(line)    
    pp.plotSegment(baseline,'red')
    pp.plotPoint(baseline.getStart(),'red') 
    
    for seg in segs:
        pp.plotSegment(seg,'red')
        pp.plotPoint(seg.getStart(),'black') 
   
    patches =[]
    
    for poly in polys:
        polygon = Polygon(poly, True)
        #print polygon
        patches.append(polygon)
     
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    colors = 100*np.random.rand(len(patches))
    colors =range(len(patches))    
    p.set_array(np.array(colors))   
    ax.add_collection(p)
    
    if show is True:
        mp.show()
    mp.close()

#Write data to .csv file    
def dataToCsv(fname,header,allVals):
    f=open(fname,'w')    
    line1='File ID,FileName'
    for i in range(len(allVals[0])):
        line1=line1+',Area '+str(i+1)
    linesOut=[line1]
    for head,vals in itertools.izip(header,allVals):
        line='\n' + str(head[0]) + ',' + str(head[1])
        for val in vals:
            line=line+','+str(val)
        linesOut.append(line)
    f.writelines(linesOut)
    f.close()


#-------------------   Calculate area/distance change  ------------------------

#This section is for calculating area loss by plotting the terminus positions 
#to a baseline. Area loss is defined as changes in the distance of the
#terminus from the baseline at a given step point. These step points can be
#defined at certain positions, or by a given number of equally-spaced step 
#points
print '\n\nCommencing area/distance change calculation'


#See plots?
plotting=True


#Steps for calculating terminus change. This can either be provided as an 
#integer which plots equally-spaced steps, or as a list which plots steps at 
#specific points along the baseline 
steps=51   
#steps=[0.0,0.3,0.55,0.9,0.96,1.0]


#Length of step line
perpendicular_length=2000


#Start (x1,y1) and end (x2,y2) of baseline co-ordinates
x1=553250
y1=8711500
x2=554900
y2=8708000


#Plotting limits. These can either be pre-defined or calculated based on the
#given baseline start and end points. Pass 'None' for each if limits are 
#undesired 
xlo=551000
xhi=556000
ylo=8706000
yhi=8712000  


#Create baseline
baseline=Segment(x1,y1,x2,y2)    

#Get datetimes from images and line objects
allareas=[]
dateheader=[]

datetime=[]
for im in terminus._imageSet:
    imn = im.getImagePath().split('\\')[1]
    datetime.append(imn)

idline = np.arange(1,(len(rline)+1),1)  
         
#Create polylines from the line objects
allrecs=polyline_reader(rline, idchar=idline, dateTime=datetime)
       
if len(allrecs)>0:

    #Swap the direction of the baseline if necssary to make sure the 
    #perpendiculars come the right way
    sortGeometry(allrecs[0],baseline)
    
    #Define a set of step lines that are perpendicular to the baseline, which 
    #intersect with the polylines
    segs=baseline.getPerpendicularSegs(steps,1.,length=perpendicular_length)
 

    count=1
    for line in allrecs:
        print ('\nProcessing line ' + str(count) +  ' with ' + str(line.size())
               + ' points')
        
        #Get around the situation where the polyline is 'short' in this case
        p0,pn=extendPolylineToMeet(line,segs)
        polys,areas=extractPolys(line,segs,p0,pn)                
        
        #Calculate distance between polylines and baseline at the given steps
        doPlots(baseline, line, polys, p0, pn, segs, 
                show=plotting,lims=[xlo,xhi,ylo,yhi])
        
        #Append data to list
        allareas.append(areas)
        dateheader.append(line.getID())        
        count=count+1


#-------------------   Write area/distance change data   ----------------------

#Define output file locations  
csvAreasOut = destination + 'area_change.csv'
csvDistancesOut = destination + 'distance_change.csv'

#Write area file if defined       
if csvAreasOut!=None:
    print '\nWriting area change data to file'
    dataToCsv(csvAreasOut,dateheader,allareas) 

#Write distance file if defined
if csvDistancesOut!=None:
    print '\nWriting distance change data to file'
    baselinedists=[]
    alldistances=[]
    p0=segs[0].getStart()
    for seg in segs[1:]:
        p1 = seg.getStart()
        baselinedists.append(p0.distance(p1))
        p0=p1
    for areas in allareas:
        distances=[]
        for basedist,area in itertools.izip(baselinedists,areas):
            dist=area/basedist
            distances.append(dist)
        alldistances.append(distances)
    dataToCsv(csvDistancesOut,dateheader,alldistances)


#------------------------------------------------------------------------------

print '\n\nFinished'