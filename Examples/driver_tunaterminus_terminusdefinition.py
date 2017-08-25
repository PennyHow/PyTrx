# -*- coding: utf-8 -*-
'''
Created on Fri Jun 24 10:07:59 2016

@author:  Penny How (p.how@ed.ac.uk)
          Nick Hulton (nick.hulton@ed.ac.uk)
    
Driver for length calculator of Tunabreen terminus from camera 2 (2015)
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


#-------------------   Functions for volume calculations   --------------------

def polyline_reader(rline, idchar=None, dateTime=None):
    polyl=[]
    for r,s,t in zip(rline,idchar,dateTime):
        if len(r)!=0:
            linecoords=[]
            for pt in r:
                linecoords.append(Point2D(float(pt[0]),float(pt[1])))
            ply=Polyline(linecoords)
            ply.setID([s,t])
            polyl.append(ply)
        else:
            pass
   
    return polyl
    
def sortGeometry(polyline,segment):
    
    x=segment.pointToRight(polyline.getStart())
    if x > 0:
        segment.reverse()
        
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
           #print 'start intersection: ',p0
        if i==polyline.size()-2:
            pne=pn

        ua=p0.getUa()
        ub=p0.getUb()
        if ua>=-0.0000000000001 and ua<=1.0000000000001 and ub>=-0.0000000000001 and ub<=1.0000000000001:
     #   if p0.getUa()>=0 and p0.getUa()<=1 and p0.getUb()>=0 and p0.getUb()<=1:
           # print i,p0.getUa(),p0.getUb()
            p0f=p0

        ua=pn.getUa()
        ub=pn.getUb()
        if ua>=-0.0000000000001 and ua<=1.0000000000001 and ub>=-0.0000000000001 and ub<=1.0000000000001:

  #      if pn.getUa()>=0 and pn.getUa()<=1 and pn.getUb()>=0 and pn.getUb()<=1:
            #print i,pn.getUa(),pn.getUb()
            pnf=pn

# Extend segment at start        
    if not p0f:
        print 'Warning: Extending start of  line since no overlap with perpendicular on baseline'
        #polyline.changePointAt(p0e,0)
        polyline.insertStartPoint(p0e)
        p0f=True
# Extend segement at end
    if not pnf:
        print 'Warning: Extending end of line since no overlap with perpendicular on baseline'
        #polyline.changePointAt(pne,polyline.size()-1)
        polyline.insertEndPoint(pne)
        pnf=True
        
    return p0f,pnf
    
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
#    pp.plotPoint(base0,'yellow')
#    pp.plotPoint(intsOnPoly0,'red')

    pSeg=polyline.getSegment(polyseg)
    allpolys=[]
    areas=[]
#now for all the other offset segments
    for seg in segs[1:]:
        
        notfound=True
        base1=seg.getStart().getCoords()
        polypoints=[]
        
        startokay=polyseg>0
        while notfound:
            
            testintersect=seg.intersectPoint(pSeg)
            ub=testintersect.getUb()
            if ub>=-0.0000000000001 and ub<=1.0000000000001:
                #print 'intersection found'
                #pp.plotPoint(testintersect,'red')
                notfound=False
                polypoints.append(testintersect.getCoords())
            else:
                addedP=pSeg.getEnd()
                #pp.plotPoint(addedP,'green')
               # print 'incrementing polyline segment'
                polyseg=polyseg+1
                pSeg=polyline.getSegment(polyseg)
                polypoints.append(addedP.getCoords())
                #print polyseg
        endokay=polyseg<polyline.size()-1
        
        plist=[base0,intsOnPoly0]+polypoints+[base1,base0]
        allpolys.append(plist)
        try:
                ring = ogr.Geometry(ogr.wkbLinearRing)   
                for p in plist:
                    ring.AddPoint(p[0],p[1])
                ring.AddPoint(p[0],p[1])
                pxpoly = ogr.Geometry(ogr.wkbPolygon)
                pxpoly.AddGeometry(ring) #create polygon ring
                pxextent = pxpoly.Area()
                #print 'area: ', pxextent
        except:
                pxextent = float('Nan')        
        
        
        if (startflag and not startokay) or (endflag and not endokay):
            pxextent = float('Nan')
        
        areas.append(pxextent)        
        
        base0=base1
        intsOnPoly0=polypoints[-1]

    return allpolys,areas
    
def calcAreas(polys):
    areas=[]
    for poly in polys:
#start and finish point should be the same to make a ring
#not trapped as of yet
        try:
                ring = ogr.Geometry(ogr.wkbLinearRing)   
                for p in poly:
                    ring.AddPoint(p[0],p[1])
                ring.AddPoint(p[0],p[1])
                pxpoly = ogr.Geometry(ogr.wkbPolygon)
                pxpoly.AddGeometry(ring) #create polygon ring
                pxextent = pxpoly.Area()
                #print 'area: ', pxextent
        except:
                pxextent = float('Nan')
    areas.append(pxextent)
    
    return areas

    
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
    
def dataToCsv(fname,header,allVals):
    print 'Writing csv file\n'
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


#---------------------------   Initialisation   -------------------------------

#Define data input directories
camdata = '../Examples/camenv_data/camenvs/CameraEnvironmentData_TU4_2015.txt'   
camimgs = 'F:/imagery/tunabreen/pytrx/TU4_terminus_2015/*.JPG'

#Define data output directory
destination = '../Examples/results/TU4_manualline/'
if not os.path.exists(destination):
    os.makedirs(destination)

#Define camera environment
cam = CamEnv(camdata)


#-----------------------   Calculate/import areas   ---------------------------

#Set up line object
terminus = Line(camimgs, cam)

##Manually plot termini
#rline, rlength = terminus.calcManualLinesXYZ()

#Import lines to terminus object
rline, rlength, pxline, pxlength = importLineData(terminus, destination)

#----------------------------   Export data   ---------------------------------

##Write area data to txt file
#writeLineFile(terminus, destination)
#
geodata = destination + 'shapefiles/'
if not os.path.exists(geodata):
    os.makedirs(geodata)

proj = 32633
writeSHPFile(terminus, destination, proj)


#----------------------------   Show results   --------------------------------

##Plot and save all extent and area images
#length=len(pxline)
#for i in range(len(pxline)):
#    plotPX(terminus, i, destination, crop=False)
#    plotXYZ(terminus, i, destination, dem=True, show=True)


#------------------------   Calculate area loss  ------------------------------

#plotting=False
# 
##steps=51   
#steps=[0.0,0.3,0.55,0.9,0.96,1.0]
#
#perpendicular_length=2000
#
##baseline co-ordinates '1' and '2' can be any way around
#x1=552920
#y1=8710720
#x2=554620
#y2=8708720
#
##x1=553181
##y1=8710240
##x2=554650
##y2=8708650
#    
#xr=(x2-x1)*.1
#yr=(y2-y1)*.1
#
#xlo=min(x1,x2)
#xhi=max(x1,x2)
#ylo=min(y1,y2)
#yhi=max(y1,y2)
#
#xlo=xlo-xr
#ylo=ylo-yr
#xhi=xhi+xr
#yhi=yhi+yr
#
#xlo=552500
#xhi=555500
#ylo=8707000
#yhi=8712000
#
#baseline=Segment(x1,y1,x2,y2)    
#
#allareas=[]
#dateheader=[]
#
#datetime=[]
#for im in terminus._imageSet:
#    imn = im.getImagePath().split('\\')[1]
#    datetime.append(imn)
#
#idline = np.arange(1,len(rline),1)  
#          
##Get the polylines from line object  
#allrecs=polyline_reader(rline, idchar=idline, dateTime=datetime)
#print 'allrecs',len(allrecs)
#print allrecs
#    
#if len(allrecs)>0:
#
#    #Swap the direction of the baseline if necssary to make sure the 
#    #perpendiculars come the right way
#    sortGeometry(allrecs[0],baseline)
#    
#    #Define a set of perpendicular from the baseline to intersect with the 
#    #polylines this has to be redone on the offchance front lines are 
#    #digitised in different directions
#    segs=baseline.getPerpendicularSegs(steps,1.,length=perpendicular_length)
# 
#    print 'segs',segs
#
#    #Now for every line you have
#    count=1
#    for line in allrecs:
#    
#        print '\nProcessing line ' +str(count)+  ' with ', line.size(),' points'
#        
#        #Get around the situation where the polyline is 'short' in this 
#        #case
#        p0,pn=extendPolylineToMeet(line,segs)
#        polys,areas=extractPolys(line,segs,p0,pn)        
#        
#        #areas=calcAreas(polys)          
#    
#        doPlots(baseline, line, polys, p0, pn, segs, 
#                show=plotting,lims=[xlo,xhi,ylo,yhi])
#        
#        allareas.append(areas)
#        dateheader.append(line.getID())
#        
#        count=count+1
#
#csvAreasOut = destination + 'volumeloss/allAreaData_subsets.csv'
#csvDistancesOut = destination + 'volumeloss/allDistData_subsets.csv'
#       
#if csvAreasOut!=None:
#    print '\nWriting area csv file'
#    dataToCsv(csvAreasOut,dateheader,allareas) 
#
#if csvDistancesOut!=None:
#    print '\nWriting length csv file'
#    baselinedists=[]
#    alldistances=[]
#    p0=segs[0].getStart()
#    for seg in segs[1:]:
#        p1 = seg.getStart()
#        baselinedists.append(p0.distance(p1))
#        p0=p1
#    #print baselinedists
#    for areas in allareas:
#        distances=[]
#        for basedist,area in itertools.izip(baselinedists,areas):
#            dist=area/basedist
#            distances.append(dist)
#        alldistances.append(distances)
#    dataToCsv(csvDistancesOut,dateheader,alldistances)


print 'Finished'
