# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 10:06:05 2017

Routines to calculate ice front areas

@author: nrjh
"""


steps=51   
#steps=[0.0,0.3,0.7,0.8,1.0]
perpendicular_length=2000
resultsDir='/Volumes/Maxtor/Tunabreen/camera2/results/AAccuracy-set2'
#/*/*line_realcoords*.csv

#resultsDir='/Volumes/Maxtor/Tunabreen/camera2/results/'
#plotting=False
plotting=True
#csvAreasOut='/Volumes/Maxtor/Tunabreen/camera2/results/allAreaData.csv'
csvAreasOut=None
#csvDistancesOut='/Volumes/Maxtor/Tunabreen/camera2/results/allDistData.csv'
csvDistancesOut='/Volumes/Maxtor/Tunabreen/camera2/results/set2accuracyDistData.csv'
 

#baseline co-ordinates '1' and '2' can be any way around
x1=553250
y1=8711000
x2=554750
y2=8709250

import sys
import itertools
import matplotlib
import matplotlib.pyplot as mp
import numpy as np
from PointPlotter import PointPlotter
from Points import Point2D
from Polylines import Polyline,Segment
from osgeo import ogr
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import glob


def polyline_reader(fname):
    file=open(fname,'r')
    print 'fname',fname
    lines=file.readlines()
    allrecs=[]
    for line in lines:
        items=line.split(',')
        idchar=items[0]
        dateTime=items[1]
        items=items[2:]
        linecoords=[]
        #print idchar,dateTime
        for i in range(0,len(items),3):
            try:
                linecoords.append(Point2D(float(items[i]),float(items[i+1])))
            except:
                print 'data item',i
                
        if len(linecoords)>2:
            ply=Polyline(linecoords)
            ply.setID([idchar,dateTime])
            print dateTime
            allrecs.append(ply)
    return allrecs
    
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
    

xr=(x2-x1)*.1
yr=(y2-y1)*.1

xlo=min(x1,x2)
xhi=max(x1,x2)
ylo=min(y1,y2)
yhi=max(y1,y2)

xlo=xlo-xr
ylo=ylo-yr
xhi=xhi+xr
yhi=yhi+yr

xlo=552000
xhi=555000
ylo=8708000
yhi=8711000



baseline=Segment(x1,y1,x2,y2)  
#allrecs=polyline_reader('./Results/cam1/2014terminus/line_realcoords_150819.csv')
#sortGeometry(allrecs[0],baseline)
#segs=baseline.getPerpendicularSegs(steps,1.,length=perpendicular_length)
   

#print ('\n')
    
def doPlots(baseline,line,polys):
    pp=PointPlotter()
    fig, ax = mp.subplots() 
    ax.set_xlim(xlo,xhi)
    ax.set_ylim(ylo,yhi)
    #pp.set_axis(xlo,xhi,ylo,yhi)        
    pp.plotPoint(p0,'blue')
    pp.plotPoint(pn,'magenta')        
    pp.plotPolylines(line)    
    pp.plotSegment(baseline,'red')
    pp.plotPoint(baseline.getStart(),'red') 
    
    for seg in segs:
        pp.plotSegment(seg,'red')
        pp.plotPoint(seg.getStart(),'black')
    
    #pp.show()      
   
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
    
    mp.show()
    
def dataToCsv(fname,header,allVals):
    print 'Writing csv file\n'
    f=open(fname,'w')
    
    line1='File ID,DateTime'
    for i in range(len(allareas[0])):
        line1=line1+',Area '+str(i+1)
    linesOut=[line1]
    for head,vals in itertools.izip(header,allVals):
        items=head[1].split(' ')
        time=items[1]
        items=items[0].split(':')
        date=items[0]+'-'+items[1]+'-'+items[2]
        line='\n'+head[0]+','+date+' '+time
        for val in vals:
            line=line+','+str(val)
        linesOut.append(line)
    f.writelines(linesOut)
    f.close()

resultSpec=resultsDir+'/*/*line_realcoords*.csv'
print resultSpec
filelist=glob.glob(resultSpec)
print filelist

filelist.sort()
print filelist


allareas=[]
dateheader=[]
# for every file containing line position data


for fname in filelist:

# get the polylines from the relevant file    
    allrecs=polyline_reader(fname)
    
    print 'allrecs',len(allrecs)
    
    if len(allrecs)>0:
# swap the direction of the baseline if necssary to make sure the perpendiculars
# come the right way
        sortGeometry(allrecs[0],baseline)
# define a set of perpendicular from the baseline to intersect with the polylines
# this has to be redone on the offchance front lines are digitised in different directions
        segs=baseline.getPerpendicularSegs(steps,1.,length=1500.)
     
        print 'segs',segs
    
# now for everyline you have
        for line in allrecs:
        
            print '\nProcessing new line with ',line.size(),' points'
# get around the situation where the polyline is 'short' in this case
            p0,pn=extendPolylineToMeet(line,segs)
            polys,areas=extractPolys(line,segs,p0,pn)        
        #areas=calcAreas(polys)          
        
            if plotting:
                doPlots(baseline,line,polys)
            
                allareas.append(areas)
                dateheader.append(line.getID())
       
if csvAreasOut!=None:
    print '\nWriting area csv file'
    dataToCsv(csvAreasOut,dateheader,allareas)   
if csvDistancesOut!=None:
    print '\nWriting length csv file'
    baselinedists=[]
    alldistances=[]
    p0=segs[0].getStart()
    for seg in segs[1:]:
        p1 = seg.getStart()
        baselinedists.append(p0.distance(p1))
        p0=p1
    #print baselinedists
    for areas in allareas:
        distances=[]
        for basedist,area in itertools.izip(baselinedists,areas):
            dist=area/basedist
            distances.append(dist)
        alldistances.append(distances)
    dataToCsv(csvDistancesOut,dateheader,alldistances)
