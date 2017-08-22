# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:04:48 2012

@author: nrjh
"""
from Points import Point2D
from Points import Dpoint
from Points import Ipoint
import math


class Polyline(object):
    
    '''A class to represent 2-D points'''

# The initialisation methods used to instantiate an instance
    def __init__(self,arg=None):
      if isinstance(arg, list):
          self._allPoints = []
          for point in arg:
              if isinstance(point, Point2D):
                  self._allPoints.append(point.clone ())
      elif isinstance(arg, Point2D):
          self._allPoints = [arg.clone()]
      else:
          self._allPoints = []
      self.id=None
          
    def setID(self,ID):
        self.ID=ID
    
    def getID(self):
        return self.ID
    

    
    def size(self):
        return len(self._allPoints)    
    
    def getPoint(self,i):
        return self._allPoints[i]
        
    def getPoints(self):
        return self._allPoints
        
    def getPointsAsLists(self):
        x=[]
        y=[]
        for p in self._allPoints:
            x.append(p.get_x())
            y.append(p.get_y())
        return (x,y)
        
# Returns start Point
    def getStart(self):
        if (len(self._allPoints)>0):
            return self.getPoint(0)
        else:
            return None 

# Returns end Point
    def getEnd(self):
        if (len(self._allPoints)>0):
            return self.getPoint(self.size()-1)
        else:
            return None


    def addPoint(self,point):
        if isinstance(point, Point2D):
            self._allPoints.append(point.clone())
        elif isinstance(point, tuple):
            self._allPoints.append(Point2D(point[0],point[1]))
#Inserts a point at the start of the chain
    def insertStartPoint(self, point):
        self._allPoints.insert(0,point.clone())
    
    def insertEndPoint(self, point):
        self._allPoints.append(point.clone())
       
    def insertAt(self,point,i):
        if isinstance(point, Point2D):
            self._allPoints=self._allPoints[:i]+[point.clone()]+self._allPoints[i:]

    def changePointAt(self,point,i):
        if isinstance(point, Point2D):
            self._allPoints[i]=point.clone()

            
    def getSegment(self,i):
        if i<0 or i>len(self._allPoints)-2:
            return None
        else:
            seg=Segment(self._allPoints[i],self._allPoints[i+1])
            return seg
            
    def getSegments(self):
        if self.size()<2:
            return None
        else:
            segs=[]
            for i in xrange(len(self._allPoints) -1):
                segs.append(self.getSegment(i))
            return segs
            
    def closest(self,point):
        minp=self._allPoints[0]
        mind=minp.distance(point)
            
        for p in self._allPoints[1:]:
            d=p.distance(point)
            if d<mind:
                mind=d
                minp=p
            
        for seg in self.getSegments():
            p=seg.getIntersect(point)
            if not(p==None):
                d=p.distance(point)
                if d<mind:
                    mind=d
                    minp=p
        return minp

# Returns Segment connecting end Points for Chains with > 1 point*/	
    def segStartEnd(self):
        if (self.size()<2):
            return None
        else:
            return Segment(self.getStart(),self.getEnd())



# Returns a 'Dpoint' which is the Point on the chain furthest from
# the connecting end segment if chain contains > 2 points
#  a 'Dpoint' is a Point together with information on what the distance is
#  and the index position along the line in which it occurs.  Both of these
# are set when the Point at maximum distance is located
#  Method principally used for line generalisation*/

    def furthestFromSeg(self):
        
        if (self.size()<3): 
            return None
        else:
            s=self.segStartEnd()
            maxi=1
            maxp=self.getPoint(maxi)
            maxd=s.pointDistance(maxp)
# iterate through all the chains points to find the one furthest from the
# connecting end segment
#			
            for i in range(2,(self.size()-1)):
                p=self.getPoint(i)
                d=s.pointDistance(p)
                if (d>maxd):
                    maxd=d
                    maxp=p
                    maxi=i
			
# store the Point, the index position it lies at and the 
# distance from the segment - then return it*/
            dp=Dpoint(maxp.get_x(),maxp.get_y(),maxd,maxi)
            return dp
	
# splits polyline on existing Point at specified index
# split Point is duplicated in sub-chains after split
# Returns a raw vector containing the two sub-chains

    def split(self, i):
        pair=[]
        if (self.size()<3):
            return None
        pair.append(self.subSet(0,i))
        pair.append(self.subSet(i,self.size()-1))
        return pair


# Extracts a sub-chain between specified indices*/
    def subSet(self,s,e):
		
#limit to existing indices extent
        if (s<0):
            s=0
        if (e>self.size()-1):
            e=self.size()-1
        c=Polyline()
        
        for i in xrange(s,e+1):
            c.addPoint(self.getPoint(i))
        return c
      
#implementation of Douglas-Peuker Line generalisation
# 't' is a bandwidth for the algorithm specifying the level 
# of generalisation
    def generalise(self, t):
		
# will only work if more than 2 points - but return this chain raw if not*/		
        if (self.size()<3):
            return self
        else:

#get the furthest point		
            dp=self.furthestFromSeg()
            
#  if the furthest Point lies within the specific bandwidth 
#  we can reduce this chain to the end segment so
#  return that segment as a Chain.
            if (dp.getD()<t):
                return self.segStartEnd().segAsPolyline()
# otherwise.....
            else:
#  else split the chain at the furthest point (why DP usefully holds
#  the index point
                v=self.split(dp.getI()) 
				
# extract the two chains independently from the 
# vector returned above*
                c1=v[0];
                c2=v[1];
                
# now the recursive bit since you can subsequently generalise 
# these two - which each part returns a generalised version
# of the sub-chain

                c1=c1.generalise(t)
                c2=c2.generalise(t)
                
# combine these two sub-chains again and return that*/
                return (combinePolyline(c1,c2));
        
        
# splits chain at a Point which may not be an existing point on the chain
# Thus creates a new Point and adds to the two separated sub-chains.
# Returns a list of the two resulting sub-chains	
    def splitAt(self, p,i):
        v=[]
        if (self.size()<2):
            return None
        else:
            c=self.subSet(0,i-1)
            if (not p.sameCoords(c.getEnd())):
                c.addPoint(p)
            v.append(c)
            c=self.subSet(i,self.size()-1)		
	if (not p.sameCoords(c.getStart())):
                c.insertStartPoint(p)
	v.append(c)
	return v
		

        
        
#calculates intersection of this chain with another.
#The other chain is unaffected.  This chain is split at intersections
#and a list returned of the relevant subchains.
#Does not split if intersection occurs at the end of this chain but will
#split if the other chain 'touches' this chain.
#
#Brute force approach initially - could be improved to inlcude range searching;

    def intersects(self, other):
        
        if (self.size()<2 or other.size()<2):
            return None
#we now need a list to return in any event so create one here		
        cv=[]
        
# i is used to record iteration along this chain's segments. There is one
# fewer sgement than points and we will create segments that
# look 'back' from the iterating Point
        i=0
# found_intersect records if we find an intersection - intially not
        found_intersect=False
        
# We may have more than one intersection with the other chain in a 
# given segment.  closestip records the intersection Point which is 
# closest to the start of the segment that is currently being examined
# in this chain.  The 'start' of the segment is the one nearest to the 
# start of the chain.
        closestip=None
        
# now iterate for all segments (by points 1>size-1 looking back) 
# in this chain
  
        while True:
# increment the segment - will initially set i to '1'
            i+=1
# k controls iteration likewise through 'other' chain'
            k=0
# Create a new Segment corresponding to the current position in this chain
            st=Segment(self.getPoint(i-1),self.getPoint(i))
# Now iterate through the segments in the other chain
            while True: 
# increment the segment in the other chain - will initially set k to 1 
                k+=1
# Create a new Segment corresponding the current position in the other chain
                so=Segment(other.getPoint(k-1),other.getPoint(k))
# see if the segments intersect (or touch)*
# ip is an 'Ipoint' which records information on where along a segment
# a segment intersects with another
                ip=st.intersectPoint(so)
# if we find an intersetion point*/
                if (ip!=None):
# check to see this is not simply touching at this chain's start or end*/
                    if (not (i==1 and ip.sameCoords(self.getStart())) 
                          and not(i==self.size()-1 and ip.sameCoords(self.getEnd())) ): 
#otherwise record we've found an intersection point
                        found_intersect=True
#set this to the the closest intersection point if we've not already found
#one and no existing intersection point is closer the the segment start
#
                        if (closestip==None):
                            closestip=ip
# if there's an intersection further up this segment with the other
# chain than the one found before
                        elif (ip.getUa()<closestip.getUa()):
                            closestip=ip     				
        	
# keep going through all the other chain's segments
                if (not (k<(other.size()-1))):
                    break
          
# end through all this chain's segment's if we don't find an intersection point*/
            if (not((not found_intersect) and i<self.size()-1)):
                break
# At this point we've may have found an intersection
# stored in point ip and the nature of the intersection*/
        
# test to see if we've found an intersection
        if (found_intersect):
# if intersection is at the end of this segment we need
# to split at this point
            if (closestip.getUa()==1):
                temp=self.split(i)
            elif (closestip.getUa()==0): 
                temp=self.split(i-1)
#  else split at the relevant segment*/
            else:      
                temp=self.splitAt(closestip, i)

# in either case we are splitting, 
# add to the Vector we want to return the chain split
# from before the intersection then..
# add in the Vector returned by any intersections with the 
# second part of the chain after the intersection, i.e. the bit we've not yet looked at.
# start again with part of chain not yet examined*/
            cv.append(temp[0])
            cv.extend(temp[1].intersects(other))

# else no intersection so assemble returning Vector with this single segment*/
        else:
            cv.append(self)
            
        return cv

  

##############################################################          
class Segment(object):
    
    def __init__(self,*args):
         if len(args)==4:
             p1=Point2D(args[0],args[1])
             p2=Point2D(args[2],args[3])
             self._segPoints=[p1,p2]
         elif len(args)==1:
             plist=args[0]
             p1=plist[0]
             p2=plist[1]
             self._segPoints=[p1.clone(),p2.clone()]
         else:
             p1=args[0]
             p2=args[1]
             self._segPoints=[p1.clone(),p2.clone()]
             
         
             
    def getStart(self):
        return self._segPoints[0]
    
    def getEnd(self):
        return self._segPoints[1]
        
    def reverse(self):
        temp=self._segPoints[0]
        self._segPoints[0]=self._segPoints[1]
        self._segPoints[1]=temp
    
    def getRanges(self):
        xr=self._segPoints[1].get_x()-self._segPoints[0].get_x()
        yr=self._segPoints[1].get_y()-self._segPoints[0].get_y()
        return xr,yr
        
    def getSlope(self):
        x1=self.getStart().get_x()
        y1=self.getStart().get_y()
        x2=self.getEnd().get_x()
        y2=self.getEnd().get_y()

#trap infinite slope
        if (x2==x1):
            return float('inf')
            
        return (y2-y1)/(x2-x1)
        
    def pointToRight(self,point):
        x1=self.getStart().get_x()
        y1=self.getStart().get_y()
        x2=self.getEnd().get_x()
        y2=self.getEnd().get_y()
        xA=point.get_x()
        yA=point.get_y()
        ''' test if point lies to right of line of Segment'''
        #v1 = {x2-x1, y2-y1}   # Vector 1
        #v2 = {x2-xA, y2-yA}   # Vector 1
        xp= ((x2 - x1)*(yA - y1) - (y2 - y1)*(xA - x1))
        #xp = v1.x*v2.y - v1.y*v2.x  # Cross product
        
        return xp
        
    def getIntersectLine(self,point):
        x1=self.getStart().get_x()
        y1=self.getStart().get_y()
        x2=self.getEnd().get_x()
        y2=self.getEnd().get_y()
        x3=point.get_x()
        y3=point.get_y()

#trap x coords same
        if (x2==x1):
            return(Point2D(x1,point.get_y()))
#trap y coords same
        if (y1==y2):
            return(Point2D(point.get_x(),y1))
            
        m1 = (y2-y1)/(x2-x1)
        c1 = y1-(m1*x1)
        c2 = y3+(x3/m1)
        x4 =(c2-c1)/(m1+(1./m1))
        y4=(m1*x4)+c1
        return Point2D(x4,y4)
        

    def inXRange(self,point):
        x1=self._segPoints[0].get_x()
        x2=self._segPoints[1].get_x()
        px=point.get_x()
        
        minx=min(x1,x2)
        maxx=max(x1,x2)
        
        return (px>=minx)and(px<=maxx)
        
                
    def getIntersect(self,point):
        ip=self.getIntersectLine(point)
        if self.inXRange(ip):
            return ip
        else:
            return None
            
    def getClosest(self,point):
        ip=self.getIntersectLine(point)
        if self.inXRange(ip):
            return ip
        else:
            d1=self._segPoints[0].distance(point)
            d2=self._segPoints[1].distance(point)
            if d1<d2:
                return self._segPoints[0]
            else:
                return self._segPoints[1]
                
    def intersects(self, other):
         return self.intersectPoint(other)!=None
         
#calculate intersection wiht another segment   
    def intersectPoint(self, other):        
        x1=self._segPoints[0].get_x()
        x2=self._segPoints[1].get_x()
        x3=other.getStart().get_x()
        x4=other.getEnd().get_x()
        y1=self._segPoints[0].get_y()
        y2=self._segPoints[1].get_y()
        y3=other.getStart().get_y()
        y4=other.getEnd().get_y()
        
        d=((x2-x1)*(y4-y3))-((y2-y1)*(x4-x3));
        
        if (d==0): 
            print 'found d==0'
            return None
        d=1./d
        ua=(((y1-y3)*(x4-x3))-((y4-y3)*(x1-x3)))*d
        ub=(((x3-x1)*(y2-y1))-((y3-y1)*(x2-x1)))*d
        
#        if (ua>=0 and ua<=1 and ub>=0 and ub<=1): 
#            x = x1 + (ua * (x2 - x1))
#            y = y1 + (ua * (y2 - y1))
#            return Ipoint(x,y,ua,ub)
#        else:
#            return None
        x = x1 + (ua * (x2 - x1))
        y = y1 + (ua * (y2 - y1))
        return Ipoint(x,y,ua,ub)

#calculate the distance to an intersection point on the Segment from the one passed
    def pointDistance(self, other):
      return self.getIntersectLine(other).distance(other)

# returns Segment as a Chain*/    
    def segAsPolyline(self):
        c=Polyline()
        c.addPoint(self._segPoints[0])
        c.addPoint(self._segPoints[1])
        return c

#Returns a single chain composed of two sub-chains
# is static to simplify parameter passing.
# param c1 Sub-chain 1
#
# Only works if end of c1, is same node as start of c2.

    def getPointOnSeg(self,offsets):
        
        if type(offsets)!=list:
            if type(offsets)==float and offsets>=0 and offsets<=1.0:
                offsets=[offsets]
            elif type(offsets)==int and offsets>0:
                o=[]
                for offset in range(offsets):
                     o.append(float(offset)/(offsets-1))
                offsets=o
            else:
                return None
        
        points=[]
        for offset in offsets:
            ranges=self.getRanges()
            xr=self._segPoints[0].get_x()+ranges[0]*offset
            yr=self._segPoints[0].get_y()+ranges[1]*offset
            points.append(Point2D(xr,yr))
        
        return points
        
    def getPerpendicularSegs(self,offsets,LR=1.,length=1.):
# set LR to -1 if you want left hand offsets
        segPoints=self.getPointOnSeg(offsets)
        x1=self.getStart().get_x()
        y1=self.getStart().get_y()
        x2=self.getEnd().get_x()
        y2=self.getEnd().get_y()
        segs=[]
        #trap x coords same
        
        if (x2==x1):
            #print 'in part 1'
            for segPoint in segPoints:
                px=segPoint.get_x()+length*LR
                segs.append(Segment(segPoint,Point2D(px,segPoint.get_y())))
            return segs
            
        #trap y coords same
        if (y1==y2):
            #print 'in part 2'
            for segPoint in segPoints:
                py=segPoint.get_y()+length*LR
                segs.append(Segment(segPoint,Point2D(segPoint.get_x(),py)))
            return segs
        
        #print 'in part 3'
        a = (x2-x1)
        b = (y2-y1)
        c=math.sqrt(a*a+b*b)
        #print a,b,c
        rat=length/c
        #print length,rat
        aoff=b*rat*LR
        boff=-a*rat*LR
        ccheck=math.sqrt(aoff*aoff+boff*boff)
        #print aoff,boff,ccheck
        for segPoint in segPoints:          
            p=Point2D(segPoint.get_x()+aoff,segPoint.get_y()+boff)
            segs.append(Segment(segPoint,p))
        return segs


def combinePolyline(c1, c2):
    
#check the two Polylines are linked by end / start nodes in sequence*/
    if (not c1.getEnd().sameCoords(c2.getStart())):
        return None
    else:
        c=Polyline()
        for i in xrange(0,c1.size()):
            c.addPoint(c1.getPoint(i))
        for i in xrange(1,c2.size()):
            c.addPoint(c2.getPoint(i))
        return c



			
##############################################################          
class Polygon(object):
    'just a container for polylines thus far'
			
	
    def __init__(self,polylines):
# here not cloning ploylines - at least for now
        if (polylines==None):
            self._polylines=[]
        else:
            self._polylines=polylines
        
        def getPolylines(self):
            return self._polylines
        
        def addPolyline(self,polyline):
            self._polylines.append(polyline)
        
         

     