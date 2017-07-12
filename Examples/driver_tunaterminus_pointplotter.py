# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:35:18 2012

@author: nrjh
"""

import matplotlib.pyplot as mp
from Polylines import Polyline
from Points import Point2D

class PointPlotter(object):

    def PointFieldScatter(self,pointField,colour="black"):
        x=[]
        y=[]
        ap=pointField.getPoints()
        
        for point in ap:
            x.append(point.get_x())
            y.append(point.get_y())
        mp.scatter(x,y,color=colour)
        
        
    def show(self):
        mp.show()
    #end of function plotPoints
    
    def plotPoint(self,point,colour='black'):
        if isinstance(point,Point2D):
            mp.scatter([point.get_x()],[point.get_y()],color=colour)
        elif isinstance(point,list):
            x=[]
            y=[]
            for p in point:
                if isinstance(p,Point2D):
                    x.append(p.get_x())
                    y.append(p.get_y())
            mp.scatter(x,y,color=colour)
    

        
    def set_axis(self,xlo,xhi,ylo,yhi):
        mp.axis([xlo,xhi,ylo,yhi])
        mp.axis("equal")
        
    def clf(self):
        mp.clf()
    
    def plotVector(self,p1,p2,colour='black'):
        xs=[p1.get_x(),p2.get_x()]
        ys=[p1.get_y(),p2.get_y()]
        mp.plot(xs,ys,color=colour)
        
    def plotSegment(self,seg,colour='black'):
        p1=seg.getStart()
        p2=seg.getEnd()
        xs=[p1.get_x(),p2.get_x()]
        ys=[p1.get_y(),p2.get_y()]
        mp.plot(xs,ys,color=colour)
     
        
    def plotPolylines(self,chains,colour='black'):
        if isinstance(chains,Polyline):
            xys=chains.getPointsAsLists()
            mp.plot(xys[0],xys[1],color=colour)
        elif isinstance(chains,list):
            for chain in chains:
                if isinstance(chain,Polyline):
                    xys=chain.getPointsAsLists()
                    mp.plot(xys[0],xys[1],color=colour)   
    
    

    def plotCircle(self,circles,colour='black'):
        
        for circle in circles:
            circle=mp.Circle((circle.get_x(),circle.get_y()),circle.getRadius(),color=colour,fill=False)
            fig = mp.gcf()
            fig.gca().add_artist(circle)