# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 14:00:35 2016

@author: nrjh
"""
import os
from shutil import copyfile
    
import glob
from datetime import date,time,timedelta

def getDateFromStr(strdate,millenium=2000):
    year=int(strdate[0:2])+millenium
    month=int(strdate[2:4])
    day=int(strdate[4:6])
    
    return date(year,month,day)

def getTimeFromStr(timestr):
    print timestr
    hour=int(timestr[0:2])
    minute=int(timestr[2:4])
    second=int(timestr[4:6])
    
    return time(hour,minute,second) 
  
def getStrFromDate(datedate):
    return str(datedate.year)[2:4]+str(datedate.month).zfill(2)+str(datedate.day).zfill(2)
    
def createNewDateDir(stemdir,datedate,prefix=''):
    
    newDir=stemdir+'\\'+prefix+datedate

    
    if not os.path.exists(newDir):
        os.makedirs(newDir)
        print 'Creating new directory: '+newDir
    return newDir

def copyFile(fromfile,tofile,copy=True):
    print '\nCopying:',fromfile,' to ',tofile
    if copy:
        copyfile(fromfile,tofile)
    else:
        print 'Dummy copy operation - copying not done'


inputdir="H:\\Tunabreen_imagery\\resequence-sample"
outputdir="H:\\Tunabreen_imagery\\sorted-by-day"
copy=True

intervalhours=1
intervalmins=0
intervalsecs=0



intervaltime=timedelta(hours=intervalhours,minutes=intervalmins,seconds=intervalsecs)
starttime=time(0,0,0)

filelist=glob.glob(inputdir+'\*')


#print filelist


start=filelist[0]

#print filelist[0].split('\\')
#print start

tokens=start.split('\\')[-1].split('_')
currentdate=tokens[1]

#print 'current date'
#print currentdate

currentDir=createNewDateDir(outputdir,currentdate,'Day_')

currentdate=getDateFromStr(currentdate)


#print currentdate.year,currentdate.month,currentdate.day

for item in filelist:
    tokens=item.split('\\')
    #print item
    fname=tokens[-1]
    #print fname
    tokens=fname.split('_')
    newdate=tokens[1]
    #print newdate
    if newdate!=currentdate:
        currentdate=newdate
        currentDir=createNewDateDir(outputdir,currentdate,'Day_')
    
    t=tokens[2].split('.')[0]
    #print t,t[2:4]
    
    if t[2:4]=='00':
        outpath=currentDir+'\\'+fname
        copyFile(item,outpath,copy)
