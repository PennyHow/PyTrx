# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 14:00:35 2016

@author: nrjh
"""
import os
from shutil import copyfile
    
import glob
from datetime import date,time

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
    
def createNewDir(newDir):
    if not os.path.exists(newDir):
        print 'Creating new directory: '+newDir
        os.makedirs(newDir)
    return newDir

def copyFile(fromfile,tofile):
    print '\nCopying:',fromfile,' to ',tofile
    copyfile(fromfile,tofile)


def extractAt(inpath,outroot,intervalList):
    path=inpath+'\\*.JPG'
    print 'Extracting from:' ,path
    filelist=glob.glob(path)
    #print filelist
    for interval in intervalList:
        newdir=outroot+'\\'+interval
        createNewDir(newdir)
        for item in filelist:
            
            fn=item.split('\\')[-1]
            
            tokens=fn.split('_')
            
            t=tokens[2].split('.')[0]
            
            #print fn,t
            if t==interval:
                outpath=newdir+'\\'+fn
                #print item,outpath
                copyFile(item,outpath)
                    
if __name__ == "__main__":
    inpath="G:\KR1_2014_Sequenced"
    outroot="G:\KR1_2014_hourly"
    intervalList=['060000','180000']
    extractAt(inpath,outroot,intervalList)