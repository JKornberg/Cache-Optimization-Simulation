# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:13:01 2021

@author: Provost
"""


#Get requests
#Get files/probabilty

import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
#from enum import Enum

#First in first out
#Least recently used
#least popular

def getFileSize(a,m):
    return (np.random.pareto(a) + 1) * m

def generateRequests(mean, n):
    #Returns structure of 
    return np.random.poisson(mean,n)

#Generate files

'''       
class RoundTrip:
    def __init__(self,speed, queue):
        self.speed = speed
        self.free = speed
        self.queue = queue
    def addTask(size):
        if free < size:
'''

class Request:
    def __init__(self,time,fileId):
        self.time
        self.fileId = fileId
        
    def executeRequest(self,time,cache,fileId):
        queue.add(cache.checkCache(fileId))
        
class Cache:
    fileIds = {}
    used = 0
    def __init__(self,queue,size):
        self.queue = queue
        self.size = size
    def checkCache(self,fileId,time):
        if fileId in self.fileIds:
            size = files[fileId][0]
            queue.add(Received(time,time+size/roundTripTime))

class Received: 
    def __init__(self, requestTime, receiveTime):
        self.time = receiveTime
    
requests = {} #
files = {} #key value pairs of {fileId: (fileSize,fileProbability)}

#Variables
numberOfFiles = 10000
reqPerSecond = 10
a, m = 8/7,1/8
cacheSize = 2000
roundTripTime = 100
queue = PriorityQueue()
cache = Cache(queue,cacheSize)
#r = RoundTrip(roundTripTime)

#Generate fileset
q = []
a = 2
b = 1/2
for i in range(numberOfFiles-1):
    q.append((np.random.pareto(a) + 1) * b)
for i in range(numberOfFiles-1):
    fileSize = (np.random.pareto(a) + 1) * b
    probSum = sum(q)
    fileProbability = q[i]/(probSum-q[i])
    files[i] = (fileSize,fileProbability)

#Is the sum of probabilities 1?
x = 0
for i in range(numberOfFiles-1):
    x += files[i][1]
print(x)

#Let's graph the file sizes just to take a look
x,y = zip(*files.items()) #I think we should use a different structure than
y1 = []                   #a dict, it's just a bit awkward to work with
for i in range(numberOfFiles-1):
    y1.append(y[i][0])
plt.plot(x,y1)
#Oh no. What's wrong here?
#My first run through has a weird spike at 8000?
#No idea why but definitely not pareto

#e1 = Event(0.5,1,1)
#queue.put((e1.time,e1))

#e2 = Event(0.25,1,1)
#queue.put((e2.time,e2))

#queue.get()