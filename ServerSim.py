# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:13:01 2021

@author: Provost
"""


#Get requests
#Get files/probabilty

import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, PriorityQueue
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
        
class Cache:
    fileIds = set() # list of fileIds
    used = 0
    cacheQueue = Queue() # queue of (fileId,fileSize)
    def __init__(self,eventQueue,size):
        self.eventQueue = eventQueue
        self.size = size
    
    def checkCache(self,fileId,fileSize,time):
        if fileId in self.fileIds:
            eventQueue.add(Received(time,time+fileSize/roundTripTime))
        else:
            #Generate arrive at queue event
            eventQueue.add(ArriveFIFO(time+400,fileId,fileSize))
            #self.insert(self,fileId,fileSize)
            
    def insert(self,fileId,fileSize):
        while(self.used + fileSize > self.size):
            removed = self.cacheQueue.pop()
            self.used -= removed[1]
            self.fileIds.remove(removed[0])
        self.fileIds.add(fileId)
        self.used += fileSize
        self.cacheQueue.push((fileId,fileSize))
        
class Request:
    def __init__(self,time,fileId):
        self.time
        self.fileId = fileId
        
    def execute(self,time,cache,fileId):
        Queue.add(cache.checkCache(fileId))

class ArriveFIFO:
    def __init__(self, time, fileId, fileSize):
        self.time = time
        self.fileId = fileId
        self.fileSize = fileSize
    def execute(self):
        fifoQueue.push((self.fileId,self.fileSize))
        
class DepartFIFO:
    def __init__(self, time, fileId, fileSize):
        self.time = time
        self.fileId = fileId
        self.fileSize = fileSize
    def execute(self):
        fifoQueue.pop()

class Received: 
    def __init__(self, requestTime, receiveTime):
        self.time = receiveTime-requestTime
    def execute(self):
        returnTimes.append(self.time)
    
requests = {} #
files = {} #key value pairs of {fileId: (fileSize,fileProbability)}

#Variables
numberOfFiles = 10000
reqPerSecond = 10
a, m = 8/7,1/8
cacheSize = 2000
roundTripTime = 100
eventQueue = PriorityQueue()
fifoQueue = Queue()
cache = Cache(eventQueue,cacheSize)
returnTimes = []
#r = RoundTrip(roundTripTime)

#Generate fileset
q = []
a = 2
b = 1/2
for i in range(numberOfFiles-1):
    q.append((np.random.pareto(a) + 1) * b)
probSum = sum(q)
fileProbabilities = np.array(q)/probSum
for i in range(numberOfFiles-1):
    fileSize = (np.random.pareto(a) + 1) * b
    files[i] = (fileSize,fileProbabilities[i])

#Is the sum of probabilities 1?
print(sum(fileProbabilities))

#Let's graph the file sizes just to take a look
x,y = zip(*files.items()) #I think we should use a different structure than
y1 = []                   #a dict, it's just a bit awkward to work with
for i in range(numberOfFiles-1):
    y1.append(y[i][0])
y1.sort()
#plt.plot(x,y1)
plt.hist(y1,bins=1000)

#e1 = Event(0.5,1,1)
#queue.put((e1.time,e1))

#e2 = Event(0.25,1,1)
#queue.put((e2.time,e2))

#queue.get()