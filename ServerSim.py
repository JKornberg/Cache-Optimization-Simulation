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
    def __init__(self,eventQueue,size, accessLink, receiver, settings):
        self.eventQueue = eventQueue
        self.size = size
        self.accessLink = accessLink
        self.receiver = receiver
        self.institutionSpeed = settings['institutionSpeed']
    def checkCache(self,fileId,fileSize,time):
        if fileId in self.fileIds:
            #(eventTime, function, (args))
            self.eventQueue.put((time+fileSize/self.institutionSpeed,self.receiver.receivedEvent,(time)))
        else:
            #Generate arrive at queue event
            self.eventQueue.put((time+400, self.accessLink.arriveFifo,(fileId,fileSize)))
            #self.insert(self,fileId,fileSize)
            
    def insert(self,fileId,fileSize):
        while(self.used + fileSize > self.size):
            removed = self.cacheQueue.get()
            self.used -= removed[1]
            self.fileIds.remove(removed[0])
        self.fileIds.add(fileId)
        self.used += fileSize
        self.cacheQueue.put((fileId,fileSize))
        
class Request:
    def __init__(self,cache, eventQueue):
        self.eventQueue = eventQueue
        self.cache = cache
    def addRequest(self,time,fileId):
        self.eventQueue.put(time, cache.checkCache, (fileId))

class AccessLink:
    fifoQueue = Queue() # holds (fileId, fileSize, requestTime)
    def __init__(self,eventQueue, receiver, settings):
        self.eventQueue = eventQueue
        self.receiver = receiver
        self.roundTripSpeed = settings['roundTripSpeed']
        self.institutionSpeed = settings['institutionSpeed']
        self.accessSpeed = settings['accessSpeed']
    def arriveFifo(self,time, requestTime, fileId, fileSize):
        self.fifoQueue.put((fileId, fileSize, requestTime))
        self.eventQueue.put((time+fileSize/self.accessSpeed, self.departFifo))
        #TODO: handle when request is in Access Link FIFO
    def departFifo(self,time):
        file = self.fifoQueue.get() #File is (fileId,fileSize,requestTime)
        #TODO: insert file into cache
        #Do we want to get cache access in here?
        requestTime = file[2]
        receiveTime = time + fileSize/self.institutionSpeed
        self.eventQueue.put((receiveTime, self.receiver.receivedEvent, (requestTime)))

class Receiver: 
    def __init__(self, results):
        self.results = results
    def receivedEvent(self,time,requestTime):
        self.results.append((requestTime,time))
    
requests = {} #
files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
results = [] #list of request, completed pairs


    

#Variables
numberOfFiles = 10000
reqPerSecond = 10
a, m = 8/7,1/8
cacheSize = 2000
results = []
settings = {'roundTripTime' : 400, 'institutionSpeed': 100, 'accessSpeed' : 15}
receiver = Receiver(results)
eventQueue = PriorityQueue()
accessLink = AccessLink(eventQueue,receiver, settings)
cache = Cache(eventQueue,cacheSize,accessLink,receiver, settings)
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


def processQueue():
    event = eventQueue.get()
    time = event[0]
    fun = event[1]
    if (len(event) > 2):
        fun(time, *event[2]) 
    else:
        fun(time)
