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
from collections import deque
#from enum import Enum

#First in first out
#Least recently used
#least popular

def getFileSize(a,m):
    return (np.random.pareto(a) + 1) * m

def generateRequest(mean):
    #Returns structure of
    return np.random.poisson(mean)

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
        self.roundTripTime = settings['roundTripTime']
    def checkCache(self,time,fileId,fileSize):
        if fileId in self.fileIds:
            #(eventTime, function, (args))
            self.eventQueue.put((time+fileSize/self.institutionSpeed,self.receiver.receivedEvent,(time)))
        else:
            #Todo: handle full cache
            #Generate arrive at queue event
            self.eventQueue.put((time+self.roundTripTime, self.accessLink.arriveFifo,(time, fileId,fileSize)))
            #self.insert(self,fileId,fileSize)
            
    def insert(self,fileId,fileSize):
        while(self.used + fileSize > self.size):
            removed = self.cacheQueue.get()
            self.used -= removed[1]
            self.fileIds.remove(removed[0])
        self.fileIds.add(fileId)
        self.used += fileSize
        self.cacheQueue.put((fileId,fileSize))

class AccessLink:
    fifoQueue = deque() # holds (fileId, fileSize, requestTime)
    def __init__(self,eventQueue, receiver, settings):
        self.eventQueue = eventQueue
        self.receiver = receiver
        self.roundTripTime = settings['roundTripTime']
        self.institutionSpeed = settings['institutionSpeed']
        self.accessSpeed = settings['accessSpeed']
    def arriveFifo(self,time, requestTime, fileId, fileSize):
        self.fifoQueue.append((fileId, fileSize, requestTime))
        if (self.fifoQueue): #Checks if fifoQueue is empty
            self.eventQueue.put((time+fileSize/self.accessSpeed, self.departFifo, (requestTime))) 
        #TODO: handle when request is in Access Link FIFO
    def departFifo(self,time,requestTime):
        f = self.fifoQueue.popleft() #File is (fileId,fileSize,requestTime)
        requestTime = f[2]
        fileSize = f[1]
        receiveTime = time + fileSize/self.institutionSpeed
        self.eventQueue.put((receiveTime, self.receiver.receivedEvent, (requestTime)))
        if(not self.fifoQueue): #if queue is not empty
            _, fileSize, requestTime = self.fifoQueue[0]
            self.eventQueue.put((time+fileSize/self.accessSpeed,self.departFifo, (requestTime)))

class Receiver: 
    def __init__(self, results):
        self.results = results
    def receivedEvent(self,time,requestTime):
        self.results.append((requestTime,time))
    
requests = {} #
files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
results = [] #list of request, completed pairs


    

#Variables
totalRequests = 1000
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
for i in range(numberOfFiles):
    q.append((np.random.pareto(a) + 1) * b)
probSum = sum(q)
fileProbabilities = np.array(q)/probSum
for i in range(numberOfFiles):
    fileSize = (np.random.pareto(a) + 1) * b
    files[i] = (fileSize,fileProbabilities[i])

#Is the sum of probabilities 1?
print(f"Total File probability: {sum(fileProbabilities)}")

def processQueue():
    event = eventQueue.get()
    time = event[0]
    fun = event[1]
    if (len(event) > 2):
        fun(time, *event[2]) 
    else:
        fun(time)

#Lets fill the eventQueue
for i in range(totalRequests):
    fileId = np.random.choice(np.arange(numberOfFiles), p=fileProbabilities)
    time = generateRequest(reqPerSecond)
    event = (time, cache.checkCache, (fileId, fileSize[fileId])) #Format: (time, func, (func args))
    eventQueue.put(event)


