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
    return np.random.exponential(1/mean)

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
    cacheQueue = deque() # queue of (fileId,fileSize)
    def __init__(self,eventQueue,size, accessLink, receiver, settings):
        self.eventQueue = eventQueue
        self.size = size
        self.accessLink = accessLink
        self.receiver = receiver
        self.institutionSpeed = settings['institutionSpeed']
        self.roundTripTime = settings['roundTripTime']
        self.requestPerSecond = settings['requestPerSecond']
        self.stopTime = settings['stopTime']
    def checkCache(self,time,fileId,fileSize):
        if fileId in self.fileIds:
            #(eventTime, function, (args))
            self.eventQueue.put((time+fileSize/self.institutionSpeed,self.receiver.receivedEvent,(time,)))
        else:
            #Generate arrive at queue event
            self.eventQueue.put((time+self.roundTripTime, self.accessLink.arriveFifo,(time, fileId,fileSize)))
        self.createRequest(time)
        
    def createRequest(self,callTime):
        if(callTime < self.stopTime):
            fileId = np.random.choice(np.arange(numberOfFiles), p=fileProbabilities)
            time = np.random.exponential(1/self.requestPerSecond) + callTime
            event = (time, self.checkCache, (fileId, files[fileId][0])) #Format: (time, func, (func args))
            self.eventQueue.put(event)
        
    def replaceFIFO(self,fileId,fileSize):
        while(self.used + fileSize > self.size):
            removed = self.cacheQueue.popleft()
            self.used -= removed[1]
            self.fileIds.remove(removed[0])
        self.fileIds.add(fileId)
        self.used += fileSize
        self.cacheQueue.append((fileId,fileSize))

class AccessLink:
    cache = None
    fifoQueue = deque() # holds (fileId, fileSize, requestTime)
    def __init__(self,eventQueue, receiver, settings):
        self.eventQueue = eventQueue
        self.receiver = receiver
        self.roundTripTime = settings['roundTripTime']
        self.institutionSpeed = settings['institutionSpeed']
        self.accessSpeed = settings['accessSpeed']
    def arriveFifo(self,time, requestTime, fileId, fileSize):
        #print(f'arrive: {requestTime}')
        if (not self.fifoQueue): #Checks if fifoQueue is empty
            self.eventQueue.put((time+fileSize/self.accessSpeed, self.departFifo, (requestTime,))) 
        self.fifoQueue.append((fileId, fileSize, requestTime))
        #TODO: handle when request is in Access Link FIFO
    def departFifo(self,time,requestTime):
        #print('depart')
        #print(f'depart: {requestTime}')
        f = self.fifoQueue.popleft() #File is (fileId,fileSize,requestTime)
        fileId = f[0]
        fileSize = f[1]
        requestTime = f[2]
        cache.replaceFIFO(fileId,fileSize)
        receiveTime = time + fileSize/self.institutionSpeed
        self.eventQueue.put((receiveTime, self.receiver.receivedEvent, (requestTime,)))
        if(self.fifoQueue): #if queue is not empty
            _, fileSize, requestTime = self.fifoQueue[0]
            self.eventQueue.put((time+fileSize/self.accessSpeed,self.departFifo, (requestTime,)))

class Receiver: 
    def __init__(self, requestTimes,turnaround):
        self.requestTimes = requestTimes
        self.turnaround = turnaround
    def receivedEvent(self,time,requestTime):
        self.requestTimes = np.append(self.requestTimes,requestTime)
        self.turnaround = np.append(self.turnaround,time-requestTime)

    
requests = {} #
files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
requestTimes = np.array([]) #list of requestTimes
turnaround = np.array([])   #list of turnaround times, endtime - requestTime


    

#Variables
#totalRequests = 1000 #Replaced with stop time
numberOfFiles = 10000
a, m = 8/7,1/8
cacheSize = 2000 #Average file size is 1, so this is 20% of total files
results = []
settings = {'roundTripTime' : 0.4, 'institutionSpeed': 100, 
            'accessSpeed' : 15, 'requestPerSecond' : 10,
            'stopTime' : 100}
receiver = Receiver(requestTimes,turnaround)
eventQueue = PriorityQueue()
accessLink = AccessLink(eventQueue,receiver, settings)
cache = Cache(eventQueue,cacheSize,accessLink,receiver, settings)
accessLink.cache = cache

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
    print(f'time: {time}')
    fun = event[1]
    if (len(event) > 2):
        fun(time, *event[2]) 
    else:
        fun(time)

#Create initial request
cache.createRequest(0)
#Run the simulation
while(not eventQueue.empty()):
    #Stop is based on time, after stoptime new requests aren't generated
    processQueue()
    
#Evaluate results
plt.scatter(receiver.requestTimes,receiver.turnaround)