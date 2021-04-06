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
from enum import Enum


class Policy(Enum):
    FIFO = 0 #First in first out
    LP = 1 #Least Popular
    BEST=2

#First in first out
#Least recently used
#least popular

def getFileSize(a,m):
    return (np.random.pareto(a) + 1) * m

def generateRequest(mean):
    #Returns structure of
    return np.random.exponential(1/mean)

#Generate files  

class Cache:
    def __init__(self,eventQueue,size, accessLink, receiver, files, fileProbabilities, settings, policy=Policy.FIFO):
        self.eventQueue = eventQueue
        self.size = size
        self.accessLink = accessLink
        self.receiver = receiver
        self.files = files
        self.fileProbabilities = fileProbabilities
        self.institutionSpeed = settings['institutionSpeed']
        self.roundTripTime = settings['roundTripTime']
        self.requestPerSecond = settings['requestPerSecond']
        self.stopTime = settings['stopTime']
        self.numberOfFiles = settings['numberOfFiles']
        self.fileIds = set()
        self.used = 0
        self.cacheQueue = deque()
        self.cacheQueue.clear()
        self.popularity = {} #{fileId: (popularity, size)}
        if not isinstance(policy, Policy):
            self.policy = Policy.FIFO
            print("Invalid policy, defaulting to FIFO")
        else:
            self.policy = policy

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
            fileId = np.random.choice(np.arange(self.numberOfFiles), p=self.fileProbabilities)
            time = np.random.exponential(1/self.requestPerSecond) + callTime
            event = (time, self.checkCache, (fileId, self.files[fileId][0])) #Format: (time, func, (func args))
            self.eventQueue.put(event)
        
    def replace(self, fileId,fileSize):
        if (self.policy == Policy.FIFO):
            self.replaceFIFO(fileId,fileSize)
        elif (self.policy == Policy.LP):
            self.replaceLeastPopular(fileId,fileSize)
        elif (self.policy == Policy.BEST):
            self.replaceBestFit(fileId,fileSize)
        else:
            print("INVALID POLICY, NOT ADDING FILE")


    def replaceFIFO(self,fileId,fileSize):
        if (fileSize > self.size):
            print("FILE TOO LARGE FOR CACHE")
            return
        if(fileId in self.fileIds): #If it was added before event executed
            return
        while(self.used + fileSize > self.size):
           # print(f"Replacing: {fileId}")
            removed = self.cacheQueue.popleft()
            self.used -= removed[1]
            try:
                self.fileIds.remove(removed[0])
            except:
                print(removed)
        self.fileIds.add(fileId)
        self.used += fileSize
        self.cacheQueue.append((fileId,fileSize))

    def activeFile(self, item):
        if item[0] in self.fileIds:
            return True
        return False

    def replaceLeastPopular(self,fileId,fileSize):
        if(fileId in self.fileIds): #If it was added before event executed
            self.popularity[fileId][0] += 1
            return
        filtered = filter(self.activeFile, self.popularity.items())
        sortedFiles = sorted(filtered, key=lambda x: x[1][0])
        i = 0
        if (fileSize > self.size):
            print("FILE TOO LARGE FOR CACHE")
            return 
        while (self.used + fileSize > self.size):
            # print(f"Replacing: {fileId}")
            removed = sortedFiles[i]
            #print(f"Replacing: {removed[0], removed[1][0]}")
            self.used -= removed[1][1]
            self.fileIds.remove(removed[0])
            i += 1

        self.fileIds.add(fileId)
        self.used += fileSize
        if fileId in self.popularity:
            self.popularity[fileId][0] += 1
        else:
            self.popularity[fileId] = [1,fileSize]
    
    def replaceBestFit(self,fileId,fileSize):
        if(fileId in self.fileIds): #If it was added before event executed
            return
        filtered = filter(self.activeFile, self.popularity.items())
        sortedFiles = sorted(filtered, key=lambda x: x[1][1], reverse=True)
        i = 0
        if (fileSize > self.size):
            print("FILE TOO LARGE FOR CACHE")
            return
        enoughSpace = False
        i = 0
        while (self.used + fileSize > self.size):
            if (sortedFiles[i][1][1] >= fileSize):
                j = i + 1
                while ( j < len(sortedFiles) and sortedFiles[j][1][1] >= fileSize):
                    i = j
                    j += 1
                removed = sortedFiles[i]
                self.used -= removed[1][1]
                self.fileIds.remove(removed[0])
            else:
                removed = sortedFiles[i]
                self.used -= removed[1][1]
                self.fileIds.remove(removed[0])
            i += 1
        self.popularity[fileId] = [0,fileSize]
        self.fileIds.add(fileId)
        self.used += fileSize
            

class AccessLink:

    def __init__(self,eventQueue, receiver, settings):
        self.eventQueue = eventQueue
        self.receiver = receiver
        self.roundTripTime = settings['roundTripTime']
        self.institutionSpeed = settings['institutionSpeed']
        self.accessSpeed = settings['accessSpeed']
        self.fifoQueue = deque()
        self.fifoQueue.clear()
        self.cache = None

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
        self.cache.replace(fileId,fileSize)
        receiveTime = time + fileSize/self.institutionSpeed
        self.eventQueue.put((receiveTime, self.receiver.receivedEvent, (requestTime,)))
        if(self.fifoQueue): #if queue is not empty
            _, fileSize, requestTime = self.fifoQueue[0]
            self.eventQueue.put((time+fileSize/self.accessSpeed,self.departFifo, (requestTime,)))

class Receiver: 
    def __init__(self):
        self.requestTimes = np.array([])
        self.turnaround =np.array([])
    def receivedEvent(self,time,requestTime):
        self.requestTimes = np.append(self.requestTimes,requestTime)
        self.turnaround = np.append(self.turnaround,time-requestTime)

def runSimulation(policy=Policy.FIFO):
    requests = {} #
    files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
    #Variables
    #totalRequests = 1000 #Replaced with stop time
    a, m = 8/7,1/8
    cacheSize = 500 #Average file size is 1, so this is 20% of total files
    numberOfFiles = 10000
    results = []
    settings = {'roundTripTime' : 0.4, 'institutionSpeed': 100, 
                'accessSpeed' : 15, 'requestPerSecond' : 10,
                'stopTime' : 100, 'numberOfFiles': numberOfFiles}

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
    #print(f"Total File probability: {sum(fileProbabilities)}")
    receiver = Receiver()
    eventQueue = PriorityQueue()
    if (not eventQueue.empty):
        print("BIG YIKES NOT EMPTY")
    accessLink = AccessLink(eventQueue,receiver, settings)
    cache = Cache(eventQueue,cacheSize,accessLink,receiver, files, fileProbabilities,settings, policy=policy)
    cache.fileIds.clear()
    accessLink.cache = cache

    #r = RoundTrip(roundTripTime)



    def processQueue():
        event = eventQueue.get()
        time = event[0]
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
    x, y = receiver.requestTimes, receiver.turnaround
    del receiver
    del cache
    del eventQueue
    del accessLink
    return x, y
    
#Evaluate results
#plt.scatter(receiver.requestTimes,receiver.turnaround)
#print(f"Average: {np.average(receiver.turnaround)}")

plt.clf()




print("***********************")

fifoX = np.array([])
fifoY = np.array([])
for j in range(10):
    fX, fY = runSimulation(policy=Policy.FIFO)
    #print(np.sum(fY))
    fifoX = np.append(fifoX, np.average(fX))
    fifoY = np.append(fifoY, np.average(fY))
print(f'Average fifo: {np.average(fifoY)}')
plt.scatter(np.arange(10),fifoY)

print("***********************")


lX = np.array([])
lY = np.array([])
for j in range(100):
    x, y = runSimulation(policy=Policy.BEST)
    #print(np.sum(y))
    lX = np.append(lX, np.average(x))
    lY = np.append(lY, np.average(y))
print(f'Average Best: {np.average(lY)}')
plt.scatter(np.arange(100),lY)

print("***********************")

#plt.scatter(lpX,lpY)