# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 19:13:01 2021
@author: Provost
@author: Jonah Kornberg Jkornberg@ufl.edu
"""


#Get requests
#Get files/probabilty

import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque
from enum import Enum
from datetime import datetime


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
    def __init__(self,eventQueue,size, accessLink, receiver, files, fileProbabilities, settings, policy=Policy.FIFO, generateRequest = True):
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
        self.generateRequest = generateRequest #For whether using same requests or not
        if not isinstance(policy, Policy):
            self.policy = Policy.FIFO
            print("Invalid policy, defaulting to FIFO")
        else:
            self.policy = policy

    def clear(self):
        self.cacheQueue.clear()
        self.fileIds.clear()
        self.popularity = {}
        self.used = 0

    def checkCache(self,time,fileId,fileSize):
        if fileId in self.fileIds:
            #(eventTime, function, (args))
            self.eventQueue.put((time+fileSize/self.institutionSpeed,self.receiver.receivedEvent,(time,)))
        else:
            #Generate arrive at queue event
            self.eventQueue.put((time+self.roundTripTime, self.accessLink.arriveFifo,(time, fileId,fileSize)))
        self.createRequest(time)
        
    def createRequest(self,callTime):
        if self.generateRequest:
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
            self.receiver.addRemoved(removed[0])

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
            self.receiver.addRemoved(removed[0])


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
            self.receiver.addRemoved(removed[0])

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
    def clear(self):
        self.fifoQueue.clear()

class Receiver: 
    def __init__(self):
        self.requestTimes = np.array([])
        self.turnaround =np.array([])
        self.removed = np.array([])
    def receivedEvent(self,time,requestTime):
        self.requestTimes = np.append(self.requestTimes,requestTime)
        self.turnaround = np.append(self.turnaround,time)
    def clear(self):
        self.requestTimes = np.array([])
        self.turnaround = np.array([])
        self.removed = np.array([])
    
    def addRemoved(self, fileId):
        self.removed = np.append(self.removed,fileId)



def runSimulation(policy=Policy.FIFO):
    def processQueue():
        event = eventQueue.get()
        time = event[0]
        fun = event[1]
        if (len(event) > 2):
            fun(time, *event[2]) 
        else:
            fun(time)
    
    files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
    #Variables
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

    receiver = Receiver()
    eventQueue = PriorityQueue()
    if (not eventQueue.empty):
        print("Event Queue not empty, leftover data in queue")
    accessLink = AccessLink(eventQueue,receiver, settings)
    cache = Cache(eventQueue,cacheSize,accessLink,receiver, files, fileProbabilities,settings, policy=policy)
    cache.fileIds.clear()
    accessLink.cache = cache

    #r = RoundTrip(roundTripTime)

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

def runIndividual():
    plt.clf()
    print("***********************")

    fifoX = np.array([])
    fifoY = np.array([])
    for j in range(1000):
        fX, fY = runSimulation(policy=Policy.FIFO)
        #print(np.sum(fY))
        fifoX = np.append(fifoX, np.average(fX))
        fifoY = np.append(fifoY, np.average(fY))
    print(f'Average fifo: {np.average(fifoY)}')
    plt.scatter(np.arange(1000),fifoY)

    print("***********************")


    lX = np.array([])
    lY = np.array([])
    for j in range(1000):
        x, y = runSimulation(policy=Policy.BEST)
        #print(np.sum(y))
        lX = np.append(lX, np.average(x))
        lY = np.append(lY, np.average(y))
    print(f'Average Best: {np.average(lY)}')
    plt.scatter(np.arange(1000),lY)

    print("***********************")

    fifoX = np.array([])
    fifoY = np.array([])
    for j in range(1000):
        fX, fY = runSimulation(policy=Policy.LP)
        #print(np.sum(fY))
        fifoX = np.append(fifoX, np.average(fX))
        fifoY = np.append(fifoY, np.average(fY))
    print(f'Average fifo: {np.average(fifoY)}')
    plt.scatter(np.arange(1000),fifoY)



def runAllSimulations(count = 1):
    def processQueue():
        event = eventQueue.get()
        time = event[0]
        fun = event[1]
        if (len(event) > 2):
            fun(time, *event[2]) 
        else:
            fun(time)
    
    allRequestTimes = np.zeros([0])
    allTurnarounds = np.zeros([3,0])
    allRemoved = [0] * 3
    for i in range(count):

        files = {} #key value pairs of {fileId: (fileSize,fileProbability)}
        a, m = 8/7,1/8
        cacheSize = 1000 #Average file size is 1, so this is 20% of total files
        numberOfFiles = 50000
        results = []
        settings = {'roundTripTime' : 0.4, 'institutionSpeed': 100, 
                    'accessSpeed' : 15, 'requestPerSecond' : 10,
                    'stopTime' : 100, 'numberOfFiles': numberOfFiles}
        totalRequests = 100000
       
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
        receiver = Receiver()
        eventQueue = PriorityQueue()
        requestTimes = np.zeros(totalRequests)
        requestTimes[0] = np.random.exponential(1/settings['requestPerSecond'])
        for i in range(len(requestTimes[1:])):
            requestTimes[i] = np.random.exponential(1/settings['requestPerSecond']) + requestTimes[i-1]
        requestFiles = np.random.choice(np.arange(numberOfFiles),size=totalRequests, p=fileProbabilities)
        accessLink = AccessLink(eventQueue,receiver, settings)
        turnArounds = np.zeros((3,totalRequests))
        #requestTimes = np.zeros((3,totalRequests))
        for i,p in enumerate(Policy):
            receiver.clear()
            accessLink.clear()
            cache = Cache(eventQueue,cacheSize,accessLink,receiver, files, fileProbabilities,settings, policy=p, generateRequest=False)
            cache.clear()
            accessLink.cache = cache
            #Create initial request
            for t,f in np.stack((requestTimes,requestFiles),axis=1):
                event = (t, cache.checkCache, (f, files[f][0])) #Format: (time, func, (func args))
                eventQueue.put(event)
            while(not eventQueue.empty()):
                #Stop is based on time, after stoptime new requests aren't generated
                processQueue()
            x,y,r = receiver.requestTimes, receiver.turnaround, receiver.removed
            turnArounds[i] = y
            allRemoved[i] = r
           # requestTimes[i]  = y
        del receiver
        del cache
        del eventQueue
        del accessLink
        allTurnarounds = np.concatenate((allTurnarounds, turnArounds), axis=1)
        allRequestTimes = np.concatenate((allRequestTimes, requestTimes))
    return (allTurnarounds, allRequestTimes)

    
Finishes, Requests  = runAllSimulations()
for i,p in enumerate(Policy):
    print(f'Average {p}: {np.average(Finishes[i]-Requests)}')
    