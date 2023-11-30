

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import *

SEGS = 2
NEIGHBORS = [(-1,0) , (1,0) , (0,-1) , (0,1)]
BETA = 1
TEMPERATURE = 40.0
ITERATIONS = 1000000
COOLRATE = 0.999


def isSafe(s,x,y):
    (a,b) = s
    return x>=0 and x<a and y>=0 and y<b
def delta(i,l):
    if(i==l):
        return -BETA
    return BETA


def reconstruct(labs):
    labels = labs
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            labels[i][j] = (labels[i][j]*255)/(SEGS-1)
    return labels

def calculateEnergy(img,variances,labels):
    energy = 0.0
    for i in range(len(img)):
        for j in range(len(img[0])):
            l = labels[i][j]
            energy += log(sqrt(variances[l]))
            for (p,q) in NEIGHBORS:
                if isSafe(img.shape,i+p,j+q):
                    energy += (delta(l,labels[i+p][j+q])/2.0)
    return energy


def variance(sums1,squares1,nos1):
    return squares1/nos1-(sums1/nos1)**2


def initialize(img):
    labels = np.zeros(shape=img.shape ,dtype=np.uint8)
    nos = [0.0]*SEGS
    sums = [0.0]*SEGS
    squares = [0.0]*SEGS
    for i in range(len(img)):
        for j in range(len(img[0])):
            l = randint(0,SEGS-1)
            sums[l] += img[i][j]
            squares[l] += img[i][j]**2
            nos[l] += 1.0
            labels[i][j] = l
    return (sums,squares,nos,labels)


original = cv2.imread('Waypoint_generation/Test.png')
# cv2.imshow('og',original)
# origflt = original.astype(np.float32)


img = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)


sums,squares,nos,labels = initialize(img)
variances = [variance(sums[i],squares[i],nos[i]) for i in range(SEGS)]


energy = calculateEnergy(img,variances,labels)

variances,sums,squares,nos


temp = TEMPERATURE
it = ITERATIONS
while it>0:
    (a,b) = img.shape
    change = False
    x = randint(0,a-1)
    y = randint(0,b-1)
    val = float(img[x][y])
    l = labels[x][y]
    newl = l
    while newl == l:
        newl = randint(0,SEGS-1)
    
    val = float(val)
    remsums = sums[l] - val
    addsums = sums[newl] + val
    
    remsquares = squares[l] - val*val
    addsquares = squares[newl] + val*val
    
    remvar = variance(remsums,remsquares,nos[l]-1)
    addvar = variance(addsums,addsquares,nos[newl]+1)

    newenergy = energy
    newenergy -= log(sqrt(variance(sums[l],squares[l],nos[l])))*(nos[l])
    newenergy += log(sqrt(remvar))*(nos[l]-1)
    newenergy -= log(sqrt(variance(sums[newl],squares[newl],nos[newl])))*(nos[newl])
    newenergy += log(sqrt(addvar))*(nos[newl]+1)
    for (p,q) in NEIGHBORS:
        if isSafe(img.shape,x+p,y+q):
            newenergy -= delta(l,labels[x+p][y+q])
            newenergy += delta(newl,labels[x+p][y+q])

    if newenergy < energy:
        change = True
    else:
        prob = 1.1
        if temp != 0:
            prob = np.exp((energy-newenergy)/temp)
        if prob >= (randint(0,1000)+0.0)/1000:
            change = True
    
    if change:
        labels[x][y] = newl
        energy = newenergy
        
        nos[l] -= 1
        sums[l] = remsums
        squares[l] = remsquares
        
        nos[newl] += 1
        sums[newl] = addsums
        squares[newl] = addsquares
        
    temp *= COOLRATE
    it -= 1


plt.imshow(reconstruct(labels) , interpolation='nearest',cmap='gray')
plt.imshow(img,cmap='gray')
cv2.imwrite("segmented.jpg",labels)
