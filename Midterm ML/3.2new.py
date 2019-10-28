#
#  Copyright 2019 Koyal Bhartia
#  @file    AdalineAlgorithm.py
#  @author  Koyal Bhartia
#  @date    03/30/2019
#  @version 1.0
#
#  @brief This is the code for Problem 3.2 from "Learning from Data"
#
#Import statments
import argparse
import numpy as np
import os, sys
from numpy import linalg as LA
import math
import pickle
import matplotlib.pyplot as plt
import random

def generate_data(sep):
    print("Generating data for sep:",sep)
    maxhori=4*(rad+thk)
    maxverti=2*(rad+thk+sep)
    c1in=0
    c1out=0
    c2in=0
    c2out=0

    X_data=np.column_stack((np.ones(N),np.ones(N)))
    fx=np.zeros((N))

    count1=0
    count2=0
    index=0
    while(count1<=999 or count2<=999):
            x1=random.uniform(-maxhori/2,maxhori/2+thk)
            x2=random.uniform(-maxverti/2-sep,maxverti/2)

            c1out=math.pow(x1,2)+math.pow(x2,2)-math.pow(rad+thk,2)
            c1in=math.pow(x1,2)+math.pow(x2,2)-math.pow(rad,2)

            c2in=math.pow(x1-(rad+thk/2),2)+math.pow(x2+sep,2)-math.pow(rad,2)
            c2out=math.pow(x1-(rad+thk/2),2)+math.pow(x2+sep,2)-math.pow(rad+thk,2)

            if(x2>=0 and c1in>=0 and c1out<=0 and count1<=999):
                X_data[index,0]=x1
                X_data[index,1]=x2
                a=index
                fx[index]=-1
                count1+=1
                index+=1
            if(x2+sep<=0 and c2in>=0 and c2out<=0 and count2<=999):
                X_data[index,0]=x1
                X_data[index,1]=x2
                b=index
                fx[index]=+1
                count2+=1
                index+=1
    return X_data,fx

def signCheck(num):
    if (num>0):
        return 1
    else:
        return -1

def misclassified(X_data,fx,w):
    break_point=1
    misclassify=[]
    X=np.column_stack((np.ones(len(X_data),dtype=int),X_data[:,0],X_data[:,1]))
    for i in range(0,len(X)):
        mul=w*X.transpose()
        sign=signCheck((mul[0,i]))
        if(sign!=(fx[i])):
            misclassify=np.append([misclassify],[i])
    if(len(misclassify)==0):
        break_point = 0
        a=-5
        length=0
    if(break_point == 1):
        point=random.randint(1,len(misclassify))
        a=int(misclassify[point-1])
    #print("Misclas. pts count",len(misclassify))
    return a,break_point

def PLA(w):
    Iterations=[]
    sep=[]
    theoretical_bound=[]
    sepvalue=0.2
    while(sepvalue<=5.1):
        w=np.mat([0,0,0])
        print("sep:",sepvalue)
        X_data,fx=generate_data(sepvalue)
        X=np.column_stack((np.ones(len(X_data),dtype=int),X_data[:,0],X_data[:,1]))
        count=0
        break_point=1
        while(break_point==1):
            count=count+1
            #print("Iterations count:",count)
            pos,break_point=misclassified(X_data,fx,w)
            if(pos!=-5):
                w_new=np.mat([0,0,0])
                w_new= w + fx[pos]*X[pos,:]
                w=w_new
        print("Iterations for above sep:",count)
        #pho=np.min(fx.transpose()*(w*X.transpose()))
        fx=np.mat([fx])

        pho=np.min(np.multiply(fx.transpose(),np.matmul(X,w.transpose())))
        R=np.max(X[:,0]+X[:,0]+X[:,0])
        w_sum=np.sum(w)
        bound=(R*R*w_sum*w_sum)/(pho*pho)
        #print(np.shape(np.matmul(X,w.transpose())),'shape of wx')

        #print(pho)
        #print(np.shape(R))
        theoretical_bound=np.append([theoretical_bound],[bound])
        Iterations=np.append([Iterations],[count])
        sep=np.append([sep],[sepvalue])
        sepvalue+=0.2
    return sep,Iterations,theoretical_bound

if __name__ == '__main__':
    centerx1,centerx2=0,0
    rad=10
    thk=5
    Total=2000
    Class=1000
    N=2000
    w=np.mat([0,0,0])
    sep,Iterations,theoretical_bound=PLA(w)
    print("Sep vales",sep)
    print("Iterations value",Iterations)
    print("theoretical_bound",theoretical_bound)

    plt.plot(sep,Iterations,'-m',label='Iteartions')
    plt.xlabel('sep')
    plt.ylabel('Iterations')
    plt.title('No. of iterations taken for PLA to converge')
    plt.legend()
    plt.show()
    plt.plot(sep,theoretical_bound,'-b',label='theoretical_bound')
    plt.xlabel('sep')
    plt.ylabel('Iterations')
    plt.title('No. of iterations taken for PLA to converge')
    plt.legend()
    plt.show()
