#
#  Copyright 2019 Koyal Bhartia
#  @file    AdalineAlgorithm.py
#  @author  Koyal Bhartia
#  @date    03/30/2019
#  @version 1.0
#
#  @brief This is the code for Problem 1.5 from "Learning from Data"
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

#Generation of random data
def generate_data(N):
    X_data=np.column_stack((np.ones(N),np.ones(N))) # X1,X2 data
    fx=np.zeros((N)) #Corresponding y
    for i in range(len(X_data)):
        X_data[i,0]=random.uniform(-1,1)
        X_data[i,1]=random.uniform(-1,1)
        if (X_data[i,0]*t_slope+t_intercept<X_data[i,1]):
            a=i
            fx[i]=1
        else:
            b=i
            fx[i]=-1
    return X_data,fx,a,b

def plotdata(X_data,fx,a,b):
    for i in range(0,len(X_data)):
        if(fx[i]==1):
            plt.plot(X_data[i,0],X_data[i,1],'*g')
        else:
            plt.plot(X_data[i,0],X_data[i,1],'oy')

    plt.plot(X_data[a,0],X_data[a,1],'*g', label='y=1 points')
    plt.plot(X_data[b,0],X_data[b,1],'oy', label='y=-1 points')


def signCheck(num):
    if (num>0):
        return 1
    else:
        return -1

#Checks for the misclassification of points for each hypothesis
def misclassified(X_data,fx,w):
    break_point=1 # Flag to keep track if all points are classified correctly
    misclassify=[]
    X=np.column_stack((np.ones(len(X_data),dtype=int),X_data[:,0],X_data[:,1]))
    for i in range(0,len(X)):
        mul=w*X.transpose()
        sign=signCheck(float(mul[0,i]))
        # Check for misclassification
        if(sign!=fx[i]):
            misclassify=np.append([misclassify],[i])
    if(len(misclassify)==0):
        break_point = 0
        a=-5
        length=0
    if(break_point == 1):
        point=random.randint(1,len(misclassify))
        a=int(misclassify[point-1])
    print("Misclas. pts count",len(misclassify))
    length=len(misclassify)
    return a,break_point,length,mul

# Runs the PLA model
def PLA(w):
    count=0
    break_point=1
    X_data,fx,a,b=generate_data(N_train)
    plotdata(X_data, fx,a,b)
    X=np.column_stack((np.ones(len(X_data),dtype=int),X_data[:,0],X_data[:,1]))
    while(break_point==1 and count<1000):
        count=count+1
        print("Iterations count:",count)
        index,break_point,length,mul=misclassified(X_data,fx,w)
        if(index!=-5):
            randd=random.randint(0,99)
            if(mul[0,randd]*fx[randd]<=1):
                w_new=np.mat([0,0,0])
                w_new= w + eta*X[randd,:]*(fx[randd]-mul[0,randd]) #Adaline algorithm
                w=w_new
            m=float(-w_new[0,1]/w_new[0,2])
            c=float(-w_new[0,0]/w_new[0,2])
            x_line1=np.mat([[-1],[1]])
            y_line1=m*x_line1+c
            plt.plot(x_line1,y_line1,'-c')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis([-1,1,-1,1])
            plt.title('Data set')
    if count==1:
        w_new=w
    return w_new

if __name__ == '__main__':
    N_train=100
    N_test=10000
    N_trials=1000
    t_slope=0.8
    t_intercept=0.2
    eta=1
    w=np.mat([0,0,0])
    w_new=PLA(w)
    # Check on the test data
    X_data,fx,a,b=generate_data(N_test)
    X=np.column_stack((np.ones(len(X_data),dtype=int),X_data[:,0],X_data[:,1]))
    misclassify=0
    for i in range(0,len(X)):
        mul=w_new*X.transpose()
        sign=signCheck(float(mul[0,i]))
        if(sign!=fx[i]):
            misclassify+=1
    print("Error on the test data:",misclassify/10000)

    #Plot for thte purpose of creating the legend
    m=float(-w_new[0,1]/w_new[0,2])
    c=float(-w_new[0,0]/w_new[0,2])
    x_line1=np.mat([[-1],[1]])
    y_line1=m*x_line1+c
    x_linet=x_line1
    y_linet=t_slope*x_linet+t_intercept
    plt.plot(x_line1,y_line1,'-c',label='PLA Lines')
    plt.plot(x_line1,y_line1,'-b',label='Best Fit Line')
    plt.plot(x_linet,y_linet,'-m',label='target_function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-1,1,-1,1])
    plt.title(' Training Data set')
    plt.legend()
    plt.show()

    #Plot the test data
    plotdata(X_data, fx,a,b)

    #Plot the best fit line and target function on the test data
    plt.plot(x_line1,y_line1,'-b',label='Best Fit Line')
    plt.plot(x_linet,y_linet,'-m',label='target_function')
    plt.axis([-1,1,-1,1])
    plt.title('Test Data set')
    plt.legend()
    plt.show()
