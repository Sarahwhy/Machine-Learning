import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
from numpy import matlib
from numpy import linalg 
import math
import random
import itertools

# Ridge Regression
def ridgereg(x,y,Lambda):
    alpha = np.mean(y)
    temp = linalg .inv(np.dot(x.transpose(), x)+Lambda*np.identity(8))
    w = np.dot(np.dot(temp,x.transpose()),y)
    coeff = [alpha,w]
    return coeff
    
# 10-fold cross validation for ridge regression
# Take x(covariates), y (labels) and lambda (penalty) as inputs
# Return average residual sum of square
def cv_10(x,y,Lambda):
    partition_x = []
    partition_y = []
    indexset = []
    error_cv = []
    ran_index = range(len(y))
    random.shuffle(ran_index)
    Rss = 0
    for i in range(0,10,1):
        temp = ran_index[(len(y)/10)*i:(len(y)/10)*(i+1)]
        indexset.append(temp)
    for i in range(0,10,1):
        partition_x.append(x[indexset[i]])
        partition_y.append(y[indexset[i]]) #random partition of 10 subsets
    for i in range(0,10,1):
        test_x_cv = partition_x[i]
        test_y_cv = partition_y[i]
        mylist = range(10)
        del mylist[i]
        temp = [partition_x[i] for i in mylist]
        train_x_cv = np.array(list(itertools.chain(*temp)))
        temp = [partition_y[i] for i in mylist]
        train_y_cv = np.array(list(itertools.chain(*temp)))
        alpha = ridgereg(train_x_cv,train_y_cv,Lambda)[0]
        w = ridgereg(train_x_cv,train_y_cv,Lambda)[1]
        expected = test_y_cv
        size = len(expected)
        predicted = np.dot(test_x_cv,w)+alpha*(np.ones(size)).reshape(size,1)
        expected = expected.reshape(len(expected))
        for i in range(len(expected)):
            Rss = Rss+(expected[i]-predicted[i])**2
    return float(Rss)/10

# Logistic regression
def sx(x):
    if x>700:
        s=1
    elif x<-700:
        s=0
    else:
        s=1/(1+math.exp(-x))
    return s

# logistic regression using batch gradient descent
def logistic(x,y,w,n,stepsize):
    temp = np.ones(len(x))
    new_x = np.c_[x, temp]
    for i in range(n):
        temp1 = np.dot(new_x,w)
        for j in range(len(temp1)):
            temp1[j] = sx(temp1[j])
        temp2 = np.dot(y-temp1.transpose(),new_x)*stepsize
        temp2 = temp2.transpose()
        w = np.array(map(sum, zip(w,temp2)))
    predict = np.dot(new_x,w)
    for k in range(len(predict)):
        if predict[k] > 0.5:
            predict[k] = 1
        else:
            predict[k] = 0
    result = [predict,w]
    return result

def risk(predict, true):
    rw_sum = 0
    for i in range(len(true)):
        temp = true[i]*math.log(sx(predict[i]))+(1-true[i])*math.log(1-sx(predict[i]))
        rw_sum = rw_sum+temp
    return -rw_sum

# logistic regression using stochastic gradient descent
def sto_logistic(x,y,w,n,stepsize):
    temp = np.ones(len(x))
    new_x = np.c_[x, temp]
    for i in range(n):
        temp1 = np.dot(new_x,w)
        for j in range(len(temp1)):
            temp1[j] = sx(temp1[j])
        index = np.random.random_integers(x.shape[0]-1)
        temp2 = (np.transpose(y)[index]-temp1[index])*stepsize*new_x[index]
        w = w+temp2
    predict = np.dot(new_x,w)
    for k in range(len(predict)):
        if predict[k] > 0.5:
            predict[k] = 1
        else:
            predict[k] = 0
    result = [predict,w]
    return result

# Logistic with kernel
# Quadratic kernel function
def qua_kernel(x,rau):
    n = x.shape[0]
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            result[i,j] = (np.dot(x[i,],x[j,].T)+rau)**2
    return result
    

#stochastic gradient descent for quadratic kernelized logistic regression
def kernel_sto_logistic(x,y,a,n,stepsize,rau):
    temp = np.ones(len(x))
    new_x = np.c_[x, temp]
    k = qua_kernel(new_x,rau)
    for i in range(n):
        temp1 = np.dot(k,a)
        for j in range(len(temp1)):
            temp1[j] = sx(temp1[j])
        index = np.random.random_integers(x.shape[0]-1)
        temp2 = (np.transpose(y)[index]-temp1[index])*stepsize
        for m in range(x.shape[0]):
            if m == index:
                a[m] = a[m]-stepsize*0.001*a[m]+temp2
            else:
                a[m] = a[m]-stepsize*0.001*a[m]
    predict = np.dot(k,a)
    for p in range(len(predict)):
        if predict[p] > 0.5:
            predict[p] = 1
        else:
            predict[p] = 0
    result = [predict,a]
    return result   

