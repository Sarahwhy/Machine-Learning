import numpy as np
from numpy import random
import scipy
from scipy import io
from sklearn import preprocessing
import math

# Hand-written digit dataset
train_data = scipy.io.loadmat('dataset/train.mat')
train_labels = train_data["train_labels"]
train_images = train_data["train_images"]
train_feature = train_images.reshape(784, 60000).transpose()

#Vectorize the labels
enc = preprocessing.OneHotEncoder()
enc.fit(train_labels)
vec_labels = enc.transform(train_labels).toarray()

#Normalize the features
scaler = preprocessing.StandardScaler().fit(train_feature)
stand_train_feature = scaler.transform(train_feature)   

# Neural Network
# logistic function
def s(x):
    for i in range(len(x)):
        x[i] = 1/(1+math.exp(-x[i]))
    return x

# Function for training two-layer neural network with tannh as activation function
# Need to modify the matrix size for other datasets
# n is the number of iterations
# alpha is the learning rate
# Use Mean Square Error 
def trainNeuralNetwork(features, labels, alpha,n):
    v = 0.1*random.randn(200,785)
    w = 0.1*random.randn(10,201)
    newrow = np.ones(785)
    v2 = np.vstack((v,newrow))
    for iter in range(n):
        index = random.randint(0,features.shape[0])
        x = features[index,:]
        y = labels[index,:]
        y = np.matrix(y)
        x2 = np.append(x,1) # append 1 as error term
        h = np.tanh(np.dot(v,x2.transpose()))
        h2 = np.matrix(np.append(h,1)) 
        z = s(np.dot(w,np.transpose(h2)))
        for k in range(10):
            w[k,:] = w[k,:]-alpha*(z[k,:]-y[:,k])*z[k,:]*(1-z[k,:])*h2
        for k in range(200):
            temp = np.dot(np.dot((z-y.transpose()),z.transpose()),(np.ones(10).reshape(10,1)-z))
            v[k,:] = v[k,:]-alpha*np.dot(temp.transpose(),w[:,k])*(1-h[k]**2)*x2.reshape(1,785)
    result = [v,w]
    return result

# Function for prediction based on trained NN hidden layer weights    
def predictNeuralNetwork(weights, images):
    v = weights[0]
    w = weights[1]
    predict = []
    for i in range(len(images)):
        x2 = np.append(images[i],1)
        h = np.tanh(np.dot(v,x2.transpose()))
        h2 = np.matrix(np.append(h,1))
        z = s(np.dot(w,np.transpose(h2)))
        max_z = max(z)
        for i in range(10):
            if z[i] == max_z:
                label = i
        predict.append(label)
    return predict

# Function for training two-layer neural network with tannh as activation function
# Need to modify the matrix size for other datasets
# n is the number of iterations
# alpha is the learning rate
# Use Cross Entropy Error

def trainNeuralNetwork2(features, labels, alpha,n):
    v = 0.1*random.randn(200,785)
    w = 0.1*random.randn(10,201)
    newrow = np.ones(785)
    v2 = np.vstack((v,newrow))
    for iter in range(n):
        index = random.randint(0,features.shape[0])
        x = features[index,:]
        y = labels[index,:]
        y = np.matrix(y)#shape of (1,10)
        x2 = np.append(x,1) # append 1 as error term
        h = np.tanh(np.dot(v,x2.transpose()))
        h2 = np.matrix(np.append(h,1)) # shape of (1,201)
        z = s(np.dot(w,np.transpose(h2)))# shape of (10,1)        
        for k in range(10):
            w[k,:] = w[k,:]-alpha*(z[k,:]-y[:,k])*h2
        for k in range(200):
            temp = np.dot(np.dot((z-y.transpose()),z.transpose()),(np.ones(10).reshape(10,1)-z))
            v[k,:] = v[k,:]-alpha*np.dot(temp.transpose(),w[:,k])*(1-h[k]**2)*x2.reshape(1,785)
    result=[v,w]
    return result

    
