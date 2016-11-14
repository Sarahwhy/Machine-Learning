# Latent Factor Model
# Updated_joke_train is the training dataset

from scipy import linalg
from numpy import random
from numpy import linalg
from numpy.linalg import inv
from numpy import matlib
import sys

# SVD for training matrix with all missing values replaced by zero 
U, s, Vh = linalg.svd(updated_joke_train)

def mean_squared_error(pred,actual):
    mse = 0
    for i in range(actual.shape[0]):
        for j in range(actual.shape[1]):
            if not math.isnan(actual[i,j]):
                mse = mse+(pred[i,j]-actual[i,j])**2
    return mse


def optim_lossfn(d,train,para,max_iter):
    n = train.shape[0]
    m = train.shape[1]
    U = 0.01*random.randn(d,n)
    V = 0.01*random.randn(d,m)
    for iteration in range(max_iter):
        for i in range(n):
            temp1 = np.zeros((1, d))
            temp2 = para*np.matlib.identity(d)
            for j in range(m):
                if not math.isnan(train[i,j]):
                    temp1 = temp1+train[i,j]*np.transpose(V[:,j])
                    temp2 = temp2+np.dot(V[:,j].reshape(d,1),np.transpose(V[:,j]).reshape(1,d))
            U[:,i] = np.transpose(np.dot(temp1,inv(temp2))).reshape(d)
        for j in range(m):
            temp1 = np.zeros((1, d))
            temp2 = para*np.matlib.identity(d)
            for i in range(n):
                if not math.isnan(train[i,j]):
                    temp1 = temp1+train[i,j]*np.transpose(U[:,i])
                    temp2 = temp2+np.dot(U[:,i].reshape(d,1),np.transpose(U[:,i]).reshape(1,d))
            V[:,j] = np.transpose(np.dot(temp1,inv(temp2))).reshape(d)
    result = [U,V]
    return result