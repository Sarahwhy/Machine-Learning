# K-means Clustering for hand-written digit classification
import numpy as np
from numpy import random
import scipy
from scipy import io
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt

mnist_data = scipy.io.loadmat('data/mnist_data/images.mat')
images = mnist_data["images"]
feature = images.reshape(784, 60000).transpose()
#Normalize the features
scaler = preprocessing.StandardScaler().fit(feature)
stand_feature = scaler.transform(feature)
   
# Initialize centroids
def InitializeCentroids(data,k):
    ini_centroid = []
    index = np.array(range(len(data)))
    np.random.shuffle(index)
    ini_centroid = data[index[0:k],].tolist()
    return ini_centroid

# Get Euclidean distance for two points
def distance(x,y):
    dist = np.sqrt(sum((x - y) ** 2))
    return dist
    
#Get clusters based on given centroids 
def GetClusters(data, centroids,k):
    clusters = [[] for i in range(k)]
    for instance in data:
        dist_list = []
        for cluster in centroids:
            dist_list.append(distance(instance,cluster))
        dist_list = np.array(dist_list)
        index = np.where(dist_list==min(dist_list))[0]
        clusters[index].append(instance)
    return clusters

def GetCentroids(clusters):
    centroid = []
    for cluster in clusters:
        centroid.append(np.mean(cluster, axis=0).tolist())
    return centroid

def stop(centroids, old_centroids, iterations):
    if iterations > 100:
        return True
    return centroids==old_centroids
    
def kmeansLoss(centroids,clusters):
    loss = 0
    for i in range(len(centroids)):
        loss = loss+sum(sum((np.array(clusters[i])-np.array(centroids[i]))**2))
    return loss

def kmeans(data,k):
    centroids = InitializeCentroids(data,k) 
    old_centroids = [[] for i in range(k)] 
    iterations = 0
    while not (stop(centroids, old_centroids, iterations)):
        iterations = iterations+1
        clusters = GetClusters(data, centroids,k)
        old_centroids = centroids
        centroids = GetCentroids(clusters)
    loss = kmeansLoss(centroids,clusters)
    result = [centroids,loss]
    return result
    

