import math
import numpy as np
import scipy
from scipy import io
import random
from collections import defaultdict
import collections

# Decision tree

# Function calculating entropy for given dataset which has the shape of (label,feature)
def entropy(data):
    freqs = np.unique(data[:,0], return_counts=True)[1]
    probs = freqs / float(len(data))
    entropy = -np.dot(probs,np.log2(probs))
    return entropy

# Function calculating information gain 
# Return the info gain by selecting ith attribute for the splitting
def infogain(data, i):
    sub_entropy = 0
    attr = np.array(data[:,i])
    freqs = np.unique(attr, return_counts=True)[1]
    values= np.unique(attr, return_counts=True)[0]
    probs = freqs / float(len(data))
    for i in range(len(values)):
        val = values[i]
        index = np.where(attr==val)
        datasubset = data[index]
        sub_entropy += probs[i] * entropy(datasubset)
    return (entropy(data) - sub_entropy)

# Function finding the best split index based on information gain
# Return type array
def best_index(data, remaining_indexes):
    infogain_list = []
    for i in remaining_indexes:
        infogain_list.append(infogain(data,i))
    infogain_list = np.array(infogain_list)
    remaining_indexes = np.array(remaining_indexes)
    best = remaining_indexes[np.where(infogain_list==max(infogain_list))][0]
    return best

# Function for splitting
# Return a list of dictionaries

def split(data, index):
    partitions = defaultdict(list)
    for temp in data:
        partitions[temp[index]].append(temp)
    return partitions
    
# Function for returning the most frequent class index in instances
def majority(data):
    freqs = np.unique(data[:,0], return_counts=True)[1]
    val = np.unique(data[:,0], return_counts=True)[0]
    index = np.where(freqs == max(freqs))
    return val[index]


# Function for building the decision tree
def build_tree(data, remaining_indexes=None,default_class=None):
    
    if remaining_indexes is None:
        remaining_indexes = [i for i in range(data.shape[1]) if i != 0]
    
    class_labels_and_counts = collections.Counter(data[:,0])
    
    if not remaining_indexes:
        return default_class
    
    elif len(class_labels_and_counts) == 1:
        class_label = class_labels_and_counts.most_common(1)[0][0]
        return class_label
    
    else:
        default_class = majority(data)
        optimal_index = best_index(data, remaining_indexes) 
        tree = {optimal_index:{}}
        partitions = split(data, optimal_index)
        updated_remaining_indexes = [i for i in remaining_indexes if i != optimal_index]
        for attribute_value in partitions:
            subtree = build_tree(np.array(partitions[attribute_value]),updated_remaining_indexes,default_class)
            tree[optimal_index][attribute_value] = subtree
    return tree

# Function for classification based on built tree  
# Return classification results, e.g. labels  
def classify(tree, data,default_class=0):
    if not isinstance(tree, dict):  
        return tree
    indexes = list(tree.keys())[0] 
    values = list(tree.values())[0]
    data_value = data[indexes]
    if data_value not in values:  
        return default_class
    return classify(values[data_value], data, default_class=0)


# Random forest 
# m is the number of randomly sampled labels for each decision tree to built on
def random_forest(data,m,index_list,default_class=None):
    
    random_indexes = np.random.choice(index_list,m)
    random_indexes = random_indexes.tolist()
    
    class_labels_and_counts = collections.Counter(data[:,0])
    
    if len(class_labels_and_counts) == 1:
        class_label = class_labels_and_counts.most_common(1)[0][0]
        return class_label
    
    else:
        default_class = majority(data)
        optimal_index = best_index(data, random_indexes) 
        best_info_gain = infogain(data,optimal_index)
        if best_info_gain == 0:
            return default_class
        else:
            tree = {optimal_index:{}}
            partitions = split(data, optimal_index)
            for attribute_value in partitions:
                subtree = random_forest(np.array(partitions[attribute_value]),m,index_list,default_class)
                tree[optimal_index][attribute_value] = subtree
    return tree

# Random forest classification
# m is the number of randomly sampled labels for each decision tree to built on
# n is the number of decision trees in the random forest
def random_forest_classify(training_data,m,index_list,test_data,n):
    prediction = []
    for i in range(n):
        tree = random_forest(training_data,m,index_list,default_class=None)
        predict_val = []
        for instance in test_data:
            predicted_label = classify(tree, instance)
            predict_val.append(predicted_label)
    
        for i in range(len(predict_val)):
            if hasattr(predict_val[i], "__len__"):
                predict_val[i] = predict_val[i][0]
        
        predict_val = np.array(predict_val)
        prediction.append(predict_val)
    
    prediction = np.array(prediction)
    vote = []
    for i in range(prediction.shape[1]):
        freqs = np.unique(prediction[:,i], return_counts=True)[1]
        val = np.unique(prediction[:,i], return_counts=True)[0]
        index = np.where(freqs==max(freqs))
        if len(index)>1:
            index = 0
        vote.append(val[index])
    
    for i in range(len(vote)):
        if hasattr(vote[i], "__len__"):
            vote[i] = vote[i][0]
    return vote


