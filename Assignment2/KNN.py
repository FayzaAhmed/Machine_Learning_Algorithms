#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.stats import mode
import statistics


# In[17]:


#Loading data
bankNote_data = pd.read_csv("BankNote_Authentication.csv") 

#shuffling 
bankNote_data = bankNote_data.sample(frac = 1)

#splitting the data vertically
features = bankNote_data.iloc[:, 0:4]
output = bankNote_data.iloc[:, -1]

# 70%
x_train = features[0:960]
y_train = output[0:960]

# 30%
x_test = features[960:]
y_test = output[960:]


#normalization
def mean_normalization(features):
    means = x_train.mean()
    stds = x_train.std()
    features = (features - means) / stds
    
mean_normalization(features)    
    
    
def eucledian_distance(row1, row2):
    distance = np.sqrt(np.sum((row1-row2)**2))
    return distance


def calc_accuracy(y_predict, y_test):
    y_predict2 = np.array(y_predict)
    y_test2 = np.array(y_test)
    sum = 0
    for i in range(len(y_test2)):
        if y_predict2[i] == y_test2[i]:
            sum += 1
    return sum / len(y_predict2) * 100 , sum


#training
def predict(x_train, y_train, x_test, k):
    y_predict = []
     
    for row in range(len(x_test)): 
        test_sample_distances = []
         
        for j in range(len(x_train)): 
            distance = eucledian_distance(x_train.iloc[j].values , x_test.iloc[row].values) 
            test_sample_distances.append(distance) 
        
        sorted_distances = np.sort(test_sample_distances) 
        length = len(sorted_distances) 
        
        #HANDLING THE TIE CASE 
        while(k < length) and (sorted_distances[k-1] == sorted_distances[k]):
            k+=1
       
        first_k_indices = np.argsort(test_sample_distances)[:k]     
        labels = y_train.iloc[first_k_indices]
        
        #the mode function always takes the first frequent number which is the nearest point to the target node
        most_frequent = statistics.mode(labels)
        y_predict.append(most_frequent)
        
    return y_predict



def printing(k):
    y_predict = predict(x_train, y_train, x_test, k)
    print("k value : ", k)
    accuracy, correct = calc_accuracy(y_test, y_predict)
    print("Number of correctly classified instances: ", correct , "total number of instances " , len(y_test))
    print("Accuracy : ", accuracy)
    print("\n")
    
    
for i in range (1,10):    
    printing(i)
    
    
printing(9)
printing(90)
printing(140)
printing(180)


# In[ ]:




