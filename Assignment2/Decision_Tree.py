#!/usr/bin/env python
# coding: utf-8

# In[10]:


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[11]:


#Loading data
bankNote_data = pd.read_csv("BankNote_Authentication.csv") 

#shuffling 
bankNote_data = bankNote_data.sample(frac = 1)

def splitDataTo_Train_Test(data, trainRatio):
    train = data.sample(frac = trainRatio)
    test = data.drop(train.index)   #drops the training rows using their indices
    x_train = train.iloc[:, 0:4]
    y_train = train.iloc[:, -1]
    x_test = test.iloc[:, 0:4]
    y_test = test.iloc[:, -1]
    return x_train, y_train , x_test , y_test


def calc_accuracy(y_predict, y_test):
    y_predict2 = np.array(y_predict)
    y_test2 = np.array(y_test)
    sum = 0
    for i in range(len(y_test2)):
        if y_predict2[i] == y_test2[i]:
            sum += 1
    return sum / len(y_predict2)


def experiment(data, ratio, iterations):
    accuracies = []
    depths = []  #the depth of the tree
    counts = []  #number of nodes in the tree
    for num in range(0, iterations):
        x_train, y_train, x_test, y_test = splitDataTo_Train_Test(data, ratio)

        tree = DecisionTreeClassifier(criterion = "entropy")
        tree.fit(x_train, y_train) #training
        
        depth = tree.tree_.max_depth
        depths.append(depth)
        
        count = tree.tree_.node_count
        counts.append(count)
        
        y_predict = tree.predict(x_test)
        accuracy = calc_accuracy(y_predict, y_test) * 100
        accuracies.append(accuracy)
     
    return accuracies, depths , counts


def measurements(data, ratio, iterations, printStat):
    accuracies, depths, counts = experiment(data, ratio, iterations)
    print("The experiment for ", iterations , " times at ratio:", ratio)  
    
    for num in range(0, iterations):
        print(num + 1 , "-> Tree Size:", depths[num] , " With Accuracy:", accuracies[num])

    if(printStat):    
        print("Min Accuracy: ", np.min(accuracies), "\nMax Accuracy: ", np.max(accuracies), "\nMean Accuracy: " , np.mean(accuracies))
        print("Min Tree size: ", np.min(depths), "\nMax Tree size: ", np.max(depths), "\nMean Tree size: " , np.mean(depths), "\n")
    print("\n")
    
    
measurements(bankNote_data, 0.25, 5, 0)
measurements(bankNote_data, 0.30, 5, 1)
measurements(bankNote_data, 0.40, 5, 1) 
measurements(bankNote_data, 0.50, 5, 1)
measurements(bankNote_data, 0.60, 5, 1)
measurements(bankNote_data, 0.70, 5, 1)


# In[12]:


ratioArray = [30, 40, 50, 60, 70]
accuracyArray = []
nodesArray = []
sizeArray = []

for ratio in ratioArray:
    accuracy, size, count = experiment(bankNote_data, ratio / 100, 1)
    accuracyArray.append(accuracy)
    sizeArray.append(size)
    nodesArray.append(count)


# In[13]:


plot.plot(ratioArray, accuracyArray, color='purple')
plot.rcParams["figure.figsize"] = (10,6)
plot.grid()
plot.xlabel('Training Set Size')
plot.ylabel('Accuracy')
plot.ylim(95, 100)


# In[14]:


plot.plot(ratioArray, nodesArray, color='red')
plot.rcParams["figure.figsize"] = (10,6)
plot.grid()
plot.xlabel('Training Set Size')
plot.ylabel('Number of nodes')


# In[ ]:





# In[ ]:




