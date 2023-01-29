#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


# In[2]:


#loading the data
customer_dataset = pd.read_csv("customer_data.csv")


#shuffling
customer_dataset = customer_dataset.sample(frac=1)

#normalization
data_x = customer_dataset[['age', 'salary']]
normalized_x = data_x.apply(lambda x: (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
purchased = customer_dataset[['purchased']]


#splitting the data (training testing)
y_train = purchased[0:320]
y_test = purchased[320:]

x_train = normalized_x[0:320]
x_test = normalized_x[320:]


# In[3]:


def sigmoid_function(x_train, theta):
        z = np.dot(x_train, theta)
        return 1 / (1 + np.exp(-z))
  
def gradient_decent(x_train, y_train, sigma):
    m = y_train.shape[0]
    return np.dot(x_train.T, (sigma - y_train)) / m


def logistic_regression(x_train, y_train, theta, alpha, iterations):
    for itr in range(iterations):
        sigma = sigmoid_function(x_train, theta)
        gradient = gradient_decent(x_train, y_train, sigma)
        theta = theta - alpha * gradient     
    return theta


def prediction(x_test, threshold, theta):
    ones = np.ones((len(x_test),1))
    x_test = np.concatenate((ones, x_test), axis=1)
    sigmoid_x = sigmoid_function(x_test, theta)
    #sigmoid_x will equal true if its value greater than or equale threshold
    sigmoid_x = sigmoid_x >= threshold
    y_predict = np.zeros(len(x_test)) #intialization with zeros
    
    for i in range(len(y_predict)):
        if sigmoid_x[i] == True:
            y_predict[i] = 1
            
    return y_predict

def calc_accuracy(y_predict, y_test):
    y_predict2 = np.array(y_predict)
    y_test2 = np.array(y_test)
    sum = 0
    for i in range(len(y_test2)):
        if y_predict2[i] == y_test2[i]:
            sum += 1
    return sum / len(y_predict2)


#perform training model 
alpha = 0.01
iterations = 1000
m = len(y_train)

# adding a new column x0 = 1 
ones = np.ones((m,1))
x_train = np.concatenate((ones, x_train), axis=1)

initial_theta = np.ones(3)  
theta = pd.DataFrame(initial_theta)

theta = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta)
accuracy = calc_accuracy(y_predict, y_test)
print("Accuracy = ", accuracy)


# In[4]:


alpha = 0.01;
theta_1 = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta_1)
accuracy1 = calc_accuracy(y_predict, y_test)
print("Learning rate = ", alpha, "\t Accuracy = ", accuracy1, "\n")


alpha = 0.03;
theta_2 = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta_2)
accuracy2 = calc_accuracy(y_predict, y_test)
print("Learning rate = ", alpha, "\t Accuracy = ", accuracy2, "\n")


alpha = 0.1;
theta_3 = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta_3)
accuracy3 = calc_accuracy(y_predict, y_test)
print("Learning rate = ", alpha, "\t Accuracy = ", accuracy3, "\n")

alpha = 0.3;
theta_4 = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta_4)
accuracy4 = calc_accuracy(y_predict, y_test)
print("Learning rate = ", alpha, "\t Accuracy = ", accuracy4, "\n")

alpha = 1;
theta_5 = logistic_regression(x_train, y_train, theta, alpha, iterations)
y_predict = prediction(x_test, 0.5, theta_5)
accuracy5 = calc_accuracy(y_predict, y_test)
print("Learning rate = ", alpha, "\t Accuracy = ", accuracy5, "\n")

