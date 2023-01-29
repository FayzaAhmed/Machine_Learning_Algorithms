#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from os import listdir


# In[2]:


#! git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset


# In[4]:


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dataset_path = './Sign-Language-Digits-Dataset/Dataset'

images = []
outputs = []

for i, label in enumerate(labels):
    datas_path = dataset_path + '/' + label
    for data in listdir(datas_path):
        img = Image.open(datas_path + '/' + data)
        new_img = img.resize((100, 100))
        imgGray = new_img.convert('L') # The L parameter is to convert the image to grayscale
        imgArray = np.array(imgGray)/255.0 
        images.append(imgArray)
        outputs.append(i) # storing the labels of each image

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(images, outputs, test_size = 413, random_state = 42)


# In[5]:


#simple NN architecture with 2 hidden layers with different number of neurons
def first_architecture():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10)
    ])       
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# In[6]:


#change number of hidden layer
def second_architecture():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'), 
    tf.keras.layers.Dense(10)
    ])       
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# In[8]:


def crossValidation(model):
    kfold = KFold(n_splits = 17, shuffle=False)
    
    accuracies = []
    
    for train, validate in kfold.split(train_data_x):
        foldX = []
        foldY = []
        for index in (train):
            foldX.append(train_data_x[index])
            foldY.append(train_data_y[index])

        validateX = []
        validateY = []
        for index in (validate):
            validateX.append(train_data_x[index])
            validateY.append(train_data_y[index])

        tensoredTrainX = []
        for i in range(len(foldX)):
            tensoredTrainX.append(tf.convert_to_tensor(foldX[i]))

        tensoredTrainX = tf.convert_to_tensor(tensoredTrainX, dtype=tf.float32) 
        tensoredTrainY = tf.convert_to_tensor(foldY) 
        
        tensoredValidX = []
        for i in range(len(validateX)):
            tensoredValidX.append(tf.convert_to_tensor(validateX[i]))
            
        tensoredValidX = tf.convert_to_tensor(tensoredValidX, dtype=tf.float32)
        tensoredValidY = tf.convert_to_tensor(validateY) 

        model.fit(tensoredTrainX, tensoredTrainY, epochs=10)
        
        loss, accurcy = model.evaluate(tensoredValidX, tensoredValidY, verbose=2)
        accuracies.append(accurcy)
        
    return accuracies


# In[9]:


def training(model) :
    tensoredTrainX = []
    for i in range(len(train_data_x)):
        tensoredTrainX.append(tf.convert_to_tensor(train_data_x[i]))

    tensoredTrainX = tf.convert_to_tensor(tensoredTrainX, dtype=tf.float32) 
    tensoredTrainY = tf.convert_to_tensor(train_data_y) 
    model.fit(tensoredTrainX, tensoredTrainY, epochs=10)    
   


# In[10]:


def testing(model):
    tensoredTestX = []
    for i in range(len(test_data_x)):
        tensoredTestX.append(tf.convert_to_tensor(test_data_x[i]))
    tensoredTestX = tf.convert_to_tensor(tensoredTestX, dtype=tf.float32)

    tensoredTestY = tf.convert_to_tensor(test_data_y) 

    test_loss, test_acc = model.evaluate(tensoredTestX, tensoredTestY, verbose=2)
    
    y_pred = model.predict(tensoredTestX, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(tensoredTestY, y_pred_bool))


# In[11]:


def runExperiment_nn(model):
    accuracies = crossValidation(model)
    sum_accuracies = sum(accuracies)
    accuracies_avg = sum_accuracies/len(accuracies)
    print("Average Accuracy : ", accuracies_avg * 100)


# In[12]:


model1 = first_architecture()

#runExperiment_nn(model1) 


# In[25]:


training(model1)


# In[26]:


testing(model1)


# In[13]:


model2 = second_architecture()

#runExperiment_nn(model2) 


# In[27]:


training(model2)


# In[28]:


testing(model2)

