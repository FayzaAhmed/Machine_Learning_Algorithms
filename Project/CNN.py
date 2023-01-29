#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from PIL import Image
from os import listdir

import tensorflow as tf
import numpy as np


# In[2]:


#! git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset


# In[3]:


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
dataset_path = './Sign-Language-Digits-Dataset/Dataset'

images = []
outputs = []

for i, label in enumerate(labels):
    datas_path = dataset_path + '/' + label
    for data in listdir(datas_path):
        img = Image.open(datas_path + '/' + data)
        new_img = img.resize((100, 100))        
        imgArray = np.array(new_img)    
        images.append(imgArray)
        outputs.append(i) # storing the labels of each image

averageImage = np.mean(images, axis = 0)
images = (images - averageImage) / 255.0

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(images, outputs, test_size = 413, random_state = 42)


# In[4]:


model_1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(200, 3, activation = 'relu', input_shape = (100,100,3)),
            tf.keras.layers.Conv2D(150, 3, activation = 'relu'),
            tf.keras.layers.MaxPool2D(4,4),
            tf.keras.layers.Conv2D(120, 3, activation = 'relu'),
            tf.keras.layers.Conv2D(80,  3, activation = 'relu'),
            tf.keras.layers.MaxPool2D(4,4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation = 'relu'),
            tf.keras.layers.Dropout(rate = 0.5),
            tf.keras.layers.Dense(10)
        ]) 
model_1.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


# In[5]:


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


# In[6]:


def training(model) :
    tensoredTrainX = []
    for i in range(len(train_data_x)):
        tensoredTrainX.append(tf.convert_to_tensor(train_data_x[i]))

    tensoredTrainX = tf.convert_to_tensor(tensoredTrainX, dtype=tf.float32) 
    tensoredTrainY = tf.convert_to_tensor(train_data_y) 
    model.fit(tensoredTrainX, tensoredTrainY, epochs=10)    


# In[7]:


def testing(model):
    tensoredTestX = []
    for i in range(len(test_data_x)):
        tensoredTestX.append(tf.convert_to_tensor(test_data_x[i]))

    tensoredTestX = tf.convert_to_tensor(tensoredTestX, dtype=tf.float32)
    tensoredTestY = tf.convert_to_tensor(test_data_y) 
    
    y_pred = model.predict(tensoredTestX, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(tensoredTestY, y_pred_bool))


# In[8]:


def runExperiment_cnn(model):
  accuracies = crossValidation(model)
  sum_accuracies = sum(accuracies)
  accuracies_avg = sum_accuracies/len(accuracies)
  print("Average Accuracy : ", accuracies_avg * 100)

#runExperiment_cnn(model_1)


# In[10]:


training(model_1)


# In[11]:


testing(model_1)


# # SVM

# In[9]:


train_data_x = np.array(train_data_x).reshape(1649,-1)

def crossValidationSVM(model):
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
          
      model.fit(foldX, foldY)

      y = model.predict(validateX)
      accuracies.append(accuracy_score(validateY,y))

  return accuracies


# In[10]:


model_2 = SVC() 

def runExperiment_svm(model): 
  accuracies = crossValidationSVM(model)
  sum_accuracies = sum (accuracies)
  accuracies_avg = sum_accuracies/len(accuracies)
  print("Average Accuracy : ", accuracies_avg * 100)

#runExperiment_svm(model_2)


# In[14]:


test_data_x = np.array(test_data_x).reshape(413, -1)

# training svm 
model_2.fit(train_data_x, train_data_y) 


# In[18]:


# testing svm
y = model_2.predict(test_data_x)
print(classification_report(test_data_y, y))

