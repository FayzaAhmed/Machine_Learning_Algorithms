#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


# In[2]:


car_dataset = pd.read_csv("car_data.csv")
#shuffling
car_dataset = car_dataset.sample(frac=1)

#normalization
numerical_Data_x = car_dataset[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
                              'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']]
numerical_Data_y = car_dataset[['price']]
numerical_Data = car_dataset[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
                              'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']]

numerical_Data_x = numerical_Data_x.apply(lambda x: (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
numerical_Data_y = numerical_Data_y.apply(lambda y: (y- y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0)))
numerical_Data = numerical_Data.apply(lambda z: (z- z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0)))

#--------------------------------------------------------------------------------------------------------------
#scatter plot
#convert the dataframe to an array
arry = np.array(numerical_Data_y[['price']])
fig = plot.figure(figsize=(20,20))
count = 0

for feature in numerical_Data_x:
    arrx = np.array(numerical_Data_x[[feature]])
    count = count + 1
    plot.subplot(6,3, count)
    plot.scatter(arrx, arry, color = 'green')
    plot.ylabel('price')
    plot.xlabel(feature)
   


# In[3]:


#choosing four features with the help of the correlation
print("Choose the best features using correlation function:")
corr_data = numerical_Data.corr()
price_column = corr_data.iloc[: , -1]
print(price_column)   

#curbweight  horsepower  enginesize  carwidth
chosen_features = car_dataset[['carwidth', 'curbweight', 'enginesize', 'horsepower']]
chosen_features = chosen_features.apply(lambda z: (z- z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0)))


# In[4]:


#------------------------------------------------------------------------------------------
#splitting the data (training tsting)
x = chosen_features  # 4 columns
y = numerical_Data['price'].to_numpy().reshape((-1,1)) #one column

y_train = y[0:160]
y_test = y[160:]

x_train = x[0:160]
x_test = x[160:]

#-------------------------------------------------------------------------------------

def cost_function(x_train, y_train, theta):
       J_theta = np.sum((x_train.dot(theta) - y_train) ** 2)/(2 * m)
       return J_theta


def linear_Regression(x_train, y_train, theta, alpha, iterations):
    cost_history = [0] * iterations
    for itr in range(iterations):
        hypothesis = x_train.dot(theta)
        error = hypothesis - y_train
        gradient = x_train.T.dot(error) / m
        theta = theta - alpha * gradient
        cost = cost_function(x_train, y_train, theta)
        cost_history[itr] = cost
         
    return theta, cost_history
 
#perform the training model
alpha = 0.01
iterations = 500
m = len(y_train)

# adding a new column x0 = 1 
ones = np.ones((m,1))
x_train = np.concatenate((ones, x_train), axis=1)

initial_theta = np.ones(5)  
theta = pd.DataFrame(initial_theta)

iterations_array = list(range(1, iterations+1))

theta, cost_history = linear_Regression(x_train, y_train, theta, alpha, iterations)


#ploting the cost against the number of iterations
plot.plot(iterations_array, cost_history, color='purple')
plot.rcParams["figure.figsize"] = (10,6)
plot.grid()
plot.xlabel('Iterations')
plot.ylabel('Cost')

#Testing Phase
m = len(x_test)
ones = np.ones((m,1))
x_test = np.concatenate((ones, x_test), axis=1)
MSE = cost_function(x_test, y_test, theta)
print('Total cost (mean squared error) = ', MSE)


# In[5]:


initial_theta = np.ones(5)  
theta = pd.DataFrame(initial_theta)

alpha = 0.001;
theta_1, cost_history_1 = linear_Regression(x_train, y_train, theta, alpha, iterations)

alpha = 0.003;
theta_2, cost_history_2 = linear_Regression(x_train, y_train, theta, alpha, iterations)

alpha = 0.01;
theta_3, cost_history_3 = linear_Regression(x_train, y_train, theta, alpha, iterations)

alpha = 0.03;
theta_4, cost_history_4 = linear_Regression(x_train, y_train, theta, alpha, iterations)

alpha = 0.1;
theta_5, cost_history_5 = linear_Regression(x_train, y_train, theta, alpha, iterations)

plot.plot(range(1, iterations +1), cost_history_1, color ='red', label = 'alpha = 0.001')
plot.plot(range(1, iterations +1), cost_history_2, color ='green', label = 'alpha = 0.003')
plot.plot(range(1, iterations +1), cost_history_3, color ='purple', label = 'alpha = 0.01')
plot.plot(range(1, iterations +1), cost_history_4, color ='orange', label = 'alpha = 0.03')

plot.rcParams["figure.figsize"] = (10,6)
plot.grid()
plot.xlabel('Iterations')
plot.ylabel('Cost J(theta)')
plot.legend()


# In[ ]:




