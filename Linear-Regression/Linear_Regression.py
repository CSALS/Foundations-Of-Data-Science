#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# <center><h1>LINEAR REGRESSION</h1></center>

# ![](assets/linear-regression.png)

# A regression problem is where the output is a real value like __stock price__, __weights__.
# 
# In linear regression we assume the __output__ is related to the input variables (which are called __features__) in linear fashion.

# Hypothesis (the value we predict) for a input vector X is given as ![](assets/hypothesis.png) when there are _2 features_ i.e. vector X = [x1 x2]

# [w0 w1 w2] is called weight vector which we need to find out so that cost is minimal

# What is cost?
# __Cost function__ is the average sum of absolute squares of the predicted value and actual value for all training input vectors

# ![](assets/cost-function.png)

# We can take partial derivates w.r.t w0, w1, w2 and equate them to 0 and solve the three simultaneous equations to find out the weight values for at which cost function is minimum. But in real world scenarios, there are lot of features so solving those many equations is impractical.

# So the approaches we are going to take :-
# 1. Gradient Descent
# 2. Stochastic Gradient Descent
# 3. Gradient Descent along with L1 and L2 regularizations
# 4. Normal Equations

# The problem statement :-
# 
# <h3>Predict altitude values given longtitude and latitude values</h3>

# The dataset we are going to use is [link](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland%2C+Denmark))

# We are going to implement all those 4 methods from scratch in python without using any external APIs like scikit-learn.
# 
# However we are going to use
# * pandas library for data reading purposes
# * numpy library for all data manipulation purposes
# * matplotlib library to plot the results

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("assets/dataset.txt", sep = ",", names = ["id", "longtitude", "latitude", "altitude"])


# In[3]:


df.head()


# In[4]:


df = (df-df.mean())/df.std()


# In[5]:


valueArray = df.to_numpy()


# In[6]:


#Splitting dataset using 70-30 cross validation technique
train_size = int(0.7 * valueArray.shape[0])
train_X = valueArray[0:train_size, 1:3]
train_X = np.insert(train_X,0,1,axis=1) #Adding bias
train_Y = valueArray[0:train_size, 3:]
test_X = valueArray[train_size + 1:, 1:3]
test_X = np.insert(test_X,0,1,axis=1) #Adding bias
test_Y = valueArray[train_size + 1:, 3:]


# In[7]:


def costFunction(X, Y, weights, norm, regParameter):
    m = X.shape[0]
    weightList = [weights[0][0], weights[0][1], weights[0][2]]
    weightList = np.asarray(weightList)
    if norm == "L1 norm":
        return np.sum(np.power(((X @ weights.T) - Y),2))/(2 * m) + regParameter/(2 * m) * np.linalg.norm(weightList, 1) 
    elif norm == "L2 norm":
        return np.sum(np.power(((X @ weights.T) - Y),2))/(2 * m) + regParameter/(2 * m) * np.linalg.norm(weightList, 2) 
    else:
        return np.sum(np.power(((X @ weights.T) - Y),2))/(2 * m)


# In[8]:


def costFunctionDerivative(X, Y, weights, norm, regParameter):
    if norm == "L1 norm":
        c = (regParameter)/(2 * X.shape[0])
        return np.sum(X * (X @ weights.T - Y), axis=0) + c * np.sign(weights)
    elif norm == "L2 norm":
        return np.sum(X * (X @ weights.T - Y), axis=0)
    else:
        return np.sum(X * (X @ weights.T - Y), axis=0)


# In[35]:


def gradientDescent(X, Y, alpha, maxIterations, weights):
    """
    This function implements all the three gradient descents with no regularization.
    At each iteration all the weights are updated with gradient calculated
    over all training points
    """
#     costs = np.zeros(maxIterations)
#     iters = np.zeros(maxIterations)
    costs = []
    iters = []
    trainX = X
    trainY = Y
    m = trainX.shape[0]
    for iteration in range(maxIterations):
        weights = weights - (alpha/m) * costFunctionDerivative(trainX, trainY, weights, "", 0)
        if iteration%20 == 0:
            costs.append(costFunction(trainX, trainY, weights, "", 0))
            iters.append(iteration + 1)
#         costs[iteration] = costFunction(trainX, trainY, weights, "", 0)
#         iters[iteration] = iteration + 1
    plt.title("Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# In[10]:


def gradientDescentWithRegularization(X, Y, alpha, maxIterations, weights, regParameter, norm):
    """
    This function implements the gradient descents with  L1 & L2 norm regularization.
    At each iteration all the weights are updated with gradient calculated
    over all training points
    """
#     costs = np.zeros(maxIterations)
#     iters = np.zeros(maxIterations)
    costs = []
    iters = []    
    trainX = X
    trainY = Y
    m = trainX.shape[0]
    for iteration in range(maxIterations):
        if norm == "L1 norm":
            weights = weights - (alpha/m) * costFunctionDerivative(trainX, trainY, weights, "L1 norm", regParameter)
        elif norm == "L2 norm":
            weights = (1 - alpha*regParameter/m)*weights - (alpha/m) * costFunctionDerivative(trainX, trainY, weights, "L2 norm", 0)
        
        if iteration%20 == 0:
            costs.append(costFunction(trainX, trainY, weights, "", 0))
            iters.append(iteration + 1)        
#         costs[iteration] = costFunction(trainX, trainY, weights, norm, regParameter)
#         iters[iteration] = iteration + 1
    plt.title("Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# In[19]:


def stochasticGradientDescent(X, Y, alpha, maxIterations, weights):
    """
    This function implements stochastic gradient descent.
    In each iteration, gradient is calculated w.r.t. each training example
    so weights are updated 'm' times in each iteration.
    """
#     costs = np.zeros(maxIterations)
#     iters = np.zeros(maxIterations)
    costs = []
    iters = []
    shuffledData = np.concatenate((X, Y), axis = 1)
    np.random.shuffle(shuffledData)
    trainX = shuffledData[:, 0 : 3]
    trainY = shuffledData[:, 3 :]
    m = trainX.shape[0]
    for iteration in range(maxIterations):
        for i in range(m):
            delta = trainX[i] @ weights.T - trainY[i]
            weights = weights - (alpha/m) * (delta * trainX[i])
        if iteration%20 == 0:
            costs.append(costFunction(trainX, trainY, weights, "", 0))
            iters.append(iteration + 1)            
#         costs[iteration] = costFunction(trainX, trainY, weights, "", 0)
#         iters[iteration] = iteration + 1   
    plt.title("Stochastic Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# In[12]:


def normalEquations(X, Y):
    """
    This function implements vectorization based linear regression.
    W = (X^T X)^-1 X^T Y - This is known as Normal Equation
    """  
    trainX = X
    trainY = Y
    weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(trainX.transpose(),trainX)),trainX.transpose()),trainY)
    return weights


# In[22]:


#Training our models


# In[33]:


batch_weights = gradientDescent(train_X, train_Y, 0.01, 1000, np.zeros([1,3]))


# In[24]:


L1_weights = gradientDescentWithRegularization(train_X, train_Y, 0.01, 1000, np.zeros([1,3]), 0.5, "L1 norm")


# In[25]:


L2_weights = gradientDescentWithRegularization(train_X, train_Y, 0.01, 1000, np.zeros([1,3]), 0.5, "L2 norm")


# In[26]:


stochastic_weights = stochasticGradientDescent(train_X, train_Y, 0.05, 200, np.zeros([1,3]))


# In[27]:


normal_weights = normalEquations(train_X, train_Y)


# In[28]:


#Testing our models


# In[ ]:




