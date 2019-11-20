#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%


df = pd.read_csv("assets/dataset.txt", sep = ",", names = ["id", "longtitude", "latitude", "altitude"])


# %%


df.head()


# %%


#Normalizing Data
df = (df-df.mean())/df.std()


# %%


valueArray = df.to_numpy()


# %%


#Splitting dataset using 70-30 cross validation technique
train_size = int(0.7 * valueArray.shape[0])
train_X = valueArray[0:train_size, 1:3]
train_X = np.insert(train_X,0,1,axis=1) #Adding bias
train_Y = valueArray[0:train_size, 3:]
test_X = valueArray[train_size + 1:, 1:3]
test_X = np.insert(test_X,0,1,axis=1) #Adding bias
test_Y = valueArray[train_size + 1:, 3:]


# %%


def gradientDescent(X, Y, alpha, maxIterations, weights):
    """
    This function implements all the batch gradient descent with no regularization.
    At each iteration all the weights are updated with gradient calculated
    over all training points
    """
    costs = []
    iters = []
    trainX = X
    trainY = Y
    m = trainX.shape[0]
    prevCost, currCost = 0.0, 0.0
    for iteration in range(maxIterations):
        gradient = np.sum(trainX * (trainX @ weights.T - trainY), axis=0)
        currCost = np.sum(np.power(((trainX @ weights.T) - trainY),2))/(2 * m)
        weights = weights - (alpha/m) * gradient
        #Plotting loss over train set for every 20 iterations
        if iteration % 20 == 0:
            costs.append(currCost)
            iters.append(iteration + 1)
        #Stopping Criterion
        if iteration != 0 and abs(prevCost - currCost) < 1e-10:
            break
        prevCost = currCost
    print("Optimal Cost is {}".format(currCost))
    print("Converting at {}th iteration with learning rate {}".format(iteration+1,alpha))
    plt.title("Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# %%


def gradientDescentWithL1Regularization(X, Y, alpha, maxIterations, weights, regParameter):
    """
    This function implements the gradient descents with L1 norm regularization.
    At each iteration all the weights are updated with gradient calculated
    over all training points
    """
    costs = []
    iters = []    
    trainX = X
    trainY = Y
    m = trainX.shape[0]
    prevCost, currCost = 0.0, 0.0
    for iteration in range(maxIterations):
        c = (regParameter)/(2 * m)
        gradient = np.sum(trainX * (trainX @ weights.T - trainY), axis=0) + c * np.sign(weights)
        weightList = [weights[0][0], weights[0][1], weights[0][2]]
        weightList = np.asarray(weightList)
        currCost = np.sum(np.power(((trainX @ weights.T) - trainY),2))/(2 * m) + regParameter/(2 * m) * np.linalg.norm(weightList, 1) 
        weights = weights - (alpha/m) * gradient
        #Plotting loss over train set for every 20 iterations
        if iteration%20 == 0:
            costs.append(currCost)
            iters.append(iteration + 1) 
        #Stopping Criterion
        if iteration != 0 and abs(prevCost - currCost) < 1e-10:
            break
        prevCost = currCost
    print("Optimal Cost is {}".format(currCost))
    print("Converting at {}th iteration with learning rate {} and {} regularization parameter".format(iteration+1,alpha,regParameter))
    plt.title("Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# %%


def gradientDescentWithL2Regularization(X, Y, alpha, maxIterations, weights, regParameter):
    """
    This function implements the gradient descents with L2 norm regularization.
    At each iteration all the weights are updated with gradient calculated
    over all training points. 
    """
    costs = []
    iters = []    
    trainX = X
    trainY = Y
    m = trainX.shape[0]
    prevCost, currCost = 0.0, 0.0
    for iteration in range(maxIterations):
        gradient = np.sum(trainX * (trainX @ weights.T - trainY), axis=0)
        weightList = [weights[0][0], weights[0][1], weights[0][2]]
        weightList = np.asarray(weightList)
        currCost = np.sum(np.power(((trainX @ weights.T) - trainY),2))/(2 * m) + regParameter/(2 * m) * np.linalg.norm(weightList, 2) 
        weights = (1 - alpha*regParameter/m)*weights - (alpha/m) * gradient
        #Plotting loss over train set for every 20 iterations
        if iteration%20 == 0:
            costs.append(currCost)
            iters.append(iteration + 1)
        #Stopping Criterion
        if iteration != 0 and abs(prevCost - currCost) < 1e-10:
            break
        prevCost = currCost
    print("Optimal Cost is {}".format(currCost))
    print("Converting at {}th iteration with learning rate {} and {} regularization parameter".format(iteration+1,alpha,regParameter))                             
    plt.title("Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# %%


def stochasticGradientDescent(X, Y, alpha, maxIterations, weights):
    """
    This function implements stochastic gradient descent.
    In each iteration, gradient is calculated w.r.t. each training example
    so weights are updated 'm' times in each iteration.
    """
    costs = []
    iters = []
    shuffledData = np.concatenate((X, Y), axis = 1)
    np.random.shuffle(shuffledData)
    trainX = shuffledData[:, 0 : 3]
    trainY = shuffledData[:, 3 :]
    m = trainX.shape[0]
    prevCost, currCost = 0.0, 0.0
    for iteration in range(maxIterations):
        for i in range(m):
            gradient = trainX[i].dot(weights.T) - trainY[i]
            weights = weights - (alpha/m) * (gradient * trainX[i])
        currCost = np.sum(np.power(((trainX @ weights.T) - trainY),2))/(2 * m)
        if iteration%20 == 0:
            costs.append(currCost)
            iters.append(iteration + 1)    
        #Stopping Criterion
        if iteration != 0 and abs(prevCost - currCost) < 1e-10:
            break
        prevCost = currCost
        
    print("Optimal Cost is {}".format(currCost))
    print("Converting at {}th iteration with learning rate {}".format(iteration+1,alpha))        
    plt.title("Stochastic Gradient Descent")
    plt.xlabel("No Of Iterations")
    plt.ylabel("Cost")
    plt.plot(iters,costs)
    plt.show()
    return weights


# %%


def normalEquations(X, Y):
    """
    This function implements vectorization based linear regression.
    AW = B where A = X^T.X, B = X^T.Y. This is known as Normal Equation
    W = inv(A).B
    """  
    trainX = X
    trainY = Y
    A = trainX.T.dot(trainX)
    B = trainX.T.dot(trainY)
    weights = np.linalg.inv(A).dot(B)
    return weights


# %%


#Training our models


# %%


batch_weights = gradientDescent(train_X, train_Y, 0.01, 2000, np.zeros([1,3]))


# %%


L1_weights = gradientDescentWithL1Regularization(train_X, train_Y, 0.02, 2000, np.zeros([1,3]), 0.7)


# %%


L2_weights = gradientDescentWithL2Regularization(train_X, train_Y, 0.015, 2000, np.zeros([1,3]), 0.8)


# %%


stochastic_weights = stochasticGradientDescent(train_X, train_Y, 0.05, 350, np.zeros([1,3]))


# %%


normal_weights = normalEquations(train_X, train_Y)


# %%


#Testing our models


# %%
