# %%
# %matplotlib inline
import numpy as np
import pandas as pd
from numpy import random as rd
import matplotlib.pyplot as pltl
import math
pltl.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D
def sign(a):
    return (a > 0) - (a < 0)
# %%
DATA = pd.read_csv("dataset.csv")
DATA_matrix = np.matrix(DATA.loc[ '0' : '434874', 'Longitude': 'Latitude'])
T = np.matrix(DATA.loc[ '0' : '434874', 'Altitude'])
ones = np.ones([DATA_matrix.shape[0],1])
DATA_matrix= np.concatenate((ones,DATA_matrix),axis=1)
w  = np.zeros([1,3])
#print(DATA_matrix)

# degree 2
sq_x=[]
sq_x=np.square(DATA_matrix[:,1:])
#print(sq_x)

# %%
DATA_matrix= np.concatenate((DATA_matrix,sq_x),axis=1)

# %%
#print(DATA_matrix)

# %%
mul=[]
mul=np.multiply(DATA_matrix[:,1:2],DATA_matrix[:,2:3])

DATA_matrix= np.concatenate((DATA_matrix,mul),axis=1)
#print(DATA_matrix)

# %%
w=np.matrix([0.,0.,0.,0.,0.,0.])
#print(w)

# %%
DATA_matrix_t = np.transpose(DATA_matrix)
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
def gradientDescent(DATA_matrix,DATA_matrix_t,w,T,iters,alpha,noc):
    #B1=np.zeros(iters)
    for k in range(0,300000):  
        b1 = np.dot((DATA_matrix[k,:]), w) - T[0,k]
        #B1[k]=(b1*b1)
        for i in range(noc):
            w[i,0] = w[i,0]- alpha * np.dot(DATA_matrix_t[i,k], b1.T)
            
            #w[i,0] = w[i,0]- alpha * np.dot(DATA_matrix_t[0,k], b1.T)
        #w[1,0] = w[1,0] - alpha * np.dot(DATA_matrix_t[1,k], b1.T)
        '''w[2,0] = w[2,0] - alpha * np.dot(DATA_matrix_t[2,k], b1.T)
        w[3,0] = w[3,0] - alpha * np.dot(DATA_matrix_t[3,k], b1.T)
        w[4,0] = w[4,0] - alpha * np.dot(DATA_matrix_t[4,k], b1.T)
        w[5,0] = w[5,0] - alpha * np.dot(DATA_matrix_t[5,k], b1.T)'''
        if(np.abs(b1) < 0.00001):
            break 
        return w
         
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2    
def rmse(Y, Y_pred):
    print(1)
    X= Y - Y_pred
    #print(X)
    for i in range(np.size(X, 0)):
        X[i]= X[i]**2
    #print(X)
    X = X.mean()
    #print(X)
    X = math.sqrt(X)
    #print(X)
    return X    

# %%

k=[]
k=np.array(DATA_matrix)
noc=len(k[0])
print(noc)
ww=gradientDescent(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-7,noc)
print("2nd degree: ",ww)
tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
#print(tr_data)
#ones = np.ones([tr_data.shape[0],1])
#tr_data= np.concatenate((ones,tr_data),axis=1)
#print(DATA_matrix)
y_pred=tr_data.dot(ww)
#print(y_pred)
y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
#y=np.array(y)
t=rmse(y.T, y_pred)
print("rmse"  ,t)
y=np.array(y)
y_pred=np.array(y_pred)
k=r2_score(y.T,y_pred)
print("r2:",k)
w=np.matrix([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# degree 3
cub_x=[]
cub_x=np.multiply(DATA_matrix[:,3:5],DATA_matrix[:,1:3])
DATA_matrix= np.concatenate((DATA_matrix,cub_x),axis=1)
#print(DATA_matrix)
mul_1=np.multiply(DATA_matrix[:,1:2],DATA_matrix[:,5:6])
DATA_matrix= np.concatenate((DATA_matrix,mul_1),axis=1)
mul_2=np.multiply(DATA_matrix[:,2:3],DATA_matrix[:,5:6])
DATA_matrix= np.concatenate((DATA_matrix,mul_2),axis=1)
DATA_matrix_t = np.transpose(DATA_matrix)

k=np.array(DATA_matrix)
noc=len(k[0])
print(noc)
ww=gradientDescent(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-10.5,noc)
print("3rd degree: ",ww)

tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
#print(tr_data)
#ones = np.ones([tr_data.shape[0],1])
#tr_data= np.concatenate((ones,tr_data),axis=1)
#print(DATA_matrix)
y_pred=tr_data.dot(ww)
#print(y_pred)
y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
#y=np.array(y)
t=rmse(y.T, y_pred)
print("rmse"  ,t)
y=np.array(y)
y_pred=np.array(y_pred)
k=r2_score(y.T,y_pred)
print("r2:",k)

# degree 4
p4_x=np.multiply(DATA_matrix[:,6:8],DATA_matrix[:,1:3])
DATA_matrix= np.concatenate((DATA_matrix,p4_x),axis=1)
mul_3=np.multiply(DATA_matrix[:,3:4],DATA_matrix[:,4:5])
DATA_matrix= np.concatenate((DATA_matrix,mul_3),axis=1)
mul_4=np.multiply(DATA_matrix[:,1:2],DATA_matrix[:,6:7])
DATA_matrix= np.concatenate((DATA_matrix,mul_4),axis=1)
mul_5=np.multiply(DATA_matrix[:,2:3],DATA_matrix[:,7:8])
DATA_matrix= np.concatenate((DATA_matrix,mul_5),axis=1)
DATA_matrix_t = np.transpose(DATA_matrix)
w=np.matrix([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
k=np.array(DATA_matrix)
noc=len(k[0])
print(noc)
ww=gradientDescent(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-14.2,noc)
print("4th degree: ",ww)
tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
#print(tr_data)
#ones = np.ones([tr_data.shape[0],1])
#tr_data= np.concatenate((ones,tr_data),axis=1)
#print(DATA_matrix)
y_pred=tr_data.dot(ww)
#print(y_pred)
y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
#y=np.array(y)
t=rmse(y.T, y_pred)
print("rmse"  ,t)
y=np.array(y)
y_pred=np.array(y_pred)
k=r2_score(y.T,y_pred)
print("r2:",k)

#degree 5

p5_x=np.multiply(DATA_matrix[:,6:8],DATA_matrix[:,3:5])
DATA_matrix= np.concatenate((DATA_matrix,p5_x),axis=1)
mul_6=np.multiply(DATA_matrix[:,6:7],DATA_matrix[:,4:5])
DATA_matrix= np.concatenate((DATA_matrix,mul_6),axis=1)
mul_7=np.multiply(DATA_matrix[:,7:8],DATA_matrix[:,3:4])
DATA_matrix= np.concatenate((DATA_matrix,mul_7),axis=1)
mul_8=np.multiply(DATA_matrix[:,10:11],DATA_matrix[:,2:3])
DATA_matrix= np.concatenate((DATA_matrix,mul_8),axis=1)
mul_9=np.multiply(DATA_matrix[:,11:12],DATA_matrix[:,1:2])
DATA_matrix= np.concatenate((DATA_matrix,mul_8),axis=1)
DATA_matrix_t = np.transpose(DATA_matrix)
w=np.zeros([1,21])
w=np.matrix(w)
k=np.array(DATA_matrix)
noc=len(k[0])
print(noc)
ww=gradientDescent(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-17.5,noc)
print("5th degree: ",ww)
tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
#print(tr_data)
#ones = np.ones([tr_data.shape[0],1])
#tr_data= np.concatenate((ones,tr_data),axis=1)
#print(DATA_matrix)
y_pred=tr_data.dot(ww)
#print(y_pred)
y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
#y=np.array(y)
t=rmse(y.T, y_pred)
print("rmse"  ,t)
y=np.array(y)
y_pred=np.array(y_pred)
k=r2_score(y.T,y_pred)
print("r2:",k)
#degree 6
p6_x=np.multiply(DATA_matrix[:,6:8],DATA_matrix[:,6:8])
DATA_matrix= np.concatenate((DATA_matrix,p6_x),axis=1)
mul_10=np.multiply(DATA_matrix[:,15:16],DATA_matrix[:,2:3])
DATA_matrix= np.concatenate((DATA_matrix,mul_10),axis=1)
mul_11=np.multiply(DATA_matrix[:,16:17],DATA_matrix[:,1:2])
DATA_matrix= np.concatenate((DATA_matrix,mul_11),axis=1)
mul_12=np.multiply(DATA_matrix[:,10:11],DATA_matrix[:,4:5])
DATA_matrix= np.concatenate((DATA_matrix,mul_12),axis=1)
mul_13=np.multiply(DATA_matrix[:,11:12],DATA_matrix[:,3:4])
DATA_matrix= np.concatenate((DATA_matrix,mul_13),axis=1)
mul_14=np.multiply(DATA_matrix[:,6:7],DATA_matrix[:,7:8])
DATA_matrix= np.concatenate((DATA_matrix,mul_14),axis=1)
DATA_matrix_t = np.transpose(DATA_matrix)
w=np.zeros([1,28])
w=np.matrix(w)
k=np.array(DATA_matrix)
noc=len(k[0])
print(noc)
ww=gradientDescent(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-21,noc)
print("6th degree: ",ww)
tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
#print(tr_data)
#ones = np.ones([tr_data.shape[0],1])
#tr_data= np.concatenate((ones,tr_data),axis=1)
#print(DATA_matrix)
y_pred=tr_data.dot(ww)
#print(y_pred)
y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
#y=np.array(y)
t=rmse(y.T, y_pred)
print("rmse"  ,t)
y=np.array(y)
y_pred=np.array(y_pred)
k=r2_score(y.T,y_pred)
print("r2:",k)

'''
print("Ridge regression  ")

def ridge_regression(DATA_matrix,DATA_matrix_t,w,T,iters,alpha,noc):
    steps = 0
    precision = 0.000001
    
    x_axis = []  # iteration
    y_axis = []  # error

    for lam in lambda_vals:

        # initialization
        w0, w1, w2 = 0, 0, 0
        w0_new, w1_new, w2_new = 1, 1, 1
        steps = 0

        while (steps < steps_count ):


            #Y_pred = (w1 * X1) + (w2 * X2) + w0 + w3*

            dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred) + (2 * lam * w0)
            dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred)) + (2 * lam * w1)
            dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred)) + (2 * lam * w2)

            w0, w1, w2 = w0_new, w1_new, w2_new

            # new values of parameters
            w0_new = w0 - L * dr_w0
            w1_new = w1 - L * dr_w1
            w2_new = w2 - L * dr_w2

            #if(abs(w-w_n) > precision)

            steps += 1
            

        x_axis.append(lam)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

#print("Final Parameters: ", w0_new, w1_new, w2_new)
def rms_calc(w):
    tr_data=np.matrix(DATA_matrix[300001 : 434874,:])
    y_pred=tr_data.dot(w)
    y=np.matrix(DATA.loc[ '300001' : '434874', 'Altitude'])
    t=rmse(y.T, y_pred)
    return t
    
def lasso_regression(DATA_matrix,DATA_matrix_t,w,T,iters,alpha,noc):
    #steps = 0
    #precision = 0.000001
    # initialization
    
    x_axis = []  # iteration
    y_axis = []  # error
    lambda_vals = [x / 1000 for x in range(1, 10)]

    for lam in lambda_vals:

        # initialization
        for k in range(0,300000):  
            b1 = np.dot((DATA_matrix[k,:]), w) - T[0,k]
        #B1[k]=(b1*b1)
            for i in range(noc):
                w[i,0] = w[i,0]- alpha * np.dot(DATA_matrix_t[i,k], b1.T)+ (2 * lam * w[i,0])
            
            
        if(np.abs(b1) < 0.00001):
            break 
            print(lam," ",w)
            

        x_axis.append(lam)
        y_axis.append(rms_calc(w))

print("L1 Regularization: Lasso Regression")
lasso_regression(DATA_matrix,DATA_matrix_t,w.T,T,100,10**-21,noc)
#print("Final Parameters: ", w0_new, w1_new, w2_new)

'''





