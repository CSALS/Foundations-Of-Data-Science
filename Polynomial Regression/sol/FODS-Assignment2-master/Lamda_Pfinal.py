import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from preprocessing import NormalScaler
from preprocessing import MinMaxScaler
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt

# After this i have got W that fits my data with costraints 
class PolynomialRegression:

    def __init__(self,degree):
        self.X = []
        self.Y = []
        self.W = []
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

        self.W_array = []
        self.plot_array = []
        self.plot_test = []
        
    
    def preprocess(self,data,degree):
        data = data.drop([3],axis=1)
        rows,columns = data.shape
        data[0] = np.ones(rows)
        power = [i+1 for i in range(columns-1)]
        val = 0
        for i in range(2,degree+1): # i range from 2 to degree
            new_col = list(combinations_with_replacement(power,i))
            for j in range(len(new_col)):
                data[columns+val] = 1 
                for k in new_col[j]:
                    data[columns+val] = data[columns+val]*data[k]
                val += 1
        
        # print("Data is of form in Data frame without Normalization: ")
        # print(data.head(2))
        # n1 = MinMaxScaler()
        n1 = NormalScaler()
        n1.fit(data)
        data = n1.transform(data)       
        data[0] = 1
        # print("Data is of form in Data frame after Normalization: ")
        # print(data.head(2))
        data = data.to_numpy()
        print("Shape of Data is :")
        print(data.shape)
        return data   

    # def cost(self,X,Y,W):
    #     cost_val = np.sum(np.square(np.matmul(X,W)-Y))
    #     return (cost_val/(2*len(X)))

    # def gradient(self,X,Y,W):
    #     grad_arr = np.matmul(np.transpose(X),(np.matmul(X,W)-Y)) 
    #     return (grad_arr/len(X))

    def cost_lasso(self,X,Y,W,lamda):
        cost_val = np.sum(np.square(np.matmul(X,W)-Y)) + np.sum(np.square(W))
        return (cost_val/(2*len(X)))

    def cost_ridge(self,X,Y,W,lamda):
        cost_val = np.sum(np.square(np.matmul(X,W)-Y)) + np.sum(np.abs(W))
        return (cost_val/(2*len(X)))

    


    def lasso_gradient(self,X,Y,W,lamda):
        gradient = np.matmul(np.transpose(X),(np.matmul(X,W)-Y)) 
        gradient = gradient/len(X)
        W_list = W.reshape(1,len(W))
        W_list = [x/abs(x) for x in W_list[0]]
        W_list = (np.array(W_list))
        lasso_term = W_list.reshape(len(W_list),1)
        return ( gradient+(lamda*lasso_term) )

    def ridge_gradient(self,X,Y,W,lamda):
        gradient = np.matmul(np.transpose(X),(np.matmul(X,W)-Y)) 
        gradient = gradient/len(X)
        ridge_term = lamda*W
        return ( gradient+(lamda*ridge_term) )/len(X)

    def train(self,X,Y,W,alpha,lamda,Regularization):
        if(Regularization=="LR"):
            init_cost = self.cost_lasso(X,Y,W,lamda)
            prev_cost = init_cost
            updated_cost = 0
            # while(abs(prev_cost-updated_cost)>1e-5):
            for i in tqdm(range(300)):
                W = W - (alpha*self.lasso_gradient(X,Y,W,lamda))
                updated_cost = self.cost_lasso(X,Y,W,lamda)

                self.W_array.append(W)
                if(abs(prev_cost-updated_cost)<1e-5):
                    print("Converged in iteration "+str(i))
                    break
        else:
            init_cost = self.cost_ridge(X,Y,W,lamda)
            prev_cost = init_cost
            updated_cost = 0
            # while(abs(prev_cost-updated_cost)>1e-5):
            for i in tqdm(range(300)):
                W = W - (alpha*self.ridge_gradient(X,Y,W,lamda))
                updated_cost = self.cost_ridge(X,Y,W,lamda)

                self.W_array.append(W)
                if(abs(prev_cost-updated_cost)<1e-5):
                    print("Converged in iteration "+str(i))
                    break
                prev_cost = updated_cost
        
        print("Final Cost is :"+str(updated_cost))
        # print("Final W values are : ")
        # print(W)
        self.W = W
        self.plot_array.append(updated_cost)

    def test(self,X,Y,W):
        predicted_Y = np.matmul(X,W)
        RMSE = self.RMSE(predicted_Y,Y)
        R2 = self.R2(X,Y,W)
        self.plot_test.append(RMSE)
        print("RMSE : "+str(RMSE))
        print("R2 : "+str(R2))

    def RMSE(self,A,B):
        MSE = np.sum(np.square(A-B))/(len(A))
        return sqrt(MSE)
    
    def R2(self,X,Y,W):
        SSR = np.sum(np.square(np.matmul(X,W)-Y)) #MSE Residual Sum of Squares
        SST = np.sum(np.square(Y-np.mean(Y)))
        return (1-(SSR/SST))

    

    


if __name__=="__main__":
    degree = 6
    predictor = PolynomialRegression(degree)
    data = pd.read_csv("./Data.csv",header=None )
    # print(data.head(5))
    # data = data.sample(frac=1).reset_index(drop=True)
    predictor.Y = np.array((data[3]))
    # print(predictor.Y)
    data = predictor.preprocess(data,degree)
    predictor.X = data
    # print(data[:5])

    np.random.seed(11)
    predictor.W = np.random.randn(data.shape[1],1) #number of columns
    predictor.Y = np.reshape(predictor.Y, (-1, 1))
    # print(predictor.Y)
    # print("Initial W values are")
    # print(predictor.W)

    rows = data.shape[0]
    train = 0.8
    cv = 1-train
    predictor.X_train= data[ :int(train*rows) ]
    predictor.Y_train= predictor.Y[ :int(train*rows) ]

    predictor.X_test = data[ int(train*rows) : ]
    predictor.Y_test = predictor.Y[ int(train*rows): ]

    # print(predictor.X_train.shape)
    # print(predictor.Y_train.shape)
    # print(predictor.X_test.shape)
    # print(predictor.Y_test.shape)
    Regularization = "RR"
    # Regularization = "LR"
    for lamda in np.linspace(1,5,25):
        predictor.train(predictor.X_train,predictor.Y_train,predictor.W,0.05,lamda,Regularization)
        predictor.test(predictor.X_test,predictor.Y_test,predictor.W)
    
    plt.scatter( np.linspace(1,5,25),predictor.plot_array)
    plt.show()
    plt.scatter( np.linspace(1,5,25),predictor.plot_test)
    plt.show()
