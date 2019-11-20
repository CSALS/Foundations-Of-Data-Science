import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from preprocessing import NormalScaler
from preprocessing import MinMaxScaler
from tqdm import tqdm
from math import sqrt

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
        self.cost_array = []
    
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

    def cost(self,X,Y,W):
        cost_val = np.sum(np.square(np.matmul(X,W)-Y))
        return (cost_val/(2*len(X)))

    def gradient(self,X,Y,W):
        grad_arr = np.matmul(np.transpose(X),(np.matmul(X,W)-Y)) 
        return (grad_arr/len(X))

    def train(self,X,Y,W,alpha):
        init_cost = self.cost(X,Y,W)
        prev_cost = init_cost
        updated_cost = 0
        prev_W = W
        # while(abs(prev_cost-updated_cost)>1e-5):
        for i in (range(1000)):
            W = W - (alpha*self.gradient(X,Y,W))
            updated_cost = self.cost(X,Y,W)
            print(updated_cost)
            self.W_array.append(W)
            self.cost_array.append(updated_cost)

            if(updated_cost<prev_cost):
                prev_W = W
                alpha += (0.4353253*alpha)
            else:
                W = prev_W
                alpha -= (0.5*alpha)
            
            if(abs(prev_cost-updated_cost)<1e-8):
                print("Converged in iteration "+str(i))
                break
            prev_cost = updated_cost
            
        print("Final Cost is :"+str(updated_cost))
        print("Final W values are : ")
        print(W)
        self.W = W

    def test(self,X,Y,W):
        predicted_Y = np.matmul(X,W)
        RMSE = self.RMSE(predicted_Y,Y)
        R2 = self.R2(X,Y,W)
        print("RMSE : "+str(RMSE))
        print("R2 : "+str(R2))

    def RMSE(self,A,B):
        MSE = np.sum(np.square(A-B))/(len(A))
        return sqrt(MSE)
    
    def R2(self,X,Y,W):
        SSR = np.sum(np.square(np.matmul(X,W)-Y)) #MSE Residual Sum of Squares
        SST = np.sum(np.square(Y-np.mean(Y)))
        return (1-(SSR/SST))

    def normal(self,X,Y):
        normal_W = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
        normal_cost = self.cost(X,Y,normal_W)
        predicted_Y = np.matmul(X,normal_W)
        normal_RMSE = self.RMSE(predicted_Y,Y)
        normal_R2 = self.R2(X,Y,normal_W)
        print("Final Cost is :"+str(normal_cost))
        print("Final W values are :")
        print(normal_W)
        print("RMSE : "+str(normal_RMSE))
        print("R2 : "+str(normal_R2))



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

    predictor.train(predictor.X_train,predictor.Y_train,predictor.W,0.05)
    predictor.test(predictor.X_test,predictor.Y_test,predictor.W)
    print("-----")
    predictor.normal(predictor.X_train,predictor.Y_train)

    # print("RMSE Error is :"+(str)(polreg_obj.RMSE(polreg_obj.X_test,polreg_obj.W,polreg_obj.Y_test)))
    # print("R2 Error is :"+(str)(polreg_obj.R2(polreg_obj.X_test,polreg_obj.W,polreg_obj.Y_test)))
    

