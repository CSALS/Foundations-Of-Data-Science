import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from preprocessing import NormalScaler
from preprocessing import MinMaxScaler
from tqdm import tqdm
from math import sqrt
class PolynomialRegression:
    def __init__(self,size):
        self.X = []
        self.Y = []
        # np.random.seed(11)
        self.W = []
        self.W_arr = []
        self.Cost_arr = []
        self.Test_cost_arr = []
        self.alpha = 5e-8
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.X_cv = []
        self.Y_cv = []
    
    def preprocess(self,data,size):
        data = data.drop([3],axis=1)
        rows,columns = data.shape
        data[0] = np.ones(rows)
        power = [i+1 for i in range(columns-1)] #1 to number of columns
        val = 0
        for i in range(2,size+1): # 2 to size -- i
            new_col = list(combinations_with_replacement(power,i))
            for j in range(len(new_col)):
                data[columns+val] = 1 
                for k in new_col[j]:
                    data[columns+val] = data[columns+val]*data[k]
                val += 1
        print("Data is of form in Data frame without Normalization: ")
        print(data.head(5))
        # n1 = MinMaxScaler() # normalizing data here
        n1 = NormalScaler()
        # n1.fit(data)
        # data = n1.transform(data)
        for i in data.columns:
            n1.fit(data[i])
            data[i] = n1.transform(data[i])        
        data[0] = 1
        print("Data is of form in Data frame after Normalization: ")
        print(data.head(5))
        data = data.to_numpy()
        print(data.shape)
        return data   

    def normal_W(self,X,Y):
        normal_W = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
        normal_err = self.error(X,normal_W,Y)
        normal_test_err = self.error(self.X_test,normal_W,self.Y_test)
        print("W after Normal eqautions method is :")
        print(normal_W)
        print("Error After Normal Equatios is :"+str(normal_err))
        print("Error After Normal Equations Testing is :"+str(normal_test_err))
        return normal_err

    def error(self,X,W,Y):
        err_val = np.sum(np.square(np.subtract(np.matmul(X,W),Y)))
        return err_val/(2*len(X)) #works for train and test because X is input matrix can be any size
    
    def grad(self,X,W,Y):
        mat = np.subtract(np.matmul(X,W),Y)
        grad = (np.matmul(np.transpose(X),mat))
        # return (grad/len(X)) grad need not to have /len(X) because alpha is so small and manages it
        return grad
    
    def train(self,data,Y):
        rows = data.shape
        train = 0.8
        cv = 0.0
        # test = 1-train-cv
        self.X_train= data[:int(train*rows)]
        self.Y_train= Y[:int(train*rows)]
        self.X_cv = data[int(train*rows):int((train+cv)*rows)]
        self.Y_cv = Y[int(train*rows):int((train+cv)*rows)]
        self.X_test = data[int((train+cv)*rows):]
        self.Y_test = Y[int((train+cv)*rows):]
        prev_err = 0

        for i in tqdm(range(10000)):
            new_err = self.error(self.X_train,self.W,self.Y_train)
            self.Cost_arr.append(new_err)
            gradient = self.grad(self.X_train,self.W,self.Y_train)
            self.W = self.W - (self.alpha*gradient)
            self.W_arr.append(self.W)
            if(i%1000==0):
                print(new_err)
            if(abs(prev_err-new_err)<1e-4):
                print("Error differece is so less and Stopped code at  iteration: "+str(i))
                break
            prev_err = new_err
        print("After training W values are :")
        print(self.W)
        print("After training error is :"+str(prev_err))


    def test(self):
        test_err = self.error(self.X_test,self.W,self.Y_test)
        print("Error after Testing in Gradiet Descent :"+str(test_err))

    def RMSE(self,X,W,Y):
        err_val = np.sum(np.square(np.subtract(np.matmul(X,W),Y)))
        return sqrt(err_val/len(X)) #root mean square error

    def R2(self,X,W,Y):
        print("\n\n")
        print("Predicted Y : ")
        print(np.matmul(X,W))
        print("\n")
        print("Original Y :")
        print(Y)
        print("\n")
        residual_err = np.sum(np.square(np.subtract(np.matmul(X,W),Y)))
        print("Residual Error is : "+str(residual_err))
        print("Mean of Y: "+str(np.mean(Y)))
        print("\n")
        print("Mean of Y : ")
        print(np.full((len(Y),1),np.mean(Y)))
        print("Individual elements Difference: ")
        print((np.subtract(np.matmul(X,W),np.full((len(Y),1),np.mean(Y)))))
        square_err = np.sum(np.square(np.subtract(Y,np.full((len(Y),1),np.mean(Y)))))
        print("\n")
        print("Mean Error is : "+str(square_err))
        return (1-(residual_err/square_err))


if __name__=="__main__":
    degree = 2
    polreg_obj = PolynomialRegression(degree)
    predictor = PolynomialRegression(degree)

    data = pd.read_csv("./Data.csv",header=None )

    # data = data.sample(frac=1).reset_index(drop=True)

    polreg_obj.Y = np.array((data[3]))
    predictor.Y = np.array((data[3]))

    data = polreg_obj.preprocess(data,degree)
    data = predictor.preprocess(data,degree)

    polreg_obj.X = data
    predictor.X = data

    np.random.seed(11)

    polreg_obj.W = np.random.randn(data.shape[1],1) #number of columns
    predictor.W = np.random.randn(data.shape[1],1) #number of columns

    polreg_obj.Y = np.reshape(polreg_obj.Y, (-1, 1))
    predictor.Y = np.reshape(polreg_obj.Y, (-1, 1))

    print("Initial W values are")
    print(polreg_obj.W)
    print(predictor.W)
 
    polreg_obj.train(data,polreg_obj.Y)
    polreg_obj.test()
    rows = data.shape[0]
    train = 0.8
    X_train= data[:int(train*rows)]
    Y_train= polreg_obj.Y[:int(train*rows)]
    polreg_obj.normal_W(X_train,Y_train)

    print("RMSE Error is :"+(str)(polreg_obj.RMSE(polreg_obj.X_test,polreg_obj.W,polreg_obj.Y_test)))
    print("R2 Error is :"+(str)(polreg_obj.R2(polreg_obj.X_test,polreg_obj.W,polreg_obj.Y_test)))
    

