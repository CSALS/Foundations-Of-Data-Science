import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from preprocessing import NormalScaler
from tqdm import tqdm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class LamdaPolyreg:
    def __init__(self):
        self.X = []
        self.Y = []
        self.W = []
        self.W_lamda = []
        self.Cost_arr = []
        self.alpha = 5e-7
        self.err = []

    
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
        n1 = NormalScaler() 
        for i in data.columns:
            n1.fit(data[i])
            data[i] = n1.transform(data[i])
        data[0] = 1
        data = data.to_numpy()
        return data 
    
    def train(self,X,Y,lamda):
        prev_err = 0
        for i in tqdm(range(1000)): #each lamda 1000 iterations
            new_err = self.MSE (X,self.W,Y)
            # self.Cost_arr.append(new_err)
            gradient = self.ridge_grad(X,self.W,Y,lamda)
            # gradient = self.lasso_grad(X,self.W,Y,lamda)
            self.W = self.W - (gradient)
            if(abs(prev_err-new_err)<1e-4):
                # i
                break
            prev_err = new_err
        # print("\n")
        # print("Lamda value is: " +str(lamda))
        final_err = self.MSE(X,self.W,Y)
        self.err.append(final_err)
        print("GD MSE Training error: "+str(final_err))
        # print("Values of W after Gradient descent with Lamda : ")
        # print(self.W)
        return None

    def test(self,X,Y,lamda):
        test_err = self.MSE(X,self.W,Y)
        # print("Lamda is : "+str(lamda))
        # print("GD MSE Testng error: "+str(test_err))
        return (test_err)

    def MSE(self,X,W,Y): # To calculate final Error
        err_val = np.sum(np.square(np.subtract(np.matmul(X,W),Y)))
        return err_val/len(X) 

    # def costFun(self,X,W,Y,lamda):
    #     cost = np.sum(np.square(np.subtract(np.matmul(X,W),Y)))+(lamda*(np.sum(np.square(W))))
    #     return cost/(2*(len(X)))

    def ridge_grad(self,X,W,Y,lamda):
        mat = np.subtract(np.matmul(X,W),Y)
        grad = (np.matmul(np.transpose(X),mat))
        ridge_mat = (lamda)*W
        return (np.add(self.alpha*grad,self.alpha*ridge_mat))
    
    def lasso_grad(self,X,W,Y,lamda):
        mat = np.subtract(np.matmul(X,W),Y)
        grad = (np.matmul(np.transpose(X),mat))

        temp = (lpr_obj.W).reshape(1,len(lpr_obj.W))
        one_list = [x/abs(x) for x in temp[0]]
        one_list = (np.array(one_list))
        one_list = one_list.reshape(len(one_list),1)

        ridge_mat = (lamda)*one_list # instead of 2D array of W created 2D array of 1 and -1's
        return (np.add(self.alpha*grad,self.alpha*ridge_mat))

    def normal(self,X,Y,lamda):
        right = np.matmul(np.transpose(X),Y)
        XXT = np.matmul(np.transpose(X),X)
        np.fill_diagonal(XXT, XXT.diagonal() + lamda)
        left = np.linalg.inv(XXT)
        W = np.matmul(left,right)
        # print(W)
        normal_err_train = self.MSE(X,W,Y)
        print("Normal MSE Training error: "+str(normal_err_train))
        # normal_err_test = self.MSE(self.X_test,W,self.Y_test)
        # print("Normal MSE Testing Error: "+str(normal_err_test))
        print("\n")



if __name__=="__main__":
    size = 2
    lpr_obj = LamdaPolyreg()
    data = pd.read_csv("./Data.csv",header=None )
    rows,columns = data.shape
    np.random.seed(11)
    lpr_obj.W = np.random.randn(data.shape[1],1) #number of columns
    lpr_obj.Y = np.array((data[3]))
    lpr_obj.Y = np.reshape(lpr_obj.Y, (-1, 1))

    data = lpr_obj.preprocess(data,size)
    lpr_obj.X = data

    train_percent = 0.8
    test_percent = 1-train_percent

    lpr_obj.X_train = data[:(int)(train_percent*rows)]
    lpr_obj.Y_train = lpr_obj.Y[:(int)(train_percent*rows)]
    lpr_obj.X_test = data[(int)(train_percent*rows):]
    lpr_obj.Y_test = lpr_obj.Y[(int)(train_percent*rows):]

    val = -1000
    lamda_list = [ x for x in range(val,1001) if(x%100==0)]
    # lamda_list = [ x/10 for x in range(val,-10)]
    print(lamda_list)

    for lamda in lamda_list:
        lpr_obj.Cost_arr = [] #initialize cost array each time to Null
        np.random.seed(11)
        lpr_obj.W = np.random.randn(data.shape[1],1) #initialize each time
        lpr_obj.train(lpr_obj.X_train,lpr_obj.Y_train,lamda)
        lpr_obj.test(lpr_obj.X_test,lpr_obj.Y_test,lamda)
        # print("\n")
        # lpr_obj.normal(lpr_obj.X_train,lpr_obj.Y_train,lamda)

    plt.scatter( lamda_list,lpr_obj.err)
    plt.show()
    
    
    



# alpha = 4e-7 and convergence = 1e-7


