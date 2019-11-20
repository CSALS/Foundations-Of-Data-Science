"""
This file contains the classes required for processing the data
before applying machine learning algorithms.
"""
import numpy as np

class NormalScaler:
    """
    Normal Scaler transforms the given data
    into a normal distribution.
    """
    def fit(self, arr):
        """
        This function calculates the mean and standard devaition
        of the data that has been passed as argument.
        """
        self.mean = np.mean(arr)
        self.std = np.std(arr)
        
    def transform(self, arr):
        """
        This function applies the normal transformation and returns the data (arr) using 
        the values of mean and std obtained from the fit function.
            x_trans = (x-mean)/std
        """
        return (arr-self.mean)/(self.std)
        

class MinMaxScaler:
    """
    MinMaxScaler transforms the given data
    such that all the data points are between 0 and 1.
    """
    def fit(self, arr):
        """
        This function calculates the minimum and maximum values
        of the given data and stores them as class attributes.
        """
        self.min = np.min(arr)
        self.max = np.max(arr)
        
    def transform(self, arr):
        """
        This function applies the min-max transformation using 
        the class attributes min and max.
            x_trans = (x-min)/(max-min)
        """
        return (arr-self.min)/(self.max-self.min)

    def inv_transform(self, arr):
        """
        This function applies the inverse transformation of minmax scaling.
        It returns the original data if the transformed data is given.
            x_orig = x * (max-min) + min
        """
        return arr*(self.max-self.min)+self.min