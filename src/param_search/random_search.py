import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import random as rand
import sys
import collections
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

#function to read in the CSV files
def read_CSV():
    train = pd.read_csv("../../data/PCA_split_data_train.csv")
    test = pd.read_csv("../../data/PCA_split_data_test.csv")
 
    #split data into "labels" and predictors (The actual training set was split 70-30 since the testing set has no outcomes)
    X_tr = [] #predictors training
    y_tr = [] #predictions training
    X_test = [] #features testing
    y_test = [] #predicitions testing
    
    for index, row in train.iterrows():
        y_tr.append(list(row.ix[1:13]))
        X_tr.append(list(row.ix[13:]))
        
    for index, row in test.iterrows():
        y_test.append(list(row.ix[1:13]))
        X_test.append(list(row.ix[13:]))
        
    return X_tr,y_tr,X_test,y_test

def get_params(algorithm):
    '''
    algorithm : name of the algorithm to get params for

    returns dict of params
    '''
    if algorithm == "kNN":
        return { 'n_neighbors' : np.arange(0, 100, 5),
                 'p' : [1, 2, 3] } #different orders of minkowski distance. 1=manhattan, 2=euclidean
    elif algorithm == "MLP": #multi-layer perceptron....not "my little pony"
        return { 'hidden_layer_sizes' = [(10,) (10, 10), (50, 50, 50)], #expand this later
                 'alpha' : [0.0001, 0.01, 1, 5, 10]}
    elif algorithm == "Decision Tree":
        return { 'criterion' : ["mse", "friedman_mse", "mae"],
                 'max_depth' : [None, 5, 10, 20],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4] }
    elif algorithm == "SVM":
        return { 'kernel' : ['rbf', 'sigmoid'],
                 'gamma' : ['scale', 'auto'],
                 'C' : [0, 0.1, 1, 5, 10],
                 'epsilon' : [0, 0,1, 1, 5, 10] }
    elif algorithm == "Random Forest":
        return { 'n_estimators' : [10, 50, 100, 200, 500],
                 'criterion' : ["mse", "friedman_mse", "mae"],
                 'max_depth' : [None, 5, 10, 20],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4] }


def random_search(algorithm):
    '''
    Testing the following algs: 

        kNN, BPNN/MLP, Decision Tree, Random Forest, SVM, Linear Reg(?)
    '''
    print("")

if __name__ == "__main__":

    X_tr,y_tr,X_test,y_test = read_CSV()
    
    algorithm = "kNN"
    param_dict = get_params(algorithm)
    print(param_dict)

