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
def read_CSV(clean_method, preprocessing):
    '''
    clean_method : integer
                   0 - deleted
                   1 - mean
                   2 - to_0
    
    preprocessing : integer
                    0 - ORIGINAL
                    1 - PCA
    '''
    
    file_paths = ["deleted", "mean", "to_0"]
    file_name = ["ORIGINAL", "PCA"]

    #print statement for my sanity
    print("\nYou have selected NaN %s, with %s data\n" % (file_paths[clean_method], file_name[preprocessing]))

    train = pd.read_csv("../../data/NAN_%s/%s_split_data_train.csv" % (file_paths[clean_method], file_name[preprocessing]))
    test = pd.read_csv("../../data/NAN_%s/%s_split_data_test.csv" % (file_paths[clean_method], file_name[preprocessing]))
 
    #split data into predictions and predictors
    X = [] #predictors training
    y = [] #predictions training
    
    for index, row in train.iterrows():
        y.append(list(row.ix[1:13]))
        X.append(list(row.ix[13:]))
        
    for index, row in test.iterrows():
        y.append(list(row.ix[1:13]))
        X.append(list(row.ix[13:]))
        
    #just gonna have one big test set for the random search, it does fold CV anyway
    return X, y

def get_params(algorithm):
    '''
    algorithm : name of the algorithm to get params for

    returns dict of params
    '''
    if algorithm == "kNN":
        return { 'n_neighbors' : np.arange(1, 100, 5),
                 'p' : [1, 2, 3] } #different orders of minkowski distance. 1=manhattan, 2=euclidean
    elif algorithm == "MLP": #broke
        return { 'hidden_layer_sizes' : [(10, 10,)], #expand this later
                 'alpha' : [0.01, 1, 5, 10]}
    elif algorithm == "Decision Tree":
        return { 'criterion' : ["mse", "friedman_mse", "mae"],
                 'max_depth' : [None, 5, 10, 20],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4] }
    elif algorithm == "SVM":
        return { 'estimator__kernel' : ['rbf', 'sigmoid'],
                 'estimator__gamma' : ['scale', 'auto'],
                 'estimator__C' : [0, 0.1, 1, 5, 10],
                 'estimator__epsilon' : [0, 0,1, 1, 5, 10] }
    elif algorithm == "Random Forest":
        return { 'n_estimators' : [10, 50, 100, 200, 500],
                 'criterion' : ["mse", "friedman_mse", "mae"],
                 'max_depth' : [None, 5, 10, 20],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4] }


def random_search(algorithm, params, X, y, iters=20):
    '''
    Testing the following algs: 

        kNN, BPNN/MLP, Decision Tree, Random Forest, SVM
    '''
    clf = None
    if algorithm == "kNN":
        clf = KNeighborsRegressor()
    elif algorithm == "MLP":
        clf = MLPRegressor()
    elif algorithm == "Decision Tree":
        clf = DecisionTreeRegressor()
    elif algorithm == "SVM":
        clf = SVR()
        clf = MultiOutputRegressor(clf)
    elif algorithm == "Random Forest":
        clf = RandomForestRegressor()

    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=iters, n_jobs=5, 
                                       scoring='neg_mean_squared_log_error', verbose=2)
    random_search.fit(X, y)
    report(random_search.cv_results_)

#stolen shamelessly off the internet
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))

            #except this part, this is the RMSLE score
            print("Mean RMSLE: {0:.3f} (std: {1:.3f})"
                  .format(np.sqrt(np.abs(results['mean_test_score'][candidate])),
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
 
    '''
    clean_method : integer
                   0 - deleted
                   1 - mean
                   2 - to_0
    
    preprocessing : integer
                    0 - ORIGINAL
                    1 - PCA
    '''
    clean_method = 0
    preprocessing = 0

    #read data from one of 6 datasets
    X, y = read_CSV(clean_method, preprocessing)

    '''
    algorithm : string
                - kNN
                - MLP ----------- (Problem, making negative predictions, cant do RMSLE)
                - Decision Tree
                - SVM ----------- (Also making negative predictions) 
                - Random Forest
    '''
    algorithm = "Random Forest"
    
    #this is where the params to test are stored
    param_dict = get_params(algorithm)

    #this where the actual searching happens
    random_search(algorithm, param_dict, X, y, iters=20)

