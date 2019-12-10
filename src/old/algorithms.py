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
from sklearn.externals import joblib

#function to read in the CSV files
def read_CSV():
    train = pd.read_csv("../../data/NAN_to_0/ORIGINAL_split_data_train.csv")
    test = pd.read_csv("../../data/NAN_to_0/ORIGINAL_split_data_test.csv")
 
    #split data into "labels" and predictors (The actual training set was split 70-30 since the testing set has no outcomes)
    X_tr = [] #predictors training
    y_tr = [] #predictions training
    X_test = [] #features testing
    y_test = [] #predicitions testing
    
    for index, row in train.iterrows():
        y_tr.append(list(row.iloc[1:13]))
        X_tr.append(list(row.iloc[13:]))
        
    for index, row in test.iterrows():
        y_test.append(list(row.iloc[1:13]))
        X_test.append(list(row.iloc[13:]))
        
    return X_tr,y_tr,X_test,y_test

#class to hold various algorithms to try on the data
class Regression_Algs:

    def __init__(self, X_tr, y_tr, X_test, y_test):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_test = X_test
        self.y_test = y_test

    def decision_tree(self):
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(self.X_tr,self.y_tr)
        y_pred = regressor.predict(self.X_test)
        return np.abs(y_pred)

    def random_forest(self):
        RFregr = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1000)
        RFregr.fit(self.X_tr,self.y_tr)
        y_pred = RFregr.predict(self.X_test)
        return np.abs(y_pred)

    def svm(self):
        clf = SVR(kernel="rbf",gamma='auto', C=1.0, epsilon=0.2)
        multi_clf = MultiOutputRegressor(clf)
        multi_clf.fit(self.X_tr,self.y_tr)
        y_pred = multi_clf.predict(self.X_test)
        return np.abs(y_pred)

    def mlp(self):
        bpnn = MLPRegressor(max_iter = 50000) #Very basic BPNN/MLP
        bpnn.fit(self.X_tr,self.y_tr)
        y_pred = bpnn.predict(self.X_test)
        return np.abs(y_pred)

    def kNN(self):
        #testing saving and opening model
        #It works!!! :D
        neigh = joblib.load("param_search/saved_models/best_kNN_ORIGINAL_deleted.joblib")
        neigh.fit(self.X_tr,self.y_tr)
        y_pred = neigh.predict(self.X_test)
        return np.abs(y_pred)

    def linear_reg(self):
        reg = LinearRegression()
        reg.fit(self.X_tr,self.y_tr)
        joblib.dump(reg, "../param_search/saved_models/best_linearRegression_ORIGINAL_to_0.joblib")
        y_pred = reg.predict(self.X_test)
        return np.abs(y_pred)
    
    #this can be expanded to include more methods of evaluation
    def eval(self, y_pred):

        multiple_err = np.sqrt(mean_squared_log_error(self.y_test, y_pred, multioutput='raw_values'))
        overall_err = np.sqrt(mean_squared_log_error(self.y_test, y_pred))
        return multiple_err, overall_err

        #cross_val_score(regressor, X, y, cv=10)

if __name__ == "__main__":

    X_tr,y_tr,X_test,y_test = read_CSV()
    algs = Regression_Algs(X_tr, y_tr, X_test, y_test)
    pred = algs.linear_reg()
    print(algs.eval(pred))

