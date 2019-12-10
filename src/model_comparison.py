import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
import re

#we want one large graph
#x-ticks are the mf'ing algorithms
#6 bars for RMSE, one for time
#(shit is gonna be sideways on the page lol)

#graphs to visualize data from the random search
def generate_rs_graphs(metric, rf=True):
    random_search_info = pd.read_csv("param_search/saved_models/Random_Search_Info.csv", index_col=0)
   
    N = 5
    algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest"]
    if metric == "Refit Time" and not rf:
        algorithms = ["kNN", "MLP", "Decision Tree", "SVM"]
        N = 4
   
    bar_values = [None] * N
    for index, row in random_search_info.iterrows():
        name = re.search("Best (.*) [^\s]+ [^\s]+", row.name).group(1)
        if not rf and name == "Random Forest":
            continue
        bv_index = algorithms.index(name)
        if bar_values[bv_index] is None:
            bar_values[bv_index] = []
        bar_values[bv_index].append(row.loc[metric])
    
    fig, ax = plt.subplots(figsize=(20,10))
    ind = np.arange(N)
    width = 0.1
    bar_values = np.array(bar_values)
    og_del = ax.bar(ind, bar_values[:,0], width)
    og_mean = ax.bar(ind + width, bar_values[:,1], width)
    og_0 = ax.bar(ind + width*2, bar_values[:,2], width)
    pca_del = ax.bar(ind + width*3, bar_values[:,3], width)
    pca_mean = ax.bar(ind + width*4, bar_values[:,4], width)
    pca_0 = ax.bar(ind + width*5, bar_values[:,5], width)
    if metric == "Refit Time":
        metric += " (s)"
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks((ind + width*2.5))
    ax.tick_params(labelsize=14)
    ax.set_xticklabels(algorithms)
    ax.legend((og_del[0], og_mean[0], og_0[0], pca_del[0], pca_mean[0], pca_0[0]), 
              ("Original NaN Deleted", "Original Mean Impute", "Original 0 Impute", "PCA NaN Deleted", "PCA Mean Impute", "PCA 0 Impute"),
              ncol=2, fontsize='large')
    title = "Performance" if metric == "Mean RMSLE" else "Fit Time"
    if not rf:
        title += " without Random Forest"
    ax.set_title("Random Search Best Algorithm %s" % title, fontsize=14)
    plt.savefig("graphs/random_search_best_alg_%s.png" % title)
    

def test_best_algs():
    file_paths = ["deleted", "mean", "to_0"]
    file_names = ["ORIGINAL", "PCA"]
    algorithms = ["kNN", "MLP", "Decision Tree", "SVM", "Random Forest"]

if __name__ == "__main__":

    #Mean RMSLE or Refit Time
    metric = "Refit Time"
    generate_rs_graphs(metric, rf=False)
