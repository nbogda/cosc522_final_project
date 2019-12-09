import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def convert_to_csv(x, y, col_name, _set, data):
    #convert split data to pandas dataframe
    tmp_y = np.array(y)
    tmp_x = np.array(x)
    pca_ = np.concatenate((tmp_y, tmp_x), axis=1)
    prediction_labels = ["Outcome %d" % i for i in range(1, tmp_y.shape[1] + 1)]
    predictor_labels = ["%s %s" % (col_name, i) for i in range(1, tmp_x.shape[1] + 1)]
    labels = np.concatenate((prediction_labels, predictor_labels), axis=0)
    pca_df = pd.DataFrame(pca_, columns=labels)
    file_name = "ORIGINAL" if col_name == "Column" else "PCA" 
    pca_df.to_csv("../data/NAN_%s/%s_split_data_%s.csv" % (data, file_name, _set))


if __name__ == "__main__":

    #choose from "mean", "to_0", or "deleted"
    data = "mean"


    train = pd.read_csv("../data/cleaned_data/nan_%s.csv" % data)
    col_names = list(train.columns)
    row, columns = train.shape
    
    #split data into "labels" and predictors
    X = [] #predictors
    y = [] #predictions

    for index, row in train.iterrows():
        y.append(list(row.iloc[1:13]))
        X.append(list(row.iloc[13:]))
    
    #splitting this so we can test our results
    #we don't have ground truth for the test set yet so we will need to use our training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    convert_to_csv(X_train, y_train, "Column", "train", data)
    convert_to_csv(X_test, y_test, "Column", "test", data)

    #create PCA object
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #get variance explained
    explained_variance = pca.explained_variance_ratio_
   
    #make first plot of just principal components
    fig1 = plt.figure()
    plt.plot(explained_variance)
    plt.title("Principal Components %s" % data)
    plt.ylabel("Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("PCA_charts/principal_comp_%s.png" % data)

    #select what percent var to keep
    desired_var = 0.9
    #select how many eigenvalues to keep
    cumsum = np.cumsum(explained_variance)
    k = np.argwhere(cumsum > desired_var)[0]
    
    #make second plot of cum var explained
    fig2 = plt.figure()
    plt.plot(cumsum)
    plt.title("Variance Explained %s" % data)
    plt.plot(k, cumsum[k], 'ro', label="Eigenvalue #%d with %.2f Variance" % (k, desired_var))
    plt.legend()
    plt.ylabel("Cumulative Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("PCA_charts/var_exp_%s.png" % data)
            
    #transform the data to lower dimension
    pca = PCA(n_components=int(k))
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    _set = "train"
    convert_to_csv(X_train, y_train, "PC", _set, data)
    _set = "test"
    convert_to_csv(X_test, y_test, "PC", _set, data)


    




