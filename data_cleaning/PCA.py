import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def convert_to_csv(x, y, _set):
    #convert split data to pandas dataframe
    tmp_y = np.array(y)
    tmp_x = np.array(x)
    pca_ = np.concatenate((tmp_y, tmp_x), axis=1)
    prediction_labels = ["Outcome %d" % i for i in range(1, tmp_y.shape[1] + 1)]
    predictor_labels = ["PC %d" % i for i in range(1, tmp_x.shape[1] + 1)]
    labels = np.concatenate((prediction_labels, predictor_labels), axis=0)
    pca_df = pd.DataFrame(pca_, columns=labels)
    pca_df.to_csv("../data/PCA_split_data_%s.csv" % _set)


if __name__ == "__main__":

    train = pd.read_csv("../data/training_set_V1.csv")
    col_names = list(train.columns)
    row, columns = train.shape
    
    #split data into "labels" and predictors
    X = [] #predictors
    y = [] #predictions

    for index, row in train.iterrows():
        y.append(list(row.ix[1:13]))
        X.append(list(row.ix[13:]))
    
    #splitting this so we can test our results
    #we don't have ground truth for the test set yet so we will need to use our training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #create PCA object
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #get variance explained
    explained_variance = pca.explained_variance_ratio_
   
    #make first plot of just principal components
    fig1 = plt.figure()
    plt.plot(explained_variance)
    plt.title("Principal Components")
    plt.ylabel("Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("principal_comp.png")

    #select what percent var to keep
    desired_var = 0.9
    #select how many eigenvalues to keep
    cumsum = np.cumsum(explained_variance)
    k = np.argwhere(cumsum > desired_var)[0]
    
    #make second plot of cum var explained
    fig2 = plt.figure()
    plt.plot(cumsum)
    plt.title("Variance Explained")
    plt.plot(k, cumsum[k], 'ro', label="Eigenvalue #%d with %.2f Variance" % (k, desired_var))
    plt.legend()
    plt.ylabel("Cumulative Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("var_exp.png")
            
    #transform the data to lower dimension
    pca = PCA(n_components=int(k))
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    _set = "train"
    convert_to_csv(X_train, y_train, _set)
    _set = "test"
    convert_to_csv(X_test, y_test, _set)


    




