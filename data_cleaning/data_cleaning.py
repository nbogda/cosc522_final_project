import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#class to read in and clean data
class Data:

    def __init__(self, data_path):
        self.train = pd.read_csv("%s/TrainingDataset.csv" % data_path)
        self.test = pd.read_csv("%s/TestDataset.csv" % data_path)
        self.train_rows, self.train_columns = self.train.shape

    #remove NaNs from dataset
    def clean_nans(self):
        #replace NaN in Outcome_ with 0 
        self.train.loc[:, self.train.columns.str.startswith('Outcome_')] = self.train.loc[:, self.train.columns.str.startswith('Outcome_')].fillna(0)

        #check for NaNs in Quan_ columns
        quan_names = list(self.train.loc[:, self.train.columns.str.startswith('Quan_')])
        quan_nulls = self.train.loc[:, quan_names].isnull().sum(axis=0)
        #if more than 60% are null, ICE THAT COLUMN in both train and test
        quan_nulls /= self.train_rows
        for i in range(0, len(quan_nulls)):
            #more than 60% is NaN
            if quan_nulls[i] > 0.6:
                self.train.drop(quan_names[i], axis=1, inplace=True)
                self.test.drop(quan_names[i], axis=1, inplace=True)
            else:
                mean = self.train[quan_names[i]].mean()
                #drop columns with all 0s
                if mean == 0:
                    self.train.drop(quan_names[i], axis=1, inplace=True)
                    self.test.drop(quan_names[i], axis=1, inplace=True)
                else:
                    self.train[quan_names[i]] = self.train[quan_names[i]].fillna(mean)
                    self.test[quan_names[i]] = self.test[quan_names[i]].fillna(mean)

        #fix date columns with NaNs by setting to 0
        self.train.loc[:, self.train.columns.str.startswith('Date_')] = self.train.loc[:, self.train.columns.str.startswith('Date_')].fillna(0)
        self.test.loc[:, self.test.columns.str.startswith('Date_')] = self.test.loc[:, self.test.columns.str.startswith('Date_')].fillna(0)

        #checking categorical columns to make sure there is some variance in the column
        cat_names = list(self.train.loc[:, self.train.columns.str.startswith("Cat_")])
        for i in range(0, len(cat_names)):
            #all values are the same, drop the column in both train and test
            if self.train[cat_names[i]].nunique() == 1:
                self.train.drop(cat_names[i], axis=1, inplace=True)
                self.test.drop(cat_names[i], axis=1, inplace=True)

        #cover everything else
        self.train = self.train.replace(np.nan, 0)
        self.test = self.test.replace(np.nan, 0)

    #normalize dataset
    def normalize_data(self):
        col_names = list(self.train.columns)
        #removing outcomes, dont normalize those
        col_names[:] = [x for x in col_names if "Outcome" not in x]
        for i in range(0, len(col_names)):
            mean = self.train[col_names[i]].mean()
            std = self.train[col_names[i]].std()
            self.train[col_names[i]] = (self.train[col_names[i]] - mean)/std
            self.test[col_names[i]] = (self.test[col_names[i]] - mean)/std
   
    #to see the distribution of the outcomes
    def plot_counts(self):
        col_names = list(self.train.loc[:, self.train.columns.str.startswith('Outcome_')])
        for i in range(0, len(col_names)):
            curr = np.array(list(self.train[col_names[i]]))
            outcome_dict = {}
            for c in curr:
                if c not in outcome_dict:
                    outcome_dict[c] = 1
                else:
                    outcome_dict[c] += 1
            keys = sorted(outcome_dict.keys())
            values = [outcome_dict[k] for k in keys]
            y_pos = np.arange(len(keys))
            plt.figure(figsize=(20, 10))
            plt.bar(y_pos, values)
            plt.xticks(y_pos, keys, rotation=90)
            plt.ylabel("Outcomes")
            plt.title(col_names[i])
            plt.savefig("%s.png" % col_names[i]) 

    #make the monetary values into class values
    def split_into_bins(self):
        bins = [0, 2000, 5000, 15000, 50000, 100000, sys.maxsize]
        labels = [1, 2, 3, 4, 5, 6]
        col_names = list(self.train.loc[:, self.train.columns.str.startswith('Outcome_')])
        for c in col_names:
            self.train[c] = pd.cut(self.train[c], bins=bins, labels=labels)

    
    #write data to csv file to be nice 
    def write_to_csv(self, version): 
        print(self.train.shape)
        print(self.test.shape)
        self.train.to_csv("../data/training_set_%s.csv" % version)
        self.test.to_csv("../data/test_set_%s.csv" % version)



if __name__ == "__main__":

    #directory that data is located in
    data = Data("../data")
    data.clean_nans()
    data.normalize_data()
    #data.plot_counts()
    #data.split_into_bins()
    data.write_to_csv("V1")
