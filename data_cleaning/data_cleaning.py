import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from mlxtend.preprocessing import minmax_scaling
#from sklearn_pandas import DataFrameMapper


#class to read in and clean data
class Data:

    def __init__(self, data_path):
        self.train = pd.read_csv("%s/TrainingDataset.csv" % data_path)
        print("Shape before cleaning:" + str(self.train.shape))

    #remove NaNs from dataset
    def clean_nans(self, type_fill):
        '''
        type_fill : integer
                    0 - fill all NA with 0
                    1 - drop column with NA
                    2 - mean imputation
        '''
        self.type_fill = type_fill
        
        #replacing NaNs with last monthly outcome
        #such sacrilege
        batman = 0 #NaN(a) counter
        outcomes_subset = self.train.loc[:, self.train.columns.str.startswith('Outcome_')]
        for index, row in outcomes_subset.iterrows():
            #find indices with NaNs
            nan_indices = np.where(np.isnan(row))[0]
            if nan_indices.size > 0:
                #count how many NaNs we had to replace
                batman += nan_indices.size
                #get the last non-NaN value
                fill_val = row.iloc[nan_indices[0] - 1]
                #replace them with this fill value
                row.iloc[nan_indices] = fill_val
                outcomes_subset.iloc[index] = row
        #put this transform back into the dataset
        self.train.loc[:, self.train.columns.str.startswith('Outcome_')] = outcomes_subset

        #replace NaN in Outcome_ with 0 
        #self.train.loc[:, self.train.columns.str.startswith('Outcome_')] = self.train.loc[:, self.train.columns.str.startswith('Outcome_')].fillna(0)
        
        #check for NaNs in Quan_ columns
        quan_names = list(self.train.loc[:, self.train.columns.str.startswith('Quan_')])
        quan_nulls = self.train.loc[:, quan_names].isnull().sum(axis=0)
        #if more than 60% are null, ICE THAT COLUMN in both train and test
        quan_nulls /= self.train.shape[0]
        for i in range(0, len(quan_nulls)):
            #more than 60% is NaN
            if quan_nulls[i] > 0.6:
                if self.type_fill != 2:
                    continue
                else:
                    self.train.drop(quan_names[i], axis=1, inplace=True)
            else:
                mean = self.train[quan_names[i]].mean()
                #drop columns with all 0s
                if mean == 0:
                    self.train.drop(quan_names[i], axis=1, inplace=True)
                elif mean != 0 and self.type_fill == 2:
                    self.train[quan_names[i]] = self.train[quan_names[i]].fillna(mean)

        #fix date columns with NaNs by setting to 0
        if self.type_fill == 2:
            self.train.loc[:, self.train.columns.str.startswith('Date_')] = self.train.loc[:, self.train.columns.str.startswith('Date_')].fillna(0)

        #checking categorical columns to make sure there is some variance in the column
        cat_names = list(self.train.loc[:, self.train.columns.str.startswith("Cat_")])
        for i in range(0, len(cat_names)):
            #all values are the same, drop the column in both train and test
            if self.train[cat_names[i]].nunique() == 1:
                self.train.drop(cat_names[i], axis=1, inplace=True)
        
        if self.type_fill == 0 or self.type_fill == 2:
            self.train = self.train.fillna(0)
        elif self.type_fill == 1:
            self.train = self.train.dropna(axis=1)

    #scale data with min/max scaling
    def scale_data(self, scale_outcomes=True):

        col_names = list(self.train.columns)
        predictor_names = [x for x in col_names if "Outcome" not in x]
        self.train[predictor_names] = minmax_scaling(self.train, columns = predictor_names)
       
        if scale_outcomes:
            outcomes = [x for x in col_names if "Outcome" in x]
            feature_range = []
            #saving the min and max values per column before standardizing
            for o in outcomes:
                feature_range.append([min(self.train[o]), max(self.train[o])])
            #save min max in numpy file for future use (to invert the transform)
            np.save("12_outcomes_feature_range_min_max.npy", np.array(feature_range))
            self.train[outcomes] = minmax_scaling(self.train, columns = outcomes)
            

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

    '''
    WE ARE NOT GOING DOWN THIS SILLY ROUTE

    #make the monetary values into class values
    def split_into_bins(self):
        bins = [0, 2000, 5000, 15000, 50000, 100000, sys.maxsize]
        labels = [1, 2, 3, 4, 5, 6]
        col_names = list(self.train.loc[:, self.train.columns.str.startswith('Outcome_')])
        for c in col_names:
            self.train[c] = pd.cut(self.train[c], bins=bins, labels=labels)
    '''
    
    #write data to csv file to be nice 
    def write_to_csv(self): 
        print("Shape after cleaning:" + str(self.train.shape))
        file_name = ["to_0", "deleted", "mean"]
        self.train.to_csv("../data/cleaned_data/nan_%s.csv" % file_name[self.type_fill])



if __name__ == "__main__":

    # 0 - convert NaN to 0
    # 1 - ruthlessly drop NaN
    # 2 - mean imputation, NaN over 60% dropped
    for type_fill in range(0, 3):
        #directory that data is located in
        data = Data("../data")
        data.clean_nans(type_fill)
        data.scale_data(scale_outcomes=False)
        #data.plot_counts()
        data.write_to_csv()
