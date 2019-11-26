import os
import pandas as pd

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
    

    #write data to csv file to be nice 
    def write_to_csv(self, version): 
        self.train.to_csv("../data/training_set_%s.csv" % version)
        self.test.to_csv("../data/test_set_%s.csv" % version)



if __name__ == "__main__":

    #directory that data is located in
    data = Data("../data")
    data.clean_nans()
    data.normalize_data()
    data.write_to_csv("V1")
