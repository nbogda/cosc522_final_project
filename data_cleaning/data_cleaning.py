import os
import pandas as pd

#class to read in and clean data
class Data:

    def __init__(self, data_path):

        self.train = pd.read_csv("%s/TrainingDataset.csv" % data_path)
        self.test = pd.read_csv("%s/TestDataset.csv" % data_path)













if __name__ == "__main__":

    #directory that data is located in
    data = Data("../data")
