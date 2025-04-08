import pandas as pd
import numpy as np
from azureml.core import Workspace, Dataset

subscription_id = 'dbbc963f-2f1e-4e47-b215-c33824a3bb74'
resource_group = 'wcd-detection-aml-eus'
workspace_name = 'dtc-aml-exp-eus'
data_path = r'/home/azureuser/cloudfiles/code/Users/t-omelamed/exai/firstrail/TestData_22_8.tsv'

class DataIterator():
    def __init__(self, df, batch_size=1) -> None:
        self.batch_size = batch_size
        self.df= df
        # self.indxs = np.random.permutation(len(self.df))

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        # res = self.df.iloc[self.indxs[self.n:self.n+self.batch_size]]
        res = self.df.iloc[self.n:self.n+self.batch_size]
        self.n = self.n+self.batch_size
        if self.n > len(self.df):
            self.n =0
        # print("next", len(res))
        return res


class TestData():
    def __init__(self, test_data_path=None, minial_malware_len = 500) -> None:
        if test_data_path is None:
            self.test_data = pd.read_csv(open(data_path,'r', encoding='utf-8'), sep='\t', header=None, error_bad_lines=False)
        else:
            # self.test_data = pd.read_csv(open(test_data_path,'r', encoding='utf-8'), sep=',', error_bad_lines=False, index_col=False)
            self.test_data = pd.read_csv(open(test_data_path,'r', encoding='utf-8'), sep=',', index_col=False)
            self.test_data = self.test_data.drop(self.test_data.columns[[0]], axis=1)

        # print(self.test_data.head())
        self.test_data.columns = ["label", "features"]
        self.malware_test_data = self.test_data[self.test_data["label"] == 1]
        self.malware_test_data=self.malware_test_data[(self.malware_test_data.features.astype(str).str.len()>minial_malware_len)]

    def get_data_iterator(self, batch_size=1):
        return DataIterator(self.test_data, batch_size)
    
    def get_malware_data_iterator(self, batch_size=1):
        return DataIterator(self.malware_test_data, batch_size)
    
    def get_balanced_data_iterator(self, batch_size=1):
        nMax = len(self.malware_test_data)
        balanced_data = self.test_data.groupby('label').apply(lambda x: x.sample(n=min(nMax, len(x))))
        return DataIterator(balanced_data, batch_size)

