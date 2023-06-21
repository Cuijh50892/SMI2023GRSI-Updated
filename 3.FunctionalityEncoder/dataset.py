import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
import glob2
import random

class ManifoldDataset(data.Dataset):
    def __init__(self,
                 split='S',
                 ):
        self.split = split
        file_name = 'codes' + split + '.csv'
        self.datalist = np.loadtxt(file_name, delimiter=",").astype(np.float32)

    def __getitem__(self, index):
        code = self.datalist[index, :]
        code = torch.from_numpy(code)
        return code

    def __len__(self):
        return len(self.datalist)

class VisualizeDataset(data.Dataset):
    def __init__(self,
                 split='S',
                 ):
        self.split = split
        file_name = 'v' + split + '.csv'
        self.datalist = np.loadtxt(file_name, delimiter=",").astype(np.float32)

    def __getitem__(self, index):
        code = self.datalist[index, :]
        code = torch.from_numpy(code)
        return code

    def __len__(self):
        return len(self.datalist)

class PairDataset(data.Dataset):
    def __init__(self,
                 ):
        file_name = 'pair.csv'
        self.datalist = np.loadtxt(file_name, delimiter=",").astype(np.int)

    def __getitem__(self, index):
        pair = self.datalist[index, :]
        pair = torch.from_numpy(pair).long()
        return pair

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    d = ManifoldDataset(split='S')
    c = d[0]
    print(c.size(), c.type())

