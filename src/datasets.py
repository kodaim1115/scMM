import time
import os
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler
import torch
from torch.utils.data import Dataset

class RNA_Dataset(Dataset):
    """
    Single-cell RNA/ADT dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.genes, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nGene number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')
        
        

def load_data(path, transpose=False):
    print("Loading  data ...")
    t0 = time.time()
    if os.path.isdir(path):
        count, peaks, barcode = read_mtx(path)
    elif os.path.isfile(path):
        count, peaks, barcode = read_csv(path)
    else:
        raise ValueError("File {} not exists".format(path))
        
    if transpose: 
        count = count.transpose()
    print('Original data contains {} cells x {} peaks'.format(*count.shape))
    assert (len(barcode), len(peaks)) == count.shape
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return count, peaks, barcode


def read_mtx(path):
    for filename in glob(path+'/*'):
        basename = os.path.basename(filename)
        if (('count' in basename) or ('matrix' in basename)) and ('mtx' in basename):
            count = mmread(filename).T.tocsr().astype('float32')
        elif 'barcode' in basename:
            if ('.txt' in basename) or ('tsv' in basename):
                sep = '\t'
            elif '.csv' in basename:
                sep = ','
            barcode = pd.read_csv(filename, sep=sep, header=None)[0].values
        elif 'gene' in basename or 'peak' in basename or 'protein' in basename:
            if ('.txt' in basename) or ('tsv' in basename):
                sep = '\t'
            elif '.csv' in basename:
                sep = ','
            feature = pd.read_csv(filename, sep=sep, header=None).iloc[:, -1].values

    return count, feature, barcode
    

def read_csv(path):
    if ('.txt' in path) or ('tsv' in path):
        sep = '\t'
    elif '.csv' in path:
        sep = ','
    else:
        raise ValueError("File {} not in format txt or csv".format(path))
    data = pd.read_csv(path, sep=sep, index_col=0).T.astype('float32')
    genes = data.columns.values
    barcode = data.index.values
    return scipy.sparse.csr_matrix(data.values), genes, barcode


class ATAC_Dataset(Dataset):
    """
    Single-cell ATAC dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.peaks, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')

