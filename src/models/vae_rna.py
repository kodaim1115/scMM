# RNA model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import prod, sqrt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from scipy.cluster.hierarchy import linkage
from sklearn.manifold import TSNE

from utils import Constants
from vis import plot_embeddings, plot_kls_df, embed_umap
from .vae import VAE
from datasets import RNA_Dataset

scale_factor = 10000

# Classes
class Enc(nn.Module):

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim): #added hidden_dim
        super(Enc, self).__init__()
        self.data_dim = data_dim
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
    
    def read_count(self, x):
        read = torch.sum(x, axis=1)
        read = read.repeat(self.data_dim, 1).t()
        return(read)

    def forward(self, x):
        read = self.read_count(x)
        x = x / read * scale_factor
        e = self.enc(x)
        lv = self.fc22(e).clamp(-12,12) #restrict to avoid torch.exp() over/underflow
        return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta 

class Dec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim): #added hidden_dim
        super(Dec, self).__init__()
        self.data_dim = data_dim
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True)))
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)
        self.fc32 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        log_r = self.fc31(d).clamp(-12,12) #restrict to avoid torch.exp() over/underflow
        r = torch.exp(log_r)
        p = self.fc32(d)
        p = torch.sigmoid(p).clamp(Constants.eps, 1 - Constants.eps) #restrict to avoid probs = 0,1
        return r, p


class RNA(VAE):
    """ Derive a specific sub-class of a VAE for RNA. """

    def __init__(self, params):
        super(RNA, self).__init__(
            dist.Laplace,
            dist.NegativeBinomial, #likelihood
            dist.Laplace,
            Enc(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim),
            Dec(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'rna'
        self.data_dim = self.params.r_dim
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(dataset, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)

        return dataloader
    
    def forward(self, x):
        read_count = self.enc.read_count(x)
        self._qz_x_params = self.enc(x)
        qz_x = self.qz_x(*self._qz_x_params)
        zs = qz_x.rsample()
        r, _ = self.dec(zs)
        r = r / scale_factor * read_count 
        px_z = self.px_z(r, _)
        return qz_x, px_z, zs