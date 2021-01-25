#RNA-ATAC multi-modal model specification
import os
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset
from torchvision.utils import save_image, make_grid

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.colors
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix

from vis import plot_embeddings, plot_kls_df, embed_umap
from .mmvae import MMVAE
from .vae_rna import RNA
from .vae_atac import ATAC

scale_factor = 10000
modal = ['r', 'm']

class RNA_ATAC(MMVAE):
    def __init__(self, params):
        prior = dist.Laplace
        super(RNA_ATAC, self).__init__(prior, params, RNA, ATAC)
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'rna-atac'

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1) 

    def getDataLoaders(self, datasets, batch_size, shuffle, drop_last, device='cuda'):
        datasets_rna_atac = TensorDataset(datasets)

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        dataloader = DataLoader(datasets_rna_atac, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs) #Shuffle here
        
        return dataloader
    
    def forward(self, x):
        qz_xs, zss = [], []
        read_counts = []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            read_counts.append(vae.enc.read_count(x[m]))
            qz_x, px_z, zs = vae(x[m])
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    if d == 0:
                        r, p = vae.dec(zs)
                        r = r / scale_factor * read_counts[d]
                        px_zs[e][d] = vae.px_z(r, p)
                    else:
                        r, p, g = vae.dec(zs)
                        r = r / scale_factor * read_counts[d]
                        px_zs[e][d] = vae.px_z(r, p, g)

        return qz_xs, px_zs, zss

    def reconstruct(self, data, train_test, runPath, sampling=False, N=1):
        if not sampling:
            recons_mat = super(RNA_ATAC, self).reconstruct(data)
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    _data = data[r].cpu()
                    recon = recon.squeeze(0).cpu().detach().numpy()                    
                    recon = csr_matrix(recon)
                    mmwrite('{}/{}_recon_{}x{}.mtx'.format(runPath, train_test, modal[r], modal[o]), recon)
        else:
            for n in range(N):
                recons_mat = super(RNA_ATAC, self).reconstruct_sample(data)
                for r, recons_list in enumerate(recons_mat):
                    for o, recon in enumerate(recons_list):
                        _data = data[r].cpu()
                        recon = recon.squeeze(0).cpu().detach().numpy()
                        recon = csr_matrix(recon)
                        mmwrite('{}/{}_recon_{}x{}.mtx'.format(runPath, train_test, modal[r], modal[o]), recon)
    
    def predict(self, data, sampling=False, N=1):
        if not sampling:
            recons_mat = super(RNA_ATAC, self).reconstruct(data)
        else:
            recons_mat = super(RNA_ATAC, self).reconstruct_sample(data)
        return(recons_mat)    

    def analyse(self, data, runPath, epoch, K=1):
        zemb, zsl, kls_df = super(RNA_ATAC, self).analyse(data, K=K)
        labels = ['Prior', *[vae.modelName.lower() for vae in self.vaes]]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch), yscale='log')
    
    
    def get_latent(self, data, train_test, runPath, sampling=False):
        lats = super(RNA_ATAC, self).latents(data, sampling)
        for m,lat in enumerate(lats):
            lat = lat.cpu().detach().numpy()
            lat = pd.DataFrame(lat)
            lat.to_csv('{}/lat_{}_{}.csv'.format(runPath, train_test, modal[m]))
        
        mean_lats = sum(lats)/len(lats)
        mean_lats = mean_lats.cpu().detach().numpy()
        mean_lats = pd.DataFrame(mean_lats)
        mean_lats.to_csv('{}/lat_{}_mean.csv'.format(runPath,train_test))
        
    
    def plot_klds(self, data, runPath):
        kls_df = super(RNA_ATAC, self).kls_df(data)
        plot_kls_df(kls_df, '{}/kl_distance.png'.format(runPath), yscale='linear')
    
    def traverse(self, runPath):
        traverse_path = runPath + '/traverse'
        traverse_dir = Path(traverse_path)
        traverse_dir.mkdir(parents=True, exist_ok=True)

        mu = self._pz_params[0].cpu().detach().numpy()
        var = torch.exp(self._pz_params[1]).cpu().detach().numpy()
        sd = np.sqrt(var)
        strt = -10
        stp = 10
        for i in range(strt,stp):
            adj_mu = mu + sd * 0.5 * i
            adj = adj_mu if i == -10 else np.vstack([adj,adj_mu])
        
        mu_ = np.tile(mu,(len(range(strt,stp)),1))

        #traverse_list = []
        for i in range(self.params.latent_dim):
            adj_dim = adj[:,i]
            traverse = np.copy(mu_)
            traverse[:,i] = np.copy(adj_dim)
            
            adj_dim = pd.DataFrame(adj_dim)
            adj_dim.to_csv(traverse_path + '/traverse_dim{}.csv'.format(i+1)) #from python to R index 

            zs = torch.from_numpy(traverse).to(device)
            px_zs = []
            for m, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.dec(zs))
                px_zs.append(px_z)  
            r_traverse = px_zs[0].mean.cpu().detach().numpy()
            p_traverse = px_zs[1].mean.cpu().detach().numpy()
            
            r_traverse = pd.DataFrame(r_traverse.numpy())
            r_traverse.to_csv(traverse_path + '/rna_traverse_dim{}.csv'.format(i+1))
            
            p_traverse = pd.DataFrame(p_traverse.numpy())
            p_traverse.to_csv(traverse_path + '/atac_traverse_dim{}.csv'.format(i+1))
