# Base MMVAE class definition

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN

from utils import get_mean, kl_divergence, Constants
from vis import embed_umap, tensors_to_df
import copy

class MMVAE(nn.Module):
    def __init__(self, prior_dist, params, *vaes):
        super(MMVAE, self).__init__()
        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])
        self.modelName = None  # filled-in per sub-class
        self.params = params
        self._pz_params = None  # defined in subclass

    @property
    def pz_params(self):
        return self._pz_params

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def get_cluster(self, data, modality='both', n_clusters=10, method='kmeans', device='cuda'):
        self.eval()
        lats = self.latents(data, sampling=False)
        if modality=='both':
            lat = sum(lats)/len(lats)
        elif modality=='rna':
            lat = lats[0]
        elif modality=='atac':
            lat = lats[1]
        
        if method=='kmeans':
            fit = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++').fit(lat.cpu().numpy())
            cluster = fit.labels_
        elif method=='dbscan':
            fit = DBSCAN(eps=0.5, min_samples=50).fit(lat.cpu().numpy())
            cluster = fit.labels_
        else:
            gamma, _, _, _, _ = self.get_gamma(lat)
            cluster = torch.argmax(gamma, axis=1)
            cluster = cluster.detach().numpy()
            fit = None

        return cluster, fit

    def forward(self, x):
        qz_xs, zss = [], []
        # initialise cross-modal matrix
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m])
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z  # fill-in diagonal
        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal
                    px_zs[e][d] = vae.px_z(*vae.dec(zs))
        return qz_xs, px_zs, zss

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions of MEANS
            recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
        return recons

    def reconstruct_sample(self, data):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(data)
            # cross-modal matrix of reconstructions of SAMPLES
            recons = [[px_z.sample() for px_z in r] for r in px_zs]
        return recons
    
    def latents(self, data, sampling=False):
        self.eval()
        with torch.no_grad():
            qz_xs, _, _ = self.forward(data)
            if not sampling:
                lats = [get_mean(qz_x) for qz_x in qz_xs]
            else:
                lats = [qz_x.sample() for qz_x in qz_xs]
        return lats
    
    def var(self, data):
        self.eval()
        with torch.no_grad():
            qz_xs, _, _ = self.forward(data)
            v = [qz_x.variance for qz_x in qz_xs]
        return v

    def analyse(self, data, K=1):
        self.eval()
        with torch.no_grad():
            qz_xs, _, zss = self.forward(data)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return embed_umap(torch.cat(zss, 0).cpu().numpy()), \
            torch.cat(zsl, 0).cpu().numpy(), \
            kls_df
    
    def kls_df(self, data):
        self.eval()
        with torch.no_grad():
            K = 1
            qz_xs, _, zss = self.forward(data)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, data[0].size(0)])).view(-1, pz.batch_shape[-1]),
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            #zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        return kls_df
    
    
