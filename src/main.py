import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset

import math
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, EarlyStopping, Constants, log_mean_exp, is_multidata, kl_divergence
from datasets import RNA_Dataset, ATAC_Dataset

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset


parser = argparse.ArgumentParser(description='scMM Hyperparameters')
parser.add_argument('--experiment', type=str, default='test', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='rna_protein', metavar='M',
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='m_elbo_naive_warmup', metavar='O',
                    help='objective to use (default: elbo)')
parser.add_argument('--llik_scaling', type=float, default=1.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                    help='learning rate (default: 1e-3)')                   
parser.add_argument('--latent_dim', type=int, default=10, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num_hidden_layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--r_hidden_dim', type=int, default=100, 
                    help='number of hidden units in enc/dec for gene')
parser.add_argument('--p_hidden_dim', type=int, default=20, 
                    help='number of hidden units in enc/dec for protein/peak')
parser.add_argument('--pre_trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn_prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--analytics', action='store_true', default=True,
                    help='disable plotting analytics')
parser.add_argument('--print_freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset_path', type=str, default="")
parser.add_argument('--r_dim', type=int, default=1)
parser.add_argument('--p_dim', type=int, default=1)
parser.add_argument('--deterministic_warmup', type=int, default=50, metavar='W',
                    help='deterministic warmup')

# args
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set up run path
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('../experiments/' + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
print(runPath)

#Data
dataset_path = args.dataset_path
if args.model == 'rna_atac':
    modal = 'ATAC-seq'
elif args.model == 'rna_protein':
    modal = 'CITE-seq'

rna_path = dataset_path + '/RNA-seq'
modal_path = dataset_path + '/{}'.format(modal)

r_dataset = RNA_Dataset(rna_path) 
args.r_dim = r_dataset.data.shape[1] 

modal_dataset = ATAC_Dataset(modal_path) if args.model == 'rna_atac' else RNA_Dataset(modal_path)
args.p_dim = modal_dataset.data.shape[1]

print("RNA-seq shape is " + str(r_dataset.data.shape))
print("{} shape is ".format(modal) + str(modal_dataset.data.shape)) 

#Split train test
if args.pre_trained:
    pretrained_path = args.pre_trained
    t_id = torch.load(pretrained_path + '/t_id.rar')
    s_id = torch.load(pretrained_path + '/s_id.rar')
else:
    num_cell = r_dataset.data.shape[0]
    t_size = np.round(num_cell*0.80).astype('int')
    t_id = np.random.choice(a=num_cell, size=t_size, replace=False) 
    s_id = np.delete(range(num_cell),t_id)
    torch.save(t_id, runPath + '/t_id.rar')
    torch.save(s_id, runPath + '/s_id.rar')

train_dataset = [Subset(r_dataset, t_id), Subset(modal_dataset, t_id)]
test_dataset = [Subset(r_dataset, s_id), Subset(modal_dataset, s_id)]

t_id = pd.DataFrame(t_id)
t_id.to_csv('{}/t_id.csv'.format(runPath))
s_id = pd.DataFrame(s_id)
s_id.to_csv('{}/s_id.csv'.format(runPath))

# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    pretrain_args = args
    #pretrain_args.learn_prior = False

    #Load model
    modelC = getattr(models, 'VAE_{}'.format(pretrain_args.model))
    model = modelC(pretrain_args).to(device)
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

else:
    # load model
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    print(args)

    model = modelC(args).to(device)
    torch.save(args,runPath+'/args.rar')

#Dataloader
train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)
test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, amsgrad=True)
objective = getattr(objectives,args.obj) 
s_objective = getattr(objectives,args.obj)

def train(epoch, agg, W):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        beta = (epoch - 1) / W  if epoch <= W else 1
        if dataT[0].size()[0] == 1:
            continue
        data = [d.to(device) for d in dataT] #multimodal
        optimizer.zero_grad()
        loss = -objective(model, data, beta)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))
    return b_loss

def test(epoch, agg, W):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            beta = (epoch - 1) / W  if epoch <= W else 1
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT]
            loss = -s_objective(model, data, beta)
            b_loss += loss.item()
    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True) 
        W = args.deterministic_warmup
        start_early_stop = W
        for epoch in range(1, args.epochs + 1):
            b_loss = train(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                break          
            test(epoch, agg, W)
            save_vars(agg, runPath + '/losses.rar')
            if epoch > start_early_stop: 
                early_stopping(agg['test_loss'][-1], model, runPath)
            if early_stopping.early_stop:
                print('Early stopping')
                break

    if args.analytics:
        def get_latent(dataloader, train_test, runPath):
            model.eval()
            with torch.no_grad():
                if args.model == 'rna_atac':
                    modal = ['rna', 'atac'] 
                elif args.model == 'rna_protein':
                    modal = ['rna', 'protein']
                pred = []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    lats = model.latents(data, sampling=False)
                    if i == 0:
                        pred = lats
                    else:
                        for m,lat in enumerate(lats):
                            pred[m] = torch.cat([pred[m], lat], dim=0) 
            
                for m,lat in enumerate(pred):
                    lat = lat.cpu().detach().numpy()
                    lat = pd.DataFrame(lat)
                    lat.to_csv('{}/lat_{}_{}.csv'.format(runPath, train_test, modal[m]))
                mean_lats = sum(pred)/len(pred)
                mean_lats = mean_lats.cpu().detach().numpy()
                mean_lats = pd.DataFrame(mean_lats)
                mean_lats.to_csv('{}/lat_{}_mean.csv'.format(runPath,train_test))

        def predict(dataloader, train_test, runPath):
            model.eval()
            with torch.no_grad():
                uni, cross = [], []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    recons_mat = model.reconstruct_sample(data)
                    for e, recons_list in enumerate(recons_mat):
                        for d, recon in enumerate(recons_list):
                            if e == d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                recon = pd.DataFrame(recon)
                                recon = sp.csr_matrix(recon)
                                if i == 0:
                                    uni.append(recon)
                                else:
                                    uni[e] = sp.vstack((uni[e], recon), format='csr')
                            if e != d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                recon = pd.DataFrame(recon)
                                recon = sp.csr_matrix(recon)
                                if i == 0:
                                    cross.append(recon)
                                else:
                                    cross[e] = sp.vstack((cross[e], recon), format='csr')
                
                mmwrite('{}/pred_{}_r_r.mtx'.format(runPath, train_test), uni[0])
                mmwrite('{}/pred_{}_p_p.mtx'.format(runPath, train_test), uni[1])
                mmwrite('{}/pred_{}_r_p.mtx'.format(runPath, train_test), cross[0])
                mmwrite('{}/pred_{}_p_r.mtx'.format(runPath, train_test), cross[1])
        
        train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
        test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

        get_latent(train_loader, 'train', runPath)
        get_latent(test_loader, 'test', runPath)

        predict(test_loader, 'test', runPath)
        
        model.traverse(runPath, device                   )


