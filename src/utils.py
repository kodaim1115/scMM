import math
import os
import shutil
import sys
import time

import torch
import torch.distributions as dist
import torch.nn.functional as F

# Classes
class Constants(object):
    eta = 1e-6
    eps =1e-8
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    #if hasattr(model, 'vaes'):
    #    for vae in model.vaes:
    #        fdir, fext = os.path.splitext(filepath)
    #        save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def vade_kld_uni(model, zs):
    n_centroids = model.params.n_centroids
    gamma, lgamma, mu_c, var_c, pi = model.get_gamma(zs) #pi, var_cは get_gammaでConstants.eta足してる
    
    mu, var = model._qz_x_params 
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    var_expand = var.unsqueeze(2).expand(var.size(0), var.size(1), n_centroids) 
    lpz_c = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           var_expand/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1) # log p(z|c)
    lpc = torch.sum(gamma*torch.log(pi), dim=1) # log p(c) #log(pi)が-inf怪しい
    lqz_x = -0.5*torch.sum(1+torch.log(var)+math.log(2*math.pi), dim=1) #see VaDE paper # log q(z|x)
    lqc_x = torch.sum(gamma*(lgamma), dim=1) # log q(c|x)
    
    kld = -lpz_c - lpc + lqz_x + lqc_x 
    
    return kld

def vade_kld(model, zs, r):
    n_centroids = model.params.n_centroids
    gamma, lgamma, mu_c, var_c, pi = model.get_gamma(zs) #pi, var_cは get_gammaでConstants.eta足してる
    
    mu, var = model.vaes[r]._qz_x_params 
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    var_expand = var.unsqueeze(2).expand(var.size(0), var.size(1), n_centroids) 
    lpz_c = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           var_expand/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1) # log p(z|c)
    lpc = torch.sum(gamma*torch.log(pi), dim=1) # log p(c) #log(pi)が-inf怪しい
    lqz_x = -0.5*torch.sum(1+torch.log(var)+math.log(2*math.pi), dim=1) #see VaDE paper # log q(z|x)
    lqc_x = torch.sum(gamma*(lgamma), dim=1) # log q(c|x)
    
    kld = -lpz_c - lpc + lqz_x + lqc_x 
    
    return kld

def pdist(sample_1, sample_2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]


class FakeCategorical(dist.Distribution):
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, locs):
        self.logits = locs
        self._batch_shape = self.logits.shape

    @property
    def mean(self):
        return self.logits

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.logits.expand([*sample_shape, *self.logits.shape]).contiguous()

    def log_prob(self, value):
        # value of shape (K, B, D)
        lpx_z = -F.cross_entropy(input=self.logits.view(-1, self.logits.size(-1)),
                                 target=value.expand(self.logits.size()[:-1]).long().view(-1),
                                 reduction='none',
                                 ignore_index=0)

        return lpx_z.view(*self.logits.shape[:-1])
        # it is inevitable to have the word embedding dimension summed up in
        # cross-entropy loss ($\sum -gt_i \log(p_i)$ with most gt_i = 0, We adopt the
        # operationally equivalence here, which is summing up the sentence dimension
        # in objective.


#from github Bjarten/early-stopping-pytorch
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, runPath):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, runPath)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, runPath) #runPath追加
            self.counter = 0

    def save_checkpoint(self, val_loss, model, runPath):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), 'checkpoint.pt')
        save_model(model, runPath + '/model.rar') #mmvaeより移植
        self.val_loss_min = val_loss


class EarlyStopping_nosave:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score =  -1e9
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, runPath):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
