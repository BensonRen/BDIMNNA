"""A module for a mixture density network layer

This file is copied from https://github.com/search?q=mixture+density+network+pytorch
MIT licensed

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
import numpy as np
#from numpy.random.Generator import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features*out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)
        self.cuda = True if torch.cuda.is_available() else False

    def forward(self, minibatch):
        #print('checki if there is nan in input:', torch.sum(torch.isnan(minibatch)))
        #print('input is: {}'.format(minibatch))
        print('size of input', minibatch.size())
        pi = self.pi(minibatch)
        #print('in forward function, pi = {}'.format(pi))
        print('self.sigma layer', self.sigma)
        sigma = self.sigma(minibatch)
        print('size of sigma', sigma.size())
        #sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features, self.out_features)
        #print('in forward function, sigma = {}'.format(sigma))
        mu = self.mu(minibatch)
        print('size of mu', mu.size())
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        #print('in forward function, mu = {}'.format(mu))
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target, eps=1e-5):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    eps = torch.tensor(eps, requires_grad=False)
    if torch.cuda.is_available():
        eps = eps.cuda()
    target = target.unsqueeze(1).expand_as(mu)
    #ll = torch.log(ONEOVERSQRT2PI / sigma) * (-0.5 * ((target - mu) / sigma)**2)
    #return torch.sum(ll, 2)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / (sigma + eps)
    #ret = 0.5*torch.matmul((target - mu)
    return torch.max(torch.prod(ret, 2), eps)

def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    #GP = gaussian_probability(sigma, mu, target)
    D = mu.size(-1)
    target = target.unsqueeze(1).expand_as(mu)
    loss = 0
    G = pi.size(1)
    B = pi.size(0)
    
    #########################
    # matrix implementation #
    #########################
    diff = target - mu
    #print('size of diff = ', diff.size())
    #print('size of sigma = ', sigma.size())
    # To make sigma symmetric add the transpose to itself
    precision_mat_diag_pos = torch.matmul(sigma.view([-1, D, D]),torch.transpose(sigma.view([-1, D, D]), 1, 2))
    #print('size of precision mat is', precision_mat_diag_pos.size())
    #precision_mat =  sigma.view([-1, D, D]) + torch.transpose(sigma.view([-1, D, D]), 1, 2)
   #diagonal_mat = torch.zeros(D,D).cuda()
    #precision_mat_diag_pos = precision_mat + diagonal_mat.fill_diagonal_(1 - torch.min(torch.diagonal(precision_mat)).detach().cpu().numpy())
    mul1 = torch.transpose(torch.diagonal(torch.matmul(diff.view([-1, D]), precision_mat_diag_pos)),0, 1)
    #print('size of mul1 = ', mul1.size())
    diff_t =  torch.transpose(diff.view([-1, D]),0, 1)
    #print('size of diff_t = ', diff_t.size())
    p_value =  torch.diagonal(torch.matmul(mul1,diff_t)).view([B, G])
    #print('size of p_value = ', p_value.size())
    #print('p_value', p_value)
    det_sigma = torch.abs(torch.det(precision_mat_diag_pos)).view([B,G])
    #det_sigma = torch.det(precision_mat_diag_pos).view([B,G])
    #print('deg_sigma', det_sigma)
    before_exp = torch.min(torch.log(pi) + 0.5*torch.log(det_sigma) - 0.5*p_value, 
                          other=torch.tensor(50.,requires_grad=False).cuda())     # capping to e^50
    #print('before_exp', before_exp)
    likelihood = torch.exp(before_exp)
    #print('likihood' , likelihood)
    #loss = torch.mean(-torch.log(torch.sum(likelihood, dim=1)))
    loss = torch.mean(-torch.log(torch.sum(likelihood, dim=1)+1e-6))
    #print('loss = ', loss)
    return loss
    """
    loss = torch.sum(0.5*p_value, dim=1)
    #print('size of det sigma', det_sigma.size())
    #print(det_sigma)
    sigma_term = -torch.sum(torch.log(pi*torch.sqrt(det_sigma.view([B, G]))), dim=1)
    #print('size of sigma term = ', sigma_term.size())
    #print('sigma term', sigma_term)
    loss += sigma_term
    mean_loss = torch.mean(loss)
    #print(mean_loss)
    #exit()
    return mean_loss
    #loss = torch.sum(torch.matmul(pi, p_value)
    """

    """
    ##########################
    # individual computation #
    ##########################
    for g in range(G):
        for b in range(B):
            diff = target[b, g, :] - mu[b, g, :]
            #print(diff.size())
            #print(sigma[:,g,:,:].size())
            p_value =  torch.matmul(torch.matmul(diff, sigma[b,g,:,:]),diff)# torch.transpose(diff))
            loss +=  0.5 * pi[b, g] * p_value                 # The diagonal part 
        loss += -torch.matmul(pi[:,g], torch.log(torch.sqrt(torch.det(sigma[:,g,:,:]))))
    print(loss.size())
    return loss
    #prob = pi*GP
    #prob = torch.log(pi)+ GP
    #print('pi part: {}, gaussian_part: {}'.format(pi, GP))
    #print('prob size = {}'.format(prob.size()))
    #for i in range(prob.size(1)):
    #    for j in range(prob.size(0)):
    #        print('prob {} = {}'.format(i, prob[j, i]))
    #print('sum(exp(prob)) = {}'.format(torch.sum(prob, dim=1)))
    #print('-log (sum(exp(prob)))={}'.format(-torch.log(torch.sum(prob, dim=1))))
    #print('mean = {}'.format(torch.mean(-torch.log(torch.sum(prob, dim=1)))))
    
    #nll =  -torch.log(torch.sum(prob, dim=1))
    #nll = -torch.sum(prob, dim=1)
    #nll = nll[torch.logical_not(torch.isinf(nll))]
    #nll = nll[torch.logical_not(torch.isnan(nll))]
    #print('mean = {}'.format(torch.mean(nll)))
    #return torch.mean(nll)
    """

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    # Original implementation
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
    """
    ######################
    # new implementation #
    ######################
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    #print('len of pis', len(pis))
    #print('pis', pis)
    print('size of sigma = ', sigma.size())
    print('size of mu = ', mu.size())
    D = mu.size(-1)
    samples = torch.zeros([len(pi), D])
    sigma_cpu_all = sigma.detach().cpu()
    mu_cpu_all = mu.detach().cpu()
    for i, idx in enumerate(pis):
        #print('i = {}'.format(i))
        sigma_cpu = sigma_cpu_all[i,idx]
        precision_mat_diag_pos = torch.matmul(sigma_cpu,torch.transpose(sigma_cpu,0,1))
        mu_cpu = mu_cpu_all[i, idx]
        #precision_mat = sigma[i, idx] + torch.transpose(sigma[i, idx], 0, 1)
        diagonal_mat = torch.tensor(np.zeros([D,D]))
        #precision_mat_diag_pos np.fill_diagonal_(diagonal_mat, 1e-7)
        precision_mat_diag_pos += diagonal_mat.fill_diagonal_(1)    # add small positive value
        #precision_mat_diag_pos = precision_mat + diagonal_mat.fill_diagonal_(1 - torch.min(torch.diagonal(precision_mat)).detach().cpu().numpy())
        #print('precision_mat = ', precision_mat_diag_pos)
        #print(precision_mat_diag_pos)
        #print(mu_cpu)
        try:
            #print('precision_mat = ', precision_mat_diag_pos)
            MVN = MultivariateNormal(loc=mu_cpu, precision_matrix=precision_mat_diag_pos)
            draw_sample = MVN.rsample()
        except:
            print("Ops, your covariance matrix is very unfortunately singular, assign loss of test_loss to avoid counting")
            draw_sample = -999*torch.ones([1, D])
        #print('sample size = ', draw_sample.size())
        samples[i, :] = draw_sample
    #print('samples', samples.size())
    return samples
def new_mdn_loss(pi, sigma, mu, target):
    """
    Copied from :
    https://github.com/sksq96/pytorch-mdn/blob/master/mdn.ipynb
    """
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(target))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)
