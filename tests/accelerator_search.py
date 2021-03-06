import matplotlib.pyplot as plt

import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.distributions.multivariate_normal import MultivariateNormal

from botorch.acquisition import UpperConfidenceBound, PosteriorMean
from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ScalarizedObjective
import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from advanced_botorch import binary_constraint
from advanced_botorch import proximal
from advanced_botorch import combine_acquisition

import models
import training


#-------------------------------------------------
#test out proximal ucb + constraint on simple test function
#-------------------------------------------------


def main():
    sigma_mult = [0.1**2, 0.5**2]
    
    n_init = 5
    x_init = torch.rand(n_init, 2)*0.5 + 0.25


    fig,ax = plt.subplots(len(sigma_mult),1)
    fig2,ax2 = plt.subplots()

    if isinstance(ax,np.ndarray):
        for a in ax.flatten():
            a.set_ylim(0,1)
            a.set_xlim(0,1)
            a.set_aspect('equal')
            
    else:
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
    
    for i in range(len(sigma_mult)):
        X, Y, mlls, cmlls = optimize(x_init, 35, torch.eye(2) * sigma_mult[i])
        ax[i].plot(X[:,0][:n_init], X[:,1][:n_init],'+C0')
        ax[i].plot(X[:,0][n_init:], X[:,1][n_init:],'-C1')
        ax[i].plot(X[:,0][-1], X[:,1][-1],'oC1')
        
        ax2.plot(np.exp(mlls) / np.arange(1,len(mlls)+1) )
        ax2.plot(np.exp(cmlls) / np.arange(1,len(cmlls)+1) )
    

def f(X):
    #Y = 1 - (X - 0.5).norm(dim=-1, keepdim=True)  # explicit output dimension
    #Y += 0.1 * torch.rand_like(Y)
    #Y = (Y - Y.mean()) / Y.std()
    mean = torch.tensor((0.4,0.4))
    sigma = torch.eye(2)
    sigma[0, 0] = 1.0**2
    sigma[1, 1] = 0.25**2
    d = MultivariateNormal(mean, sigma)
    Y = torch.exp(d.log_prob(X)) / torch.exp(d.log_prob(mean))

    #second index serves as the constraint
    #C = X[:,0].numpy() < 0.75
    C = np.all(np.vstack(((X[:,0] - 0.5)**2 + (X[:,1] - 0.5)**2 < 0.35**2,
                          X[:,1] < X[:,0])), axis = 0)
    
    C = torch.from_numpy(C).float()
    
    result = torch.cat((Y.reshape(-1,1),C.reshape(-1,1)), axis = 1)    
    return result


def optimize(x_initial, n_steps, sigma_matrix):
    train_X = x_initial
    train_Y = f(train_X)

    mlls = []
    cmlls = []
    for i in range(n_steps):
        print(i)

        #define gp model for objective(s)
        gp = SingleTaskGP(train_X, train_Y[:,0].reshape(-1,1))
        mll_val = fit_gp(gp)
        mlls += [mll_val]

        #define constraint GP
        cgp = SingleTaskGP(train_X, train_Y[:,1].reshape(-1,1))
        cmll_val = fit_gp(cgp)
        cmlls += [cmll_val]
        #get candidate for observation and add to training data
        if i % 10 == 0:
            plot = True
        else:
            plot = False
            
        candidate = max_acqf(gp, cgp, sigma_matrix, plot = plot)
        train_X = torch.cat((train_X, candidate))
        train_Y = f(train_X)

    #plot_model(cgp, lk, train_X)
        
    candidate = max_acqf(gp, cgp, sigma_matrix, plot = True)
    print(gp.covar_module.base_kernel.lengthscale)
    print(cgp.covar_module.base_kernel.lengthscale)

    return train_X, train_Y, np.array(mlls), np.array(cmlls)

    

def max_acqf(gp, cgp, sigma_matrix, plot = False):
    #finds new canidate point based on EHVI acquisition function

    constr = binary_constraint.BinaryConstraint(cgp)
    #constr = PosteriorMean(cgp)
    prox = proximal.ProximalAcqusitionFunction(gp, sigma_matrix)
    UCB = UpperConfidenceBound(gp, beta = 1e6)
    comb = combine_acquisition.MultiplyAcquisitionFunction(gp, [constr, prox, UCB])
    
    bounds = torch.stack([torch.zeros(2), torch.ones(2)])

    if plot:
        #plot_acq(comb, bounds.numpy().T, gp.train_inputs[0])
        plot_acq(constr, bounds.numpy().T, gp.train_inputs[0])
        #plot_acq(UCB, bounds.numpy().T, gp.train_inputs[0])
        
    candidate, acq_value = optimize_acqf(
        comb, bounds = bounds, q = 1, num_restarts = 10, raw_samples = 20)

    return candidate

        
def fit_gp(gp):
    #fits GP model
    mll = ExactMarginalLogLikelihood(gp.likelihood,
                                     gp)

    train_x = gp.train_inputs[0]
    train_y = gp.train_targets.reshape(-1,1)
    
    mll = fit_gpytorch_model(mll)

    gp.train()
    mll_val = mll(gp(train_x), train_y.flatten())
    return mll_val
    

def plot_model(model, lk, obs):
    fig, ax = plt.subplots()

    n = 25
    x = [np.linspace(0, 1, n) for e in [0,1]]
    xx,yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()
    
    with torch.no_grad():
        pred = lk(model(pts))
        f = pred.mean

    c = ax.pcolor(xx,yy,f.detach().reshape(n,n),vmin = 0.0, vmax = 1.0)
    ax.plot(*obs.detach().numpy().T,'+')

    fig.colorbar(c)

def plot_acq(func, bounds, obs):
    fig, ax = plt.subplots()

    n = 25
    x = [np.linspace(*bnds, n) for bnds in bounds]
    xx,yy = np.meshgrid(*x)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    pts = torch.from_numpy(pts).float()

    #print(pts)
    
    f = torch.zeros(n**2)
    for i in range(pts.shape[0]):
        f[i] = func(pts[i].reshape(1,-1))
    c = ax.pcolor(xx,yy,f.detach().reshape(n,n))
    ax.plot(*obs.detach().numpy().T,'+')

    ax.set_title(type(func))
    fig.colorbar(c)

main()
plt.show()

