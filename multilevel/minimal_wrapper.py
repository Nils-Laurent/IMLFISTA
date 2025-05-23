#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
"""
import deepinv as dinv
import torch
import torch.nn as nn
from torchvision.io import read_image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F
import copy
from deepinv.loss.metric import PSNR
from multilevel.multilevel import ParametersMultilevel, MultiLevel
perf_psnr = PSNR()

class MultilevelSolver():
    def __init__(self, x_true, y, physics, device='cpu'):
        super(MultilevelSolver, self).__init__()
        self.x = x_true
        self.y = y
        self.physics = physics
        self.device = device
        self.max_iter = 50
        self.Config = Config(self.x, self.y, self.physics, self.device)
        self.IMLFB_solver = IMLFB_solver(self.Config)
        self.IMLFISTA_solver = IMLFISTA_solver(self.Config)
        self.FB_solver = FB_solver(self.Config)
        self.FISTA_solver = FISTA_solver(self.Config)
        self.PNP_solver = PNP_solver(self.Config)
        self.IMLPNP_solver = IMLPNP_solver(self.Config)
    def compute_minimum(self, init_iter, max_iter = 300):
        return self.FISTA_solver.compute_minimum(init_iter, max_iter)
    def compute_metrics_at_init(self, init_iter):
        xk = init_iter.clone()
        F_init = self.FISTA_solver.data_fidelity(xk, self.y, self.physics) + self.FISTA_solver.regularization * self.FISTA_solver.prior.fn(xk)
        PSNR_init = perf_psnr(xk,self.x).item()
        return F_init, PSNR_init
    def FB(self, init_iter):
        return self.FB_solver.solve(init_iter, max_iter = self.max_iter)
    def IMLFB(self, init_iter):
        return self.IMLFB_solver.solve(init_iter, max_iter = self.max_iter)
    def FISTA(self, init_iter):
        return self.FISTA_solver.solve(init_iter, max_iter = self.max_iter)
    def IMLFISTA(self, init_iter):
        return self.IMLFISTA_solver.solve(init_iter, max_iter = self.max_iter)
    def PNP(self, init_iter):
        return self.PNP_solver.solve(init_iter, max_iter = self.max_iter)
    def IMLPNP(self, init_iter):
        return self.IMLPNP_solver.solve(init_iter, max_iter = self.max_iter)
    
class Config():
    def __init__(self, x, y, physics, device):
        if x.shape[1]>3:
            raise ValueError("Input image must have at most 3 channels for Plug and Play denoisers")
        self.device = device
        self.physics = physics
        self.y = y
        self.x = x
        self.data_fidelity = dinv.optim.L2()
        self.prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=150)
        self.denoiser = self.prior.prox
        if isinstance(self.physics.noise_model, dinv.physics.GaussianNoise):
            self.regularization = 2*self.physics.noise_model.sigma**2
        else:
            raise ValueError("Noise model not supported")
        self.gamma = torch.ones(1, device= device)/ self.physics.compute_norm(torch.randn(self.x.shape).to(device))
        self.max_iter = 5


class IMLFB_solver():
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=150)
        self.denoiser = self.prior.prox
        self.regularization = config.regularization
        self.gamma = config.gamma*1.95
        self.max_iter = config.max_iter
        self.gamma_ML = self.gamma
        self.levels = 3
        self.coarse_iter = 5
        self.max_ML_steps = 10
        self.info_transfer = "daubechies8"
        self.args_multilevel = ParametersMultilevel(target_shape = self.x.shape[-3:], levels = self.levels, max_ML_steps = 1, param_coarse_iter = self.coarse_iter, step_size=self.gamma_ML, info_transfer=self.info_transfer, prior=self.prior, denoiser=self.denoiser, data_fidelity= self.data_fidelity, physics = self.physics, observation = self.y, device = self.device)
    def solve(self, init_iter, max_iter = None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        cst_grad = None
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                if k<self.max_ML_steps:
                    xk = MultiLevel(xk, self.levels, self.levels-1, self.args_multilevel, self.regularization, cst_grad, self.device)
                xk = xk - self.gamma*self.data_fidelity.grad(xk, self.y, self.physics)
                xk = self.prior.prox(xk, gamma = self.regularization*self.gamma)
                metrics.crit[k] = self.data_fidelity(xk, self.y, self.physics) + self.regularization * self.prior.fn(xk)
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"IMLFB[{k+1}]: crit {metrics.crit[k]} / psnr {metrics.psnr[k]}")
        return xk, metrics

class FB_solver():
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=150)
        self.denoiser = self.prior.prox
        self.regularization = config.regularization
        self.gamma = config.gamma*1.95
        self.max_iter = config.max_iter
    def solve(self, init_iter, max_iter=None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                xk = xk - self.gamma*self.data_fidelity.grad(xk, self.y, self.physics)
                xk = self.prior.prox(xk, gamma = self.regularization*self.gamma)
                metrics.crit[k] = self.data_fidelity(xk, self.y, self.physics) + self.regularization * self.prior.fn(xk)
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"FB[{k+1}]: crit {metrics.crit[k]} / psnr {metrics.psnr[k]}")
        return xk, metrics

class IMLFISTA_solver():
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=150)
        self.denoiser = self.prior.prox
        self.regularization = config.regularization
        self.gamma = config.gamma*0.95
        self.max_iter = config.max_iter
        self.a = 3
        self.gamma_ML = config.gamma*1.95
        self.levels = 3
        self.coarse_iter = 5
        if self.x.shape[2]<=256 or self.x.shape[3]<=256:
            self.max_ML_steps = 5
        else:
            self.max_ML_steps = 10
        self.info_transfer = "daubechies8"
        self.args_multilevel = ParametersMultilevel(target_shape = self.x.shape[-3:], levels = self.levels, max_ML_steps = 1, param_coarse_iter = self.coarse_iter, step_size=self.gamma_ML, info_transfer=self.info_transfer, prior=self.prior, denoiser=self.denoiser, data_fidelity= self.data_fidelity, physics = self.physics, observation = self.y, device = self.device)
    def solve(self, init_iter, max_iter=None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        zk = init_iter.clone()
        cst_grad = None
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                xk_prev = xk.clone()
                if k<self.max_ML_steps:
                    zk = MultiLevel(zk, self.levels, self.levels-1, self.args_multilevel, self.regularization, cst_grad, self.device)
                xk = zk - self.gamma*self.data_fidelity.grad(zk, self.y, self.physics)
                xk = self.prior.prox(xk, gamma = self.regularization*self.gamma)
                metrics.crit[k] = self.data_fidelity(xk, self.y, self.physics) + self.regularization * self.prior.fn(xk)
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"IML FISTA[{k+1}]: crit {metrics.crit[k]} / psnr {metrics.psnr[k]}")
                zk = xk + (k) / (k+1+self.a) * (xk - xk_prev)
        return xk, metrics
    
class FISTA_solver():
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.prior = dinv.optim.TVPrior(def_crit=1e-6, n_it_max=150)
        self.denoiser = self.prior.prox
        self.regularization = config.regularization
        self.gamma = config.gamma*0.95
        self.max_iter = config.max_iter
        self.a = 3
    def solve(self, init_iter, max_iter=None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        zk = init_iter.clone()
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                xk_prev = xk.clone()
                xk = zk - self.gamma*self.data_fidelity.grad(zk, self.y, self.physics)
                xk = self.prior.prox(xk, gamma = self.regularization*self.gamma)
                metrics.crit[k] = self.data_fidelity(xk, self.y, self.physics) + self.regularization * self.prior.fn(xk)
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"FISTA[{k+1}]: crit {metrics.crit[k]} / psnr {metrics.psnr[k]}")
                zk = xk + (k) / (k+1+self.a) * (xk - xk_prev)
        return xk, metrics
    def compute_minimum(self, init_iter, max_iter=300):
        xk, metrics = self.solve(init_iter, max_iter, display = False)
        return xk, metrics.crit[-1], metrics.psnr[-1]

class PNP_solver(): # Plug and Play
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.denoiser = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz", device=self.device)
        self.prior = dinv.optim.prior.PnP(denoiser=self.denoiser)
        self.regularization = config.regularization
        self.gamma = config.gamma*0.95
        self.max_iter = config.max_iter
    def solve(self, init_iter, max_iter=None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                xk_prev = xk.clone()
                xk = xk - self.gamma*self.data_fidelity.grad(xk, self.y, self.physics)
                xk = self.denoiser(xk, self.regularization*self.gamma)
                metrics.crit[k] = torch.linalg.norm(xk.flatten()-xk_prev.flatten())
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"PNP[{k+1}]: xk-xprev {metrics.crit[k]} / psnr {metrics.psnr[k]}")
        return xk, metrics

class IMLPNP_solver():
    def __init__(self, config):
        self.device = config.device
        self.y = config.y
        self.x = config.x
        self.physics = config.physics
        self.data_fidelity = config.data_fidelity
        self.denoiser = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz", device=self.device)
        self.prior = dinv.optim.prior.PnP(denoiser=self.denoiser)
        self.regularization = config.regularization
        self.gamma = config.gamma*0.95
        self.max_iter = config.max_iter
        self.gamma_ML = config.gamma*1.95
        self.levels = 3
        self.coarse_iter = 5
        self.max_ML_steps = 5
        self.info_transfer = "daubechies8"
        self.args_multilevel = ParametersMultilevel(target_shape = self.x.shape[-3:], levels = self.levels, max_ML_steps = 1, param_coarse_iter = self.coarse_iter, step_size=self.gamma_ML, info_transfer=self.info_transfer, prior=self.prior, denoiser=self.denoiser, data_fidelity= self.data_fidelity, physics = self.physics, observation = self.y, device = self.device)
    def solve(self, init_iter, max_iter=None, display=True):
        if max_iter is None:
            max_iter = self.max_iter
        xk = init_iter.clone()
        cst_grad = None
        metrics = Metrics(max_iter)
        metrics.crit = 1e10*np.ones(max_iter)
        metrics.psnr = 1e10*np.ones(max_iter)
        with torch.no_grad():
            for k in range(max_iter):
                xk_prev = xk.clone()
                if k<self.max_ML_steps:
                    xk = MultiLevel(xk, self.levels, self.levels-1, self.args_multilevel, self.regularization, cst_grad, self.device)
                xk = xk - self.gamma*self.data_fidelity.grad(xk, self.y, self.physics)
                xk = self.denoiser(xk, self.regularization*self.gamma)
                metrics.crit[k] = torch.linalg.norm(xk.flatten()-xk_prev.flatten())
                metrics.psnr[k] = perf_psnr(xk,self.x).item()
                if display and k % 10 == 0: 
                    print(f"IMLPNP[{k+1}]: xk-xprev {metrics.crit[k]} / psnr {metrics.psnr[k]}")
        return xk, metrics
class Metrics():
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.crit = 1e10*np.ones(self.max_iter)
        self.psnr = 1e10*np.ones(self.max_iter)

def PlotSLvsML(x, y, F_init, PSNR_init, x_hat, F_min, PSNR_hat, x_FB=None, metrics_FB=None, x_IMLFB=None, metrics_IMLFB=None, x_FISTA=None, metrics_FISTA=None, x_IMLFISTA=None, metrics_IMLFISTA=None, x_PNP=None, metrics_PNP=None, x_IMLPNP=None, metrics_IMLPNP=None):
    if metrics_FB is None and metrics_FISTA is None and metrics_IMLFB is None and metrics_IMLFISTA is None and metrics_PNP is None and metrics_IMLPNP is None:
        raise ValueError("No metrics provided for FB, IMLFB, FISTA, IMLFISTA, PNP or IMLPNP \n Run the algorithm first")
    best_x, best_psnr, best_alg = identify_best_reconstruction(x_FB=x_FB, x_FISTA=x_FISTA, x_IMLFB=x_IMLFB, x_IMLFISTA=x_IMLFISTA, x_PNP=x_PNP, x_IMLPNP=x_IMLPNP, metrics_FB=metrics_FB, metrics_FISTA=metrics_FISTA, metrics_IMLFB=metrics_IMLFB, metrics_IMLFISTA=metrics_IMLFISTA, metrics_PNP=metrics_PNP, metrics_IMLPNP=metrics_IMLPNP)
    dinv.utils.plot([x, y, best_x], titles=["Original","Measurements PSNR = {:.2f}dB".format(PSNR_init),f"Best alg: {best_alg} - PSNR = {best_psnr:.2f}dB"],figsize=[10,3], save_fn="images.png") 
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
    if x_FB is None and x_FISTA is None and x_IMLFB is None and x_IMLFISTA is None:
        raise ValueError("No x_FB, x_FISTA, x_IMLFB or x_IMLFISTA provided")
    if metrics_FB is None and metrics_IMLFB is None and metrics_FISTA is None and metrics_IMLFISTA is None:
        raise ValueError("No metrics provided for FB, IMLFB, FISTA or IMLFISTA")
    if x_FB is not None and metrics_FB is not None:
        axs[0].plot(np.concatenate((np.array(F_init),metrics_FB.crit))/F_init-F_min/F_init, label='FB')
    if x_IMLFB is not None and metrics_IMLFB is not None:
        axs[0].plot(np.concatenate((np.array(F_init),metrics_IMLFB.crit))/F_init-F_min/F_init, label='IMLFB')
    if x_FISTA is not None and metrics_FISTA is not None:
        axs[0].plot(np.concatenate((np.array(F_init),metrics_FISTA.crit))/F_init-F_min/F_init, label='FISTA')
    if x_IMLFISTA is not None and metrics_IMLFISTA is not None:
        axs[0].plot(np.concatenate((np.array(F_init),metrics_IMLFISTA.crit))/F_init-F_min/F_init, label='IMLFISTA')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].set_title('F-F* w.r.t. iter')
    if x_FB is not None and metrics_FB is not None:
        axs[1].plot(np.concatenate((np.array(F_init),metrics_FB.crit)), label='FB')
    if x_IMLFB is not None and metrics_IMLFB is not None:
        axs[1].plot(np.concatenate((np.array(F_init),metrics_IMLFB.crit)), label='IMLFB')
    if x_FISTA is not None and metrics_FISTA is not None:
        axs[1].plot(np.concatenate((np.array(F_init),metrics_FISTA.crit)), label='FISTA')
    if x_IMLFISTA is not None and metrics_IMLFISTA is not None:
        axs[1].plot(np.concatenate((np.array(F_init),metrics_IMLFISTA.crit)), label='IMLFISTA')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_title('F w.r.t. iter')
    if x_FB is not None and metrics_FB is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_FB.psnr)), label='FB')
    if x_IMLFB is not None and metrics_IMLFB is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_IMLFB.psnr)), label='IMLFB')
    if x_FISTA is not None and metrics_FISTA is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_FISTA.psnr)), label='FISTA')
    if x_IMLFISTA is not None and metrics_IMLFISTA is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_IMLFISTA.psnr)), label='IMLFISTA')
    if x_PNP is not None and metrics_PNP is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_PNP.psnr)), label='PNP')
    if x_IMLPNP is not None and metrics_IMLPNP is not None:
        axs[2].plot(np.concatenate((np.array([PSNR_init]),metrics_IMLPNP.psnr)), label='IMLPNP')
    axs[2].legend()
    axs[2].set_title('PSNR w.r.t. iter')
    plt.tight_layout()
    plt.savefig("metrics.png")

def identify_best_reconstruction(x_FB=None, x_FISTA = None, x_IMLFB = None, x_IMLFISTA = None, x_PNP = None, x_IMLPNP = None, metrics_FB=None, metrics_IMLFB=None, metrics_FISTA=None, metrics_IMLFISTA=None, metrics_PNP=None, metrics_IMLPNP=None):
    if metrics_FB is None and metrics_FISTA is None and metrics_IMLFB is None and metrics_IMLFISTA is None and metrics_PNP is None and metrics_IMLPNP is None:
        raise ValueError("No metrics provided for FB, IMLFB, FISTA, IMLFISTA, PNP or IMLPNP \n Run the algorithm first")
    if metrics_FB is not None:
        PSNR_FB = metrics_FB.psnr[-1]
    else:
        PSNR_FB = None
    if metrics_IMLFB is not None:
        PSNR_IMLFB = metrics_IMLFB.psnr[-1]
    else:
        PSNR_IMLFB = None
    if metrics_FISTA is not None:
        PSNR_FISTA = metrics_FISTA.psnr[-1]
    else:
        PSNR_FISTA = None
    if metrics_IMLFISTA is not None:
        PSNR_IMLFISTA = metrics_IMLFISTA.psnr[-1]
    else:
        PSNR_IMLFISTA = None
    if metrics_PNP is not None:
        PSNR_PNP = metrics_PNP.psnr[-1]
    else:
        PSNR_PNP = None
    if metrics_IMLPNP is not None:
        PSNR_IMLPNP = metrics_IMLPNP.psnr[-1]
    else:
        PSNR_IMLPNP = None
    values = [PSNR_FB, PSNR_IMLFB, PSNR_FISTA, PSNR_IMLFISTA, PSNR_PNP, PSNR_IMLPNP]
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    i_max, v_max = max(valid, key=lambda x: x[1])
    if i_max == 0:
        return x_FB, PSNR_FB, 'FB'
    elif i_max == 1:
        return x_IMLFB, PSNR_IMLFB, 'IMLFB'
    elif i_max == 2:
        return x_FISTA, PSNR_FISTA, 'FISTA'
    elif i_max == 3:
        return x_IMLFISTA, PSNR_IMLFISTA, 'IMLFISTA'
    elif i_max == 4:
        return x_PNP, PSNR_PNP, 'PNP'
    elif i_max == 5:
        return x_IMLPNP, PSNR_IMLPNP, 'IMLPNP'
    