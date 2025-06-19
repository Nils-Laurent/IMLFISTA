#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2025
@author: Nils Laurent
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv


import torch
from torchvision.io import read_image
from multilevel.minimal_wrapper import MultilevelSolver, PlotSLvsML
from multilevel.multilevel import create_information_transfer, create_grad_prior
from multilevel.multilevel_initialization import ml_pnp_init_with_solver
import deepinv as dinv

device = 'cpu'

# Load image from deepinv
x = dinv.utils.load_url_image(url=dinv.utils.get_image_url("butterfly.png"), img_size=256).to(device)

# Define linear operator
physics = dinv.physics.Inpainting(
    tensor_size=x.shape[1:], mask=0.5, device=device, noise_model=dinv.physics.GaussianNoise(0.1)
)

# generate measurement
y = physics(x)
back = physics.A_adjoint(y)
model = MultilevelSolver(x, y, physics, device=device)

# ML PnP paper settings
denoiser = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download")
model.IMLPNP_solver.denoiser = dinv.models.EquivariantDenoiser(denoiser, random=True)
model.IMLPNP_solver.args_multilevel.denoiser = model.IMLPNP_solver.denoiser
model.IMLPNP_solver.args_multilevel.grad_prior = create_grad_prior(model.IMLPNP_solver.prior, denoiser, device)
model.IMLPNP_solver.data_fidelity = dinv.optim.L2()
model.IMLPNP_solver.max_iter = 20
model.IMLPNP_solver.levels = 3  # was 4, but the image is small
model.IMLPNP_solver.args_multilevel.levels = model.IMLPNP_solver.levels
model.IMLPNP_solver.max_ML_steps = 2
model.IMLPNP_solver.args_multilevel.max_ML_steps = model.IMLPNP_solver.max_ML_steps
model.IMLPNP_solver.info_transfer = "sinc"
model.IMLPNP_solver.args_multilevel.information_transfer = create_information_transfer(model.IMLPNP_solver.info_transfer, device)
model.IMLPNP_solver.args_multilevel.step_size = 1.0
model.IMLPNP_solver.regularization = 0.0801

# gamma : ML PnP does not depend on gamma

# run
with torch.no_grad():
    init = back.clone()
    F_init, PSNR_init = model.compute_metrics_at_init(init)
    print("init in progress ...")
    model.IMLPNP_solver.coarse_iter = 5
    model.IMLPNP_solver.args_multilevel.param_coarse_iter = model.IMLPNP_solver.coarse_iter
    ml_init = ml_pnp_init_with_solver(init, model.IMLPNP_solver)
    _, PSNR_ML_init = model.compute_metrics_at_init(ml_init)
    print("solver is running ...")
    model.IMLPNP_solver.coarse_iter = 3
    model.IMLPNP_solver.args_multilevel.param_coarse_iter = model.IMLPNP_solver.coarse_iter
    x_IMLPNP, metrics_IMLPNP = model.IMLPNP_solver.solve(ml_init)
    _, PSNR_out = model.compute_metrics_at_init(x_IMLPNP)
    print("done.")

dinv.utils.plot(
    [x, y, ml_init, x_IMLPNP],
    titles=[
        "Original",
        f"Measurements PSNR = {PSNR_init:.2f}dB",
        f"ML init PSNR = {PSNR_ML_init:.2f}dB",
        f"IMLPNP - PSNR = {PSNR_out:.2f}dB"
    ],
    figsize=[10, 3]
)

# Tracer les courbes de PSNR
