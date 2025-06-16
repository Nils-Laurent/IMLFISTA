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
# model.IMLPNP_solver.denoiser = dinv.models.DRUNet(in_channels=3, out_channels=3, pretrained="download")

# ML PnP custom settings
model.max_iter = 20

# run
with torch.no_grad():
    init = back.clone()
    F_init, PSNR_init = model.compute_metrics_at_init(init)
    print("init in progress ...")
    ml_init = ml_pnp_init_with_solver(init, model.IMLPNP_solver)
    _, PSNR_ML_init = model.compute_metrics_at_init(ml_init)
    print("solver is running ...")
    x_IMLPNP, metrics_IMLPNP = model.IMLPNP(ml_init)
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
