#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025
@author: Guillaume Lauga
"""

#pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv


import torch
from torchvision.io import read_image
import deepinv as dinv
from multilevel.minimal_wrapper import MultilevelSolver, PlotSLvsML

device = 'cpu'

# Load image from local
# image_path = "path_to_image/image.png"  # Replace with your image path
# image_file = read_image(image_path)
# x = image_file.unsqueeze(0).to(torch.float32).to(device)/255

# Load image from deepinv
x = dinv.utils.load_url_image(url=dinv.utils.get_image_url("butterfly.png"), img_size=256).to(device)
# Reduce image size for testing
# x = x[:, :, ::4, ::4]

# Define the Forward Operator: study case of deblurring + Gaussian noise
#-----------------------------------------------------------------------
# Load a forward operator $A$ and generate some (noisy) measurements.
# The full list of operators is available here:
# (https://deepinv.github.io/deepinv/deepinv.physics.html).

# The forward operator needs to live in the image domain (i.e. the observation is an image), e.g., a blurring operator. The multilevel solver that is used here is not suited for Tomography or other forward operators that do not live in the image domain.

# Define linear operator
filter_0 = dinv.physics.blur.gaussian_blur(sigma=(4, 4), angle=0.0)
physics = dinv.physics.Blur(filter_0, device=device, padding='reflect', noise_model=dinv.physics.GaussianNoise(0.01))

# generate measurement
y = physics(x)
back = physics.A_adjoint(y)
model = MultilevelSolver(x, y, physics, device=device)

# run
with torch.no_grad():
    init = back.clone()
    F_init, PSNR_init = model.compute_metrics_at_init(init)
    x_hat, F_min, PSNR_hat   = model.compute_minimum(init)
    init = back.clone()
    x_IMLFB, metrics_IMLFB = model.IMLFB(init)
    init = back.clone()
    x_IMLFISTA, metrics_IMLFISTA = model.IMLFISTA(init)
    init = back.clone()
    x_FB, metrics_FB = model.FB(init)
    init = back.clone()
    x_FISTA, metrics_FISTA = model.FISTA(init)
    init = back.clone()
    x_PNP, metrics_PNP = model.PNP(init)
    init = back.clone()
    x_IMLPNP, metrics_IMLPNP = model.IMLPNP(init)


PlotSLvsML(x, y, F_init, PSNR_init, x_hat, F_min, PSNR_hat, x_FB=x_FB, metrics_FB=metrics_FB, x_IMLFB=x_IMLFB, metrics_IMLFB=metrics_IMLFB, x_FISTA=x_FISTA, metrics_FISTA=metrics_FISTA, x_IMLFISTA=x_IMLFISTA, metrics_IMLFISTA=metrics_IMLFISTA, x_PNP=x_PNP, metrics_PNP=metrics_PNP, x_IMLPNP=x_IMLPNP, metrics_IMLPNP=metrics_IMLPNP)