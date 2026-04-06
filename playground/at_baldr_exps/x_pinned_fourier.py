# %%
import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime

from bcam import Bcam

from astropy.io import fits
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hcipy
from asgard_alignment import FLI_Cameras as FLI
import scipy.optimize as opt

# %%
n_act = 12
n_beam = 10

act_grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
cam_grid = hcipy.make_pupil_grid(32, diameter=32)

# %%


def fourier_basis(n_modes):
    max_freq = n_modes  # lambda/D
    freqs = hcipy.make_pupil_grid(
        n_modes,
        max_freq,
    )

    basis = hcipy.make_fourier_basis(act_grid, freqs.scaled(2 * np.pi))

    fourier = basis

    # if odd number of modes, remove piston
    if n_modes % 2 == 1:
        fourier = hcipy.ModeBasis(fourier.transformation_matrix[:, 1:], act_grid)
    return fourier


def pin_outer_edge(basis):
    """
    Pin the outermost row and column of pixels of each basis mode
    to the value of the mode inwards from it
    """
    transformation_matrix = basis.transformation_matrix.copy()
    for i in range(transformation_matrix.shape[1]):
        mode = transformation_matrix[:, i].reshape(act_grid.shape)
        mode[0, :] = mode[1, :]
        mode[-1, :] = mode[-2, :]
        mode[:, 0] = mode[:, 1]
        mode[:, -1] = mode[:, -2]
        transformation_matrix[:, i] = mode.flatten()
    return hcipy.ModeBasis(transformation_matrix, act_grid)


# %%
fourier_full = fourier_basis(4)

fourier_pinned = pin_outer_edge(fourier_full)

# %%
idx = 5
plt.subplot(121)
hcipy.imshow_field(fourier_full[idx])
plt.subplot(122)
hcipy.imshow_field(fourier_pinned[idx])

# %%
