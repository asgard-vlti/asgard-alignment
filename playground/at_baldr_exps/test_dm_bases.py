# %%
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt
import numpy as np

import hcipy

# %%
n_act = 12
n_beam = 10

grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
n_modes = 6
max_freq = n_modes  # lambda/D
probe_max_freq = max_freq

freqs = hcipy.make_pupil_grid(
    n_modes,
    max_freq,
)

print(len(freqs.x), freqs.x)

basis = hcipy.make_fourier_basis(grid, freqs.scaled(2 * np.pi))
basis.transformation_matrix.shape

hc_fourier = basis.transformation_matrix
indices_to_remove = [0, 11, 132, 143]
hc_fourier = np.delete(hc_fourier, indices_to_remove, axis=0)
hc_fourier.shape
# %%
plt.imshow(dmbases.get_DM_command_in_2D(hc_fourier[:, 5]))


# %%
xcor = hc_fourier.T @ hc_fourier
plt.imshow(xcor, vmin=np.min(xcor), vmax=np.max(xcor), cmap="bwr")
