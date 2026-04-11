# %%
import DM_modes2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

act_grid = DM_modes2.make_hc_act_grid()
fourier, freqs_used = DM_modes2.fourier_basis(
    act_grid,
    min_freq_HO=1.1,
    max_freq_HO=5.01,
    spacing_HO=1.0,
    start_HO=0.0,
    orthogonalise=False,
    pin_edges=True,
)

plt.figure()
plt.scatter(freqs_used[:, 0] / (2 * np.pi), freqs_used[:, 1] / (2 * np.pi))

plt.figure()
DM_modes2.plot_basis_summary(fourier, -1)

plt.figure()
xcor = fourier.transformation_matrix.T @ fourier.transformation_matrix
plt.imshow(xcor, norm=mcolors.CenteredNorm(), cmap="RdBu")
plt.colorbar()

# %%
import hcipy

hcipy.imshow_field(fourier[0])

# %%
basis = fourier.transformation_matrix


def rms(x, aperture=None):
    if aperture is not None:
        x = x[aperture]
    return np.sqrt(np.mean(x**2))


rms(basis[:, 3])

# aperture is circle of radius 5 actuators
aperture = np.linalg.norm(act_grid.points, axis=1) < 0.5
plt.imshow(aperture.reshape(act_grid.shape).T)

rms(basis[:, 0], aperture=aperture), rms(basis[:, 3], aperture=aperture)