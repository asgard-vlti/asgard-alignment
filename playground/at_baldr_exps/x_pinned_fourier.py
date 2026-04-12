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

radii = [1.1, 2.1, 3.1, 4.1, 5.1]
# circles of different radii to show the spacing
for r in radii:
    circle = plt.Circle((0, 0), r, color="k", fill=False, ls="--")
    plt.gca().add_patch(circle)

plt.axis("equal")

# print the number of modes in each annulus
for i in range(len(radii) - 1):
    r_min = radii[i]
    r_max = radii[i + 1]
    count = np.sum((freqs_used[:, 0] / (2 * np.pi)) ** 2 + (freqs_used[:, 1] / (2 * np.pi)) ** 2 >= r_min**2)
    count -= np.sum((freqs_used[:, 0] / (2 * np.pi)) ** 2 + (freqs_used[:, 1] / (2 * np.pi)) ** 2 >= r_max**2)
    print(f"Annulus {i}: {count} modes")

# %%
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
