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
    count = np.sum(
        (freqs_used[:, 0] / (2 * np.pi)) ** 2 + (freqs_used[:, 1] / (2 * np.pi)) ** 2
        >= r_min**2
    )
    count -= np.sum(
        (freqs_used[:, 0] / (2 * np.pi)) ** 2 + (freqs_used[:, 1] / (2 * np.pi)) ** 2
        >= r_max**2
    )
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

# hcipy.imshow_field(fourier[0])

# coeffs = 0.02*np.random.randn(fourier.num_modes)
coeffs = np.zeros(fourier.num_modes)
coeffs[0] = 0.3
coeffs[3] = 0.3
hcipy.imshow_field(fourier.linear_combination(coeffs))
plt.colorbar()

"""
Add an explicit slope limiter. Or if too difficult, a Laplacian limiter. i.e. a sparse matrix:

    f_k,k is the value of the k,kth actuator.

    Laplacian = -4*f_k,k  + f_k,k+1 + f_k,k-1 + f_k+1,k + f_k-1,k. If near an edge, replace f_k,k-1 with f_k,k+1.
    If Laplacian > L_max, add (Laplacian - L_max)/4.

"""
# %%
import matplotlib.colors as mcolors

coeffs = 0.02 * np.random.randn(fourier.num_modes)

coeffs = np.zeros(fourier.num_modes)
coeffs[0] = -0.5
coeffs[3] = -0.3


def laplacian_limiter(surface, L_max, return_L=False):
    # surface is a 2D array of actuator values
    # we will modify surface in place
    new_surface = surface.copy()

    laplacian = (
        -4 * surface[1:-1, 1:-1]
        + surface[1:-1, 2:]
        + surface[1:-1, :-2]
        + surface[2:, 1:-1]
        + surface[:-2, 1:-1]
    )

    new_surface[1:-1, 1:-1] += np.clip((laplacian - L_max) / 4, 0, None)
    # and the opposite for negative Laplacian
    new_surface[1:-1, 1:-1] += np.clip((laplacian + L_max) / 4, None, 0)

    # retain pinning
    new_surface[0, 1:-1] = new_surface[1, 1:-1]
    new_surface[-1, 1:-1] = new_surface[-2, 1:-1]
    new_surface[1:-1, 0] = new_surface[1:-1, 1]
    new_surface[1:-1, -1] = new_surface[1:-1, -2]

    if return_L:
        L_values = np.zeros_like(surface)
        L_values[1:-1, 1:-1] = laplacian
        return new_surface, L_values
    return new_surface


original = fourier.linear_combination(coeffs).reshape(act_grid.shape)
filtered, L = laplacian_limiter(original, L_max=0.1, return_L=True)

plt.subplot(131)
plt.imshow(original, norm=mcolors.CenteredNorm(), cmap="RdBu")
plt.subplot(132)
plt.imshow(filtered, norm=mcolors.CenteredNorm(), cmap="RdBu")
plt.subplot(133)
plt.imshow(original - filtered, norm=mcolors.CenteredNorm(), cmap="bwr")
plt.colorbar()

plt.figure()
plt.imshow(L, norm=mcolors.CenteredNorm(), cmap="RdBu")

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
