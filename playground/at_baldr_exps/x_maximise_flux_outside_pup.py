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

beam = 1


# %%
def mds_connect(host: str, port: int = 5555, timeout_ms: int = 5000):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{port}")
    return ctx, sock


def mds_send(sock, msg: str) -> str:
    sock.send_string(msg)
    return sock.recv_string().strip()


ctx, sock = mds_connect("mimir")

# %%
dm = dmclass(beam)

cam = Bcam(beam)


# %%
mds_send(sock, "off SBB")
# mds_send(sock, "b_shut close all")
time.sleep(3)
# %%
cam.take_dark(256)
plt.imshow(cam.dark)
plt.colorbar()
# %%
# mds_send(sock, "b_shut open all")
mds_send(sock, "on SBB")

# %%
offset = 200.0
mds_send(sock, f"moverel BMX{beam} {offset}")
mds_send(sock, f"moverel BMY{beam} {offset}")
time.sleep(1)


pupil_only = cam.take_stack(1000).mean(0)

mds_send(sock, f"moverel BMX{beam} {-offset}")
mds_send(sock, f"moverel BMY{beam} {-offset}")
time.sleep(1)
# %%
plt.imshow(pupil_only)
# %%
# make a pupil mask. Need to find the pixels in the circle
n_act = 12
n_beam = 10

act_grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
cam_grid = hcipy.make_pupil_grid(32, diameter=32)


def smooth_circle(grid, radius, softening=0.1, centre=(0, 0)):
    r = np.sqrt((grid.x - centre[0]) ** 2 + (grid.y - centre[1]) ** 2)
    return 1 / (1 + np.exp((r - radius) / softening))


def xcor_sum_model(params, args):
    img, grid, softening = args
    img /= np.sum(img)
    model = smooth_circle(
        grid, radius=params[0], softening=softening, centre=(params[1], params[2])
    ).reshape(grid.shape)
    model /= model.sum()
    return -np.sum(img * model)


def xcor_sum(params, args):
    (img,) = args
    img /= np.sum(img)


ideal_pupil = smooth_circle(cam_grid, radius=10, softening=0.5)

hcipy.imshow_field(ideal_pupil, cam_grid)
# %%
res = opt.minimize(
    xcor_sum_model,
    x0=[8, 0, 0],
    args=((pupil_only, cam_grid, 0.5),),
    bounds=((8, 8), (-10, 10), (-10, 10)),
)

pupil_mask = smooth_circle(
    cam_grid, radius=res.x[0], softening=0.5, centre=(res.x[1], res.x[2])
).reshape(32, 32)
pupil_center = (res.x[1], res.x[2])

# %%
plt.imshow(pupil_only)
plt.contour(pupil_mask, levels=[0.5], color="r")


# %%
# pupil_mask =
scattered_flux_mask_r_outer = 12
scattered_flux_mask_r_inner = 9.5
scattered_flux_mask = (
    smooth_circle(
        cam_grid, scattered_flux_mask_r_outer, centre=pupil_center, softening=0.01
    )
    - smooth_circle(
        cam_grid, scattered_flux_mask_r_inner, centre=pupil_center, softening=0.01
    )
).reshape(cam_grid.shape)

zern_in_img = cam.take_stack(256).mean(0)

# plt.imshow(scattered_flux_mask)
plt.figure()
plt.imshow(pupil_only)
plt.contour(scattered_flux_mask, levels=[0.5], colors="r")
plt.contour(scattered_flux_mask, ":", levels=[0.1], colors="w")
# %%
act_reg_mask = 1 - smooth_circle(act_grid, radius=0.5, softening=0.03)
act_reg_mask /= act_reg_mask.sum()

hcipy.imshow_field(act_reg_mask, grid=act_grid)
plt.colorbar()



# %%
def L1_masked_reg(cmd, mask):
    # the l1 regularisation term, evaluated with weights given by the mask
    return np.sum(mask * np.abs(cmd))


def flux_outside_pupil(img, scatter_mask):
    return np.sum(img * scatter_mask)


def uniformity_in_pupil(img, pupil_mask):
    img_in_pupil = img * pupil_mask
    mean_in_pupil = np.sum(img_in_pupil) / np.sum(pupil_mask)
    # want a uniform distribution in the pupil, so penalise the variance
    return np.sqrt(np.sum(pupil_mask * (img_in_pupil - mean_in_pupil) ** 2))


def loss(cmd, lamb_unif, lamb_reg, scatter_mask, act_mask):
    dm.set_data(cmd)
    time.sleep(0.01)
    img = cam.take_stack(64).mean(0)

    f = flux_outside_pupil(img, scatter_mask=scatter_mask)
    u = uniformity_in_pupil(img, pupil_mask=pupil_mask)
    l1 = L1_masked_reg(cmd, act_mask)
    l = float(-f + lamb_unif * u + lamb_reg * l1)
    print(np.sqrt(np.mean(cmd**2)), f"{l:.3f}")
    return l


init_cmd = np.zeros(144)
scattered_flux_mask /= scattered_flux_mask.sum()
act_reg_mask /= act_reg_mask.sum()

dm.set_data(init_cmd)
time.sleep(0.01)
img = cam.take_stack(64).mean(0)

pupil_hard_mask = pupil_mask>0.6

flux_outside_pupil(img, scatter_mask=scattered_flux_mask), uniformity_in_pupil(img, pupil_hard_mask), L1_masked_reg(
    init_cmd, act_reg_mask
)

# %%
# while True:
#     print(loss(init_cmd, 10.0, scattered_flux_mask, act_reg_mask),end="\r")
#     time.sleep(0.01)
print(loss(init_cmd, 0.1, 10.0, scattered_flux_mask, act_reg_mask))
# %%
loss(np.random.randn(144) * 0.02, 0.1, 0.0, scattered_flux_mask, act_reg_mask)

# %%
res = opt.minimize(
    loss,
    init_cmd,
    (10.0, scattered_flux_mask, act_reg_mask),
    method="Powell",
    options={"disp": True},
    bounds=[[-0.25, 0.25] for _ in range(144)],
)
# %%
plt.imshow(res.x.reshape(12, 12))

# %%
# n_zern = 9
# zern = hcipy.make_zernike_basis(n_zern, 1.5, act_grid, starting_mode=2)

# # hcipy.imshow_field(zern[2])
# hcipy.imshow_field(zern.linear_combination(np.random.randn(n_zern) * 0.01), act_grid)


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


fourier_small = fourier_basis(2)

n_offset_modes = fourier_small.num_modes
# %%
hcipy.imshow_field(fourier_small[0])


# %%
def basis_loss(coeffs, basis, lamb_unif, scatter_mask, act_mask, scale=0.05):
    coeffs_scaled = coeffs * scale
    cmd = basis.linear_combination(coeffs_scaled)
    return loss(cmd, lamb_unif, 0.0, scatter_mask, act_mask)

# %%
res = opt.minimize(
    basis_loss,
    np.zeros(n_offset_modes),
    (fourier_small, 0.5, scattered_flux_mask, act_reg_mask),
    method="COBYLA",
    options={"disp": True, "maxiter": 50},
    # bounds=[[-0.05, 0.05] for _ in range(n_offset_modes)],
)
# res = opt.minimize(
#     basis_loss,
#     np.zeros(n_offset_modes),
#     (fourier_small, 0.0, scattered_flux_mask, act_reg_mask),
#     method="Powell",
#     options={"disp": True, "maxiter":10},
#     bounds=[[-0.15, 0.15] for _ in range(n_offset_modes)],
# )
# %%
sol = res.x.copy()
basis_loss(sol, fourier_small, 0.0, scattered_flux_mask, act_reg_mask)
len(sol)


# %%
fourier_middle = fourier_basis(4)

n_terms = fourier_middle.num_modes
init_coeffs = fourier_middle.coefficients_for(fourier_small.linear_combination(sol))
init_coeffs

# %%
hcipy.imshow_field(fourier_small.linear_combination(sol))
plt.colorbar()
# %%
hcipy.imshow_field(fourier_middle.linear_combination(init_coeffs))
plt.colorbar()

# %%
basis_loss(
    init_coeffs * 5, fourier_middle, 0.0, scattered_flux_mask, act_reg_mask, 0.01
)
# %%

res = opt.minimize(
    basis_loss,
    init_coeffs * 5,
    (fourier_middle, 0.2, scattered_flux_mask, act_reg_mask, 0.01),
    method="COBYLA",
    options={"disp": True, "maxiter": 120},
    # bounds=[[-0.05, 0.05] for _ in range(n_offset_modes)],
)
# %%
basis_loss(res.x, fourier_middle, 0.2, scattered_flux_mask, act_reg_mask, 0.01)
# %%
basis_loss(np.zeros(len(res.x)), fourier_middle, 0.2, scattered_flux_mask, act_reg_mask, 0.01)


# %%
sol = res.x.copy()
fourier_large = fourier_basis(6)

n_terms = fourier_large.num_modes
init_coeffs = fourier_large.coefficients_for(fourier_middle.linear_combination(sol))

# %%

res = opt.minimize(
    basis_loss,
    init_coeffs,
    (fourier_large, 0.3, scattered_flux_mask, act_reg_mask, 0.01),
    method="COBYLA",
    options={"disp": True, "maxiter": 320},
    # bounds=[[-0.05, 0.05] for _ in range(n_offset_modes)],
)
# %%
basis_loss(res.x, fourier_large, 0.2, scattered_flux_mask, act_reg_mask, 0.01)

# %%
flat_img = cam.take_stack(256).mean(0)
# %%
# np.savez("beam3_good_flat2.npz",
#          flat=fourier_large.linear_combination(res.x),
#          flat_img=flat_img,
#          n_fourier_modes=8,
#          lamb_unif=0.3)

# %%

img = cam.take_stack(256).mean(0)
plt.imshow(img*(pupil_mask>0.5))

# %%
dm.set_data(np.zeros(144))

# %%
# basis of cutting the pupil grid into sub squares, starting from 4 squares total
n_squares_accross = 2


def block_basis(n_squares_accross, act_grid):
    n = int(n_squares_accross)
    if n < 1:
        raise ValueError("n_squares_accross must be >= 1")

    x = np.asarray(act_grid.x)
    y = np.asarray(act_grid.y)

    x_edges = np.linspace(np.min(x), np.max(x), n + 1)
    y_edges = np.linspace(np.min(y), np.max(y), n + 1)

    modes = []
    for iy in range(n):
        y_in = (y >= y_edges[iy]) & (
            y <= y_edges[iy + 1] if iy == n - 1 else y < y_edges[iy + 1]
        )

        for ix in range(n):
            x_in = (x >= x_edges[ix]) & (
                x <= x_edges[ix + 1] if ix == n - 1 else x < x_edges[ix + 1]
            )
            mode = (x_in & y_in).astype(float)
            modes.append(mode)

    return hcipy.ModeBasis(np.column_stack(modes), act_grid)


square_basis_2 = block_basis(2, act_grid)
hcipy.imshow_field(square_basis_2[3])

plt.figure()
square_basis_2 = block_basis(4, act_grid)
hcipy.imshow_field(square_basis_2[0])
plt.figure()
square_basis_2 = block_basis(6, act_grid)
hcipy.imshow_field(square_basis_2[0])

# %%

n_squares = [2, 3, 4]
n_max_iters = [60, 120, 480]
init_coeffs = None
for n, max_it in zip(n_squares, n_max_iters):
    square_basis = block_basis(n, act_grid)
    n_modes = square_basis.num_modes

    if init_coeffs is None:
        init_coeffs = np.zeros(n_modes)
    else:
        init_coeffs = square_basis.coefficients_for(
            prev_square_basis.linear_combination(res.x)
        )

    basis_loss(init_coeffs,square_basis, 0.0, scattered_flux_mask, act_reg_mask, 0.01)

    time.sleep(5)
    res = opt.minimize(
        basis_loss,
        init_coeffs,
        (square_basis, 0.1, scattered_flux_mask, act_reg_mask, 0.01),
        method="COBYLA",
        options={"disp": True, "maxiter": max_it},
        # bounds=[[-0.05, 0.05] for _ in range(n_offset_modes)],
    )

    prev_square_basis = square_basis

# %%
