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

beam = 3


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
dm = dmclass(3)

cam = Bcam(3)


# %%
mds_send(sock, "off SBB")
time.sleep(3)
# %%
cam.take_dark(256)
plt.imshow(cam.dark)
plt.colorbar()
# %%
mds_send(sock, "on SBB")

offset = 200.0
mds_send(sock, f"moverel BMX{beam} {offset}")
mds_send(sock, f"moverel BMY{beam} {offset}")
time.sleep(1)


pupil_only = cam.take_stack(1000).mean(0)

mds_send(sock, f"moverel BMX{beam} {-offset}")
mds_send(sock, f"moverel BMY{beam} {-offset}")
time.sleep(1)
# %%
# make a pupil mask. Need to find the pixels in the circle
n_act = 12
n_beam = 10

act_grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
cam_grid = hcipy.make_pupil_grid(32, diameter=32)


def smooth_circle(grid, radius, softening=0.1, centre=(0, 0)):
    r = np.sqrt((grid.x - centre[0]) ** 2 + (grid.y - centre[1]) ** 2)
    return 1 / (1 + np.exp((r - radius) / softening))


def xcor_sum(params, args):
    img, grid, softening = args
    img /= np.sum(img)
    model = smooth_circle(
        grid, radius=params[0], softening=softening, centre=(params[1], params[2])
    )
    return -np.sum(img * model)


ideal_pupil = smooth_circle(cam_grid, radius=10, softening=0.5)

hcipy.imshow_field(ideal_pupil, cam_grid)
# %%
res = opt.minimize(
    xcor_sum,
    x0=[10, 0, 0],
    args=(pupil_only, cam_grid, 0.5),
    bounds=((5, 15), (-10, 10), (-10, 10)),
)

pupil_mask = smooth_circle(
    cam_grid, radius=res.x[0], softening=0.5, centre=(res.x[1], res.x[2])
)
pupil_center = (res.x[1], res.x[2])


# %%


mask = 1 - smooth_circle(act_grid, radius=0.5, softening=0.03)

hcipy.imshow_field(mask, grid=act_grid)
plt.colorbar()

# %%


# %%
def L1_masked_reg(cmd, mask):
    # the l1 regularisation term, evaluated with weights given by the mask
    return np.sum(mask * np.abs(cmd))


def flux_outside_pupil(cmd):
    pass
