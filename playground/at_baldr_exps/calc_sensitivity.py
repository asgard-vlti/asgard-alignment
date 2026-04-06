# %%
import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime

from bcam import Bcam

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

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
time.sleep(1)
# %%
# %%
vals = np.random.randn(144) * 0.1
dm.set_data(vals)

# %%
dm.set_data(np.zeros(144))
time.sleep(0.01)

# %%
zero_point_file = np.load("beam3_good_flat.npz")
zero_point = zero_point_file["flat"]*0.01

dm.shms[1].set_data(zero_point)
dm.shm0.post_sems(1)

imgs = cam.take_stack(1000)
plt.imshow(imgs.mean(0))
plt.colorbar()

ref = imgs.mean(0)
# %%
diffs = np.diff(imgs, axis=0)
# plt.imshow(diffs.std(0))
plt.plot(diffs[:, 15, 15])

# %%
import hcipy

n_act = 12
n_beam = 10

grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
n_modes = 8
max_freq = n_modes  # lambda/D
probe_max_freq = max_freq

freqs = hcipy.make_pupil_grid(
    n_modes,
    max_freq,
)

basis = hcipy.make_fourier_basis(grid, freqs.scaled(2 * np.pi))
basis.transformation_matrix.shape

hc_fourier = basis.transformation_matrix

# if odd number of modes, remove piston
if n_modes % 2 == 1:
    hc_fourier = hc_fourier[:, 1:]


hc_fourier.shape

# %%
plt.imshow(hc_fourier[:, 0].reshape(12, 12))


# %%
def compute_IM(dm, cam, basis, amp, sleep=0.01, n_im=1, n_pokes=5, n_discard=2):
    n_modes = basis.shape[-1]
    responses = []

    for mode_idx in range(n_modes):
        res = 0.0
        for pk_idx in range(n_pokes):
            imgs = []
            for sp in [-1, 1]:
                cmd = np.zeros((n_modes, 1))
                cmd[mode_idx] = sp * amp
                cmd = basis @ cmd
                dm.set_data(cmd.flatten())

                time.sleep(sleep)

                cam.take_stack(n_discard)
                ims = cam.take_stack(n_im)

                imgs.append(cam.normalise(ims).mean(0))
            res += (imgs[1] - imgs[0]) / (2 * amp * n_pokes)
        responses.append(res)

    dm.set_data(np.zeros(144))
    return np.array(responses)


start = time.time()
im = compute_IM(dm, cam, hc_fourier, amp=0.005, sleep=0.02, n_im=10)
print(f"interaction matrix took {time.time() - start:.2f}s")

# %%
im.shape
# %%
import matplotlib.colors as mcolor

im = im.reshape(im.shape[0], 32,32)
idx = 0
plt.subplot(121)
plt.imshow(hc_fourier[:, idx].reshape(12, 12), norm=mcolor.CenteredNorm(), cmap="bwr")
plt.subplot(122)
plt.imshow(im[idx], norm=mcolor.CenteredNorm(), cmap="bwr")
plt.colorbar()

# %%
im = im.reshape(im.shape[0], -1)
# %%
xcor = im @ im.T
plt.imshow(xcor, norm=mcolor.CenteredNorm(), cmap="bwr")
plt.colorbar()
# %%
# FIM = (im) @ im.T
FIM = (im / ref.flatten()) @ im.T
Cov = np.linalg.inv(FIM)

plt.imshow(Cov, norm=mcolor.CenteredNorm(), cmap="bwr")
plt.colorbar()

metric = np.trace(Cov)
metric

# %%
plt.plot(np.diag(Cov),'x')
ax2 = plt.twinx()
ax2.plot(np.diag(FIM), 'x', c="r")

# %%
def im_FIM_metric(dm, basis, ref, metric_type="avg_cov_ph"):
    start = time.time()
    im = compute_IM(dm, cam, basis, amp=0.01, sleep=0.01, n_im=2, n_pokes=4)
    print(f"interaction matrix took {time.time() - start:.2f}s")

    im = im.reshape(im.shape[0], -1)
    FIM = (im / ref.flatten()) @ im.T

    if metric_type == "avg_cov_ph":
        Cov = np.linalg.inv(FIM)
        return np.trace(Cov), im, Cov
    else:
        raise ValueError()


covs = []
n_runs = 5
for i in range(n_runs):
    metric, im, cov = im_FIM_metric(dm, hc_fourier, ref)
    print(f"{metric:.2e}")
    covs.append(cov)

for i in range(n_runs):
    plt.subplot(1, n_runs, i + 1)
    plt.imshow(covs[i], norm=mcolor.CenteredNorm(), cmap="bwr")

# %%

offset_ch = 1


def write_offset(dm: dmclass, offset_cmd):
    dm.shms[offset_ch].set_data(offset_cmd)


n_act = 12
n_beam = 10

grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
n_modes = 3
max_freq = n_modes  # lambda/D
probe_max_freq = max_freq

freqs = hcipy.make_pupil_grid(
    n_modes,
    max_freq,
)

basis = hcipy.make_fourier_basis(grid, freqs.scaled(2 * np.pi))
basis.transformation_matrix.shape

fourier_small = basis.transformation_matrix

# if odd number of modes, remove piston
if n_modes % 2 == 1:
    fourier_small = fourier_small[:, 1:]

# %%
plt.imshow(fourier_small[:, 1].reshape(12, 12))

# %%
# offsets = np.array([0.05,0.0,0.0,0.0])
offsets = np.zeros(fourier_small.shape[1])
offsets[1] = 0.01

write_offset(dm, fourier_small @ (offsets[:, None]))
time.sleep(0.03)
ref = cam.take_stack(1000).mean(0)

res, im, cov = im_FIM_metric(dm, hc_fourier, ref)

print(f"{res:.3e}")

# %%
import scipy.optimize


def loss(offsets, args):
    fourier_small, hc_fourier = args
    write_offset(dm, fourier_small @ (offsets[:, None]))
    time.sleep(0.03)
    ref = cam.take_stack(1000).mean(0)

    res, im, cov = im_FIM_metric(dm, hc_fourier, ref)
    return res


offsets = np.zeros(fourier_small.shape[1])
offsets[1] = 0.01

res = scipy.optimize.minimize(
    loss,
    offsets,
    args=((fourier_small, hc_fourier),),
    method="Nelder-Mead",
    options={"disp": True, "maxfev": 120},
)
# %%
res.x
