# %%
import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime

from bcam import Bcam
from tqdm.auto import tqdm

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

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
# mds_send(sock, "off SBB")
# mds_send(sock, f"b_shut close {beam}")
cur_bmy = mds_send(sock,f"read BMY{beam}")
mds_send(sock,f"moveabs BMY{beam} 500.0")
time.sleep(3)
# %%
cam.take_dark(256)
plt.imshow(cam.dark)
plt.colorbar()
# %%
mds_send(sock, f"moveabs BMY{beam} {cur_bmy}")
# mds_send(sock, "on SBB")
time.sleep(2)
# %%
# %%
vals = np.random.randn(144) * 0.1
dm.set_data(vals)

# %%
dm.set_data(np.zeros(144))
time.sleep(0.01)

# %%
# zero_point_file = np.load("beam3_good_flat.npz")
# zero_point = zero_point_file["flat"]*0.01

zero_point = np.zeros(144)

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

import DM_modes2


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

hc_fourier = fourier.transformation_matrix

# %%
plt.imshow(hc_fourier[:, 0].reshape(12, 12))


# %%
def compute_IM(dm, cam, basis, amp, sleep=0.01, n_im=1, n_pokes=5, n_discard=2):
    n_modes = basis.shape[-1]
    responses = []

    for mode_idx in range(n_modes):
    # for mode_idx in tqdm(range(n_modes)):
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
im = compute_IM(dm, cam, hc_fourier, amp=0.03, sleep=0.01, n_im=2,n_discard=1, n_pokes=10,)
print(f"interaction matrix took {time.time() - start:.2f}s")

# %%
im.shape
# %%
import matplotlib.colors as mcolor

im = im.reshape(im.shape[0], 32,32)
idx = 1
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
print(f"metric: {metric:.3e}")

# %%
plt.plot(np.diag(Cov),'x')
plt.ylabel("Cov", color="C0")
ax2 = plt.twinx()
ax2.plot(np.diag(FIM), 'x', c="r")
ax2.set_ylabel("Fisher info", color="r")

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
dm.set_data(np.zeros(144))
time.sleep(0.03)
ref = cam.take_stack(1000).mean(0)


#%%
start = time.time()
im = compute_IM(dm, cam, hc_fourier, amp=0.03, sleep=0.01, n_im=2,n_discard=1, n_pokes=10,)
print(f"interaction matrix took {time.time() - start:.2f}s")

im = im.reshape(im.shape[0], -1)

# %%
recon_matrix = hcipy.inverse_tikhonov(
    im.T, rcond=1e-3
)

#%%
t_start = time.time()
dur = 5

recons = []
ref_flat = cam.normalise(ref).flatten()
while time.time() - t_start < dur:
    img = cam.get_img()
    recon = recon_matrix.dot(cam.normalise(img).flatten()-ref_flat)
    recons.append(recon)

recons = np.array(recons)
# %%
fps = 500.0
times = np.arange(len(recons))/500.0
plt.plot(times,recons[:,0])
plt.plot(times,recons[:,1])
plt.figure()
plt.psd(recons[:,0], Fs=500.0)
plt.psd(recons[:,1], Fs=500.0)



# %%
def rms(vec):
    return np.sqrt(np.mean(vec**2))

def run_cl(dur, gains=None, leakage = None, print_every=0.2):
    if gains is None:
        gains = 0.1*np.ones(len(recon_matrix))
    if leakage is None:
        leakage = 0.99*np.ones(len(recon_matrix))

    t_start = time.time()
    dm_acts = np.zeros(len(recon_matrix))
    last_print = 0
    recons = []
    i = 0

    cmds =[]

    has_started = False

    while True:
        img = cam.get_img()
        i += 1
        recon = recon_matrix.dot(cam.normalise(img).flatten()-ref_flat)
        recons.append(recon)

        dm_acts = leakage*dm_acts - gains*recon
        cmd = hc_fourier@dm_acts
        # cmd -= np.mean(cmd)

        if rms(cmd) > 0.4:
            print("\nopening loop")
            time.sleep(1)
            dm.set_data(np.zeros(144))
            break

        dm.set_data(cmd)

        cmds.append([cmd])

        cur = time.time()
        if cur - t_start > dur:
            dm.set_data(np.zeros(144))
            break
        if cur - last_print > print_every:
            print(f"\rRecon rms: {rms(recon):.3e}, FPS: {i/(cur - t_start):.2f}", end='')
            last_print = cur
    
    return recons,cmds


n_modes = len(recon_matrix)
LO_cut = 2
block_2 = 40

# run_cl(5, 0.1*np.ones(n_modes),0.99*np.ones(n_modes))
cl_recons,cmds = run_cl(
    20, 
    np.concatenate([0.2*np.ones(LO_cut),0.15*np.ones(block_2),0.05*np.ones(n_modes - (LO_cut+block_2))],),
    np.concatenate([0.99*np.ones(LO_cut),0.998*np.ones(block_2),0.999*np.ones(n_modes - (LO_cut+block_2))],),
)

print("done")
cl_recons = np.array(cl_recons)
np.std(cl_recons[:,0]),np.std(recons[:,0])

# %%
cmds = np.array(cmds)
cmds = cmds.reshape(-1, 144)

# plt.plot(cmds[:,70])
plt.plot(np.mean(cmds, axis=1))
# %%
dm.set_data(np.zeros(144))
# %%
fps = 500.0
times = np.arange(len(recons))/500.0
plt.plot(times,recons[:,0])
plt.plot(times,recons[:,1])
times = np.arange(len(cl_recons))/500.0
plt.plot(times,cl_recons[:,0])
plt.plot(times,cl_recons[:,1])
plt.figure()
plt.psd(recons[:,0], Fs=500.0)
plt.psd(recons[:,1], Fs=500.0)
plt.psd(cl_recons[:,0], Fs=500.0)
plt.psd(cl_recons[:,1], Fs=500.0)
plt.xscale("log")
    
# %%
dm.set_data(np.zeros(144))

# %%
dm.shms[3].set_data(np.zeros(144))
dm.shm0.post_sems(1)

# %%

hc_fourier.shape
