#!/usr/bin/env python
import numpy as np
import zmq
import time
import toml
import os
import argparse
import matplotlib.pyplot as plt
import subprocess
import glob
import datetime
import common.DM_basis_functions as dmbases
import pyBaldr.utilities as util

"""
Updated build_baldr_control_matrix.py with:
- dimensionally-correct MAP / SVD handling
- consistent internal operator convention:
    I2M_* internal shape = (Nmodes, Nsignal)
  and transpose only when writing TOML
- measurement-space TT projection applied consistently
- original verification steps reintroduced and updated so they run with the
  corrected conventions

Important note:
This script is internally consistent, but hardware-facing verification sections
still depend on live SHM camera/DM objects existing and being reachable on the
target system.
"""

tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")

parser = argparse.ArgumentParser(description="build control model")

parser.add_argument("--toml_file", type=str, default=default_toml)
parser.add_argument("--beam_id", type=int, default=3)
parser.add_argument(
    "--phasemask",
    type=str,
    default="H4",
    choices=[f"H{i+1}" for i in range(5)] + [f"J{i+1}" for i in range(5)],
)
parser.add_argument("--signal_space", type=str, default='dm', choices=['dm', 'pix'])
parser.add_argument("--inverse_method_LO", type=str, default="pinv")
parser.add_argument("--inverse_method_HO", type=str, default="zonal")

parser.add_argument(
    "--no_project_TT_out_HO",
    dest="project_TT_out_HO",
    action="store_false",
    help="Disable projecting TT out of HO (default: enabled)"
)
parser.set_defaults(project_TT_out_HO=True)

parser.add_argument("--project_waffle_out_HO", dest="project_waffle_out_HO", action="store_true")
parser.add_argument("--no_filter_edge_actuators", dest="filter_edge_actuators", action="store_false")
parser.set_defaults(filter_edge_actuators=True)

parser.add_argument(
    "--fig_path",
    type=str,
    default=f'/home/asg/ben_bld_data/{tstamp_rough}/',
    help="path/to/output/image/ for the saved figures"
)

args = parser.parse_args()
os.makedirs(args.fig_path, exist_ok=True)

print("filter_edge_actuators = ", args.filter_edge_actuators)

with open(args.toml_file.replace('#', f'{args.beam_id}'), "r") as f:
    config_dict = toml.load(f)

pupil_mask = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None)
).astype(bool)

I2A = np.array(config_dict[f'beam{args.beam_id}']['I2A'], dtype=float)
IM = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("IM", None),
    dtype=float
)
M2C = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("M2C", None),
    dtype=float
)

I0 = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("I0", None)
)
N0 = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("N0", None)
)
norm_pupil = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("norm_pupil", None)
)

dark_cov = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("dark_cov", None)
if dark_cov is not None:
    dark_cov = np.array(dark_cov, dtype=float)

intrn_flx_I0 = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("intrn_flx_I0", None)
)
print(f"ENSURE FLUX NORMALIZATION EXISTS AND IS NOT LOW (i.e. ~ 1). intrn_flx_I0={intrn_flx_I0}")

I2rms_sec = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", {}).get(f"{args.phasemask}", {}).get("secondary", None)
).astype(float)
I2rms_ext = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", {}).get(f"{args.phasemask}", {}).get("exterior", None)
).astype(float)

if not np.all(np.isfinite(I2rms_sec)):
    print("\n WARNING: No secondary strehl modes found in config file, using 2x2 I matrix instead.")
    I2rms_sec = np.eye(2)
if not np.all(np.isfinite(I2rms_ext)):
    print("\n WARNING: No exterior strehl modes found in config file, using 2x2 I matrix instead.")
    I2rms_ext = np.eye(2)

LO = int(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("LO", None)
)

inside_edge_filt = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("inner_pupil_filt", None)
)
N0 = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("N0", None)
)
sec = np.array(
    config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("secondary", None)
)
poke_amp = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("poke_amp", None)
camera_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", {}).get("camera_config", None)

Nsig_pix = IM.shape[1]
if dark_cov is not None:
    if dark_cov.ndim == 1:
        if dark_cov.size == Nsig_pix * Nsig_pix:
            dark_cov = dark_cov.reshape(Nsig_pix, Nsig_pix)
            print(f"reshaped dark_cov to {dark_cov.shape}")
        elif dark_cov.size == Nsig_pix:
            dark_cov = np.diag(dark_cov)
            print(f"interpreted dark_cov as diagonal and reshaped to {dark_cov.shape}")
        else:
            print(f"WARNING: could not interpret dark_cov with shape {dark_cov.shape}, using None")
            dark_cov = None
    elif dark_cov.ndim != 2:
        print(f"WARNING: dark_cov ndim={dark_cov.ndim} unsupported, using None")
        dark_cov = None

if args.signal_space.lower() == "dm":
    print('projecting IM to registered DM space via I2A')
    IM = np.array([I2A @ ii for ii in IM], dtype=float)
elif args.signal_space.lower() == "pix":
    print('keeping IM in pixel (measurement) space')
else:
    raise UserWarning('invalid signal space! --signal_space must be dm | pix')

IM_LO = IM[:LO]
IM_HO = IM[LO:]

M2C_LO = M2C[:, :LO].copy()
M2C_HO = M2C[:, LO:].copy()

print(f"IM_LO.shape = {IM_LO.shape}")
print(f"IM_HO.shape = {IM_HO.shape}")

pup_im_std = np.std(IM_HO, axis=0)
den = np.max(pup_im_std) - np.min(pup_im_std)
if den > 0:
    pup_mask = (pup_im_std - np.min(pup_im_std)) / den
else:
    pup_mask = np.zeros_like(pup_im_std)
pup_mask[pup_mask < 0.2] = 0

print("pup_mask.shape", pup_mask.shape)

if args.signal_space.lower() == "dm":
    dm_mask = pup_mask.reshape(-1)
    util.nice_heatmap_subplots(
        im_list=[util.get_DM_command_in_2D(pup_mask), util.get_DM_command_in_2D(dm_mask)],
        title_list=['pupil mask\nmeas space', 'pupil mask\nproj to dm space']
    )
    plt.show()
elif args.signal_space.lower() == "pix":
    dm_mask = I2A @ pup_mask.reshape(-1)
    util.nice_heatmap_subplots(
        im_list=[pup_mask.reshape(32, 32), util.get_DM_command_in_2D(dm_mask)],
        title_list=['pupil mask\nmeas space', 'pupil mask\nproj to dm space']
    )
    plt.show()
else:
    raise UserWarning('invalid signal space! --signal_space must be dm | pix')

# ---------------- LO ----------------
if args.inverse_method_LO.lower() == 'pinv':
    I2M_LO = np.linalg.pinv(IM_LO).T

elif args.inverse_method_LO.lower() == 'map':
    Ca_LO = np.eye(IM_LO.shape[0], dtype=float)
    if args.signal_space.lower() == 'pix':
        Cn_LO = dark_cov if (dark_cov is not None and dark_cov.shape == (IM_LO.shape[1], IM_LO.shape[1])) else np.eye(IM_LO.shape[1], dtype=float)
    else:
        if dark_cov is not None:
            if dark_cov.shape == (Nsig_pix, Nsig_pix):
                Cn_LO = I2A @ dark_cov @ I2A.T
            elif dark_cov.shape == (IM_LO.shape[1], IM_LO.shape[1]):
                Cn_LO = dark_cov
            else:
                Cn_LO = np.eye(IM_LO.shape[1], dtype=float)
        else:
            Cn_LO = np.eye(IM_LO.shape[1], dtype=float)

    reg = 1e-12 * np.eye(IM_LO.shape[1], dtype=float)
    I2M_LO = Ca_LO @ IM_LO @ np.linalg.inv(IM_LO.T @ Ca_LO @ IM_LO + Cn_LO + reg)

elif 'svd_truncation' in args.inverse_method_LO.lower():
    k = int(args.inverse_method_LO.split('truncation_')[-1])
    U, S, Vt = np.linalg.svd(IM_LO, full_matrices=False)
    k_eff = min(k, len(S))
    Sinv = np.zeros_like(S, dtype=float)
    for ii in range(k_eff):
        if S[ii] > 0:
            Sinv[ii] = 1.0 / S[ii]
    I2M_LO = ((Vt.T * Sinv) @ U.T).T
else:
    raise UserWarning('no inverse method provided for LO')

# ---------------- HO ----------------
if args.inverse_method_HO.lower() == 'pinv':
    I2M_HO = np.linalg.pinv(IM_HO).T

elif args.inverse_method_HO.lower() == 'map':
    Ca_HO = np.eye(IM_HO.shape[0], dtype=float)

    if args.signal_space.strip().lower() == 'pix':
        if dark_cov is not None and dark_cov.shape == (IM_HO.shape[1], IM_HO.shape[1]):
            Cn_HO = dark_cov
        else:
            if dark_cov is not None:
                print(f'input dark_cov shape {dark_cov.shape} incompatible with pixel MAP, using identity matrix')
            else:
                print('input dark_cov is None, using identity matrix')
            Cn_HO = np.eye(IM_HO.shape[1], dtype=float)

    elif args.signal_space.strip().lower() == 'dm':
        if dark_cov is not None:
            if dark_cov.shape == (Nsig_pix, Nsig_pix):
                Cn_HO = I2A @ dark_cov @ I2A.T
            elif dark_cov.shape == (IM_HO.shape[1], IM_HO.shape[1]):
                Cn_HO = dark_cov
            else:
                print(f'input dark_cov shape {dark_cov.shape} incompatible with dm MAP, using identity matrix')
                Cn_HO = np.eye(IM_HO.shape[1], dtype=float)
        else:
            print('input dark_cov is None, using identity matrix')
            Cn_HO = np.eye(IM_HO.shape[1], dtype=float)
    else:
        raise UserWarning("invalid signal_space., --signal_space = dm | pix")

    reg = 1e-12 * np.eye(IM_HO.shape[1], dtype=float)
    I2M_HO = Ca_HO @ IM_HO @ np.linalg.inv(IM_HO.T @ Ca_HO @ IM_HO + Cn_HO + reg)

elif args.inverse_method_HO.lower() == 'zonal':
    if IM_HO.shape[0] != IM_HO.shape[1]:
        raise ValueError(f"zonal HO inversion requires square HO block in chosen signal space, got {IM_HO.shape}")
    diag_vals = []
    for i in range(len(IM_HO)):
        v = IM_HO[i][i]
        if np.isfinite(v) and v != 0:
            diag_vals.append(dm_mask.astype(bool)[i] / v)
        else:
            diag_vals.append(0.0)
    I2M_HO = np.diag(np.array(diag_vals, dtype=float))

elif 'svd_truncation' in args.inverse_method_HO.lower():
    k = int(args.inverse_method_HO.split('truncation_')[-1])
    U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)
    k_eff = min(k, len(S))
    Sinv = np.zeros_like(S, dtype=float)
    for ii in range(k_eff):
        if S[ii] > 0:
            Sinv[ii] = 1.0 / S[ii]
    I2M_HO = ((Vt.T * Sinv) @ U.T).T

elif "eigen" in args.inverse_method_HO.lower():
    try:
        k_eff_req = int(args.inverse_method_HO.lower().split("eigen_")[-1])
    except Exception:
        raise ValueError("For eigen HO inversion use --inverse_method_HO eigen_<k>, e.g. eigen_30")

    eps = 1e-12
    H_HO = IM_HO.T
    U, S, Vt = np.linalg.svd(H_HO, full_matrices=False)

    k_eff = min(k_eff_req, S.shape[0])

    U_k = U[:, :k_eff]
    S_k = S[:k_eff]
    Vt_k = Vt[:k_eff, :]
    V_k = Vt_k.T

    I2M_HO = U_k.T
    M2C_HO_raw = M2C[:, LO:]
    M2C_HO = M2C_HO_raw @ (V_k @ np.diag(1.0 / (S_k + eps)))

    modes2plot = min(5, I2M_HO.shape[0])
    if args.signal_space.strip().lower() == 'dm':
        try:
            util.nice_heatmap_subplots(
                im_list=[util.get_DM_command_in_2D(I2M_HO[ii]) for ii in range(modes2plot)],
                title_list=[f"eigenmode {ii}" for ii in range(modes2plot)]
            )
            plt.show()
        except Exception:
            print("SOMETHING WENT WRONG WITH PLOTTING EIGEN MODES")
    else:
        try:
            util.nice_heatmap_subplots(
                im_list=[I2M_HO[ii].reshape(32, 32) for ii in range(modes2plot)],
                title_list=[f"eigenmode {ii}" for ii in range(modes2plot)]
            )
            plt.show()
        except Exception:
            print("SOMETHING WENT WRONG WITH PLOTTING EIGEN MODES")
else:
    raise UserWarning('no inverse method provided for HO')

# TT projection in measurement space
if args.project_TT_out_HO and LO > 0 and I2M_HO.shape[0] > 0:
    Umeas, Smeas, _ = np.linalg.svd(IM_LO.T, full_matrices=False)
    r_lo = int(np.sum(Smeas > 1e-12))
    Qlo = Umeas[:, :r_lo]
    P_LO = Qlo @ Qlo.T
    print("P_LO.shape", P_LO.shape)

    leak_mat = I2M_HO @ IM_LO.T
    rel_leak = np.linalg.norm(leak_mat) / (np.linalg.norm(I2M_HO) * np.linalg.norm(IM_LO.T) + 1e-30)
    print("--\nsanity check project TT out of HO:\nrelative HO response to LO subspace before projection:", rel_leak)

    I2M_HO = I2M_HO @ (np.eye(P_LO.shape[0]) - P_LO)
    print("I2M_HO.shape", I2M_HO.shape)

# write TOML
dict2write = {
    f"beam{args.beam_id}": {
        f"{args.phasemask}": {
            "ctrl_model": {
                "inverse_method_LO": args.inverse_method_LO,
                "inverse_method_HO": args.inverse_method_HO,
                "controller_type": "leaky",
                "signal_space": args.signal_space.lower(),
                "sza": np.array(M2C).shape[0],
                "szm": np.array(M2C).shape[1],
                "szp": np.array(I2M_HO).shape[1],
                "I2A": np.array(I2A).tolist(),
                "I2M_LO": np.array(I2M_LO.T).tolist(),
                "I2M_HO": np.array(I2M_HO.T).tolist(),
                "M2C_LO": np.array(M2C_LO).tolist(),
                "M2C_HO": np.array(M2C_HO).tolist(),
                "I2rms_sec": np.array(I2rms_sec).tolist(),
                "I2rms_ext": np.array(I2rms_ext).tolist(),
                "telemetry": 0,
                "auto_close": 0,
                "auto_open": 1,
                "auto_tune": 0,
                "close_on_strehl_limit": 10,
                "open_on_strehl_limit": 0,
                "open_on_flux_limit": 0,
                "open_on_dm_limit": 0.3,
                "LO_offload_limit": 1,
            }
        }
    }
}

if os.path.exists(args.toml_file.replace('#', f'{args.beam_id}')):
    try:
        current_data = toml.load(args.toml_file.replace('#', f'{args.beam_id}'))
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        current_data = {}
else:
    current_data = {}

current_data = util.recursive_update(current_data, dict2write)

with open(args.toml_file.replace('#', f'{args.beam_id}'), "w") as f:
    toml.dump(current_data, f)

print(f"updated configuration file {args.toml_file.replace('#', f'{args.beam_id}')}")

# --------------------------------------------------
# QUICK LOOK (kept from original, updated for operator convention)
# --------------------------------------------------
if 0:
    for beam_id in [args.beam_id]:
        im_list = [I0.reshape(32, 32), np.array(N0).reshape(32, 32), np.array(norm_pupil).reshape(32, 32), util.get_DM_command_in_2D(dm_mask)]
        title_list = ['<I0>', '<N0>', 'normalized pupil', 'mask']
        cbar_list = ["UNITLESS"] * len(im_list)
        util.nice_heatmap_subplots(im_list, title_list=title_list, cbar_label_list=cbar_list)
        plt.savefig(f'{args.fig_path}' + f'reference_intensities_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
        plt.show()

        modes2look = [0, 1, min(65, len(IM)-1), min(67, len(IM)-1)]
        if args.signal_space.lower() == "dm":
            im_list = [util.get_DM_command_in_2D(IM[m]) for m in modes2look]
        else:
            im_list = [IM[m].reshape(32, 32) for m in modes2look]
        util.nice_heatmap_subplots(im_list, title_list=[f'mode {m}' for m in modes2look], cbar_label_list=["UNITLESS"] * len(im_list))
        plt.savefig(f'{args.fig_path}' + f'IM_some_modes_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
        plt.show()

        U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)

        plt.figure(figsize=(6, 4))
        plt.semilogy(S, 'o-')
        plt.title("Singular Values of IM_HO")
        plt.xlabel("Index")
        plt.ylabel("Singular value (log scale)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_singular_values_beam{beam_id}.png', bbox_inches='tight', dpi=200)

        plt.figure(figsize=(15, 3))
        for i in range(min(5, Vt.shape[0])):
            ax = plt.subplot(1, 5, i + 1)
            if args.signal_space.lower() == "dm":
                im = ax.imshow(util.get_DM_command_in_2D(Vt[i]), cmap='viridis')
            else:
                im = ax.imshow(Vt[i].reshape(32, 32), cmap='viridis')
            ax.set_title(f"Vt[{i}]")
            plt.colorbar(im, ax=ax)
        plt.suptitle("First 5 intensity eigenmodes (Vt)")
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_first5_intensity_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)

        plt.figure(figsize=(15, 3))
        for i in range(min(5, U.shape[1])):
            ax = plt.subplot(1, 5, i + 1)
            try:
                im = ax.imshow(util.get_DM_command_in_2D(U[:, i]), cmap='plasma')
            except Exception:
                im = ax.imshow(np.atleast_2d(U[:, i]), cmap='plasma', aspect='auto')
            ax.set_title(f"U[:, {i}]")
            plt.colorbar(im, ax=ax)
        plt.suptitle("First 5 system eigenmodes (U)")
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_first5_system_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)
        plt.show()
        plt.close("all")

# --------------------------------------------------
# VERIFICATION STEPS (kept from original, corrected for conventions)
# --------------------------------------------------
test_reco = input("press enter to continue recon tests. 0 to finish ...")

if test_reco != '0':
    from asgard_alignment.DM_shm_ctrl import dmclass
    from xaosim.shmlib import shm

    TEST_BEAM = int(args.beam_id)
    N_TRIALS = 40
    AMP_STD = 0.05
    CAM_SHM = f"/dev/shm/baldr{args.beam_id}.im.shm"
    FIG_DIR = os.path.expanduser(args.fig_path or "~/Downloads/")

    with open(args.toml_file.replace('#', f'{TEST_BEAM}'), "r") as f:
        cfg = toml.load(f)

    top = cfg[f"beam{TEST_BEAM}"]
    ctrl = top[args.phasemask]["ctrl_model"]

    I2A_test = np.array(top["I2A"], dtype=float)
    I2M_LO = np.array(ctrl["I2M_LO"], dtype=float).T   # restore operator convention
    M2C_LO = np.array(ctrl["M2C_LO"], dtype=float)
    LO_count = int(ctrl.get("LO", 2))
    sigspace = str(ctrl.get("signal_space", "dm")).lower()
    inner_pupil_filt = np.array(ctrl["inner_pupil_filt"]).astype(bool)
    I0_flat = np.array(ctrl["I0"], dtype=float)
    N0_flat = np.array(ctrl["N0"], dtype=float)
    dark = np.array(ctrl["dark"], dtype=float)

    assert LO_count >= 2, "LO must include at least tip & tilt (LO>=2)."
    assert I2M_LO.shape[0] >= 2, "I2M_LO must have at least 2 rows for tip/tilt."

    cam = shm(f"/dev/shm/baldr{args.beam_id}.im.shm")
    dm = dmclass(beam_id=TEST_BEAM, main_chn=3)

    amps = np.linspace(-0.2, 0.2, 20)
    tip_true, tip_rec, tilt_true, tilt_rec = [], [], [], []
    zero144 = np.zeros(144, dtype=float)

    try:
        for a in amps:
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[0] = a
            u_cmd = M2C_LO @ a_lo
            dm.set_data(u_cmd)
            time.sleep(0.1)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp, axis=0)

            s_pix = I_norm_flat / np.mean(N0_flat[inner_pupil_filt]) - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            s = I2A_test @ s_pix if sigspace == "dm" else s_pix
            a_hat_lo = I2M_LO @ s
            tip_true.append(a)
            tip_rec.append(a_hat_lo[0])

        dm.set_data(zero144)
    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    tip_true = np.array(tip_true, dtype=float)
    tip_rec = np.array(tip_rec, dtype=float)
    tip_err = tip_rec - tip_true

    plt.figure(figsize=(6, 4))
    plt.plot(tip_true, tip_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(tip_true, tip_true, "--", label="ideal")
    p = np.polyfit(tip_true, tip_rec, 1)
    plt.title(
        f"Tip ramp response (beam{TEST_BEAM}, {args.phasemask})\n"
        f"slope={p[0]:.3g}  offset={p[1]:.3g}  RMSE={np.sqrt(np.mean(tip_err**2)):.3g}"
    )
    plt.xlabel("commanded tip amplitude [DM units]")
    plt.ylabel("reconstructed tip [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_tip = os.path.join(args.fig_path, f"tip_ramp_response_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_tip, dpi=180)
    plt.show()
    plt.close()

    try:
        for a in amps:
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[1] = a
            u_cmd = M2C_LO @ a_lo
            dm.set_data(u_cmd)
            time.sleep(0.1)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp, axis=0)

            s_pix = I_norm_flat / np.mean(N0_flat[inner_pupil_filt]) - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            s = I2A_test @ s_pix if sigspace == "dm" else s_pix
            a_hat_lo = I2M_LO @ s
            tilt_true.append(a)
            tilt_rec.append(a_hat_lo[1])

        dm.set_data(zero144)
    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    tilt_true = np.array(tilt_true, dtype=float)
    tilt_rec = np.array(tilt_rec, dtype=float)
    tilt_err = tilt_rec - tilt_true

    plt.figure(figsize=(6, 4))
    plt.plot(tilt_true, tilt_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(tilt_true, tilt_true, "--", label="ideal")
    p = np.polyfit(tilt_true, tilt_rec, 1)
    plt.title(
        f"Tilt ramp response (beam{TEST_BEAM}, {args.phasemask})\n"
        f"slope={p[0]:.3g}  offset={p[1]:.3g}  RMSE={np.sqrt(np.mean(tilt_err**2)):.3g}"
    )
    plt.xlabel("commanded tilt amplitude [DM units]")
    plt.ylabel("reconstructed tilt [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_tilt = os.path.join(args.fig_path, f"tilt_ramp_response_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_tilt, dpi=180)
    plt.show()
    plt.close()

    print(f"[OK] Saved TT ramp plots:\n  {out_png_tip}\n  {out_png_tilt}")

    rng = np.random.default_rng(0)
    true_tt, rec_tt = [], []
    zero144 = np.zeros(144, dtype=float)

    try:
        for k in range(N_TRIALS):
            a_tt = rng.normal(0.0, AMP_STD, size=2)
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[:2] = a_tt
            u_cmd = M2C_LO @ a_lo
            dm.set_data(u_cmd)
            time.sleep(0.1)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp, axis=0)
            s_pix = I_norm_flat / np.mean(N0_flat[inner_pupil_filt]) - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            s = I2A_test @ s_pix if sigspace == "dm" else s_pix
            a_hat_lo = I2M_LO @ s
            true_tt.append(a_tt)
            rec_tt.append(a_hat_lo[:2])

        dm.set_data(zero144)
    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    true_tt = np.array(true_tt)
    rec_tt = np.array(rec_tt)
    err = rec_tt - true_tt

    rmse_tip = np.sqrt(np.mean(err[:, 0] ** 2))
    rmse_tilt = np.sqrt(np.mean(err[:, 1] ** 2))
    rmse_all = np.sqrt(np.mean(err ** 2))

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(6, 3))
    for i, name in enumerate(["Tip", "Tilt"]):
        plt.subplot(1, 2, i + 1)
        plt.scatter(true_tt[:, i], rec_tt[:, i], s=18)
        m = max(np.max(np.abs(true_tt[:, i])), np.max(np.abs(rec_tt[:, i]))) * 1.1 + 1e-6
        plt.plot([-m, m], [-m, m], '--', lw=1)
        plt.xlabel(f"True {name} [DM units]")
        plt.ylabel(f"Reconstructed {name} [DM units]")
        plt.title(f"{name}  RMSE={np.sqrt(np.mean(err[:, i]**2)):.3g}")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
    out_png = os.path.join(args.fig_path, f"recon_LO_TT_sanity_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.show()
    plt.close('all')

    print("\n=== LO (Tip/Tilt) reconstructor sanity test ===")
    print(f"Beam: {TEST_BEAM} | Signal space: {sigspace} | Trials: {N_TRIALS}")
    print(f"Std of commanded TT: {AMP_STD} (per mode)")
    print(f"RMSE Tip : {rmse_tip:.4g}")
    print(f"RMSE Tilt: {rmse_tilt:.4g}")
    print(f"RMSE All : {rmse_all:.4g}")
    print(f"Saved plot: {out_png}")

ho_in = input(f"\nEnter HO mode index to test  (or '0' to skip): ").strip()

if ho_in != '0':
    from asgard_alignment.DM_shm_ctrl import dmclass
    from xaosim.shmlib import shm

    TEST_BEAM = int(args.beam_id)

    with open(args.toml_file.replace('#', f'{TEST_BEAM}'), "r") as f:
        cfg = toml.load(f)

    top = cfg[f"beam{TEST_BEAM}"]
    ctrl = top[args.phasemask]["ctrl_model"]
    I2A_test = np.array(top["I2A"], dtype=float)
    I2M_LO = np.array(ctrl["I2M_LO"], dtype=float).T
    M2C_LO = np.array(ctrl["M2C_LO"], dtype=float)
    LO_count = int(ctrl.get("LO", 2))
    sigspace = str(ctrl.get("signal_space", "dm")).lower()
    inner_pupil_filt = np.array(ctrl["inner_pupil_filt"]).astype(bool)
    I0_flat = np.array(ctrl["I0"], dtype=float)
    N0_flat = np.array(ctrl["N0"], dtype=float)
    dark = np.array(ctrl["dark"], dtype=float)

    I2M_HO = np.array(ctrl["I2M_HO"], dtype=float).T
    M2C_HO = np.array(ctrl["M2C_HO"], dtype=float)

    cam = shm(f"/dev/shm/baldr{args.beam_id}.im.shm")
    dm = dmclass(beam_id=TEST_BEAM, main_chn=3)

    n_meas = 140 if sigspace == "dm" else int(I0_flat.size)
    N_HO_CTRL = int(I2M_HO.shape[0])
    N_HO_CMD = int(M2C_HO.shape[1])

    print(f"\nHO sanity: I2M_HO shape = {I2M_HO.shape} (Nmodes_ctrl x Nmeas)")
    print(f"HO sanity: M2C_HO shape = {M2C_HO.shape} (Nact x Nmodes_cmd)")

    ho_idx = int(ho_in)
    if ho_idx < 0 or ho_idx >= N_HO_CTRL:
        raise ValueError(f"HO index {ho_idx} out of range [0..{N_HO_CTRL-1}]")

    ho_cmd_idx = min(ho_idx, N_HO_CMD - 1)

    amps_ho = np.linspace(-0.2, 0.2, 21)
    ho_true, ho_rec, lo_leak = [], [], []
    zero144 = np.zeros(144, dtype=float)

    try:
        for a in amps_ho:
            a_ho = np.zeros(N_HO_CMD, dtype=float)
            a_ho[ho_cmd_idx] = a
            u_cmd = M2C_HO @ a_ho
            dm.set_data(u_cmd)
            time.sleep(0.12)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_flat = np.mean(img_tmp, axis=0)

            s_pix = I_flat / np.mean(N0_flat[inner_pupil_filt]) - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            s = I2A_test @ s_pix if sigspace == "dm" else s_pix

            a_hat_ho = I2M_HO @ s

            if LO_count > 0:
                a_hat_lo = I2M_LO @ s
                lo_leak.append(np.linalg.norm(a_hat_lo[:min(2, LO_count)]))
            else:
                lo_leak.append(0.0)

            ho_true.append(a)
            ho_rec.append(a_hat_ho[ho_idx])

        dm.set_data(zero144)
    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    ho_true = np.array(ho_true, dtype=float)
    ho_rec = np.array(ho_rec, dtype=float)
    ho_err = ho_rec - ho_true
    lo_leak = np.array(lo_leak, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(ho_true, ho_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(ho_true, ho_true, "--", label="ideal")
    p = np.polyfit(ho_true, ho_rec, 1)
    plt.title(
        f"HO single-mode ramp (beam{TEST_BEAM}, {args.phasemask})\n"
        f"ctrl_idx={ho_idx}, cmd_idx={ho_cmd_idx} | slope={p[0]:.3g} offset={p[1]:.3g} "
        f"RMSE={np.sqrt(np.mean(ho_err**2)):.3g}\n"
        f"median LO leakage={np.median(lo_leak):.3g}"
    )
    plt.xlabel("commanded HO amplitude [DM units]")
    plt.ylabel("reconstructed HO coefficient [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_ho_ramp = os.path.join(args.fig_path, f"recon_HO_mode{ho_idx}_ramp_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_ho_ramp, dpi=180)
    plt.show()
    plt.close()

    print(f" Saved HO ramp plot: {out_png_ho_ramp}")

    N_TRIALS_HO = 40
    AMP_STD_HO = 0.05

    rng = np.random.default_rng(1)
    true_ho, rec_ho, leak_lo = [], [], []

    try:
        for k in range(N_TRIALS_HO):
            a = float(rng.normal(0.0, AMP_STD_HO))
            a_ho = np.zeros(N_HO_CMD, dtype=float)
            a_ho[ho_cmd_idx] = a
            u_cmd = M2C_HO @ a_ho
            dm.set_data(u_cmd)
            time.sleep(0.12)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_flat = np.mean(img_tmp, axis=0)

            s_pix = I_flat / np.mean(N0_flat[inner_pupil_filt]) - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            s = I2A_test @ s_pix if sigspace == "dm" else s_pix

            a_hat_ho = I2M_HO @ s
            true_ho.append(a)
            rec_ho.append(a_hat_ho[ho_idx])

            if LO_count > 0:
                a_hat_lo = I2M_LO @ s
                leak_lo.append(np.linalg.norm(a_hat_lo[:min(2, LO_count)]))
            else:
                leak_lo.append(0.0)

        dm.set_data(zero144)
    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    true_ho = np.array(true_ho)
    rec_ho = np.array(rec_ho)
    err_ho = rec_ho - true_ho
    rmse_ho = np.sqrt(np.mean(err_ho**2))
    leak_lo = np.array(leak_lo)

    plt.figure(figsize=(5, 4))
    plt.scatter(true_ho, rec_ho, s=18)
    m = max(np.max(np.abs(true_ho)), np.max(np.abs(rec_ho))) * 1.1 + 1e-6
    plt.plot([-m, m], [-m, m], "--", lw=1)
    plt.xlabel("True HO amplitude [DM units]")
    plt.ylabel("Reconstructed HO coefficient [DM units]")
    plt.title(
        f"HO single-mode random (ctrl_idx={ho_idx}, cmd_idx={ho_cmd_idx})\n"
        f"RMSE={rmse_ho:.3g} | median LO leakage={np.median(leak_lo):.3g}"
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    out_png_ho_rand = os.path.join(args.fig_path, f"recon_HO_mode{ho_idx}_random_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_ho_rand, dpi=180)
    plt.show()
    plt.close()

    print(f"\n=== HO single-mode reconstructor sanity test ===")
    print(f"Beam: {TEST_BEAM} | Signal space: {sigspace}")
    print(f"Control mode idx tested: {ho_idx} | Command-basis idx excited: {ho_cmd_idx}")
    print(f"Trials: {N_TRIALS_HO} | Std commanded: {AMP_STD_HO}")
    print(f"RMSE HO: {rmse_ho:.4g}")
    print(f"Saved plot: {out_png_ho_rand}")
