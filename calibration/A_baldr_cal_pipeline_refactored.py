#!/usr/bin/env python3
"""
Refactored Baldr / ZWFS calibration pipeline with explicit support for calibrating
multiple beams that share:
  - one detector / camera,
  - one internal calibration source,
and have:
  - independent DMs,
  - independent phase-mask stages.

Key design change versus the original script:
shared resources are handled in synchronized "global epochs". Any action that affects
all beams at once (source state, camera configuration assumptions, clear-pupil / ZWFS
reference state acquisition) is performed once for the whole selected beam set.
Beam-local operations (DM pokes) are applied to all selected beams first, then all beams
are read back, so every acquisition corresponds to one coherent system state.

This preserves the original calibration products and TOML / FITS outputs as closely as
possible, while fixing the main multi-beam failure mode in the DM-registration stage and
making the reference acquisition consistent with a single shared source and detector.

Notes:
- This script keeps the original interactive checkpoints.
- It assumes DM flats are already applied externally, matching the original script.
- It was syntax-checked, but it cannot be hardware-verified in this environment.



Traceback (most recent call last):
  File "/home/asg/Progs/repos/asgard-alignment/calibration/A_baldr_cal_pipeline_refactored.py", line 827, in <module>
    main()
  File "/home/asg/Progs/repos/asgard-alignment/calibration/A_baldr_cal_pipeline_refactored.py", line 816, in main
    build_opd_model(rt, dark_dict)
  File "/home/asg/Progs/repos/asgard-alignment/calibration/A_baldr_cal_pipeline_refactored.py", line 536, in build_opd_model
    N0_norm = np.mean(N0[inner_pupil_filt])
IndexError: boolean index did not match indexed array along dimension 0; dimension is 32 but corresponding boolean dimension is 1024



"""

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import toml
import zmq
from astropy.io import fits
from scipy.ndimage import binary_erosion, median_filter
from xaosim.shmlib import shm

from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import common.DM_registration as DM_registration
import common.phasescreens as ps
from pyBaldr import utilities as util


def ask_continue(prompt: str) -> None:
    usr_input = input(prompt)
    if usr_input.strip() == "0":
        raise SystemExit(0)


def get_bad_pixel_indicies(imgs, std_threshold=20, mean_threshold=6):
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)
    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (
        (np.abs(mean_frame - global_mean) > mean_threshold * global_std)
        | (std_frame > std_threshold * np.median(std_frame))
    )
    return bad_pixel_map


def interpolate_bad_pixels(img, bad_pixel_map):
    filtered_image = img.copy()
    filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
    return filtered_image


@dataclass
class RuntimeContext:
    args: argparse.Namespace
    socket: zmq.Socket
    camclient: object
    cam_shm: Dict[int, object]
    dm_shm_dict: Dict[int, dmclass]
    message_history: List[str]

    def send_and_get_response(self, message: str) -> str:
        self.message_history.append(f":blue[Sending message to server: ] {message}\n")
        self.socket.send_string(message)
        response = self.socket.recv_string()
        colour = "red" if ("NACK" in response or "not connected" in response) else "green"
        self.message_history.append(
            f":{colour}[Received response from server: ] {response}\n"
        )
        return response.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baldr Calibration pipeline to IM (multi-beam refactor).")
    default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")
    parser.add_argument("--toml_file", type=str, default=default_toml)
    parser.add_argument("--host", type=str, default="192.168.100.2")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=int, default=5000)
    parser.add_argument(
        "--beam_id",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[3],
        help="Comma-separated beam IDs, e.g. 1,2,3,4",
    )
    parser.add_argument(
        "--global_camera_shm",
        type=str,
        default="/dev/shm/cred1.im.shm",
    )
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument(
        "--fig_path",
        type=str,
        default="/home/asg/Progs/repos/asgard-alignment/calibration/reports/",
    )
    parser.add_argument("--phasemask", type=str, default="H4")
    parser.add_argument("--mode", type=str, default="bright")
    parser.add_argument("--lobe_threshold", type=float, default=0.03)
    parser.add_argument("--LO", type=int, default=2)
    parser.add_argument("--basis_name", type=str, default="zonal")
    parser.add_argument("--poke_amp", type=float, default=0.05)
    parser.add_argument("--signal_space", type=str, default="dm")
    parser.add_argument("--DM_flat", type=str, default="baldr")
    parser.add_argument("--no_imgs", type=int, default=10)
    parser.add_argument("--dark_frames", type=int, default=2000)
    parser.add_argument("--frame_sleep", type=float, default=0.01)
    parser.add_argument("--settle_time", type=float, default=0.2)
    parser.add_argument("--rel_offset", type=float, default=-200.0)
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> RuntimeContext:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout)
    socket.connect(f"tcp://{args.host}:{args.port}")

    camclient = FLI.fli(args.global_camera_shm, roi=[None, None, None, None])
    cam_shm = {b: shm(f"/dev/shm/baldr{b}.im.shm") for b in args.beam_id}
    dm_shm_dict = {b: dmclass(beam_id=b) for b in args.beam_id}

    return RuntimeContext(
        args=args,
        socket=socket,
        camclient=camclient,
        cam_shm=cam_shm,
        dm_shm_dict=dm_shm_dict,
        message_history=[],
    )


def acquire_mean_image(rt: RuntimeContext, beam_id: int, n_imgs: int, subtract=None) -> np.ndarray:
    imgs = []
    for _ in range(n_imgs):
        img = rt.cam_shm[beam_id].get_data()
        if subtract is not None:
            img = img - subtract
        imgs.append(img)
        time.sleep(rt.args.frame_sleep)
    return np.mean(imgs, axis=0)


def acquire_mean_images_all_beams(
    rt: RuntimeContext,
    n_imgs: int,
    subtract_map: Mapping[int, np.ndarray] | None = None,
) -> Dict[int, np.ndarray]:
    out = {}
    for b in rt.args.beam_id:
        subtract = None if subtract_map is None else subtract_map[b]
        out[b] = acquire_mean_image(rt, b, n_imgs=n_imgs, subtract=subtract)
    return out


def move_all_beams_to_mask(rt: RuntimeContext, phasemask: str) -> None:
    for beam_id in rt.args.beam_id:
        res = rt.send_and_get_response(f"fpm_movetomask phasemask{beam_id} {phasemask}")
        print(f"beam {beam_id}: moved to phasemask {phasemask}: {res}")


def move_all_beams_relative(rt: RuntimeContext, dx: float = 0.0, dy: float = 0.0) -> None:
    for beam_id in rt.args.beam_id:
        if dx != 0:
            print(rt.send_and_get_response(f"moverel BMX{beam_id} {dx}"))
            time.sleep(1)
        if dy != 0:
            print(rt.send_and_get_response(f"moverel BMY{beam_id} {dy}"))
            time.sleep(1)


def update_toml(rt: RuntimeContext, beam_id: int, new_data: dict) -> None:
    toml_path = rt.args.toml_file.replace("#", f"{beam_id}")
    if os.path.exists(toml_path):
        try:
            current_data = toml.load(toml_path)
        except Exception as exc:
            print(f"Error loading TOML file {toml_path}: {exc}")
            current_data = {}
    else:
        current_data = {}

    current_data = util.recursive_update(current_data, new_data)
    with open(toml_path, "w") as f:
        toml.dump(current_data, f)


def collect_darks(rt: RuntimeContext) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    ask_continue(
        "\n======================\npress enter when ready to get a dark frame (will turn BB source off), enter 0 to exit\n"
    )
    print("turning off internal SBB source for bias")
    rt.send_and_get_response("off SBB")
    time.sleep(10)

    dark_dict: Dict[int, np.ndarray] = {}
    Cn_dict: Dict[int, np.ndarray] = {}
    for beam_id in rt.args.beam_id:
        dark_tmp = []
        for _ in range(rt.args.dark_frames):
            dark_tmp.append(rt.cam_shm[beam_id].get_data())
            time.sleep(rt.args.frame_sleep)
        dark_dict[beam_id] = np.mean(dark_tmp, axis=0)
        Dtmp = np.asarray(dark_tmp, dtype=float).reshape(len(dark_tmp), -1)
        Cn_dict[beam_id] = np.cov(Dtmp.T)
        print(f"beam {beam_id} dark covariance shape = {Cn_dict[beam_id].shape}")

    rt.send_and_get_response("on SBB")
    time.sleep(3)

    util.nice_heatmap_subplots(
        im_list=[dark_dict[b] for b in rt.args.beam_id],
        title_list=[f"beam{b} dark" for b in rt.args.beam_id],
    )
    plt.show()
    util.nice_heatmap_subplots(
        im_list=[Cn_dict[b] for b in rt.args.beam_id],
        title_list=[f"beam{b} dark covariance" for b in rt.args.beam_id],
    )
    plt.show()
    return dark_dict, Cn_dict


def register_pupils(rt: RuntimeContext) -> None:
    ask_continue(
        f"\n======================\nmoving to phasemask {rt.args.phasemask} reference position, enter 0 to exit\n"
    )
    move_all_beams_to_mask(rt, rt.args.phasemask)
    time.sleep(1)

    print("Moving all FPMs out to get clear pupils")
    move_all_beams_relative(rt, dx=rt.args.rel_offset, dy=rt.args.rel_offset)
    time.sleep(1)

    clear_imgs = acquire_mean_images_all_beams(rt, n_imgs=rt.args.no_imgs)

    for beam_id in rt.args.beam_id:
        savepath = None
        if rt.args.fig_path is None:
            savepath = f"delme{beam_id}.png"
        else:
            os.makedirs(rt.args.fig_path, exist_ok=True)
            savepath = os.path.join(rt.args.fig_path, f"pupil_reg_beam{beam_id}")

        ell1 = util.detect_pupil(
            clear_imgs[beam_id], sigma=2, threshold=0.5, plot=rt.args.plot, savepath=savepath
        )
        cx, cy, a, b, theta, pupil_mask = ell1
        rad_est = np.sqrt((1 / np.pi) * np.sum(pupil_mask))
        exterior = util.filter_exterior_annulus(pupil_mask, inner_radius=rad_est + 1, outer_radius=rad_est + 5)

        secondary = np.zeros_like(pupil_mask, dtype=bool)
        y_indices, x_indices = np.where(pupil_mask)
        center_x = int(round(np.mean(x_indices)))
        center_y = int(round(np.mean(y_indices)))
        secondary[center_y, center_x] = True

        new_data = {
            "io": {"mode": "shm"},
            f"beam{beam_id}": {
                "pupil_ellipse_fit": {
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "a": float(a),
                    "b": float(b),
                    "theta": float(theta),
                },
                "pupil_mask": {
                    "mask": pupil_mask.tolist(),
                    "exterior": exterior.tolist(),
                    "secondary": secondary.tolist(),
                },
            },
        }
        update_toml(rt, beam_id, new_data)
        print(f"pupil detection for beam {beam_id} finished")

    print(f"\n======================\nmoving back to phasemask {rt.args.phasemask} reference position")
    move_all_beams_to_mask(rt, rt.args.phasemask)


def register_dm(rt: RuntimeContext) -> Dict[int, np.ndarray]:
    move_all_beams_to_mask(rt, rt.args.phasemask)
    time.sleep(1)
    ask_continue(
        "\n======================\npress enter when ready to register DM (ensure phasemask is aligned!). enter 0 to exit\n"
    )

    number_of_pokes = 8
    poke_amplitude = 0.05
    sleeptime = rt.args.settle_time
    dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4)

    pupil_mask = {}
    for beam_id in rt.args.beam_id:
        with open(rt.args.toml_file.replace("#", f"{beam_id}")) as file:
            config_dict = toml.load(file)
            pupil_mask[beam_id] = np.array(
                config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)
            ).astype(bool)

    img_4_corners: Dict[int, List[np.ndarray]] = {b: [] for b in rt.args.beam_id}
    print(f"GOING VERY SLOW ({sleeptime}s delays) DUE TO SHM DELAY DM")

    for act in dm_4_corners:
        print(f"actuator {act}")
        img_list_push = {b: [] for b in rt.args.beam_id}
        img_list_pull = {b: [] for b in rt.args.beam_id}

        for nn in range(number_of_pokes):
            print(f"poke {nn}")
            poke_vector = np.zeros(140)
            poke_vector[act] = ((-1) ** nn) * poke_amplitude

            for beam_id in rt.args.beam_id:
                rt.dm_shm_dict[beam_id].set_data(
                    rt.dm_shm_dict[beam_id].cmd_2_map2D(poke_vector, fill=0)
                )

            time.sleep(sleeptime)

            imgs = acquire_mean_images_all_beams(rt, n_imgs=10)
            for beam_id in rt.args.beam_id:
                if nn % 2:
                    img_list_push[beam_id].append(imgs[beam_id])
                else:
                    img_list_pull[beam_id].append(imgs[beam_id])

            for beam_id in rt.args.beam_id:
                rt.dm_shm_dict[beam_id].set_data(
                    rt.dm_shm_dict[beam_id].cmd_2_map2D(np.zeros_like(poke_vector), fill=0)
                )
            time.sleep(sleeptime)

        for beam_id in rt.args.beam_id:
            delta_img = np.abs(
                np.mean(img_list_push[beam_id], axis=0) - np.mean(img_list_pull[beam_id], axis=0)
            )
            img_4_corners[beam_id].append(pupil_mask[beam_id].astype(float) * delta_img)

    transform_dicts: Dict[int, dict] = {}
    bilin_interp_matricies: Dict[int, np.ndarray] = {}

    for beam_id in rt.args.beam_id:
        beam_fig_dir = os.path.join(rt.args.fig_path, f"beam{beam_id}")
        os.makedirs(beam_fig_dir, exist_ok=True)
        plt.close("all")
        transform_dicts[beam_id] = DM_registration.calibrate_transform_between_DM_and_image(
            dm_4_corners, img_4_corners[beam_id], debug=True, fig_path=beam_fig_dir + "/"
        )
        plt.close("all")

        img = img_4_corners[beam_id][0].copy()
        coords = transform_dicts[beam_id]["actuator_coord_list_pixel_space"]
        x_target = np.array([x for x, _ in coords])
        y_target = np.array([y for _, y in coords])
        x_grid = np.arange(img.shape[0])
        y_grid = np.arange(img.shape[1])
        M = DM_registration.construct_bilinear_interpolation_matrix(
            image_shape=img.shape,
            x_grid=x_grid,
            y_grid=y_grid,
            x_target=x_target,
            y_target=y_target,
        )
        _ = M @ img.reshape(-1)
        bilin_interp_matricies[beam_id] = M
        update_toml(rt, beam_id, {f"beam{beam_id}": {"I2A": M.tolist()}})

        tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path_tmp = f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/dm_registration/beam{beam_id}/"
        os.makedirs(path_tmp, exist_ok=True)
        file_tmp = f"dm_reg_beam{beam_id}_{tstamp}.json"
        with open(path_tmp + file_tmp, "w") as json_file:
            json.dump(util.convert_to_serializable(transform_dicts[beam_id]), json_file)
        print(f"saved dm registration json : {path_tmp + file_tmp}")

    return bilin_interp_matricies


def plot_strehl_pixel_registration(data, exterior_filter, secondary_filter, savefig=None):
    label = "I0-N0"
    fs = 18
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_main = plt.subplot(gs[1, 0])
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)

    im = ax_main.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    ax_main.text(0, 0, label, fontsize=25, color="white")
    ax_main.set_xlabel("X (pixels)", fontsize=fs)
    ax_main.set_ylabel("Y (pixels)", fontsize=fs)

    x_counts = np.sum(data, axis=0)
    y_counts = np.sum(data, axis=1)
    ax_xhist.bar(np.arange(len(x_counts)), x_counts, color="gray", edgecolor="black")
    ax_yhist.barh(np.arange(len(y_counts)), y_counts, color="gray", edgecolor="black")
    ax_yhist.set_xlabel("ADU", fontsize=fs)
    ax_xhist.set_ylabel("ADU", fontsize=fs)
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)
    ax_xhist.set_xlim(ax_main.get_xlim())
    ax_yhist.set_ylim(ax_main.get_ylim())

    if np.sum(exterior_filter):
        ax_main.contour(exterior_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5], colors="red", linestyles="-", linewidths=2, origin="lower")
        ex_coords = np.argwhere(exterior_filter)
        ax_main.scatter(ex_coords[:, 1], ex_coords[:, 0], marker="x", color="red", alpha=0.4, label="Exterior Filter")

    if np.sum(secondary_filter):
        ax_main.contour(secondary_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5], colors="blue", linestyles="-", linewidths=2, origin="lower")
        sec_coords = np.argwhere(secondary_filter)
        ax_main.scatter(sec_coords[:, 1], sec_coords[:, 0], marker="x", color="blue", alpha=0.4, label="Secondary Filter")

    ax_main.legend(fontsize=fs)
    ax_xhist.tick_params(labelsize=15)
    ax_yhist.tick_params(labelsize=15)
    ax_main.tick_params(labelsize=15)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight", dpi=200)
        print(f"saving image {savefig}")
    plt.show()


def register_strehl_pixels(rt: RuntimeContext, dark_dict: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    ask_continue("\n======================\npress enter when ready to register Strehl pixels, enter 0 to exit\n")

    zwfs_pupils = acquire_mean_images_all_beams(rt, n_imgs=10)
    util.nice_heatmap_subplots([zwfs_pupils[b] for b in rt.args.beam_id], savefig="delme.png")

    print("Moving all FPMs out to get clear pupils")
    move_all_beams_relative(rt, dx=rt.args.rel_offset, dy=0.0)
    time.sleep(2)
    clear_pupils = acquire_mean_images_all_beams(rt, n_imgs=10)

    print("Moving all FPMs back in beam")
    move_all_beams_relative(rt, dx=-rt.args.rel_offset, dy=0.0)
    time.sleep(2)

    secondary_filter_dict = {}
    exterior_filter_dict = {}

    for beam_id in rt.args.beam_id:
        center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(
            clear_pupils[beam_id], sigma=2, threshold=0.5, plot=False, savepath=None
        )
        secondary_filter = util.get_secondary_mask(pupil_mask, (center_x, center_y))

        if rt.args.mode == "bright":
            pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=7, outer_radius=100)
            pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=11, outer_radius=100)
        elif rt.args.mode == "faint":
            pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=4, outer_radius=100)
            pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=8, outer_radius=100)
        else:
            raise UserWarning("invalid mode. Must be either 'bright' or 'faint'")

        exterior_filter = (
            np.abs(zwfs_pupils[beam_id] - clear_pupils[beam_id])
            > rt.args.lobe_threshold * np.mean(clear_pupils[beam_id][pupil_mask])
        ) * (~pupil_mask) * pupil_edge_filter * pupil_limit_filter

        exterior_filter_dict[beam_id] = exterior_filter
        secondary_filter_dict[beam_id] = secondary_filter

        update_toml(
            rt,
            beam_id,
            {
                f"beam{beam_id}": {
                    "pupil_mask": {
                        "exterior": exterior_filter.astype(int).tolist(),
                        "secondary": secondary_filter.astype(int).tolist(),
                    }
                }
            },
        )

    for beam_id in rt.args.beam_id:
        savepath = f"delme{beam_id}.png" if rt.args.fig_path is None else os.path.join(rt.args.fig_path, f"strehl_pixel_filter{beam_id}.png")
        plot_strehl_pixel_registration(
            data=np.array(zwfs_pupils[beam_id]) - np.array(clear_pupils[beam_id]),
            exterior_filter=exterior_filter_dict[beam_id],
            secondary_filter=secondary_filter_dict[beam_id],
            savefig=savepath,
        )
        plt.close("all")

    return zwfs_pupils, clear_pupils, exterior_filter_dict


def build_opd_model(rt: RuntimeContext, dark_dict: Dict[int, np.ndarray]) -> None:
    ask_continue("\n======================\npress enter when ready to start calibrating the OPD model for Baldr, enter 0 to exit\n")
    opd_per_cmd = 3000
    r0 = 0.1 * (1.65 / 0.5) ** (6 / 5)
    L0 = 0.1

    scrn_list = [
        ps.PhaseScreenKolmogorov(nx_size=24, pixel_scale=1.8 / 24, r0=r0, L0=L0, random_seed=None)
        for _ in range(50)
    ]

    for beam_id in rt.args.beam_id:
        with open(rt.args.toml_file.replace("#", f"{beam_id}"), "r") as f:
            config_dict = toml.load(f)
            ctrl_model = config_dict.get(f"beam{beam_id}", {}).get(f"{rt.args.phasemask}", {}).get("ctrl_model", {})
            inner_pupil_filt = np.array(ctrl_model.get("inner_pupil_filt", None)).astype(bool)
            N0 = np.array(ctrl_model.get("N0", None)).reshape(rt.cam_shm[beam_id].get_data().shape)
            N0_norm = np.mean(N0.flatten()[inner_pupil_filt.flatten()])

        telem = {"N0": N0, "N0_norm": N0_norm, "i": [], "s": [], "opd_rms_est": []}
        scrn_scaling_grid = np.logspace(-1, 0.2, 5)
        for it, scrn in enumerate(scrn_list):
            print(f"beam {beam_id}: input aberration {it}/{len(scrn_list)}")
            for ph_scale in scrn_scaling_grid:
                cmd = util.create_phase_screen_cmd_for_DM(scrn, scaling_factor=ph_scale, drop_indicies=None, plot_cmd=False)
                cmd = np.array(cmd).reshape(12, 12)
                rt.dm_shm_dict[beam_id].set_data(cmd)
                time.sleep(rt.args.frame_sleep)
                i = rt.cam_shm[beam_id].get_data() - dark_dict[beam_id]
                s = i / N0_norm
                opd_est = np.std(opd_per_cmd * cmd)
                telem["i"].append(i)
                telem["s"].append(s)
                telem["opd_rms_est"].append(opd_est)

        rt.dm_shm_dict[beam_id].set_data(np.zeros_like(cmd))

        correlation_map = util.compute_correlation_map(np.array(telem["s"]), np.array(telem["opd_rms_est"]))
        yy, xx = np.ogrid[:telem["s"][0].shape[0], :telem["s"][0].shape[1]]
        snr = np.mean(np.array(telem["s"]), axis=0) / np.std(np.array(telem["s"]), axis=0)
        radial_constraint = (
            ((xx - telem["s"][0].shape[1] // 2) ** 2 + (yy - telem["s"][0].shape[0] // 2) ** 2 <= 20**2)
            * ((xx - telem["s"][0].shape[1] // 2) ** 2 + (yy - telem["s"][0].shape[0] // 2) ** 2 >= 6**2)
        )
        strehl_filt = (correlation_map < -0.7) & (snr > 1.0) & radial_constraint

        util.nice_heatmap_subplots(im_list=[correlation_map, strehl_filt], cbar_label_list=["Pearson R", "filt"])
        plt.figure()
        plt.plot([np.mean(ss[strehl_filt]) for ss in telem["s"]], np.array(telem["opd_rms_est"]), ".", label="est")
        plt.xlabel("<s>")
        plt.ylabel("OPD RMS [nm RMS]")
        plt.legend()
        plt.show()

        filtered_sigs = np.array([np.mean(ss[strehl_filt]) for ss in telem["s"]])
        opd_nm_est = np.array(telem["opd_rms_est"])
        opd_model_params = util.fit_piecewise_continuous(x=filtered_sigs, y=opd_nm_est, n_grid=80, trim=0.15)
        print(f"using util.fit_piecewise_continuous, opd_model_params = {opd_model_params}")

        x = filtered_sigs
        y = opd_nm_est
        opd_pred = util.piecewise_continuous(
            x,
            interc=opd_model_params["interc"],
            slope_1=opd_model_params["slope_1"],
            slope_2=opd_model_params["slope_2"],
            x_knee=opd_model_params["x_knee"],
        )
        plt.figure()
        plt.scatter(x, y, label="meas")
        plt.scatter(x, opd_pred, label="pred")
        plt.xlabel("signal (i_ext / <N0>)")
        plt.ylabel("OPD [nm RMS]")
        plt.legend()
        plt.show()

        update_toml(
            rt,
            beam_id,
            {
                f"beam{beam_id}": {
                    f"{rt.args.phasemask}": {
                        "ctrl_model": {
                            "strehl_filter": np.array(strehl_filt).astype(int).reshape(-1).tolist(),
                            "opd_m_interc": opd_model_params["interc"],
                            "opd_m_slope_1": opd_model_params["slope_1"],
                            "opd_m_slope_2": opd_model_params["slope_2"],
                            "opd_m_x_knee": opd_model_params["x_knee"],
                        }
                    }
                }
            },
        )
        print(f"updated configuration file {rt.args.toml_file.replace('#', f'{beam_id}')}")


def build_im(rt: RuntimeContext, dark_dict: Dict[int, np.ndarray], Cn_dict: Dict[int, np.ndarray]) -> None:
    ask_continue("\n======================\npress enter when ready to start calibrating the Interaction Matrix for Baldr, enter 0 to exit\n")
    no_imgs = rt.args.no_imgs

    I2A_dict = {}
    pupil_mask = {}
    secondary_mask = {}
    exterior_mask = {}
    baldr_pupils = None
    for beam_id in rt.args.beam_id:
        with open(rt.args.toml_file.replace("#", f"{beam_id}"), "r") as f:
            config_dict = toml.load(f)
            baldr_pupils = config_dict["baldr_pupils"]
            I2A_dict[beam_id] = config_dict[f"beam{beam_id}"]["I2A"]
            pupil_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)).astype(bool)
            secondary_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None)).astype(bool)
            exterior_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None)).astype(bool)

    move_all_beams_to_mask(rt, rt.args.phasemask)
    time.sleep(1)

    print("Moving all FPMs out to get clear pupils")
    move_all_beams_relative(rt, dx=rt.args.rel_offset, dy=rt.args.rel_offset)
    time.sleep(0.5)
    clear_pupils = {b: [] for b in rt.args.beam_id}
    inner_pupil_filt = {}
    normalized_pupils = {}

    for beam_id in rt.args.beam_id:
        for _ in range(no_imgs):
            clear_pupils[beam_id].append(rt.cam_shm[beam_id].get_data() - dark_dict[beam_id])
            time.sleep(rt.args.frame_sleep)

        inner_pupil_filt[beam_id] = binary_erosion(
            pupil_mask[beam_id] * (~secondary_mask[beam_id]), structure=np.ones((3, 3), dtype=bool)
        )
        pixel_filter = secondary_mask[beam_id] | (~util.remove_boundary(np.array(pupil_mask[beam_id])).astype(bool))
        normalized_pupils[beam_id] = np.mean(clear_pupils[beam_id], axis=0)
        normalized_pupils[beam_id][pixel_filter] = np.mean(np.mean(clear_pupils[beam_id], axis=0)[~pixel_filter])

    print("Moving all FPMs back in beam")
    move_all_beams_relative(rt, dx=-rt.args.rel_offset, dy=-rt.args.rel_offset)
    time.sleep(3)

    input("\n======================\nphasemasks aligned? ensure alignment then press enter")
    print("Getting ZWFS pupils")
    zwfs_pupils = {b: [] for b in rt.args.beam_id}
    for beam_id in rt.args.beam_id:
        for _ in range(no_imgs):
            zwfs_pupils[beam_id].append(rt.cam_shm[beam_id].get_data() - dark_dict[beam_id])
            time.sleep(rt.args.frame_sleep)

    LO_basis = dmbases.zer_bank(2, rt.args.LO + 1)
    if "zonal" in rt.args.basis_name.lower().strip():
        zonal_basis = np.array([rt.dm_shm_dict[rt.args.beam_id[0]].cmd_2_map2D(ii) for ii in np.eye(140)])
    elif "zernike" in rt.args.basis_name.lower().strip():
        zonal_basis = dmbases.zer_bank(4, 143)
    else:
        raise UserWarning(f"invalid --basis_name={rt.args.basis_name} input. must be 'zonal' or 'zernike'")

    modal_basis = np.array(LO_basis.tolist() + zonal_basis.tolist())
    M2C = modal_basis.copy().reshape(modal_basis.shape[0], -1).T

    n_modes = modal_basis.shape[0]
    number_of_pokes_per_cmd = 8
    signs = [(-1) ** n for n in range(number_of_pokes_per_cmd)]
    n_plus = sum(s > 0 for s in signs)
    n_minus = sum(s < 0 for s in signs)

    frame0 = rt.cam_shm[rt.args.beam_id[0]].get_data()
    ny, nx = frame0.shape
    Iplus_stack = {b: np.zeros((n_modes, n_plus, ny, nx), dtype=np.float32) for b in rt.args.beam_id}
    Iminus_stack = {b: np.zeros((n_modes, n_minus, ny, nx), dtype=np.float32) for b in rt.args.beam_id}
    IM_mat = {b: np.zeros((n_modes, ny * nx), dtype=np.float32) for b in rt.args.beam_id}

    for i, m in enumerate(modal_basis):
        print(f"executing cmd {i}/{n_modes - 1}")
        plus_k = 0
        minus_k = 0
        for s in signs:
            cmd = (s * rt.args.poke_amp / 2) * m
            for b in rt.args.beam_id:
                rt.dm_shm_dict[b].set_data(cmd)
            for b in rt.args.beam_id:
                imgs = []
                for _ in range(no_imgs):
                    imgs.append(rt.cam_shm[b].get_data() - dark_dict[b])
                    time.sleep(rt.args.frame_sleep)
                img_tmp = np.mean(imgs, axis=0).astype(np.float32)
                if s > 0:
                    Iplus_stack[b][i, plus_k] = img_tmp
                else:
                    Iminus_stack[b][i, minus_k] = img_tmp
            if s > 0:
                plus_k += 1
            else:
                minus_k += 1

        for b in rt.args.beam_id:
            N0_tmp = np.mean(clear_pupils[b], axis=0)
            norm_factor = float(np.mean(N0_tmp[inner_pupil_filt[b]]))
            I_plus = np.mean(Iplus_stack[b][i], axis=0).reshape(-1) / norm_factor
            I_minus = np.mean(Iminus_stack[b][i], axis=0).reshape(-1) / norm_factor
            errsig = (I_plus - I_minus) / rt.args.poke_amp
            IM_mat[b][i] = errsig.astype(np.float32).reshape(-1)

    for beam_id in rt.args.beam_id:
        rt.dm_shm_dict[beam_id].set_data(np.zeros_like(cmd))

    hdul = fits.HDUList()
    phdr = fits.Header()
    phdr["DATE"] = datetime.datetime.utcnow().isoformat()
    phdr["PHMASK"] = rt.args.phasemask
    phdr["POKEAMP"] = float(rt.args.poke_amp)
    phdr["LO"] = int(rt.args.LO)
    phdr["NOIMGS"] = int(no_imgs)
    phdr["NSIGN"] = number_of_pokes_per_cmd
    phdr["BEAMS"] = ",".join(map(str, rt.args.beam_id))
    hdul.append(fits.PrimaryHDU(header=phdr))
    hdul.append(fits.ImageHDU(np.asarray(modal_basis, dtype=np.float32), name="MODES"))
    hdul.append(fits.ImageHDU(np.asarray(M2C, dtype=np.float32), name="M2C"))

    for b in rt.args.beam_id:
        hdul.append(fits.ImageHDU(np.asarray(IM_mat[b], dtype=np.float32), name=f"IM_B{b}"))
        I0 = np.mean(zwfs_pupils[b], axis=0).astype(np.float32)
        N0 = np.mean(clear_pupils[b], axis=0).astype(np.float32)
        hdul.append(fits.ImageHDU(I0, name=f"I0_B{b}"))
        hdul.append(fits.ImageHDU(N0, name=f"N0_B{b}"))
        hdul.append(fits.ImageHDU(np.asarray(I2A_dict[b], dtype=np.float32), name=f"I2A_B{b}"))
        hdul.append(fits.ImageHDU(np.asarray(pupil_mask[b], dtype=np.uint8), name=f"PUPIL_B{b}"))
        hdul.append(fits.ImageHDU(np.asarray(secondary_mask[b], dtype=np.uint8), name=f"SEC_B{b}"))
        hdul.append(fits.ImageHDU(np.asarray(exterior_mask[b], dtype=np.uint8), name=f"EXT_B{b}"))
        hdul.append(fits.ImageHDU(np.asarray(inner_pupil_filt[b], dtype=np.uint8), name=f"INNER_B{b}"))
        hdul.append(fits.CompImageHDU(np.asarray(Iplus_stack[b], dtype=np.float32), name=f"IPLUS_B{b}", compression_type="RICE_1"))
        hdul.append(fits.CompImageHDU(np.asarray(Iminus_stack[b], dtype=np.float32), name=f"IMINUS_B{b}", compression_type="RICE_1"))
        hdul.append(fits.CompImageHDU(np.asarray(IM_mat[b], dtype=np.float32), name=f"IM_FINAL_B{b}", compression_type="RICE_1"))

    tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")
    fits_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/IM/{tstamp_rough}/IM_{rt.args.phasemask}_{tstamp}.fits"
    os.makedirs(os.path.dirname(fits_path), exist_ok=True)
    hdul.writeto(fits_path, overwrite=True)
    print(f"saved fits file with IM telemetry {fits_path}")

    for beam_id in rt.args.beam_id:
        dict2write = {
            f"beam{beam_id}": {
                f"{rt.args.phasemask}": {
                    "ctrl_model": {
                        "build_method": "double-sided-poke",
                        "DM_flat": rt.args.DM_flat.lower(),
                        "crop_pixels": np.array(baldr_pupils[f"{beam_id}"]).tolist(),
                        "pupil_pixels": np.where(np.array(pupil_mask[beam_id]).reshape(-1))[0].tolist(),
                        "interior_pixels": np.where(np.array(inner_pupil_filt[beam_id]).reshape(-1))[0].tolist(),
                        "secondary_pixels": np.where(np.array(secondary_mask[beam_id]).reshape(-1))[0].tolist(),
                        "exterior_pixels": np.where(np.array(exterior_mask[beam_id]).reshape(-1))[0].tolist(),
                        "IM": np.array(IM_mat[beam_id]).tolist(),
                        "poke_amp": rt.args.poke_amp,
                        "LO": rt.args.LO,
                        "M2C": np.nan_to_num(np.array(M2C), 0).tolist(),
                        "I0": np.mean(zwfs_pupils[beam_id], axis=0).reshape(-1).tolist(),
                        "intrn_flx_I0": float(np.sum(np.mean(zwfs_pupils[beam_id], axis=0))),
                        "N0": np.mean(clear_pupils[beam_id], axis=0).reshape(-1).tolist(),
                        "norm_pupil": np.array(normalized_pupils[beam_id]).reshape(-1).tolist(),
                        "camera_config": {k: str(v) for k, v in rt.camclient.config.items()},
                        "pupil": np.array(pupil_mask[beam_id]).astype(int).reshape(-1).tolist(),
                        "secondary": np.array(secondary_mask[beam_id]).astype(int).reshape(-1).tolist(),
                        "exterior": np.array(exterior_mask[beam_id]).astype(int).reshape(-1).tolist(),
                        "inner_pupil_filt": np.array(inner_pupil_filt[beam_id]).astype(int).reshape(-1).tolist(),
                        "bias": np.zeros([32, 32]).reshape(-1).astype(int).tolist(),
                        "dark": np.array(dark_dict[beam_id]).astype(int).reshape(-1).tolist(),
                        "dark_cov": np.array(Cn_dict[beam_id]).astype(int).reshape(-1).tolist(),
                        "bad_pixel_mask": np.ones([32, 32]).reshape(-1).astype(int).tolist(),
                        "bad_pixels": [],
                    }
                }
            }
        }
        update_toml(rt, beam_id, dict2write)
        print(f"updated configuration file {rt.args.toml_file.replace('#', f'{beam_id}')}")

    for beam_id in rt.args.beam_id:
        _, S, _ = np.linalg.svd(IM_mat[beam_id], full_matrices=True)
        plt.figure()
        plt.semilogy(S)
        plt.legend()
        plt.xlabel("mode index")
        plt.ylabel("singular values")
        os.makedirs(rt.args.fig_path, exist_ok=True)
        plt.savefig(os.path.join(rt.args.fig_path, f"IM_singularvalues_beam{beam_id}.png"), bbox_inches="tight", dpi=200)
        plt.show()
        plt.close()


def main() -> None:
    args = parse_args()
    rt = build_runtime(args)
    dark_dict, Cn_dict = collect_darks(rt)
    register_pupils(rt)
    register_dm(rt)
    register_strehl_pixels(rt, dark_dict)
    build_opd_model(rt, dark_dict)
    build_im(rt, dark_dict, Cn_dict)

    for b in rt.cam_shm:
        try:
            rt.cam_shm[b].close(erase_file=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
