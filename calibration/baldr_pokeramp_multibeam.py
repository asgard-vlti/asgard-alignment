#!/usr/bin/env python3
"""
Multi-beam Baldr poke-ramp acquisition using the same SHM interfaces as
calibration/A_baldr_cal_pipeline_refactored.py.

Key changes versus the original script:
- Uses per-beam camera SHM handles:
    cam_shm = {b: shm(f"/dev/shm/baldr{b}.im.shm") for b in args.beam_id}
- Uses per-beam DM SHM handles:
    dm_shm_dict = {b: dmclass(beam_id=b) for b in args.beam_id}
- Applies the same poke step to all selected beams at once, then reads all beams back.
- Saves a separate FITS file per beam, preserving the original FITS content as closely
  as possible.

Notes:
- Shared resources (source state, phase-mask moves) are handled once per selected beam set.
- Beam-specific references, reconstructors, and outputs are loaded/written independently.
- This script was refactored from the user-provided poke-ramp script and was not
  hardware-verified in this environment.
"""

import argparse
import datetime
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import toml
import zmq
from astropy.io import fits
from xaosim.shmlib import shm

from asgard_alignment import FLI_Cameras as FLI
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import pyBaldr.utilities as util


@dataclass
class BeamConfig:
    beam_id: int
    config_dict: dict
    pupil_mask: np.ndarray
    I2A: np.ndarray
    IM: np.ndarray
    M2C_LO: np.ndarray
    M2C_HO: np.ndarray
    I2M_LO: np.ndarray
    I2M_HO: np.ndarray
    I0: np.ndarray
    N0: np.ndarray
    LO: int
    inner_pupil_filt: np.ndarray
    camera_config: dict
    N0_runtime: float
    i_setpoint_runtime: np.ndarray


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
    parser = argparse.ArgumentParser(description="Baldr multi-beam poke-ramp acquisition.")
    default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")

    parser.add_argument(
        "--global_camera_shm",
        type=str,
        default="/dev/shm/cred1.im.shm",
        help="Camera shared memory path. Default: /dev/shm/cred1.im.shm",
    )
    parser.add_argument(
        "--toml_file",
        type=str,
        default=default_toml,
        help="TOML file pattern with # replaced by beam_id.",
    )
    parser.add_argument(
        "--beam_id",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[3],
        help="Comma-separated beam IDs, e.g. 1,2,3,4",
    )
    parser.add_argument("--phasemask", type=str, default="H4")
    parser.add_argument("--LO", type=int, default=2)
    parser.add_argument("--basis_name", type=str, default="zonal")
    parser.add_argument("--Nmodes", type=int, default=10, help="Number of modes to probe")
    parser.add_argument("--amp_max", type=float, default=0.2, help="Max poke amplitude")
    parser.add_argument("--no_amp_samples", type=int, default=20, help="Number of amplitude samples")
    parser.add_argument("--no_samples_per_cmd", type=int, default=20, help="Frames averaged per command")
    parser.add_argument(
        "--signal_space",
        type=str,
        default="pix",
        help="Signal space: 'dm' uses I2A, 'pix' uses pixel space",
    )
    parser.add_argument(
        "--DM_flat",
        type=str,
        default="baldr",
        help="Metadata only. DM flat is assumed already applied externally.",
    )
    parser.add_argument(
        "--fig_path",
        type=str,
        default="/home/asg/Progs/repos/asgard-alignment/calibration/reports/test/",
        help="Path for saved figures",
    )
    parser.add_argument("--host", type=str, default="192.168.100.2")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=int, default=5000)
    parser.add_argument("--frame_sleep", type=float, default=0.01)
    parser.add_argument("--dark_frames", type=int, default=1000)
    parser.add_argument("--rel_offset", type=float, default=-200.0)
    parser.add_argument("--settle_time", type=float, default=0.01)
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


def ask_continue(prompt: str) -> None:
    usr_input = input(prompt)
    if usr_input.strip() == "0":
        raise SystemExit(0)


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


def load_beam_configs(rt: RuntimeContext) -> Dict[int, BeamConfig]:
    beam_cfg: Dict[int, BeamConfig] = {}
    for beam_id in rt.args.beam_id:
        with open(rt.args.toml_file.replace("#", f"{beam_id}"), "r") as f:
            config_dict = toml.load(f)

        ctrl_model = config_dict.get(f"beam{beam_id}", {}).get(rt.args.phasemask, {}).get("ctrl_model", {})
        pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)).astype(bool)
        I2A = np.array(config_dict[f"beam{beam_id}"]["I2A"], dtype=float)
        IM = np.array(ctrl_model.get("IM", None), dtype=float)

        M2C_LO = np.array(ctrl_model.get("M2C_LO", None), dtype=float)
        M2C_HO = np.array(ctrl_model.get("M2C_HO", None), dtype=float)
        I2M_LO = np.array(ctrl_model.get("I2M_LO", None), dtype=float)
        I2M_HO = np.array(ctrl_model.get("I2M_HO", None), dtype=float)

        frame_shape = rt.cam_shm[beam_id].get_data().shape
        I0 = np.array(ctrl_model.get("I0", None), dtype=float).reshape(frame_shape)
        N0 = np.array(ctrl_model.get("N0", None), dtype=float).reshape(frame_shape)

        LO = int(ctrl_model.get("LO", rt.args.LO))
        inner_pupil_filt = np.array(ctrl_model.get("inner_pupil_filt", None)).astype(bool).reshape(frame_shape)
        camera_config = ctrl_model.get("camera_config", None)

        N0_runtime = float(np.mean(N0[inner_pupil_filt]))
        i_setpoint_runtime = I0 / N0_runtime

        beam_cfg[beam_id] = BeamConfig(
            beam_id=beam_id,
            config_dict=config_dict,
            pupil_mask=pupil_mask,
            I2A=I2A,
            IM=IM,
            M2C_LO=M2C_LO,
            M2C_HO=M2C_HO,
            I2M_LO=I2M_LO,
            I2M_HO=I2M_HO,
            I0=I0,
            N0=N0,
            LO=LO,
            inner_pupil_filt=inner_pupil_filt,
            camera_config=camera_config,
            N0_runtime=N0_runtime,
            i_setpoint_runtime=i_setpoint_runtime,
        )
    return beam_cfg


def acquire_dark_all_beams(rt: RuntimeContext) -> Dict[int, np.ndarray]:
    print("turning off internal SBB source for bias")
    rt.send_and_get_response("off SBB")
    time.sleep(10)

    dark_current: Dict[int, np.ndarray] = {}
    for beam_id in rt.args.beam_id:
        dark_tmp = []
        for _ in range(rt.args.dark_frames):
            dark_tmp.append(rt.cam_shm[beam_id].get_data())
            time.sleep(rt.args.frame_sleep)
        dark_current[beam_id] = np.mean(dark_tmp, axis=0)

    rt.send_and_get_response("on SBB")
    time.sleep(3)
    print("turning back on internal SBB source, check plot that darks are ok")

    util.nice_heatmap_subplots(
        im_list=[dark_current[b] for b in rt.args.beam_id],
        title_list=[f"beam{b}" for b in rt.args.beam_id],
    )
    plt.show()

    return dark_current


def acquire_reference_pupils(
    rt: RuntimeContext,
    beam_cfg: Dict[int, BeamConfig],
    dark_current: Dict[int, np.ndarray],
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float], Dict[int, np.ndarray]]:
    print(f"moving to phasemask {rt.args.phasemask} reference position")
    move_all_beams_to_mask(rt, rt.args.phasemask)
    time.sleep(1)

    rel_offset = rt.args.rel_offset
    print("Moving all FPMs out to get clear pupils")
    move_all_beams_relative(rt, dx=rel_offset, dy=rel_offset)
    time.sleep(0.5)

    print("getting clear pupils")
    N0_current: Dict[int, np.ndarray] = {}
    for beam_id in rt.args.beam_id:
        N0s = []
        for _ in range(rt.args.no_samples_per_cmd):
            N0s.append(rt.cam_shm[beam_id].get_data() - dark_current[beam_id])
            time.sleep(rt.args.frame_sleep)
        N0_current[beam_id] = np.mean(N0s, axis=0)

    print("Moving all FPMs back in beam")
    move_all_beams_relative(rt, dx=-rel_offset, dy=-rel_offset)
    time.sleep(3)

    ask_continue("phasemasks aligned? ensure alignment then press enter (or 0 to exit)\n")

    print("Getting ZWFS pupils")
    I0_current: Dict[int, np.ndarray] = {}
    N0_runtime_current: Dict[int, float] = {}
    i_setpoint_runtime_current: Dict[int, np.ndarray] = {}
    for beam_id in rt.args.beam_id:
        I0s = []
        for _ in range(rt.args.no_samples_per_cmd):
            I0s.append(rt.cam_shm[beam_id].get_data() - dark_current[beam_id])
            time.sleep(rt.args.frame_sleep)
        I0_current[beam_id] = np.mean(I0s, axis=0)
        N0_runtime_current[beam_id] = float(np.mean(N0_current[beam_id][beam_cfg[beam_id].inner_pupil_filt]))
        i_setpoint_runtime_current[beam_id] = I0_current[beam_id] / N0_runtime_current[beam_id]

    return N0_current, I0_current, N0_runtime_current, i_setpoint_runtime_current


def build_modal_basis(rt: RuntimeContext) -> np.ndarray:
    LO_basis = dmbases.zer_bank(2, rt.args.LO + 1)

    if "zonal" in rt.args.basis_name.lower().strip():
        zonal_basis = np.array([rt.dm_shm_dict[rt.args.beam_id[0]].cmd_2_map2D(ii) for ii in np.eye(140)])
    elif "zernike" in rt.args.basis_name.lower().strip():
        zonal_basis = dmbases.zer_bank(4, 143)
    else:
        raise UserWarning("basis_name must be 'zonal' or 'zernike'")

    modal_basis = np.array(LO_basis.tolist() + zonal_basis.tolist())
    return modal_basis[: int(rt.args.Nmodes)]


def acquire_pokeramp(
    rt: RuntimeContext,
    beam_cfg: Dict[int, BeamConfig],
    dark_current: Dict[int, np.ndarray],
    N0_runtime_current: Dict[int, float],
    modal_basis: np.ndarray,
):
    if rt.args.signal_space.lower() not in ["dm", "pix"]:
        raise UserWarning("signal space must either be 'dm' or 'pix'")

    probe_amps = np.linspace(-rt.args.amp_max, rt.args.amp_max, int(rt.args.no_amp_samples))

    test_frame = rt.cam_shm[rt.args.beam_id[0]].get_data()
    ny, nx = test_frame.shape
    n_mode = int(len(modal_basis))
    n_amp = int(len(probe_amps))

    data = {}
    for beam_id in rt.args.beam_id:
        n_lo = int(beam_cfg[beam_id].I2M_LO.shape[0])
        n_ho = int(beam_cfg[beam_id].I2M_HO.shape[0])
        data[beam_id] = {
            "imgs_cube": np.zeros((n_mode, n_amp, ny, nx), dtype=np.float32),
            "signal_cube": np.zeros((n_mode, n_amp, ny, nx), dtype=np.float32),
            "eLO_cube": np.zeros((n_mode, n_amp, n_lo), dtype=np.float32),
            "eHO_cube": np.zeros((n_mode, n_amp, n_ho), dtype=np.float32),
        }

    for idx, mode in enumerate(modal_basis):
        print(f"executing mode {idx + 1}/{n_mode}")
        for ai, amp in enumerate(probe_amps):
            time.sleep(rt.args.settle_time)

            cmd = amp * mode
            for beam_id in rt.args.beam_id:
                rt.dm_shm_dict[beam_id].set_data(cmd)

            for beam_id in rt.args.beam_id:
                subframes = []
                for _ in range(int(rt.args.no_samples_per_cmd)):
                    subframes.append(rt.cam_shm[beam_id].get_data() - dark_current[beam_id])
                    time.sleep(rt.args.frame_sleep)

                subframe_avg = np.mean(subframes, axis=0).astype(np.float32)
                signal = (
                    subframe_avg / N0_runtime_current[beam_id] - beam_cfg[beam_id].i_setpoint_runtime
                ).astype(np.float32)

                if rt.args.signal_space == "dm":
                    e_LO = (beam_cfg[beam_id].I2M_LO @ (beam_cfg[beam_id].I2A @ signal.reshape(-1))).astype(np.float32)
                    e_HO = (beam_cfg[beam_id].I2M_HO @ (beam_cfg[beam_id].I2A @ signal.reshape(-1))).astype(np.float32)
                elif rt.args.signal_space == "pix":
                    e_LO = (beam_cfg[beam_id].I2M_LO @ signal.reshape(-1)).astype(np.float32)
                    e_HO = (beam_cfg[beam_id].I2M_HO @ signal.reshape(-1)).astype(np.float32)
                else:
                    raise ValueError("signal_space must be 'dm' or 'pix'")

                data[beam_id]["imgs_cube"][idx, ai] = subframe_avg
                data[beam_id]["signal_cube"][idx, ai] = signal
                data[beam_id]["eLO_cube"][idx, ai] = e_LO
                data[beam_id]["eHO_cube"][idx, ai] = e_HO

    zero_cmd = np.zeros_like(modal_basis[0])
    for beam_id in rt.args.beam_id:
        rt.dm_shm_dict[beam_id].set_data(zero_cmd)

    return probe_amps, data


def write_beam_fits(
    rt: RuntimeContext,
    beam_cfg: Dict[int, BeamConfig],
    dark_current: Dict[int, np.ndarray],
    N0_current: Dict[int, np.ndarray],
    I0_current: Dict[int, np.ndarray],
    N0_runtime_current: Dict[int, float],
    modal_basis: np.ndarray,
    probe_amps: np.ndarray,
    data: Dict[int, dict],
    camera_config_current: dict,
) -> Dict[int, str]:
    tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tstamp_rough = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/pokeramp/{tstamp_rough}"
    os.makedirs(out_dir, exist_ok=True)

    fits_paths: Dict[int, str] = {}

    for beam_id in rt.args.beam_id:
        imgs_cube = data[beam_id]["imgs_cube"]
        signal_cube = data[beam_id]["signal_cube"]
        eLO_cube = data[beam_id]["eLO_cube"]
        eHO_cube = data[beam_id]["eHO_cube"]

        fits_path = f"{out_dir}/pokeramp_beam{beam_id}_{rt.args.phasemask}_{rt.args.basis_name}_{tstamp}.fits"
        fits_paths[beam_id] = fits_path

        hdr = fits.Header()
        hdr["DATE"] = tstamp
        hdr["BEAMID"] = int(beam_id)
        hdr["PHMASK"] = str(rt.args.phasemask)
        hdr["BASIS"] = str(rt.args.basis_name)
        hdr["SIGSPC"] = str(rt.args.signal_space)
        hdr["NMODE"] = int(imgs_cube.shape[0])
        hdr["NAMP"] = int(imgs_cube.shape[1])
        hdr["AMPMAX"] = float(rt.args.amp_max)
        hdr["NSPCMD"] = int(rt.args.no_samples_per_cmd)
        hdr["N0RUN0"] = float(beam_cfg[beam_id].N0_runtime)
        hdr["N0RUNC"] = float(N0_runtime_current[beam_id])
        hdr["TOML"] = rt.args.toml_file.replace("#", f"{beam_id}")[:68]

        hdus = [fits.PrimaryHDU(header=hdr)]
        hdus.append(fits.ImageHDU(data=dark_current[beam_id].astype(np.float32), name="DARK_CURR"))
        hdus.append(fits.ImageHDU(data=N0_current[beam_id].astype(np.float32), name="N0_CURR"))
        hdus.append(fits.ImageHDU(data=I0_current[beam_id].astype(np.float32), name="I0_CURR"))
        hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].N0).astype(np.float32), name="N0_TOML"))
        hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].I0).astype(np.float32), name="I0_TOML"))
        hdus.append(fits.ImageHDU(data=np.array(probe_amps, dtype=np.float32), name="PROBE_AMPS"))
        hdus.append(fits.ImageHDU(data=np.array(modal_basis, dtype=np.float32), name="MODAL_BASIS"))
        hdus.append(fits.ImageHDU(data=imgs_cube, name="IMGS"))
        hdus.append(fits.ImageHDU(data=signal_cube, name="SIGNAL"))
        hdus.append(fits.ImageHDU(data=eLO_cube, name="E_LO"))
        hdus.append(fits.ImageHDU(data=eHO_cube, name="E_HO"))

        if beam_cfg[beam_id].pupil_mask is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].pupil_mask, dtype=np.uint8), name="PUPIL_MASK"))
        if beam_cfg[beam_id].inner_pupil_filt is not None:
            hdus.append(
                fits.ImageHDU(data=np.array(beam_cfg[beam_id].inner_pupil_filt, dtype=np.uint8), name="INNER_PUPF")
            )

        if beam_cfg[beam_id].I2A is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].I2A, dtype=np.float32), name="I2A"))
        if beam_cfg[beam_id].I2M_LO is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].I2M_LO, dtype=np.float32), name="I2M_LO"))
        if beam_cfg[beam_id].I2M_HO is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].I2M_HO, dtype=np.float32), name="I2M_HO"))
        if beam_cfg[beam_id].M2C_LO is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].M2C_LO, dtype=np.float32), name="M2C_LO"))
        if beam_cfg[beam_id].M2C_HO is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].M2C_HO, dtype=np.float32), name="M2C_HO"))
        if beam_cfg[beam_id].IM is not None:
            hdus.append(fits.ImageHDU(data=np.array(beam_cfg[beam_id].IM, dtype=np.float32), name="IM"))

        cam_keys = np.array(list(camera_config_current.keys()), dtype="S64")
        cam_vals = np.array([str(v) for v in camera_config_current.values()], dtype="S128")
        cam_cols = [
            fits.Column(name="KEY", format="64A", array=cam_keys),
            fits.Column(name="VALUE", format="128A", array=cam_vals),
        ]
        hdus.append(fits.BinTableHDU.from_columns(cam_cols, name="CAMCFG_CURR"))

        toml_dump = toml.dumps(beam_cfg[beam_id].config_dict)
        toml_lines = np.array(toml_dump.splitlines(), dtype="S200")
        toml_col = fits.Column(name="TOML", format="200A", array=toml_lines)
        hdus.append(fits.BinTableHDU.from_columns([toml_col], name="TOML_DUMP"))

        fits.HDUList(hdus).writeto(fits_path, overwrite=True)
        print(f"wrote pokeramp FITS: {fits_path}")

    return fits_paths


def make_summary_plots(
    rt: RuntimeContext,
    probe_amps: np.ndarray,
    data: Dict[int, dict],
) -> None:
    os.makedirs(rt.args.fig_path, exist_ok=True)

    for beam_id in rt.args.beam_id:
        imgs_cube = data[beam_id]["imgs_cube"]
        eLO_cube = data[beam_id]["eLO_cube"]
        eHO_cube = data[beam_id]["eHO_cube"]

        n_mode = int(eLO_cube.shape[0])
        n_amp = int(eLO_cube.shape[1])
        n_lo = int(eLO_cube.shape[2])
        n_ho = int(eHO_cube.shape[2])

        i0 = int(np.argmin(np.abs(probe_amps)))
        k = 2
        i_lo = max(0, i0 - k)
        i_hi = min(len(probe_amps), i0 + k + 1)
        probe_amps_local = probe_amps[i_lo:i_hi]

        n_lo_plot = min(4, n_lo)
        n_ho_plot = min(4, n_ho)
        n_mode_plot = min(4, n_mode)

        for r in range(n_lo_plot):
            plt.figure(figsize=(6, 4))
            for m in range(n_mode_plot):
                y = eLO_cube[m, :, r]
                plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"poke mode {m}")
            plt.xlabel("poke amplitude")
            plt.ylabel(f"e_LO[{r}]")
            plt.title(f"LO error vs amplitude (full ramp)\nreconstructed LO index {r}  (beam{beam_id}, {rt.args.phasemask})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        for r in range(n_ho_plot):
            plt.figure(figsize=(6, 4))
            for m in range(n_mode_plot):
                y = eHO_cube[m, :, r]
                plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"poke mode {m}")
            plt.xlabel("poke amplitude")
            plt.ylabel(f"e_HO[{r}]")
            plt.title(f"HO error vs amplitude (full ramp)\nreconstructed HO index {r}  (beam{beam_id}, {rt.args.phasemask})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        n_diag = min(8, n_mode)
        for m in range(n_diag):
            if m >= n_lo:
                break
            y = eLO_cube[m, :, m]
            p = np.polyfit(probe_amps, y, 1)
            yfit = p[0] * probe_amps + p[1]
            plt.figure(figsize=(6, 4))
            plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"data  slope={p[0]:.3g}")
            plt.plot(probe_amps, yfit, linestyle="--", label="linear fit")
            plt.xlabel("poke amplitude")
            plt.ylabel(f"e_LO[{m}]")
            plt.title(f"LO diagonal response (full ramp)\npoke mode {m}  (beam{beam_id}, {rt.args.phasemask})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        for m in range(n_diag):
            if m >= n_ho:
                break
            y = eHO_cube[m, :, m]
            p = np.polyfit(probe_amps, y, 1)
            yfit = p[0] * probe_amps + p[1]
            plt.figure(figsize=(6, 4))
            plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"data  slope={p[0]:.3g}")
            plt.plot(probe_amps, yfit, linestyle="--", label="linear fit")
            plt.xlabel("poke amplitude")
            plt.ylabel(f"e_HO[{m}]")
            plt.title(f"HO diagonal response (full ramp)\npoke mode {m}  (beam{beam_id}, {rt.args.phasemask})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        slopes_LO = np.zeros((n_mode, n_lo), dtype=np.float32)
        slopes_HO = np.zeros((n_mode, n_ho), dtype=np.float32)
        slopes_LO_local = np.zeros((n_mode, n_lo), dtype=np.float32)
        slopes_HO_local = np.zeros((n_mode, n_ho), dtype=np.float32)

        for m in range(n_mode):
            for r in range(n_lo):
                slopes_LO[m, r] = np.polyfit(probe_amps, eLO_cube[m, :, r], 1)[0]
                slopes_LO_local[m, r] = np.polyfit(probe_amps_local, eLO_cube[m, i_lo:i_hi, r], 1)[0]
            for r in range(n_ho):
                slopes_HO[m, r] = np.polyfit(probe_amps, eHO_cube[m, :, r], 1)[0]
                slopes_HO_local[m, r] = np.polyfit(probe_amps_local, eHO_cube[m, i_lo:i_hi, r], 1)[0]

        plt.figure(figsize=(7, 5))
        plt.imshow(slopes_LO, aspect="auto")
        plt.colorbar(label="global slope (reco / amp)")
        plt.xlabel("LO reconstructed index")
        plt.ylabel("poked mode index")
        plt.title(f"LO slope matrix (global, full ramp)\nbeam{beam_id}, {rt.args.phasemask}")
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.imshow(slopes_LO_local, aspect="auto")
        plt.colorbar(label=f"local slope (±{k} samples) (reco / amp)")
        plt.xlabel("LO reconstructed index")
        plt.ylabel("poked mode index")
        plt.title(f"LO slope matrix (local around 0)\nbeam{beam_id}, {rt.args.phasemask}")
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.imshow(slopes_HO, aspect="auto")
        plt.colorbar(label="global slope (reco / amp)")
        plt.xlabel("HO reconstructed index")
        plt.ylabel("poked mode index")
        plt.title(f"HO slope matrix (global, full ramp)\nbeam{beam_id}, {rt.args.phasemask}")
        plt.tight_layout()

        plt.figure(figsize=(7, 5))
        plt.imshow(slopes_HO_local, aspect="auto")
        plt.colorbar(label=f"local slope (±{k} samples) (reco / amp)")
        plt.xlabel("HO reconstructed index")
        plt.ylabel("poked mode index")
        plt.title(f"HO slope matrix (local around 0)\nbeam{beam_id}, {rt.args.phasemask}")
        plt.tight_layout()

        n_img_show = min(4, int(imgs_cube.shape[0]))
        for m in range(n_img_show):
            for ai, tag in [(0, "amin"), (n_amp - 1, "amax")]:
                plt.figure(figsize=(5, 4))
                plt.imshow(imgs_cube[m, ai, :, :])
                plt.colorbar()
                plt.title(f"IMG poke mode {m} {tag}\nbeam{beam_id}, {rt.args.phasemask}")
                plt.tight_layout()

    print(f"summary plots saved to: {rt.args.fig_path}")
    plt.show()


def main() -> None:
    args = parse_args()
    rt = build_runtime(args)

    beam_cfg = load_beam_configs(rt)
    camera_config_current = {k: str(v) for k, v in rt.camclient.config.items()}

    dark_current = acquire_dark_all_beams(rt)
    N0_current, I0_current, N0_runtime_current, _ = acquire_reference_pupils(rt, beam_cfg, dark_current)
    modal_basis = build_modal_basis(rt)
    probe_amps, data = acquire_pokeramp(rt, beam_cfg, dark_current, N0_runtime_current, modal_basis)
    fits_paths = write_beam_fits(
        rt,
        beam_cfg,
        dark_current,
        N0_current,
        I0_current,
        N0_runtime_current,
        modal_basis,
        probe_amps,
        data,
        camera_config_current,
    )
    make_summary_plots(rt, probe_amps, data)

    print("Saved FITS files:")
    for beam_id, path in fits_paths.items():
        print(f"  beam {beam_id}: {path}")

    for b in rt.cam_shm:
        try:
            rt.cam_shm[b].close(erase_file=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
