"""Sweep one Heimdallr HPOL motor and record baseline squared visibilities."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
import zmq


beam_number = 1
scan_width = 300
scan_nsteps = 11
fringe_band = "K1"
fringe_srange = 8
fringe_step = 0.5
status_message_count = 50

MDS_ENDPOINT = "tcp://192.168.100.2:5555"
HEIMDALLR_ENDPOINT = "tcp://192.168.100.2:6660"
BASELINES = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
MOVE_TIMEOUT_SECONDS = 60


def open_socket(context, endpoint):
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.SNDTIMEO, 10000)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    return socket


def request(socket, command):
    socket.send_string(command)
    return socket.recv_string().strip()


def read_hpol(socket, beam):
    return int(request(socket, f"read HPOL{beam}"))


def move_hpol(socket, beam, position):
    response = request(socket, f"moveabs HPOL{beam} {position}")
    if response != "ACK":
        raise RuntimeError(f"MDS rejected HPOL{beam} move to {position}: {response}")

    deadline = time.monotonic() + MOVE_TIMEOUT_SECONDS
    while read_hpol(socket, beam) != position:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"HPOL{beam} did not reach {position} steps")
        time.sleep(0.2)


def sample_visibilities(socket, baseline_indices):
    samples = onp.empty((status_message_count, 2, 3), dtype=float)
    for sample_index in range(status_message_count):
        status = json.loads(request(socket, "status"))
        for band_index, key in enumerate(("v2_K1", "v2_K2")):
            values = onp.asarray(status[key], dtype=float)
            if values.shape != (len(BASELINES),) or not onp.all(onp.isfinite(values)):
                raise ValueError(f"Invalid {key} in Heimdallr status: {values}")
            samples[sample_index, band_index] = values[baseline_indices]
    return samples.mean(axis=0)


def run_scan(mds_socket, heimdallr_socket, run_command=subprocess.run):
    if beam_number not in (1, 2, 3, 4):
        raise ValueError("beam_number must be between 1 and 4")
    if scan_width <= 0 or scan_nsteps < 2:
        raise ValueError("scan_width must be positive and scan_nsteps must be at least 2")

    starting_position = read_hpol(mds_socket, beam_number)
    positions = onp.rint(
        onp.linspace(
            starting_position - scan_width / 2,
            starting_position + scan_width / 2,
            scan_nsteps,
        )
    ).astype(int)
    if len(onp.unique(positions)) != scan_nsteps:
        raise ValueError("scan_nsteps produces duplicate integer HPOL positions")

    baseline_indices = [
        index for index, pair in enumerate(BASELINES) if beam_number in pair
    ]
    baseline_labels = onp.array(
        [f"{BASELINES[index][0]}-{BASELINES[index][1]}" for index in baseline_indices]
    )
    averaged_v2 = onp.empty((scan_nsteps, 2, 3), dtype=float)
    fringe_command = [
        "find-fringes",
        fringe_band,
        str(fringe_srange),
        str(fringe_step),
    ]

    try:
        for index, position in enumerate(positions):
            print(f"Sample {index + 1}/{scan_nsteps}: HPOL{beam_number} = {position}")
            move_hpol(mds_socket, beam_number, int(position))
            run_command(fringe_command, check=True)
            run_command(["h-tilts"], check=True)
            run_command(fringe_command, check=True)
            averaged_v2[index] = sample_visibilities(
                heimdallr_socket, baseline_indices
            )
    finally:
        print(f"Restoring HPOL{beam_number} to {starting_position}")
        try:
            move_hpol(mds_socket, beam_number, starting_position)
        except zmq.ZMQError:
            restore_socket = open_socket(mds_socket.context, MDS_ENDPOINT)
            try:
                move_hpol(restore_socket, beam_number, starting_position)
            finally:
                restore_socket.close()

    return starting_position, positions, baseline_labels, averaged_v2


def save_results(starting_position, positions, baseline_labels, averaged_v2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_base = Path.cwd() / f"linbo_search_beam{beam_number}_{timestamp}"
    data_path = output_base.with_suffix(".npz")
    onp.savez(
        data_path,
        averaged_v2=averaged_v2,
        hpol_positions=positions,
        baseline_labels=baseline_labels,
        starting_hpol_position=starting_position,
        beam_number=beam_number,
        scan_width=scan_width,
        scan_nsteps=scan_nsteps,
        fringe_band=fringe_band,
        fringe_srange=fringe_srange,
        fringe_step=fringe_step,
        status_message_count=status_message_count,
    )
    print(f"Saved {data_path}")

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for band_index, band in enumerate(("K1", "K2")):
        for baseline_index, label in enumerate(baseline_labels):
            axes[band_index].plot(
                positions,
                averaged_v2[:, band_index, baseline_index],
                marker="o",
                label=f"B{label}",
            )
        axes[band_index].set_ylabel(f"{band} V²")
        axes[band_index].legend()
        axes[band_index].grid(True)
    axes[-1].set_xlabel(f"HPOL{beam_number} position (steps)")
    figure.tight_layout()
    figure_path = output_base.with_suffix(".png")
    figure.savefig(figure_path)
    print(f"Saved {figure_path}")
    plt.show()
    plt.close(figure)


def main():
    context = zmq.Context()
    mds_socket = open_socket(context, MDS_ENDPOINT)
    heimdallr_socket = open_socket(context, HEIMDALLR_ENDPOINT)
    try:
        results = run_scan(mds_socket, heimdallr_socket)
        save_results(*results)
    finally:
        mds_socket.close()
        heimdallr_socket.close()
        context.term()


if __name__ == "__main__":
    main()
