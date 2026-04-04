# TODO: move as many of each set at once to the average relative offset, then a fine tune move abs
# TODO: check if any adc-track processes are running on the machine and warn if so. If confirmed, kill the processes and continue
import argparse
import os
import sys
import time
from dataclasses import dataclass

import zmq
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import adc_fns as af

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib


def send_and_recv(socket, message):
    socket.send_string(message)
    return socket.recv_string().strip()


def load_config(config_path):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    return [
        float(cfg["adcs_zeropos"]["BADCU1"]),
        float(cfg["adcs_zeropos"]["BADCL1"]),
        float(cfg["adcs_zeropos"]["BADCU2"]),
        float(cfg["adcs_zeropos"]["BADCL2"]),
        float(cfg["adcs_zeropos"]["BADCU3"]),
        float(cfg["adcs_zeropos"]["BADCL3"]),
        float(cfg["adcs_zeropos"]["BADCU4"]),
        float(cfg["adcs_zeropos"]["BADCL4"]),
    ]


def wait_until_reached(
    socket, adc_name, abs_target, timeout_s=120, poll_interval_s=0.5
):
    print(f"Waiting for {adc_name} to reach {abs_target}...")
    total_time = 0.0

    while total_time < timeout_s:
        time.sleep(poll_interval_s)
        response = send_and_recv(socket, f"read {adc_name}")
        try:
            current = float(response)
        except ValueError:
            print(f"WARN: Could not parse position for {adc_name}: {response!r}")
            total_time += poll_interval_s
            continue

        if current == abs_target:
            print(f"{adc_name} reached the target position.")
            return

        total_time += poll_interval_s

    print(f"ERROR: {adc_name} did not reach {abs_target} within {timeout_s} seconds.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Move ADCs to zero positions")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config_files",
            "adc_slew_config.toml",
        ),
        help="Path to the ADC zero positions config file",
    )
    args = parser.parse_args()

    adc_zeropos = load_config(args.config)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://mimir:5555")

    relpos_upper = af.read_relative_positions(socket, af.ADC_UPPER_INDICES, adc_zeropos)
    relpos_lower = af.read_relative_positions(socket, af.ADC_LOWER_INDICES, adc_zeropos)

    # measure the average relative offset for each group
    avg_relpos_upper = np.mean(relpos_upper)
    avg_relpos_lower = np.mean(relpos_lower)

    # check if any values are different from the mean, print a warning if so
    if not np.allclose(relpos_upper, avg_relpos_upper, atol=1.0):
        print(
            "WARN: Not all upper ADCs are at the same relative position. "
            f"Relative positions: {relpos_upper}, average: {avg_relpos_upper}"
        )
    if not np.allclose(relpos_lower, avg_relpos_lower, atol=1.0):
        print(
            "WARN: Not all lower ADCs are at the same relative position. "
            f"Relative positions: {relpos_lower}, average: {avg_relpos_lower}"
        )

    # relmove the group by the int(average)
    relmove_upper = int(avg_relpos_upper)
    relmove_lower = int(avg_relpos_lower)

    group_label = "adc_upper"
    message = f"rotm_slew {group_label} {-relmove_upper}"
    print(f"Relmoving {group_label} by {-relmove_upper} steps...")
    response = send_and_recv(socket, message)
    print(f"Response: {response}")

    group_label = "adc_lower"
    message = f"rotm_slew {group_label} {-relmove_lower}"
    print(f"Relmoving {group_label} by {-relmove_lower} steps...")
    response = send_and_recv(socket, message)
    print(f"Response: {response}")

    for i, zeropos in enumerate(adc_zeropos):
        upper_or_lower = "upper" if i % 2 == 0 else "lower"
        adc_num = (i // 2) + 1
        adc_name = f"BAC{upper_or_lower[0].upper()}{adc_num}"
        print(f"Moving {adc_name} to zero position: {zeropos}")
        response = send_and_recv(socket, f"moveabs {adc_name} {zeropos}")
        print(f"Response: {response}")
        wait_until_reached(socket, adc_name, zeropos)


if __name__ == "__main__":
    main()
