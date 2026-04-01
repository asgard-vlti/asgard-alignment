# TODO: make a similar script that moves them all to zeros
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


def wait_until_reached(client, motor_index, abs_target, timeout_s=60):
    motor_name = adc_names[motor_index]
    print(f"Waiting for {motor_name} to slew to {abs_target} centidegrees...")
    total_time = 0.0
    current = None
    while total_time < timeout_s:
        time.sleep(0.5)
        response = client.send_and_recv(f"read {motor_name}")
        current = float(response)
        if current == abs_target:
            print(f"{motor_name} reached the target position.")
            return
        total_time += 0.5

    print(
        f"ERROR: {motor_name} did not reach the target position within {timeout_s} seconds."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Move ADCs to zero positions")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config_files",
            "adc_zeropos_config.toml",
        ),
        help="Path to the ADC zero positions config file",
    )
    args = parser.parse_args()

    adc_zeropos = load_config(args.config)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://mimir:5555")

    for i, zeropos in enumerate(adc_zeropos):
        upper_or_lower = "upper" if i % 2 == 0 else "lower"
        adc_num = (i // 2) + 1
        adc_name = f"BAC{upper_or_lower[0].upper()}{adc_num}"
        print(f"Moving {adc_name} to zero position: {zeropos}")
        response = send_and_recv(socket, f"moveabs {adc_name} {zeropos}")
        print(f"Response: {response}")
