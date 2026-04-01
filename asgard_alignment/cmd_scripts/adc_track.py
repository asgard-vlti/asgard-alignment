"""
Code to slew and track the ADCs.

Runs with a script, with inputs
RA = hhmmss.ssss as a string
Dec = +ddmmss.ss as a string
"""

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
from astropy.utils import iers

iers.conf.auto_download = False
print(iers.conf.iers_degraded_accuracy)
iers.conf.iers_degraded_accuracy = "warn"

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib

# TODO: check where this should live
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config_files",
    "adc_slew_config.toml",
)
SERVER_ENDPOINT = "tcp://mimir:5555"
REQUEST_TIMEOUT_MS = 10000
REQUEST_RETRIES = 3

adc_names = [
    "BADCU1",
    "BADCL1",
    "BADCU2",
    "BADCL2",
    "BADCU3",
    "BADCL3",
    "BADCU4",
    "BADCL4",
]

ADC_UPPER_INDICES = [0, 2, 4, 6]
ADC_LOWER_INDICES = [1, 3, 5, 7]


@dataclass
class ADCConstants:
    lat: float
    lon: float
    el: float
    const_a: float
    sign1: float
    adc_zeropos: list[float]


class ZmqLazyPirateClient:
    def __init__(self, context, endpoint, timeout_ms=REQUEST_TIMEOUT_MS):
        self.context = context
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.socket = self._create_socket()

    def _create_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def send_and_recv(self, message, retries=REQUEST_RETRIES):
        for attempt in range(1, retries + 1):
            self.socket.send_string(message)
            if self.socket.poll(self.timeout_ms, zmq.POLLIN):
                return self.socket.recv_string().strip()

            print(
                f"WARN: Timeout waiting for reply to '{message}' "
                f"(attempt {attempt}/{retries})."
            )
            self.socket.close()
            self.socket = self._create_socket()

        print(
            f"ERROR: No reply from ADC server after {retries} attempts for '{message}'."
        )
        sys.exit(1)

    def close(self):
        self.socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Slew/track ADCs from target RA/Dec")
    parser.add_argument("ra", help="Target RA as hhmmss.ssss")
    parser.add_argument("dec", help="Target Dec as [+/-]ddmmss.ss")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to ADC slew config TOML",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not send any move command"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--slew", action="store_true", help="Run one-shot slew")
    mode.add_argument(
        "--track", action="store_true", help="Continuously recompute and slew"
    )
    parser.add_argument(
        "--track-interval",
        type=float,
        default=5.0,
        help="Seconds between track updates (only with --track)",
    )

    # Optional CLI overrides for constants. If omitted, values come from config.
    parser.add_argument("--lat", type=float, help="Observatory latitude in deg")
    parser.add_argument(
        "--long", dest="lon", type=float, help="Observatory longitude in deg"
    )
    parser.add_argument("--el", type=float, help="Observatory elevation in m")
    parser.add_argument("--const-a", type=float, help="Dispersion equation constant a")
    parser.add_argument("--sign1", type=float, help="Dispersion equation sign1")
    parser.add_argument(
        "--adc-zeropos",
        nargs=8,
        type=float,
        metavar=(
            "BADCU1",
            "BADCL1",
            "BADCU2",
            "BADCL2",
            "BADCU3",
            "BADCL3",
            "BADCU4",
            "BADCL4",
        ),
        help="ADC zero positions in centidegrees",
    )
    return parser.parse_args()


def parse_ra_dec(ra_str, dec_str):
    if "." not in ra_str and "." not in dec_str:
        # case where the format is hhmmss or ddmmss are ints
        ra_int = int(ra_str)
        dec_int = int(dec_str)
        ra = ra_int // 10000 + (ra_int % 10000) // 100 / 60 + (ra_int % 100) / 3600
        sign = -1 if dec_int < 0 else 1
        dec_int = abs(dec_int)
        dec = sign * (
            dec_int // 10000 + (dec_int % 10000) // 100 / 60 + (dec_int % 100) / 3600
        )

    elif "." in ra_str and "." in dec_str:
        ra = float(ra_str[:2]) + float(ra_str[2:4]) / 60 + float(ra_str[4:]) / 3600
        sign = -1 if dec_str[0] == "-" else 1
        if dec_str[0] in ["-", "+"]:
            dec_str = dec_str[1:]
        dec = float(dec_str[:2]) + float(dec_str[2:4]) / 60 + float(dec_str[4:]) / 3600
        dec *= sign
    else:
        print(
            "ERROR: RA and Dec must both be in the same format (either hhmmss/ddmmss or hhmmss.ssss/ddmmss.ss)."
        )
        sys.exit(1)
    ra *= 15

    return ra, dec


def load_config(config_path):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    return ADCConstants(
        lat=float(cfg["observatory"]["lat"]),
        lon=float(cfg["observatory"]["long"]),
        el=float(cfg["observatory"]["el"]),
        const_a=float(cfg["dispersion_eqns"]["const_a"]),
        sign1=float(cfg["dispersion_eqns"]["sign1"]),
        adc_zeropos=[
            float(cfg["adcs_zeropos"]["BADCU1"]),
            float(cfg["adcs_zeropos"]["BADCL1"]),
            float(cfg["adcs_zeropos"]["BADCU2"]),
            float(cfg["adcs_zeropos"]["BADCL2"]),
            float(cfg["adcs_zeropos"]["BADCU3"]),
            float(cfg["adcs_zeropos"]["BADCL3"]),
            float(cfg["adcs_zeropos"]["BADCU4"]),
            float(cfg["adcs_zeropos"]["BADCL4"]),
        ],
    )


def resolve_constants(args):
    need_config = any(
        value is None
        for value in [
            args.lat,
            args.lon,
            args.el,
            args.const_a,
            args.sign1,
            args.adc_zeropos,
        ]
    )
    cfg = load_config(args.config) if need_config else None
    if need_config:
        assert cfg is not None
    return ADCConstants(
        lat=args.lat if args.lat is not None else cfg.lat,
        lon=args.lon if args.lon is not None else cfg.lon,
        el=args.el if args.el is not None else cfg.el,
        const_a=args.const_a if args.const_a is not None else cfg.const_a,
        sign1=args.sign1 if args.sign1 is not None else cfg.sign1,
        adc_zeropos=(
            args.adc_zeropos if args.adc_zeropos is not None else cfg.adc_zeropos
        ),
    )


def ra_dec_to_altaz(ra_deg, dec_deg, constants):
    location = EarthLocation(
        lat=constants.lat * u.deg,
        lon=constants.lon * u.deg,
        height=constants.el * u.m,
    )
    obstime = Time(time.time(), format="unix")
    skycoord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = skycoord.transform_to(AltAz(obstime=obstime, location=location))
    return float(altaz.alt.to_value(u.deg)), float(altaz.az.to_value(u.deg))


def read_relative_positions(client, indices, zeropos):
    positions = []
    for i in indices:
        response = client.send_and_recv(f"read {adc_names[i]}")
        positions.append(float(response) - zeropos[i])
    return positions


def ensure_common_position(label, positions):
    if not all(pos == positions[0] for pos in positions):
        print(
            f"ERROR: Not all '{label}' motors are at the same position. Need to zero."
        )
        sys.exit(1)


def calculate_adc_targets(alt, az, constants):
    dispersion = 1.0 / np.tan(np.radians(alt))
    acos_arg = 1 - constants.const_a * dispersion
    if acos_arg < -1 or acos_arg > 1:
        print(
            f"ERROR: Invalid acos input {acos_arg:.6f}. Target not reachable for current model."
        )
        sys.exit(1)

    delta = np.degrees(np.arccos(acos_arg))
    adc_a_target = constants.sign1 * (az - alt - 12.98) + delta
    adc_b_target = constants.sign1 * (az - alt - 12.98) - delta
    return int(adc_a_target * 100), int(adc_b_target * 100)


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


def slew_group(client, group_label, adc_target, current_positions, zeropos):
    if group_label == "adc_upper":
        group_indices = ADC_UPPER_INDICES
    elif group_label == "adc_lower":
        group_indices = ADC_LOWER_INDICES
    else:
        print(f"ERROR: Unknown ADC group label '{group_label}'.")
        sys.exit(1)

    current_reference = current_positions[0]
    relative_target = int(adc_target - current_reference)
    message = f"adc_slew {group_label} {relative_target}"

    client.send_and_recv(message)
    for motor_index in group_indices:
        abs_target = adc_target + zeropos[motor_index]
        wait_until_reached(client, motor_index, abs_target)


def perform_slew_cycle(client, constants, alt, az):
    adc_a_positions = read_relative_positions(
        client, ADC_UPPER_INDICES, constants.adc_zeropos
    )
    ensure_common_position("adc_upper", adc_a_positions)

    adc_b_positions = read_relative_positions(
        client, ADC_LOWER_INDICES, constants.adc_zeropos
    )
    ensure_common_position("adc_lower", adc_b_positions)

    adc_a_target, adc_b_target = calculate_adc_targets(alt, az, constants)
    print(
        "Calculated ADC target positions: "
        f"adc_upper={adc_a_target} centidegrees, adc_lower={adc_b_target} centidegrees"
    )

    slew_group(
        client,
        "adc_upper",
        adc_a_target,
        adc_a_positions,
        constants.adc_zeropos,
    )
    slew_group(
        client,
        "adc_lower",
        adc_b_target,
        adc_b_positions,
        constants.adc_zeropos,
    )


def connect_socket():
    context = zmq.Context()
    client = ZmqLazyPirateClient(context, SERVER_ENDPOINT)
    return context, client


def main():
    args = parse_args()
    constants = resolve_constants(args)

    ra, dec = parse_ra_dec(args.ra, args.dec)
    print(f"Parsed RA: {ra:.2f} degrees, Dec: {dec:.2f} degrees")

    alt, az = ra_dec_to_altaz(ra, dec, constants)

    if args.dry_run:
        print(
            f"Dry run complete. Computed Alt: {alt:.2f} degrees, Az: {az:.2f} degrees"
        )
        return

    context, client = connect_socket()
    try:
        print(f"Target Altitude: {alt:.2f} degrees, Azimuth: {az:.2f} degrees")
        if args.track:
            print("Starting ADC tracking loop")
            while True:
                alt, az = ra_dec_to_altaz(ra, dec, constants)
                if alt < 20:
                    print("ERROR: Target is below 20 degrees altitude.")
                    sys.exit(1)
                perform_slew_cycle(client, constants, alt, az)
                if args.dry_run:
                    break
                time.sleep(args.track_interval)
        else:
            if alt < 20:
                print("ERROR: Target is below 20 degrees altitude.")
                sys.exit(1)
            perform_slew_cycle(client, constants, alt, az)

        if not args.dry_run:
            client.send_and_recv("adc_disable")
        print("ADC command sequence complete.")
    finally:
        client.close()
        context.term()


if __name__ == "__main__":
    main()
