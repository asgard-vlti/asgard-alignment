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

import asgard_alignment.adc_fns as af

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
    "BACU1",
    "BACL1",
    "BACU2",
    "BACL2",
    "BACU3",
    "BACL3",
    "BACU4",
    "BACL4",
]

ADC_UPPER_INDICES = [0, 2, 4, 6]
ADC_LOWER_INDICES = [1, 3, 5, 7]


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
        default=1.0,
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


def main():
    args = parse_args()
    constants = af.resolve_constants(args)

    ra, dec = af.parse_ra_dec(args.ra, args.dec)
    print(f"Parsed RA: {ra:.2f} degrees, Dec: {dec:.2f} degrees")

    alt, az = af.ra_dec_to_altaz(ra, dec, constants)

    context, client = af.connect_socket()
    try:
        print(f"Target Altitude: {alt:.2f} degrees, Azimuth: {az:.2f} degrees")
        if args.track:
            print("Starting ADC tracking loop")
            while True:
                alt, az = af.ra_dec_to_altaz(ra, dec, constants)

                if alt < 20:
                    print("ERROR: Target is below 20 degrees altitude.")
                    sys.exit(1)
                af.perform_slew_cycle(client, constants, alt, az)
                if args.dry_run:
                    break
                time.sleep(args.track_interval)
        else:
            if alt < 20:
                print("ERROR: Target is below 20 degrees altitude.")
                sys.exit(1)

            if args.dry_run:
                print("Dry run mode - not sending slew command")
            else:
                af.perform_slew_cycle(client, constants, alt, az)

        if not args.dry_run:
            client.send_and_recv("adc_disable")
        print("ADC command sequence complete.")
    finally:
        client.close()
        context.term()


if __name__ == "__main__":
    main()
