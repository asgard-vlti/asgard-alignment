"""
Code to slew and track the ADCs.

Runs with a script, with inputs
RA = hhmmss.ssss as a string
Dec = +ddmmss.ss as a string
"""

# TODO: make a similar script that moves them all to zeros
# TODO: argparse/config of slew (init guess only) or track (continuous loop)

import sys
import time
import zmq
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

# Observatory coordinates
lat = -24.62743941
long = -70.40498689
el = 2669

# TODO: make argparse, or if there are no argparse use a config file
# Constants - to be over-written by keywords
const_a = 1.0
sign1 = 1.0

# !!! A global that should come from /usr/local/etc.
# TODO: only this script knows the zerospos
adc_zeropos = [-2000, -14800, -7400, -16500, -7800, 1200, 13200, -13200]
adc_names = [
    "ADC1A",
    "ADC1B",
    "ADC2A",
    "ADC2B",
    "ADC3A",
    "ADC3B",
    "ADC4A",
    "ADC4B",
]


def ra_dec_to_altaz(ra_deg, dec_deg):
    location = EarthLocation(lat=lat * u.deg, lon=long * u.deg, height=el * u.m)
    obstime = Time(time.time(), format="unix")
    skycoord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz = skycoord.transform_to(AltAz(obstime=obstime, location=location))
    return altaz.alt.deg, altaz.az.deg


def main():
    # Use argv to determine the mode (FAINT or STANDARD) and other parameters
    if len(sys.argv) != 3:
        print("Usage: python adc_track.py RA Dec")
        sys.exit(1)
    # Convert RA string into RA in degrees.
    ra_str = sys.argv[1]
    ra = float(ra_str[:2]) + float(ra_str[2:4]) / 60 + float(ra_str[4:]) / 3600
    ra *= 15  # Convert hours to degrees
    dec_str = sys.argv[2]
    if dec_str[0] == "-":
        sign = -1
        dec_str = dec_str[1:]  # Remove the negative sign for parsing
    else:
        sign = 1
    if dec_str[0] == "+":
        dec_str = dec_str[1:]  # Remove the positive sign for parsing
    dec = float(dec_str[:2]) + float(dec_str[2:4]) / 60 + float(dec_str[4:]) / 3600
    dec *= sign
    print(f"Parsed RA: {ra:.2f} degrees, Dec: {dec:.2f} degrees")
    # Convert RA/Dec to Alt/Az
    alt, az = ra_dec_to_altaz(ra, dec)
    print(f"Target Altitude: {alt:.2f} degrees, Azimuth: {az:.2f} degrees")
    if alt < 20:
        print("ERROR: Target is below 20 degrees altitude.")
        sys.exit(1)
    # Connect to MDS and find where the ADCs currently are
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    server_address = "tcp://mimir:5555"
    socket.connect(server_address)
    # Verify that the "A" motors all have the same position.
    adc_a_positions = []
    for i in range(4):
        message = f"read {adc_names[2*i]}"
        socket.send_string(message)
        response = socket.recv_string().strip()
        adc_a_positions.append(float(response) - adc_zeropos[2 * i])
    if not all(pos == adc_a_positions[0] for pos in adc_a_positions):
        print("ERROR: Not all 'A' motors are at the same position. Need to zero.")
        sys.exit(1)
    # Very that the "B" motors all have the same position.
    adc_b_positions = []
    for i in range(4):
        message = f"read {adc_names[2*i+1]}"
        socket.send_string(message)
        response = socket.recv_string().strip()
        # zeropos is taken into account from the MDS
        adc_b_positions.append(float(response) - adc_zeropos[2 * i + 1])
    if not all(pos == adc_b_positions[0] for pos in adc_b_positions):
        print("ERROR: Not all 'B' motors are at the same position. Need to zero.")
        sys.exit(1)
    # Calculate the required ADC positions to track the target.
    dispersion = np.cot(np.radians(alt))  # Simplified dispersion model
    delta = np.degrees(np.acos(1 - const_a * dispersion))
    # TODO: if delta < -1, stop the loop
    adc_a_target = sign1 * (az - alt - 12.98) + delta
    adc_b_target = sign1 * (az - alt - 12.98) - delta
    # Convert to centidegrees as an integer.
    adc_a_target = int(adc_a_target * 100)
    adc_b_target = int(adc_b_target * 100)
    print(
        f"Calculated ADC target positions: A={adc_a_target} centidegrees, B={adc_b_target} centidegrees"
    )

    # Slew the ADCs to the target positions. This is done as a
    # special relative move.
    for AB, adc_target, adc_positions in zip(
        ["A", "B"], [adc_a_target, adc_b_target], [adc_a_positions, adc_b_positions]
    ):
        relative_target = adc_target - adc_positions[0]
        message = f"adc_slew {AB} {relative_target}"
        socket.send_string(message)
        # We have to wait for the A slew to finish before starting the B slew, otherwise we get weird behavior where the B motors start moving and then the A motors start moving again.
        adc_1_abs_target = adc_a_target + adc_zeropos[0]
        adc_1_currentpos = adc_a_positions[0] + adc_zeropos[0]
        print(f"Waiting for ADC {AB}1 to slew to {adc_1_abs_target} centidegrees...")
        totaltime = 0
        while (adc_1_currentpos != adc_1_abs_target) and (totaltime < 60):
            time.sleep(0.5)
            message = f"read {adc_names[0]}"
            socket.send_string(message)
            response = socket.recv_string().strip()
            adc_1_currentpos = float(response)
            totaltime += 0.5
        if adc_1_currentpos != adc_1_abs_target:
            print(
                f"ERROR: ADC {AB}1 did not reach the target position within 60 seconds."
            )
            sys.exit(1)
        print(f"ADC {AB}1 reached the target position.")
    message = f"adc_disable"
    socket.send_string(message)
    response = socket.recv_string().strip()
    print("ADC slew complete. Now tracking the target... (not implemented yet)")


if __name__ == "__main__":
    main()
