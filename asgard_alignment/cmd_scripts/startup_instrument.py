"""
A full system startup, starting from the case where only mimir is on
"""

import os
from asgard_alignment.PDU_telnet import AtenEcoPDU
import sys
import time
import subprocess


def ping_test(ip_address):
    response = os.system(f"ping -c 1 {ip_address} > /dev/null 2>&1")
    return response == 0


def power_on_all(power_on_camera=False):
    # Box power ON through 192.168.100.11, port [05]
    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()
    outlets_to_power = [5]
    if power_on_camera:
        print("Powering on the camera...")
        pdu.switch_outlet_status(6, "on")
        outlets_to_power.append(6)

    print("Powering on the box...")
    pdu.switch_outlet_status(5, "on")

    is_on = {outlet: False for outlet in outlets_to_power}

    while not all(is_on.values()):
        for outlet in is_on.keys():
            res = pdu.read_outlet_status(outlet)
            if res == "on":
                is_on[outlet] = True
            else:
                print(f"Outlet {outlet} status: {res}")
        if not all(is_on.values()):
            print("Waiting for the power on...")
            time.sleep(1)  # wait for the box to power on

    print(
        "Box and camera powered on successfully."
        if power_on_camera
        else "Box powered on successfully."
    )

    time.sleep(5)

    # Ping test 192.168.100.10
    if not ping_test("192.168.100.10"):
        print(
            "Ping test failed for 192.168.100.10 (controllino). Trying once more time..."
        )
        time.sleep(5)
        if not ping_test("192.168.100.10"):
            print("Ping test failed for 192.168.100.10 (controllino). Exiting.")
            sys.exit(1)

    # Start the installed status GUI in its own session so it outlives this terminal.
    print("Starting status-mimir...")
    subprocess.Popen(
        ["/home/asg/.conda/envs/asgard/bin/status-mimir"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Sleep a tiny bit more, for MDS boot-up.
    time.sleep(0.5)

    # Start services as subprocesses so they remain independent of this script.
    print("Starting MDS, expect ~5s delay...")
    subprocess.run("/usr/local/bin/run_mds")
    time.sleep(5)
    print("Starting engineering GUI...")
    subprocess.run("/usr/local/bin/run_eng_gui")

    print("Starting camera & DM servers, expect ~30s delay...")
    subprocess.run("/usr/local/bin/run_cam_server")
    time.sleep(15)
    subprocess.run("/usr/local/bin/run_DM_server")
    time.sleep(5)

    print("Starting RTTs (Heimdallr and Baldr, if not already started)...")
    subprocess.run("/usr/local/bin/run_heimdallr")
    subprocess.run(["/usr/local/bin/run_baldr_tt", "1"])
    subprocess.run(["/usr/local/bin/run_baldr_tt", "2"])
    subprocess.run(["/usr/local/bin/run_baldr_tt", "3"])
    subprocess.run(["/usr/local/bin/run_baldr_tt", "4"])
    subprocess.run(["/usr/local/bin/run_baldr", "1"])
    subprocess.run(["/usr/local/bin/run_baldr", "2"])
    subprocess.run(["/usr/local/bin/run_baldr", "3"])
    subprocess.run(["/usr/local/bin/run_baldr", "4"])
    time.sleep(1)

    print("Starting DCS (back-end) server and clients/telemetry...")
    # The back_end_server is DCS as far as wag is concerned.
    subprocess.run("/usr/local/bin/run_back_end_server")

    # The MCS client passes information to WAG. If wag isn't started up, hopefully it is robust!
    # As this is program that communicates with everything, it is used for the "mimir status" gui.
    # Currently does not exist
    print("Starting MCS client...")
    subprocess.run("/usr/local/bin/run_mcs_client")

    # This is telemetry for heimdally and baldr_tt
    print("Starting telemetry...")
    subprocess.run("/usr/local/bin/run_telem")

    # Loading the laboratory (internal) flats
    time.sleep(1)
    print("Loading laboratory flats...")
    subprocess.run(["/home/asg/.conda/envs/asgard/bin/flat-load", "-1", "lab"])

    print("All commands executed successfully. Instrument startup complete.")
    print("Load a state using the gui, and run 'fetch' on the camera server")


def main():
    # check which mode to run
    inp = (
        input("Do you want to power on the C RED one camera too? (y/n): ")
        .strip()
        .lower()
    )
    if inp == "y":
        power_on_all(power_on_camera=True)
    elif inp == "n":
        power_on_all()
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
