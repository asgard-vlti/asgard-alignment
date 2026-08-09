import argparse
from asgard_alignment.PDU_telnet import AtenEcoPDU
import zmq
import asgard_alignment.controllino as co
import time
import socket
import subprocess

LOWER_BOX_OUTLET = 5
C_RED_OUTLET = 6
MDS_HOST = "192.168.100.2"
MDS_PORT = 5555
MDS_WAIT_TIMEOUT_S = 45
MDS_POLL_INTERVAL_S = 1
C_RED_PORT = 6667
C_RED_WAIT_TIMEOUT_S = 10


def ping_device(ip):
    result = subprocess.run(
        ["ping", "-c", "1", ip],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def open_zmq_connection(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    server_address = f"tcp://{MDS_HOST}:{port}"
    socket.connect(server_address)
    return socket


def send_and_get_response(socket, string):
    socket.send_string(string)
    res = socket.recv_string()
    return res


def is_tcp_port_open(host, port, timeout_s=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def restart_mds_from_path():
    # Policy constraint: MDS restart must happen only via run_mds on PATH.
    subprocess.run(["run_mds"], check=True)


def wait_for_mds(host, port, total_wait_s=MDS_WAIT_TIMEOUT_S, poll_s=MDS_POLL_INTERVAL_S):
    deadline = time.time() + total_wait_s
    while time.time() < deadline:
        if is_tcp_port_open(host, port):
            return True
        time.sleep(poll_s)
    return False


def wait_for_tcp_port(host, port, total_wait_s, poll_s=1):
    deadline = time.time() + total_wait_s
    while time.time() < deadline:
        if is_tcp_port_open(host, port):
            return True
        time.sleep(poll_s)
    return False


def get_mds_connection_or_recover():
    if not is_tcp_port_open(MDS_HOST, MDS_PORT):
        print("MDS is not reachable. Restarting MDS using run_mds...")
        try:
            restart_mds_from_path()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "run_mds was not found on PATH. Install/activate it before running shutdown."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"run_mds failed with exit code {exc.returncode}.") from exc

        print("Waiting for MDS to become reachable...")
        if not wait_for_mds(MDS_HOST, MDS_PORT):
            raise RuntimeError(
                f"MDS did not come up within {MDS_WAIT_TIMEOUT_S}s after run_mds restart attempt."
            )

    return open_zmq_connection(MDS_PORT)


def send_with_mds_recovery(mds_connection, command):
    try:
        return send_and_get_response(mds_connection, command), mds_connection
    except zmq.error.Again:
        print(f"MDS timeout while sending '{command}'. Attempting MDS recovery...")
        mds_connection = get_mds_connection_or_recover()
        try:
            return send_and_get_response(mds_connection, command), mds_connection
        except zmq.error.Again as exc:
            raise RuntimeError(
                f"MDS remained unresponsive after recovery while sending '{command}'."
            ) from exc


def shutdown(inc_CRED):
    try:
        mds_connection = get_mds_connection_or_recover()
    except RuntimeError as exc:
        print(f"Unable to establish MDS connection: {exc}")
        print("Aborting shutdown.")
        return

    date = time.strftime("%Y-%m-%d %H:%M:%S")

    # try:
    #     res = send_and_get_response(mds_connection, f"save all before_shutdown_{date}")
    #     print("saved", res)
    # except zmq.error.Again:
    #     inp = input(
    #         "MDS did not respond (and hence state is not saved). Do you want to continue with shutdown? (y/n): "
    #     )
    #     if inp.lower() != "y":
    #         print("Aborting shutdown.")
    #         return
    #     print("Proceeding with shutdown...")

    if inc_CRED:
        if not wait_for_tcp_port(MDS_HOST, C_RED_PORT, C_RED_WAIT_TIMEOUT_S):
            print(
                f"Warning: C-RED connection not reachable after {C_RED_WAIT_TIMEOUT_S}s; "
                "continuing without C-RED shutdown."
            )
            inc_CRED = False
        else:
            c_red_connection = open_zmq_connection(C_RED_PORT)

    cc = co.PowerControllino("192.168.100.10", init_motors=False)

    # turn off all sources: SRL, SGL and SBB
    lamps = ["SRL", "SGL", "SBB"]

    for lamp in lamps:
        try:
            _, mds_connection = send_with_mds_recovery(mds_connection, f"off {lamp}")
        except RuntimeError as exc:
            print(f"Failed to send MDS shutdown command for {lamp}: {exc}")
            print("Aborting shutdown.")
            return
        time.sleep(1)  # wait for the command to be processed

    # # flippers up
    # names = [f"SSF{i}" for i in range(1, 5)]
    # for i, flipper in enumerate(names):
    #     message = f"moveabs {flipper} 1.0"
    #     send_and_get_response(mds_connection, message)
    #     time.sleep(2)  # wait for the command to be processed

    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()
    pre_shutdown_current = float(pdu.read_power_value("olt", LOWER_BOX_OUTLET, "curr"))
    print(f"Pre-shutdown current: {pre_shutdown_current} A")

    devices = [
        "USB upper coms power",
        "X-MCC (BMX,BMY)",
        "X-MCC (BFO,SDL,BDS,SSS)",
        "LS16P (HFO)",
        "DM1",
        "DM2",
        "DM3",
        "DM4",
        "USB hubs",
    ]
    for device in devices:
        cc.turn_off(device)
    print("Waiting for all devices to turn off...")

    time.sleep(5)  # wait for devices to turn off

    post_shutdown_current = float(pdu.read_power_value("olt", LOWER_BOX_OUTLET, "curr"))
    print(f"Post-shutdown current: {post_shutdown_current} A")

    input(
        "Type 'exit' in text client for DM server. Kill the MDS and engineering GUI, then press Enter to continue..."
    )

    if not inc_CRED:
        input(
            "In C red server text client, type 'stop' and then type 'exit'. Then press Enter here to continue..."
        )

    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()

    if inc_CRED:
        print("Closing C RED...")
        send_and_get_response(c_red_connection, "stop")
        send_and_get_response(c_red_connection, 'cli "set cooling off"')
        send_and_get_response(c_red_connection, 'cli "shutdown"')
        print("C RED shutdown command sent.")

        pdu.switch_outlet_status(C_RED_OUTLET, "off")
        time.sleep(7)
        res = pdu.read_outlet_status(C_RED_OUTLET)
        if res == "off":
            print("C RED outlet is off.")

        pdu = AtenEcoPDU("192.168.100.11")
        pdu.connect()

    print("Turning off PDU outlet...")
    pdu.switch_outlet_status(LOWER_BOX_OUTLET, "off")
    time.sleep(7)  # wait for PDU to turn off

    res = pdu.read_outlet_status(LOWER_BOX_OUTLET)
    if res == "off":
        print("PDU outlet is off.")
    else:
        print(f"PDU outlet status: {res}. Please check the PDU manually.")

    # Verify no response from ping 192.168.100.111, 192.168.100.10
    ips_to_check = ["192.168.100.111", "192.168.100.10"]

    for ip in ips_to_check:
        if ping_device(ip):
            print(f"Device {ip} is still reachable. Please check manually.")
        else:
            print(f"Device {ip} is not reachable, as expected.")

    print("Shutdown procedure completed successfully.")


def main():
    """Run the instrument shutdown sequence, optionally including C-RED."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "include_cred",
        nargs="?",
        choices=["inc_CRED"],
        help="Pass inc_CRED to include the C-RED camera in the shutdown",
    )
    args = parser.parse_args()
    shutdown(args.include_cred == "inc_CRED")
