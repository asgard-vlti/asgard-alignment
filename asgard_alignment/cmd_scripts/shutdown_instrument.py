import argparse
import os
import signal
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
DM_PORT = 6666
C_RED_PORT = 6667
C_RED_WAIT_TIMEOUT_S = 10
ENG_GUI_PORT = 8501
MDS_EXECUTABLE = "/home/asg/.conda/envs/asgard/bin/mds"
PROCESS_STOP_TIMEOUT_S = 5
PROCESS_POLL_INTERVAL_S = 0.1


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


def get_standby_status():
    connection = open_zmq_connection(MDS_PORT)
    try:
        return send_and_get_response(connection, "standby_status").strip()
    finally:
        connection.close(linger=0)


def is_tcp_port_open(host, port, timeout_s=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def restart_mds_from_path():
    # Policy constraint: MDS restart must happen only via run_mds on PATH.
    subprocess.run(["run_mds"], check=True)


def wait_for_mds(
    host, port, total_wait_s=MDS_WAIT_TIMEOUT_S, poll_s=MDS_POLL_INTERVAL_S
):
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
            raise RuntimeError(
                f"run_mds failed with exit code {exc.returncode}."
            ) from exc

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


def send_server_shutdown_commands(server_name, port, commands):
    if not is_tcp_port_open(MDS_HOST, port):
        print(f"{server_name} is already unreachable; skipping its shutdown commands.")
        return True

    try:
        connection = open_zmq_connection(port)
    except zmq.ZMQError as exc:
        print(f"Could not connect to {server_name}: {exc}")
        return False
    try:
        for command in commands:
            print(f"Sending '{command}' to {server_name}...")
            try:
                response = send_and_get_response(connection, command)
            except zmq.ZMQError as exc:
                print(f"{server_name} did not acknowledge '{command}': {exc}")
                return False
            print(f"{server_name} response: {response}")
            if response.lstrip().lower().startswith("error:"):
                print(f"{server_name} rejected '{command}'.")
                return False
    finally:
        connection.close(linger=0)

    return True


def read_process_command(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as process_file:
            return [os.fsdecode(arg) for arg in process_file.read().split(b"\0") if arg]
    except FileNotFoundError:
        return []


def matches_service_command(service_name, command):
    if service_name == "MDS":
        return MDS_EXECUTABLE in command
    return (
        "run" in command
        and any(os.path.basename(arg) == "streamlit" for arg in command)
        and any(
            arg.endswith("/asgard_alignment/cmd_scripts/engineering_GUI.py")
            for arg in command
        )
    )


def stop_locked_service(service_name, lock_path, host, port):
    try:
        with open(lock_path, encoding="utf-8") as lock_file:
            pid = int(lock_file.read().strip())
    except FileNotFoundError:
        pid = None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read a valid {service_name} PID from {lock_path}: {exc}") from exc

    if pid is not None and pid <= 0:
        raise RuntimeError(f"Invalid {service_name} PID {pid} in {lock_path}.")

    try:
        command = read_process_command(pid) if pid is not None else []
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect {service_name} PID {pid}: {exc}") from exc

    if not command:
        if not is_tcp_port_open(host, port):
            print(f"{service_name} is already stopped.")
            return
        raise RuntimeError(f"{service_name} is still reachable, but has no live PID in {lock_path}.")

    if not matches_service_command(service_name, command):
        raise RuntimeError(f"{service_name} lock PID {pid} belongs to another process.")

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        if not is_tcp_port_open(host, port):
            print(f"{service_name} is already stopped.")
            return
        raise RuntimeError(f"{service_name} PID {pid} disappeared while its port is still open.")
    except OSError as exc:
        raise RuntimeError(f"Could not kill {service_name} PID {pid}: {exc}") from exc

    deadline = time.monotonic() + PROCESS_STOP_TIMEOUT_S
    while True:
        try:
            command = read_process_command(pid)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect {service_name} PID {pid}: {exc}") from exc
        if not command and not is_tcp_port_open(host, port, timeout_s=0.2):
            print(f"Stopped {service_name} (PID {pid}).")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{service_name} PID {pid} or port {port} is still active after "
                f"{PROCESS_STOP_TIMEOUT_S}s."
            )
        time.sleep(PROCESS_POLL_INTERVAL_S)


def shutdown(inc_CRED):
    try:
        mds_connection = get_mds_connection_or_recover()
    except RuntimeError as exc:
        print(f"Unable to establish MDS connection: {exc}")
        print("Aborting shutdown.")
        return

    try:
        standby_status = get_standby_status()
    except zmq.ZMQError as exc:
        print(f"Could not determine MDS standby status: {exc}")
        standby_status = None

    if standby_status == "STANDBY":
        print("MDS is in standby; skipping state save.")
    elif standby_status == "ACTIVE":
        date = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            res = send_and_get_response(mds_connection, f"save all before_shutdown_{date}")
            print("saved", res)
        except zmq.error.Again:
            inp = input(
                "MDS did not respond (and hence state is not saved). Do you want to continue with shutdown? (y/n): "
            )
            if inp.lower() != "y":
                print("Aborting shutdown.")
                return
            print("Proceeding with shutdown...")
    else:
        print(f"Unknown MDS standby status: {standby_status!r}")
        inp = input(
            "State will not be saved. Continue with shutdown? (y/n): "
        )
        if inp.lower() != "y":
            print("Aborting shutdown.")
            return
        print("Proceeding with shutdown...")

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

    if not send_server_shutdown_commands("DM server", DM_PORT, ("exit",)):
        print("Aborting shutdown before switching off the lower-box outlet.")
        return

    mds_connection.close(linger=0)
    try:
        stop_locked_service(
            "engineering GUI", "/tmp/asg.eng_gui.lock", "127.0.0.1", ENG_GUI_PORT
        )
        stop_locked_service("MDS", "/tmp/asg.mds.lock", MDS_HOST, MDS_PORT)
    except RuntimeError as exc:
        print(f"Could not stop instrument services: {exc}")
        print("Aborting shutdown before switching off the lower-box outlet.")
        return

    if not inc_CRED:
        if not send_server_shutdown_commands(
            "C-RED server", C_RED_PORT, ("stop", "exit")
        ):
            print("Aborting shutdown before switching off the lower-box outlet.")
            return

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
