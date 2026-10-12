import argparse
import logging
import os
import sys

# Update PATH and PYTHONPATH before continuing
os.environ["PATH"] += os.pathsep + os.path.abspath("/home/asg/.conda/envs/asgard/bin/")
os.environ["PYTHONPATH"] += os.pathsep + os.path.abspath("/home/asg/Progs/repos/asgard-alignment")
sys.path.append(os.path.abspath("/home/asg/Progs/repos/asgard-alignment"))

from asgard_alignment.MultiDeviceServer import MultiDeviceServer

config_file = "/home/asg/.config/asgard-alignment/motor_info_full_system.json"
LOCK_FILE = "/tmp/asg.mds.lock"

def main():
    try:
        with open(LOCK_FILE,"r") as f:
            existing_pid = int(f.read().strip())
    except:
        existing_pid=None
    if existing_pid:
        try:
            os.kill(existing_pid, 0)
            print(f"ERROR: Process already running under pid {existing_pid}")
            return
        except:
            print(f"Stale lock file found. Ignoring (PID {existing_pid})")

    parser = argparse.ArgumentParser(description="Run the MDS server.")
    # parser.add_argument(
    #     "-c", "--config", type=str, required=True, help="Path to the configuration file"
    # )
    parser.add_argument(
        "--host", type=str, default="192.168.100.2", help="Host address"
    )
    parser.add_argument("-p", "--port", type=int, default=5555, help="Port number")

    args = parser.parse_args()

    # Remove all handlers associated with the root logger object (if any)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler with millisecond timestamps
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Set the lock
    with open(LOCK_FILE, "w") as file_obj:
        file_obj.write(str(os.getpid()))

    serv = MultiDeviceServer(args.port, args.host, config_file=config_file)
    # p = multiprocessing.Process(target=serv.run)
    # p.start()
    serv.run()


    return

if __name__ == "__main__":
    main()
