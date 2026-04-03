import zmq
import psutil
import json
import time

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")


def get_running_scripts():
    status = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Look for python scripts; filter by specific keywords in your filenames
            if "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"])
                print(cmdline)
                # if "my_experiment_script.py" in cmdline:
                #     status["Experiment_1"] = "Running"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return status


if __name__ == "__main__":
    last = time.time()
    while True:
        if time.time() - last > 5:  # Check every 5 seconds
            print("Current running scripts:", get_running_scripts())
            last = time.time()

# while True:
# message = socket.recv_string()
# if message == "STATUS":
#     socket.send_json(get_running_scripts())
