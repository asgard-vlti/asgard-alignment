import zmq
import psutil
import json
import time

# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")

scripts_of_interest = {
    "MDS": "python asgard_alignment/MultiDeviceServer.py",
    "Eng gui":"streamlit run asgard_alignment/cmd_scripts/engineering_GUI.py",
    "CRED1": "/bin/cred1_server",
    "DM": "/bin/cred1_server",
    "BTT1": "/usr/local/bin/baldr_tt /usr/local/etc/def1.toml --socket 'tcp://*:6671'",
    "BTT2": "/usr/local/bin/baldr_tt /usr/local/etc/def2.toml --socket 'tcp://*:6672'",
    "BTT3": "/usr/local/bin/baldr_tt /usr/local/etc/def3.toml --socket 'tcp://*:6673'",
    "BTT4": "/usr/local/bin/baldr_tt /usr/local/etc/def4.toml --socket 'tcp://*:6674'",
    "Heimdallr": "/usr/local/bin/heimdallr",
    "MCS" : "/home/asg/.conda/envs/asgard/bin/mcs-client",
    "Heim Telem":"/home/asg/.conda/envs/asgard/bin/save-ft-performance",
    "DCS (back end)":"/home/asg/.conda/envs/asgard/bin/back-end-server"
}


def get_running_scripts(scripts_of_interest):
    status = {k:"closed" for k in scripts_of_interest.keys()}
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"])
            if "xterm" in cmdline:
                for k,v in scripts_of_interest.items():
                    if v in cmdline:
                        status[k] = "open"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return status


if __name__ == "__main__":
    last = 0
    while True:
        if time.time() - last > 5:  # Check every 5 seconds
            print("Current running scripts:", get_running_scripts(scripts_of_interest))
            last = time.time()


# while True:
# message = socket.recv_string()
# if message == "STATUS":
#     socket.send_json(get_running_scripts())
