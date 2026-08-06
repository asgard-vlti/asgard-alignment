import os
import sys
import subprocess
import pathlib
import shutil

os.environ["PATH"] += os.pathsep + os.path.abspath("/home/asg/.conda/envs/asgard/bin/")
os.environ["PYTHONPATH"] += os.pathsep + os.path.abspath("/home/asg/Progs/repos/asgard-alignment")
sys.path.append(os.path.abspath("/home/asg/Progs/repos/asgard-alignment"))

LOCK_FILE = "/tmp/asg.eng_gui.lock"
def main(redirect=None):
    if redirect is not None:
        redirect_file = open(redirect, 'a')
    else:
        redirect_file=subprocess.DEVNULL
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

    this_dir = pathlib.Path(__file__).parent
    launch_gui_command = [
        # "conda", "run", "-n", "asgard", "--no-capture-output",
        "streamlit",
        "run",
        f"{str(this_dir / 'engineering_GUI.py')}",
        "--server.port", str(8501),
        "--server.headless", "true"
    ]
    print("Running command:")
    print(" ".join(launch_gui_command))
    print(f"streamlit is found at: {shutil.which('streamlit')}")
    print(f"PATH is {os.getenv('PATH')}")

    process = subprocess.Popen(launch_gui_command)

    # Write PID out to lock file
    with open(LOCK_FILE, "w") as fileobj:
        fileobj.write(str(process.pid))
    
    return

if __name__ == "__main__":

    main(redirect="/home/asg/logs/eng_gui/log.txt")
