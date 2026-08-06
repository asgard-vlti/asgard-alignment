import os
import sys
import subprocess
import pathlib

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

    process = subprocess.Popen([
        "streamlit",
        "run",
        f"{str(this_dir / 'engineering_GUI.py')}",
        "--server.port", str(8501),
        "--server.headless", "true"
    ], 
    # stdout=redirect_file, stderr=redirect_file
    )

    # Write PID out to lock file
    with open(LOCK_FILE, "w") as fileobj:
        fileobj.write(str(process.pid))
    
    return

if __name__ == "__main__":
    main(redirect="/home/asg/logs/eng_gui/log.txt")
