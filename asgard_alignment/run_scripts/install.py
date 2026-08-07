from pathlib import Path
import subprocess
import sys

def main():
    """Install run_* commands for asgard-alignment."""

    pwd = Path(__file__).parent
    run_scripts = [str(_) for _ in pwd.glob("run_*")]

    try:
        subprocess.run(
            ["sudo", "cp", ]  + 
            run_scripts + 
            ["/usr/local/bin/.", ]
            )
    except: 
        return 1
    
    return 0

if __name__ == "__main__":
    r = main()
    sys.exit(r)
