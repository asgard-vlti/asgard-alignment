# Change the Baldr mode from STANDARD to FAINT or back.

import zmq
import sys, os, shutil
import time

class BSaveMode:
    def __init__(self, mode):

        self.mds = self._open_mds_connection()
        self.mode = mode


    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def _send_and_get_response(self, message):
        # print("sending", message)
        self.mds.send_string(message)
        response = self.mds.recv_string()
        # print("response", response)
        return response.strip()

    def _archive_current_state(self):
        """
        Insert a time-stamp before ".json" in the file path, 
        and copy to the subdirectory "OLD" of the same directory.
        """
        timestamp = time.strftime("%y%m%d_%H%M%S")
        load_dir = "/home/asg/.config/asgard-alignment/stable_states"
        save_dir = f"{load_dir}/OLD"
        filename = f"baldr_ONLY_{self.mode.lower()}_{timestamp}.json"
        load_path = f"{load_dir}/baldr_ONLY_{self.mode.lower()}.json"
        save_path = f"{save_dir}/{filename}"
        os.makedirs(save_dir, exist_ok=True)
        try:
            shutil.copy(load_path, save_path)
            print(f"Archived current state to {save_path}")
        except Exception as e:
            print(f"Error archiving current state: {e}")

    def _save_all_BLF_beams(self):
        #Save path is relative
        save_path = f"../stable_states/baldr_ONLY_{target_pos.lower()}.json"
        message = f"save {target} baldr"
        try:
            res = self._send_and_get_response(message)
        except Exception as e:
            res = f"ERROR: {e}"
        return res

    def run(self):
        self._archive_current_state
        self._save_all_BLF_beams()
        
def main():
    # Use argv to determine the mode (FAINT or STANDARD) and other parameters
    if len(sys.argv) != 2:
        print("Usage: python b_savemode.py [FAINT|STANDARD]")
        sys.exit(1)
    mode = sys.argv[1].upper()
    if mode not in ["FAINT", "STANDARD"]:
        print("Invalid mode. Please specify either 'FAINT' or 'STANDARD'.")
        sys.exit(1)
    saveit = BMode(mode)
    saveit.run()

if __name__ == "__main__":    
    main()