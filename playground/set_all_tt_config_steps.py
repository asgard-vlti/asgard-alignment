import zmq
host = "mimir"
port = 5555
timeout = 2000


new_value = 15

# Create a ZeroMQ context
context = zmq.Context()

# Create a socket to communicate with the server
socket = context.socket(zmq.REQ)

# Set the receive timeout
socket.setsockopt(zmq.RCVTIMEO, timeout)

# Connect to the server
server_address = f"tcp://{host}:{port}"
socket.connect(server_address)


motors = ["HTTP", "HTTI"]
beams = list(range(1,5))

for m in motors:
    for b in beams:
        for s in [-1,1]:
            message = f"tt_config_step {m}{b} {s*new_value}"
            socket.send_string(message)

            try:
                # Wait for a response from the server
                response = socket.recv_string()
                print(f"Received response from server: {response}")
            except zmq.Again as e:
                print(f"Timeout waiting for response from server: {e}")