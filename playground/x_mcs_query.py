
# %%
import argparse
import zmq
import json
import datetime

# Parse command line arguments
# parser = argparse.ArgumentParser(description="ZeroMQ Client")
# parser.add_argument("--host", type=str, default="localhost", help="Server host")
# parser.add_argument("--port", type=int, default=5555, help="Server port")
# parser.add_argument("--timeout", type=int, default=5000, help="Response timeout in milliseconds")
# args = parser.parse_args()

args = {
    "host" : "wag",
    "port" : 7050,
    "timeout" : 5000
}

# Create a ZeroMQ context
context = zmq.Context()

# Create a socket to communicate with the server
socket = context.socket(zmq.REQ)

# Set the receive timeout
socket.setsockopt(zmq.RCVTIMEO, args['timeout'])

# Connect to the server
server_address = f"tcp://{args['host']}:{args['port']}"
socket.connect(server_address)


timeNow = datetime.datetime.now(datetime.timezone.utc)
# timeNow = timeNow+datetime.timedelta(hours=3)
timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")

msg = {
    "command":{
        "name" : "read",
        "time" : timeStamp,
        "parameter": [{
            # "name" : "OCS.TEL.NAME"
            "name" : "ao_status" ,
            # "name" : "OCS.MCS.NBPARAMS"
            # "name" : "MCS.MCU.mimir.aoloopStatus"
        },
        {
            "name" : "seeing"
            }]
    }
}

print(json.dumps(msg))
socket.send_string(json.dumps(msg)+'\n')
s = socket.recv_string()
print(s)

