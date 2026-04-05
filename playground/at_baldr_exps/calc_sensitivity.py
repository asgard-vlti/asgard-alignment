# %%
import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime

from astropy.io import fits
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from asgard_alignment import FLI_Cameras as FLI

beam = 3


# %%
def mds_connect(host: str, port: int = 5555, timeout_ms: int = 5000):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{port}")
    return ctx, sock


def mds_send(sock, msg: str) -> str:
    sock.send_string(msg)
    return sock.recv_string().strip()

# %%
