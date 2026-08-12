#!/usr/bin/env python3

import argparse
import datetime
import os
import time

import numpy as np
import zmq
from astropy.io import fits
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass


parser = argparse.ArgumentParser(
    description="Acquire BALDR I0, zonal DM poke-amplitude scan, N0, and dark references."
)
parser.add_argument(
    "--beam_id",
    type=int,
    default=1,
    choices=[1, 2, 3, 4],
    help="BALDR beam ID. Default: 1",
)
args = parser.parse_args()


# -----------------------------------------------------------------------------
# Hard-coded acquisition parameters
# -----------------------------------------------------------------------------
N_ACTUATORS = 140
POKE_MIN = -0.1
POKE_MAX = +0.1
N_POKE_AMPLITUDES = 10
N_FRAMES_PER_POKE = 20

N_DARK_FRAMES = 1000
N_REFERENCE_FRAMES = 20

# The phase mask is assumed to be correctly aligned BEFORE this script starts.
# The script does not command a named mask position. I0 and the poke scan are
# acquired at the starting mask position. Only at the end of the poke scan are
# BMX/BMY moved relatively to obtain N0, after which the inverse relative move
# restores the starting position.
FPM_CLEAR_OFFSET_UM = -200.0

# dmclass uses zero-based indexing for the DM SHM subchannels.
# main_chn=2 writes only to the dedicated poke channel and does not touch the
# flat conventionally held on channel 0.
DM_CHANNEL = 2

DM_SETTLE_S = 0.02
SOURCE_OFF_SETTLE_S = 10.0
SOURCE_ON_SETTLE_S = 3.0
FPM_SETTLE_S = 1.0
FRAME_TIMEOUT_S = 5.0

CAMERA_SHM = f"/dev/shm/baldr{args.beam_id}.im.shm"
OUTPUT_ROOT = "/home/asg/Progs/repos/asgard-alignment/calibration/reports/pokeramp"

# Multi-device server used only for SBB on/off and relative FPM motion.
# No commands are sent to the camera.
MDS_HOST = "192.168.100.2"
MDS_PORT = 5555

probe_amps = np.linspace(POKE_MIN, POKE_MAX, N_POKE_AMPLITUDES)


# -----------------------------------------------------------------------------
# Small hardware helpers
# -----------------------------------------------------------------------------
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
socket.setsockopt(zmq.SNDTIMEO, 5000)
socket.connect(f"tcp://{MDS_HOST}:{MDS_PORT}")


def send_and_get_response(message):
    print(f"> {message}")
    socket.send_string(message)
    response = socket.recv_string().strip()
    print(f"< {response}")
    if "NACK" in response or "not connected" in response.lower():
        raise RuntimeError(f"Hardware command failed: {message!r} -> {response!r}")
    return response


def acquire_new_frames(n_frames):
    """Acquire n_frames distinct raw frames from the BALDR camera SHM."""
    data = np.empty((n_frames, ny, nx), dtype=camera_dtype)
    last_counter = cam_shm.get_counter()

    for i in range(n_frames):
        frame = np.array(
            cam_shm.get_data(
                check=last_counter,
                sleepT=0.001,
                timeout=FRAME_TIMEOUT_S,
            ),
            copy=True,
        )
        new_counter = cam_shm.get_counter()

        if new_counter <= last_counter:
            raise TimeoutError(
                f"No new BALDR camera frame within {FRAME_TIMEOUT_S:.1f} s."
            )

        data[i] = frame
        last_counter = new_counter

    return data


# -----------------------------------------------------------------------------
# Initialise camera SHM and DM poke channel
# -----------------------------------------------------------------------------
print(f"\nBeam {args.beam_id}")
print(f"Camera SHM: {CAMERA_SHM}")
print(f"DM poke channel index: {DM_CHANNEL}")
print("This script does NOT apply or modify the DM flat.")
print("Ensure the required DM flat and phase-mask alignment are already in place before running.\n")

cam_shm = shm(CAMERA_SHM)
dm_shm = dmclass(beam_id=args.beam_id, main_chn=DM_CHANNEL)

if getattr(cam_shm, "empty", False):
    raise RuntimeError(f"Camera SHM does not exist: {CAMERA_SHM}")
if dm_shm.nch <= DM_CHANNEL:
    raise RuntimeError(
        f"DM beam {args.beam_id} has only {dm_shm.nch} subchannels; "
        f"cannot use channel index {DM_CHANNEL}."
    )

# Read one frame only to establish image shape and native dtype.
test_frame = np.array(cam_shm.get_data(), copy=True)
if test_frame.ndim != 2:
    raise RuntimeError(
        f"Expected a 2-D BALDR camera SHM frame, got shape {test_frame.shape}."
    )
ny, nx = test_frame.shape
camera_dtype = test_frame.dtype

# Match the exact dtype of the DM SHM when writing commands.
dm_dtype = np.dtype(dm_shm.shms[DM_CHANNEL].npdtype)

# Build 140 zonal actuator maps. The four inactive 12x12 corners are explicitly
# zero rather than the dmclass default of NaN.
eye140 = np.eye(N_ACTUATORS)
zonal_basis = np.array(
    [dm_shm.cmd_2_map2D(cmd, fill=0.0) for cmd in eye140],
    dtype=dm_dtype,
)
zero_cmd = np.zeros((12, 12), dtype=dm_dtype)

# Raw acquisition arrays only. No dark subtraction, averaging or normalization.
frames = np.empty(
    (N_ACTUATORS, N_POKE_AMPLITUDES, N_FRAMES_PER_POKE, ny, nx),
    dtype=camera_dtype,
)
dm_commands = np.empty(
    (N_ACTUATORS, N_POKE_AMPLITUDES, 12, 12),
    dtype=dm_dtype,
)

frames_gib = frames.nbytes / 1024**3
print(
    f"Poke scan: {N_ACTUATORS} actuators x {N_POKE_AMPLITUDES} amplitudes "
    f"x {N_FRAMES_PER_POKE} frames."
)
print(f"Raw poke frame cube allocation: {frames_gib:.2f} GiB\n")


# -----------------------------------------------------------------------------
# Acquisition order
#   1. I0 at the pre-aligned starting mask position
#   2. Full actuator/amplitude poke scan at that same mask position
#   3. N0 after a relative BMX/BMY offset
#   4. Restore the starting mask position
#   5. Darks with SBB off
# -----------------------------------------------------------------------------
sbb_is_off = False
fpm_x_displaced = False
fpm_y_displaced = False

try:
    # All references use zero on this script's DM poke channel as their
    # baseline. Other DM channels, including the user's pre-applied flat,
    # remain untouched.
    dm_shm.set_data(zero_cmd)
    time.sleep(DM_SETTLE_S)

    # -------------------------------------------------------------------------
    # I0: starting phase-mask position, SBB on, zero poke command
    # -------------------------------------------------------------------------
    # IMPORTANT: no phase-mask command is sent here. The script assumes the
    # phase mask has already been aligned before execution.
    print(f"Acquiring {N_REFERENCE_FRAMES} raw I0 frames at the starting mask position...")
    i0_frames = acquire_new_frames(N_REFERENCE_FRAMES)

    # -------------------------------------------------------------------------
    # POKE RAMP: leave the phase mask completely untouched
    # -------------------------------------------------------------------------
    print("\nStarting actuator poke ramp...")

    for actuator_idx in range(N_ACTUATORS):
        print(f"Actuator {actuator_idx + 1:3d}/{N_ACTUATORS}")

        for amp_idx, amp in enumerate(probe_amps):
            cmd = (amp * zonal_basis[actuator_idx]).astype(dm_dtype, copy=False)

            # Absolute command on the dedicated poke channel. Previous pokes
            # cannot accumulate because set_data overwrites this channel.
            dm_shm.set_data(cmd)
            dm_commands[actuator_idx, amp_idx] = cmd
            time.sleep(DM_SETTLE_S)

            frames[actuator_idx, amp_idx] = acquire_new_frames(
                N_FRAMES_PER_POKE
            )

        # Explicitly remove this actuator's poke before moving to the next one.
        dm_shm.set_data(zero_cmd)
        time.sleep(DM_SETTLE_S)

    # Ensure the poke channel is zero before taking N0.
    dm_shm.set_data(zero_cmd)
    time.sleep(DM_SETTLE_S)

    # -------------------------------------------------------------------------
    # N0: only now move the phase mask clear using relative BMX/BMY offsets
    # -------------------------------------------------------------------------
    print("\nPoke scan complete. Moving phase mask out of the beam for N0...")

    send_and_get_response(
        f"moverel BMX{args.beam_id} {FPM_CLEAR_OFFSET_UM}"
    )
    fpm_x_displaced = True

    send_and_get_response(
        f"moverel BMY{args.beam_id} {FPM_CLEAR_OFFSET_UM}"
    )
    fpm_y_displaced = True

    time.sleep(FPM_SETTLE_S)

    print(f"Acquiring {N_REFERENCE_FRAMES} raw N0 frames...")
    n0_frames = acquire_new_frames(N_REFERENCE_FRAMES)

    # Restore exactly the position at which the script started. We deliberately
    # use inverse relative moves rather than fpm_movetomask: the initial mask
    # alignment is assumed to have been done by the user beforehand.
    print("Restoring the starting phase-mask position...")

    send_and_get_response(
        f"moverel BMY{args.beam_id} {-FPM_CLEAR_OFFSET_UM}"
    )
    fpm_y_displaced = False

    send_and_get_response(
        f"moverel BMX{args.beam_id} {-FPM_CLEAR_OFFSET_UM}"
    )
    fpm_x_displaced = False

    time.sleep(FPM_SETTLE_S)

    # -------------------------------------------------------------------------
    # DARK: take these last so turning the SBB off/on cannot perturb I0 or pokes
    # -------------------------------------------------------------------------
    print(f"Acquiring {N_DARK_FRAMES} dark frames with SBB off...")
    send_and_get_response("off SBB")
    sbb_is_off = True
    time.sleep(SOURCE_OFF_SETTLE_S)
    dark_frames = acquire_new_frames(N_DARK_FRAMES)

    print("Turning SBB back on...")
    send_and_get_response("on SBB")
    sbb_is_off = False
    time.sleep(SOURCE_ON_SETTLE_S)

finally:
    # Bench-safe cleanup: only zero this script's dedicated poke channel.
    # Never call zero_all(), because that would also erase the pre-applied flat.
    try:
        dm_shm.set_data(zero_cmd)
        print("\nDM poke channel returned to zero.")
    except Exception as exc:
        print(f"\nWARNING: could not zero DM poke channel: {exc}")

    # If acquisition is interrupted after one or both relative N0 moves, undo
    # only the axes that actually moved. No named-mask command is ever used.
    if fpm_y_displaced:
        try:
            print("Restoring BMY after interruption...")
            send_and_get_response(
                f"moverel BMY{args.beam_id} {-FPM_CLEAR_OFFSET_UM}"
            )
            fpm_y_displaced = False
        except Exception as exc:
            print(f"WARNING: could not restore BMY: {exc}")

    if fpm_x_displaced:
        try:
            print("Restoring BMX after interruption...")
            send_and_get_response(
                f"moverel BMX{args.beam_id} {-FPM_CLEAR_OFFSET_UM}"
            )
            fpm_x_displaced = False
        except Exception as exc:
            print(f"WARNING: could not restore BMX: {exc}")

    # Do not leave the internal source off if acquisition is interrupted during
    # the dark sequence.
    if sbb_is_off:
        try:
            print("Restoring SBB source after interruption...")
            send_and_get_response("on SBB")
        except Exception as exc:
            print(f"WARNING: could not turn SBB source back on: {exc}")


# -----------------------------------------------------------------------------
# Save raw references, raw poke frames, and exact applied DM commands to FITS
# -----------------------------------------------------------------------------
now_utc = datetime.datetime.now(datetime.timezone.utc)
tstamp = now_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
date_dir = now_utc.strftime("%Y-%m-%d")
out_dir = os.path.join(OUTPUT_ROOT, date_dir)
os.makedirs(out_dir, exist_ok=True)

fits_path = os.path.join(
    out_dir,
    f"pokeramp_raw_beam{args.beam_id}_{tstamp}.fits",
)

hdr = fits.Header()
hdr["DATE"] = (now_utc.isoformat(), "UTC time file was written")
hdr["BEAMID"] = (args.beam_id, "BALDR beam ID")
hdr["CAMSHM"] = (CAMERA_SHM, "Camera shared-memory stream")
hdr["DMCHIDX"] = (DM_CHANNEL, "Zero-based DM poke subchannel index")
hdr["MASKINIT"] = ("PREALIGNED", "I0/pokes use phase-mask position at script start")
hdr["FPMOFF"] = (FPM_CLEAR_OFFSET_UM, "Relative BMX/BMY offset used for N0 [um]")
hdr["NACT"] = (N_ACTUATORS, "Number of active DM actuators scanned")
hdr["AMPMIN"] = (POKE_MIN, "Minimum DM command amplitude")
hdr["AMPMAX"] = (POKE_MAX, "Maximum DM command amplitude")
hdr["NAMP"] = (N_POKE_AMPLITUDES, "Number of amplitudes per actuator")
hdr["NFRAME"] = (N_FRAMES_PER_POKE, "Raw frames per poke state")
hdr["NDARK"] = (N_DARK_FRAMES, "Raw dark frames")
hdr["NREF"] = (N_REFERENCE_FRAMES, "Raw frames in each I0/N0 reference")
hdr["SETTLE"] = (DM_SETTLE_S, "DM settling delay [s]")
hdr["SBBOFFST"] = (SOURCE_OFF_SETTLE_S, "SBB-off settling delay [s]")
hdr["SBBONST"] = (SOURCE_ON_SETTLE_S, "SBB-on settling delay [s]")
hdr["FPMSTL"] = (FPM_SETTLE_S, "FPM settling delay [s]")
hdr["TIMEOUT"] = (FRAME_TIMEOUT_S, "New-frame timeout [s]")
hdr["FLAT"] = ("PREAPPLIED", "Script does not apply or alter DM flat")
hdr["REFCMD"] = (0.0, "DM poke-channel command during DARK/I0/N0")
hdr["DATAPROC"] = ("RAW", "No dark subtraction/averaging/normalisation")
hdr["FRMAXIS"] = ("ACT,AMP,SAMP,Y,X", "Logical FRAMES array axis order")
hdr["DMCAXIS"] = ("ACT,AMP,Y,X", "Logical DM_CMDS array axis order")
hdr["REFAXIS"] = ("SAMP,Y,X", "Logical DARK/I0/N0 array axis order")
hdr["COMMENT"] = "Amplitude values are np.linspace(AMPMIN, AMPMAX, NAMP)."
hdr["COMMENT"] = "FRAMES[i,j,:,:,:] corresponds to DM_CMDS[i,j,:,:]."
hdr["COMMENT"] = "DARK/I0/N0 are saved as raw frame stacks, not means."
hdr["COMMENT"] = "Order: I0 -> poke scan -> relative-offset N0 -> darks."
hdr["COMMENT"] = "No named phase-mask position is commanded by this script."

hdul = fits.HDUList(
    [
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(data=dark_frames, name="DARK_FRAMES"),
        fits.ImageHDU(data=i0_frames, name="I0_FRAMES"),
        fits.ImageHDU(data=n0_frames, name="N0_FRAMES"),
        fits.ImageHDU(data=zero_cmd, name="REF_DM_CMD"),
        fits.ImageHDU(data=frames, name="FRAMES"),
        fits.ImageHDU(data=dm_commands, name="DM_CMDS"),
    ]
)
hdul.writeto(fits_path, overwrite=False)

print(f"\nSaved raw BALDR reference + poke-ramp data to:\n{fits_path}")
