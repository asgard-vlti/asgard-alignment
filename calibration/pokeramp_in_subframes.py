#!/usr/bin/env python
import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse


from astropy.io import fits
#from scipy.signal import TransferFunction, bode
from scipy.ndimage import binary_erosion 
#from types import SimpleNamespace
import asgard_alignment.controllino as co # for turning on / off source \
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import common.phasemask_centering_tool as pct
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI




"""
for each mode we ramp the mode amplitude (+/-amp_max) on DM and record response curve in camera in the given space (pix or dm interpolated)
argunments to add 
    --amp_max
    --no_amp_samples
e.g. amp = np.linspace(-amp_max, amp_max, no_amp_samples)
store response images in datacube with indexing corresponding to [mode, amp] = <image>
where <image> is N sampled mean image for given mode and amp 

also record errors from e_LO = I2M_LO @ signal , e_HO = I2M_HO @ signal  
write to fits file with added telemetry and meta data 
make some summary plots (for a few i)
    - e_LO[i] vs mode i amplitude 
    - e_HO[i] vs mode i amplitude 

"""


MDS_port = 5555
MDS_host = "192.168.100.2" # simmode : "127.0.0.1" #'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}




def send_and_get_response(message):
    # st.write(f"Sending message to server: {message}")
    state_dict["message_history"].append(
        f":blue[Sending message to server: ] {message}\n"
    )
    state_dict["socket"].send_string(message)
    response = state_dict["socket"].recv_string()
    if "NACK" in response or "not connected" in response:
        colour = "red"
    else:
        colour = "green"
    # st.markdown(f":{colour}[Received response from server: ] {response}")
    state_dict["message_history"].append(
        f":{colour}[Received response from server: ] {response}\n"
    )

    return response.strip()



def plot2d( thing ):
    plt.figure()
    plt.imshow(thing)
    plt.colorbar()
    plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
    plt.close()

# split_mode 1 
#aa = shm("/dev/shm/baldr1.im.shm")
#util.nice_heatmap_subplots( [ aa.get_data() ],savefig='delme.png')

parser = argparse.ArgumentParser(description="Interaction and control matricies.")

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 

#Camera shared memory path (used to get camera settings )
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# TOML file path; default is relative to the current file's directory.
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=int, #lambda s: [int(item) for item in s.split(",")],
    default=3, #[3], # 1, 2, 3, 4],
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H4",
    help="Comma-separated beam IDs to apply. Default: 1,2,3,4"
)

parser.add_argument(
    "--LO",
    type=int,
    default=2,
    help="Up to what zernike order do we consider Low Order (LO). 2 is for tip/tilt, 3 would be tip,tilt,focus etc). Default: %(default)s"
)


parser.add_argument(
    "--basis_name",
    type=str,
    default="zonal",
    help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
)

parser.add_argument(
    "--Nmodes",
    type=int,
    default=10,
    help="number of modes to probe"
)

parser.add_argument(
    "--amp_max",
    type=float,
    default=0.2,
    help="max amplitude to poke ramp DM modes "
)

parser.add_argument(
    "--no_amp_samples",
    type=int,
    default=20,
    help="max amplitude to poke ramp DM modes "
)

parser.add_argument(
    "--no_samples_per_cmd",
    type=float,
    default=20,
    help="max amplitude to poke ramp DM modes "
)

parser.add_argument(
    "--signal_space",
    type=str,
    default='pix',
    help="what space do we consider the signal on. either dm (uses I2A) or pixel"
)

parser.add_argument(
    "--DM_flat",
    type=str,
    default="baldr",
    help="What flat do we use on the DM during the calibration. either 'baldr' or 'factory'. Default: %(default)s"
)


# parser.add_argument(
#     '--cam_fps',
#     type=int,
#     default=1000,
#     help="frames per second on camera. Default: %(default)s"
# )


# parser.add_argument(
#     '--cam_gain',
#     type=int,
#     default=10,
#     help="camera gain. Default: %(default)s"
# )

parser.add_argument("--fig_path", 
                    type=str, 
                    default='/home/asg/Progs/repos/asgard-alignment/calibration/reports/test/', 
                    help="path/to/output/image/ for the saved figures")



args=parser.parse_args()



#for beam_id in args.beam_id:

with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    #  read in the current calibrated matricies 
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)

    M2C_LO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_LO", None) ).astype(float)
    M2C_HO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C_HO", None) ).astype(float)
    I2M_LO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
    I2M_HO = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)

    
    I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) )
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)
    

    # # define our Tip/Tilt or lower order mode index on zernike DM basis 
    LO = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("LO", None)

    # tight (non-edge) pupil filter
    inner_pupil_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)

    camera_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None)

    N0_runtime = np.mean( N0[inner_pupil_filt] )
    i_setpoint_runtime = I0 / N0_runtime 


# for reading camera config ! 
camclient = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])
camera_config_current = {k:str(v) for k,v in camclient.config.items()} # current configuration of test 

# Open SHM once for each subframe 
cam_shm = shm(f"/dev/shm/baldr{args.beam_id}.im.shm") #{b: shm(f"/dev/shm/baldr{b}.im.shm") for b in args.beam_id}


# set up DM SHMs 
print( 'setting up DMs')
#dm_shm = {}
#for beam_id in args.beam_id:
dm_shm = dmclass( beam_id=args.beam_id, main_chn=2 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
    
    ###     UP TO USER TO PUT THE FLAT ON!!!
    # zero all channels
    # dm_shm[beam_id].zero_all()
    
    # if args.DM_flat.lower() == 'factory':
    #     # activate flat (does this on channel 1)
    #     dm_shm[beam_id].activate_flat()
    # elif args.DM_flat.lower() == 'baldr':
    #     # apply dm flat + calibrated offset (does this on channel 1)
    #     dm_shm[beam_id].activate_calibrated_flat()
        
    # else:
    #     print( "Unknow flat option. Valid options are 'factory' or 'baldr'. Using baldr flat as default")
    #     args.DM_flat == 'baldr'
    #     dm_shm[beam_id].activate_calibrated_flat()




# Get bias in each subframe to account for aduoffset pixelwize
    # from asgard_alignment import controllino as co
    # myco = co.Controllino('172.16.8.200')

    # myco.turn_off("SBB")
    # time.sleep(10)
    
    # dark_raw = c.get_data()

    # myco.turn_on("SBB")
    # time.sleep(10)

print("turning off internal SBB source for bias")
send_and_get_response("off SBB")

time.sleep(10)

dark_dict = {}
#for beam_id in args.beam_id:
dark_tmp = []
for _ in range( 1000 ):
    dark_tmp.append( cam_shm.get_data() )
    time.sleep(0.01) #
dark_current = np.mean( dark_tmp , axis=0)

send_and_get_response("on SBB")
time.sleep(3)
print("turning back on internal SBB source, check plot that darks are ok")

# check 
util.nice_heatmap_subplots( im_list=[dark_current], title_list=[f"beam{args.beam_id}"] )
plt.show() 


print(f"moving to phasemask {args.phasemask} reference position")
# Move to phase mask
#for beam_id in args.beam_id:
message = f"fpm_movetomask phasemask{args.beam_id} {args.phasemask}"
res = send_and_get_response(message)
print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)


# WE USE OUR READ IN I0, N0 HERE 

# Get reference pupils (later this can just be a SHM address)
zwfs_pupils = {}
clear_pupils = {}
normalized_pupils = {}
# for new phasemask H band is close to edge so offset MUST be negative direction
rel_offset = -200.0 #um phasemask offset for clear pupil
print( 'Moving FPM out to get clear pupils')
#for beam_id in args.beam_id:
message = f"moverel BMX{args.beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 
time.sleep( 1 )
message = f"moverel BMY{args.beam_id} {rel_offset}"
res = send_and_get_response(message)
print(res) 


time.sleep(0.5)


############
#Clear Pupil (new for comparison to what was used in build IM)
print( 'gettin clear pupils')


#for beam_id in args.beam_id:


N0s = []
for _ in range( args.no_samples_per_cmd ):
    N0s.append( cam_shm.get_data()  - dark_current )
    time.sleep(0.01) #

N0_current = np.mean( N0s ,axis=0) 


# move phase mask back in
print( 'Moving FPM back in beam.')
message = f"moverel BMX{args.beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 
time.sleep(1)
message = f"moverel BMY{args.beam_id} {-rel_offset}"
res = send_and_get_response(message)
print(res) 
time.sleep(3)

#############
# ZWFS Pupil( new for comparison to what was used in build IM)
input("phasemasks aligned? ensure alignment then press enter")


print( 'Getting ZWFS pupils')

I0s = []
for _ in range( args.no_samples_per_cmd ):
    I0s.append( cam_shm.get_data() - dark_current )
    time.sleep(0.01) #

I0_current = np.mean( I0s, axis=0) 


N0_runtime_currnt = np.mean( N0_current[inner_pupil_filt] )
i_setpoint_runtime_current = I0_current / N0_runtime_currnt



# Defining our modal basis for probing 

LO_basis = dmbases.zer_bank(2, args.LO+1 )
zonal_basis = np.array([dm_shm.cmd_2_map2D(ii) for ii in np.eye(140)]) 
#zonal_basis = dmbases.zer_bank(4, 143 )
modal_basis = np.array( LO_basis.tolist() +  zonal_basis.tolist() ) 
# should be 144 x 140 (we deal with errors in 140 actuator space (columns), but SHM takes 144 vector as input (rows)) 
# this is why we do transpose 
#M2C = modal_basis.copy().reshape(modal_basis.shape[0],-1).T # mode 2 command matrix 


if args.signal_space.lower() not in ["dm", "pix"] :
    raise UserWarning("signal space must either be 'dm' or 'pixel'")




############# ACQUIRE POKE RAMP DATA 


probe_amps = np.linspace(-args.amp_max, args.amp_max, int(args.no_amp_samples))

# infer dimensions from one frame and matrices
test_frame = cam_shm.get_data()
ny, nx = test_frame.shape

n_mode = int(len(modal_basis))
n_amp  = int(len(probe_amps))

# reconstructor output sizes
n_lo = int(I2M_LO.shape[0])
n_ho = int(I2M_HO.shape[0])

# allocate cubes
imgs_cube   = np.zeros((n_mode, n_amp, ny, nx), dtype=np.float32)
signal_cube = np.zeros((n_mode, n_amp, ny, nx), dtype=np.float32)

eLO_cube    = np.zeros((n_mode, n_amp, n_lo), dtype=np.float32)
eHO_cube    = np.zeros((n_mode, n_amp, n_ho), dtype=np.float32)

# (optional) keep a dict mirror of your old structure if you still want it
# pokeramp_list = {}

for idx, mode in enumerate(modal_basis):

    # pokeramp_list[idx] = {'imgs': [], 'signal': [], 'e_LO': [], 'e_HO': []}

    for ai, amp in enumerate(probe_amps):

        # ---- apply dm poke ----
        time.sleep(0.01)
        dm_shm.set_data(amp * mode)  # poke channel handled by dm_shm object config

        # ---- get frames ----
        subframes = []
        for _ in range(int(args.no_samples_per_cmd)):
            subframes.append(cam_shm.get_data() - dark_current)
            time.sleep(0.01)

        # ---- average image ----
        subframe_avg = np.mean(subframes, axis=0).astype(np.float32)

        # ---- signal  ----
        # for the signal we use the updated measured N0_runtime_currnt (since we typically update onsky)
        # but we keep the internal i_setpoint_runtime
        signal = (subframe_avg / N0_runtime_currnt - i_setpoint_runtime).astype(np.float32)

        # ---- reconstructor telemetry (same logic as your original) ----
        if args.signal_space == "dm":
            e_LO = (I2M_LO @ (I2A @ signal)).astype(np.float32)
            e_HO = (I2M_HO @ (I2A @ signal)).astype(np.float32)
        elif args.signal_space == "pix":
            e_LO = (I2M_LO @ signal).astype(np.float32)
            e_HO = (I2M_HO @ signal).astype(np.float32)
        else:
            raise ValueError("args.signal_space must be 'dm' or 'pix'")

        # ---- store in cubes ----
        imgs_cube[idx, ai, :, :]   = subframe_avg
        signal_cube[idx, ai, :, :] = signal

        eLO_cube[idx, ai, :] = e_LO
        eHO_cube[idx, ai, :] = e_HO

        # ---- optional: also append to legacy dict structure ----
        # pokeramp_list[idx]['imgs'].append(subframe_avg)
        # pokeramp_list[idx]['signal'].append(signal)
        # pokeramp_list[idx]['e_LO'].append(e_LO)
        # pokeramp_list[idx]['e_HO'].append(e_HO)



# -----------------------------------------
# WRITE TO FITS WITH METADATA + NEW REFERENCES

tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
tstamp_rough = datetime.datetime.now().strftime("%Y-%m-%d")

out_dir = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/pokeramp/{tstamp_rough}"
os.makedirs(out_dir, exist_ok=True)

fits_path = (
    f"{out_dir}/pokeramp_beam{args.beam_id}_{args.phasemask}_{args.basis_name}_{tstamp}.fits"
)

# ---- primary header: minimal but useful scalar metadata ----
hdr = fits.Header()

# identifiers / provenance
hdr["DATE"]   = tstamp
hdr["BEAMID"] = int(args.beam_id)
hdr["PHMASK"] = str(args.phasemask)
hdr["BASIS"]  = str(args.basis_name)
hdr["SIGSPC"] = str(args.signal_space)

# probing setup
hdr["NMODE"]  = int(imgs_cube.shape[0])
hdr["NAMP"]   = int(imgs_cube.shape[1])
hdr["AMPMAX"] = float(args.amp_max)
hdr["NSPCMD"] = int(args.no_samples_per_cmd)

# calibration runtime scalars (these are important!)
hdr["N0RUN0"] = float(N0_runtime)          # from TOML references (if you want it)
hdr["N0RUNC"] = float(N0_runtime_currnt)   # current measured runtime N0

# store TOML path (truncate to FITS card limit)
toml_path_used = args.toml_file.replace("#", f"{args.beam_id}")
hdr["TOML"] = toml_path_used[:68]

# ---- build HDUs ----
hdus = []
hdus.append(fits.PrimaryHDU(header=hdr))

# ---- measured during this run (new references) ----
hdus.append(fits.ImageHDU(data=dark_current.astype(np.float32), name="DARK_CURR"))
hdus.append(fits.ImageHDU(data=N0_current.astype(np.float32),   name="N0_CURR"))
hdus.append(fits.ImageHDU(data=I0_current.astype(np.float32),   name="I0_CURR"))

# also store the references that were loaded from TOML (for direct comparison)
# (these can be big; but you asked to keep them)
hdus.append(fits.ImageHDU(data=np.array(N0).astype(np.float32), name="N0_TOML"))
hdus.append(fits.ImageHDU(data=np.array(I0).astype(np.float32), name="I0_TOML"))

# ---- probing definitions ----
hdus.append(fits.ImageHDU(data=np.array(probe_amps, dtype=np.float32), name="PROBE_AMPS"))
hdus.append(fits.ImageHDU(data=np.array(modal_basis, dtype=np.float32), name="MODAL_BASIS"))

# ---- main data products ----
# dims:
#   IMGS:   (mode, amp, y, x)
#   SIGNAL: (mode, amp, y, x)
#   E_LO:   (mode, amp, n_lo)
#   E_HO:   (mode, amp, n_ho)
hdus.append(fits.ImageHDU(data=imgs_cube,   name="IMGS"))
hdus.append(fits.ImageHDU(data=signal_cube, name="SIGNAL"))
hdus.append(fits.ImageHDU(data=eLO_cube,    name="E_LO"))
hdus.append(fits.ImageHDU(data=eHO_cube,    name="E_HO"))

# ---- masks / filters used (critical for reproducibility) ----
# store as uint8 to keep it FITS-friendly
if pupil_mask is not None:
    hdus.append(fits.ImageHDU(data=np.array(pupil_mask, dtype=np.uint8), name="PUPIL_MASK"))
if inner_pupil_filt is not None:
    hdus.append(fits.ImageHDU(data=np.array(inner_pupil_filt, dtype=np.uint8), name="INNER_PUPF"))

# ---- store calibration matrices used (optional but recommended) ----
# these can be large; if you want smaller files, comment out what you don't need
if I2A is not None:
    hdus.append(fits.ImageHDU(data=np.array(I2A, dtype=np.float32), name="I2A"))
if I2M_LO is not None:
    hdus.append(fits.ImageHDU(data=np.array(I2M_LO, dtype=np.float32), name="I2M_LO"))
if I2M_HO is not None:
    hdus.append(fits.ImageHDU(data=np.array(I2M_HO, dtype=np.float32), name="I2M_HO"))
if M2C_LO is not None:
    hdus.append(fits.ImageHDU(data=np.array(M2C_LO, dtype=np.float32), name="M2C_LO"))
if M2C_HO is not None:
    hdus.append(fits.ImageHDU(data=np.array(M2C_HO, dtype=np.float32), name="M2C_HO"))
if IM is not None:
    hdus.append(fits.ImageHDU(data=np.array(IM, dtype=np.float32), name="IM"))

# ---- store camera_config_current (current camera settings) ----
# keep as a simple 2-column binary table: KEY, VALUE
cam_keys = np.array(list(camera_config_current.keys()), dtype="S64")
cam_vals = np.array([str(v) for v in camera_config_current.values()], dtype="S128")
cam_cols = [
    fits.Column(name="KEY",   format="64A",  array=cam_keys),
    fits.Column(name="VALUE", format="128A", array=cam_vals),
]
hdus.append(fits.BinTableHDU.from_columns(cam_cols, name="CAMCFG_CURR"))

# ---- store the TOML config dict (full) as text for exact provenance ----
# this avoids trying to flatten nested dicts into FITS keywords
toml_dump = toml.dumps(config_dict)
toml_lines = np.array(toml_dump.splitlines(), dtype="S200")
toml_col = fits.Column(name="TOML", format="200A", array=toml_lines)
hdus.append(fits.BinTableHDU.from_columns([toml_col], name="TOML_DUMP"))

# ---- write file ----
fits.HDUList(hdus).writeto(fits_path, overwrite=True)
print(f"wrote pokeramp FITS: {fits_path}")





# -----------------------------------------
# SUMMARY PLOTS (full updated block)
#   - error vs amplitude plots (LO + HO) over full ramp
#   - global slope matrices (full ramp)
#   - local slope matrices around zero amplitude (±k samples)
#   - optional sanity images at amp endpoints
# -----------------------------------------

os.makedirs(args.fig_path, exist_ok=True)

# ----- basic dims -----
probe_amps = np.asarray(probe_amps, dtype=float)

n_mode = int(eLO_cube.shape[0])
n_amp  = int(eLO_cube.shape[1])
n_lo   = int(eLO_cube.shape[2])
n_ho   = int(eHO_cube.shape[2])

# -----------------------------------------
# Identify zero-amplitude index and window
# -----------------------------------------
i0 = int(np.argmin(np.abs(probe_amps)))

# half-width of local window around zero (±k samples)
k = 2  # change to 3,4,... if you want a wider local fit

i_lo = max(0, i0 - k)
i_hi = min(len(probe_amps), i0 + k + 1)

probe_amps_local = probe_amps[i_lo:i_hi]

# -----------------------------------------
# (1) Error vs amplitude plots (full ramp)
#   - plot a few reconstructed LO/HO indices
#   - overlay a few poked modes
# -----------------------------------------

# how many reconstructed modes to show
n_lo_plot = min(4, n_lo)
n_ho_plot = min(4, n_ho)

# how many poked modes to overlay
n_mode_plot = min(4, n_mode)

# LO: for each reconstructed LO index r, overlay responses for a few poked modes m
for r in range(n_lo_plot):
    plt.figure(figsize=(6, 4))
    for m in range(n_mode_plot):
        y = eLO_cube[m, :, r]
        plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"poke mode {m}")

    plt.xlabel("poke amplitude")
    plt.ylabel(f"e_LO[{r}]")
    plt.title(
        f"LO error vs amplitude (full ramp)\n"
        f"reconstructed LO index {r}  (beam{args.beam_id}, {args.phasemask})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(args.fig_path, f"eLO_vs_amp_reco{r:02d}.png"), dpi=150)
    #plt.close()

# HO: for each reconstructed HO index r, overlay responses for a few poked modes m
for r in range(n_ho_plot):
    plt.figure(figsize=(6, 4))
    for m in range(n_mode_plot):
        y = eHO_cube[m, :, r]
        plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"poke mode {m}")

    plt.xlabel("poke amplitude")
    plt.ylabel(f"e_HO[{r}]")
    plt.title(
        f"HO error vs amplitude (full ramp)\n"
        f"reconstructed HO index {r}  (beam{args.beam_id}, {args.phasemask})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(args.fig_path, f"eHO_vs_amp_reco{r:02d}.png"), dpi=150)
    #plt.close()


# -----------------------------------------
# (2) Diagonal response plots (poke m -> reco m)
#   - show a few LO and a few HO diagonals, full ramp
# -----------------------------------------
n_diag = min(8, n_mode)

# LO diagonals
for m in range(n_diag):
    if m >= n_lo:
        break

    y = eLO_cube[m, :, m]
    p = np.polyfit(probe_amps, y, 1)
    yfit = p[0] * probe_amps + p[1]

    plt.figure(figsize=(6, 4))
    plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"data  slope={p[0]:.3g}")
    plt.plot(probe_amps, yfit, linestyle="--", label="linear fit")
    plt.xlabel("poke amplitude")
    plt.ylabel(f"e_LO[{m}]")
    plt.title(f"LO diagonal response (full ramp)\npoke mode {m}  (beam{args.beam_id}, {args.phasemask})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(args.fig_path, f"diag_LO_mode{m:03d}.png"), dpi=150)
    #plt.close()

# HO diagonals
for m in range(n_diag):
    if m >= n_ho:
        break

    y = eHO_cube[m, :, m]
    p = np.polyfit(probe_amps, y, 1)
    yfit = p[0] * probe_amps + p[1]

    plt.figure(figsize=(6, 4))
    plt.plot(probe_amps, y, marker="o", linestyle="-", label=f"data  slope={p[0]:.3g}")
    plt.plot(probe_amps, yfit, linestyle="--", label="linear fit")
    plt.xlabel("poke amplitude")
    plt.ylabel(f"e_HO[{m}]")
    plt.title(f"HO diagonal response (full ramp)\npoke mode {m}  (beam{args.beam_id}, {args.phasemask})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(args.fig_path, f"diag_HO_mode{m:03d}.png"), dpi=150)
    #plt.close()


# -----------------------------------------
# (3) Global slope matrices (full ramp)
#   slopes_LO[m, r] = slope of e_LO[r] when poking mode m (fit over full probe_amps)
# -----------------------------------------
slopes_LO = np.zeros((n_mode, n_lo), dtype=np.float32)
slopes_HO = np.zeros((n_mode, n_ho), dtype=np.float32)

for m in range(n_mode):
    for r in range(n_lo):
        y = eLO_cube[m, :, r]
        slopes_LO[m, r] = np.polyfit(probe_amps, y, 1)[0]

for m in range(n_mode):
    for r in range(n_ho):
        y = eHO_cube[m, :, r]
        slopes_HO[m, r] = np.polyfit(probe_amps, y, 1)[0]


# -----------------------------------------
# (4) Local slope matrices (near zero amplitude)
#   fit only within indices [i_lo:i_hi]
# -----------------------------------------
slopes_LO_local = np.zeros((n_mode, n_lo), dtype=np.float32)
slopes_HO_local = np.zeros((n_mode, n_ho), dtype=np.float32)

for m in range(n_mode):
    for r in range(n_lo):
        y = eLO_cube[m, i_lo:i_hi, r]
        slopes_LO_local[m, r] = np.polyfit(probe_amps_local, y, 1)[0]

for m in range(n_mode):
    for r in range(n_ho):
        y = eHO_cube[m, i_lo:i_hi, r]
        slopes_HO_local[m, r] = np.polyfit(probe_amps_local, y, 1)[0]


# -----------------------------------------
# (5) Plot slope matrices (global + local)
# -----------------------------------------

plt.figure(figsize=(7, 5))
plt.imshow(slopes_LO, aspect="auto")
plt.colorbar(label="global slope (reco / amp)")
plt.xlabel("LO reconstructed index")
plt.ylabel("poked mode index")
plt.title(f"LO slope matrix (global, full ramp)\nbeam{args.beam_id}, {args.phasemask}")
plt.tight_layout()
#plt.savefig(os.path.join(args.fig_path, "slopes_LO_global.png"), dpi=150)
#plt.close()

plt.figure(figsize=(7, 5))
plt.imshow(slopes_LO_local, aspect="auto")
plt.colorbar(label=f"local slope (±{k} samples) (reco / amp)")
plt.xlabel("LO reconstructed index")
plt.ylabel("poked mode index")
plt.title(f"LO slope matrix (local around 0)\nbeam{args.beam_id}, {args.phasemask}")
plt.tight_layout()
#plt.savefig(os.path.join(args.fig_path, "slopes_LO_local.png"), dpi=150)
#plt.close()

plt.figure(figsize=(7, 5))
plt.imshow(slopes_HO, aspect="auto")
plt.colorbar(label="global slope (reco / amp)")
plt.xlabel("HO reconstructed index")
plt.ylabel("poked mode index")
plt.title(f"HO slope matrix (global, full ramp)\nbeam{args.beam_id}, {args.phasemask}")
plt.tight_layout()
#plt.savefig(os.path.join(args.fig_path, "slopes_HO_global.png"), dpi=150)
#plt.close()

plt.figure(figsize=(7, 5))
plt.imshow(slopes_HO_local, aspect="auto")
plt.colorbar(label=f"local slope (±{k} samples) (reco / amp)")
plt.xlabel("HO reconstructed index")
plt.ylabel("poked mode index")
plt.title(f"HO slope matrix (local around 0)\nbeam{args.beam_id}, {args.phasemask}")
plt.tight_layout()
#plt.savefig(os.path.join(args.fig_path, "slopes_HO_local.png"), dpi=150)
#plt.close()


# -----------------------------------------
# (6)  endpoints of ramp
# -----------------------------------------
n_img_show = min(4, int(imgs_cube.shape[0]))
for m in range(n_img_show):
    for ai, tag in [(0, "amin"), (n_amp - 1, "amax")]:
        plt.figure(figsize=(5, 4))
        plt.imshow(imgs_cube[m, ai, :, :])
        plt.colorbar()
        plt.title(f"IMG poke mode {m} {tag}\nbeam{args.beam_id}, {args.phasemask}")
        plt.tight_layout()
        #plt.savefig(os.path.join(args.fig_path, f"img_mode{m:03d}_{tag}.png"), dpi=150)
        #plt.close()

print(f"summary plots saved to: {args.fig_path}")

plt.show()


# ############ my original 
# probe_amps = np.linspace( -args.amp_max , args.amp_max, args.no_amp_samples)
# pokeramp_list = {}
# for idx, mode in enumerate(modal_basis):
#     pokeramp_list[idx] = {  'imgs':[],
#                             'signal':[],
#                             'e_LO':[],
#                             'e_HO':[],
#                             }

#     for amp in probe_amps:

#         # apply dm 

#         time.sleep(0.01)
#         dm_shm.set_data( amp * mode )  # we apply on shm channel 2. DM flat should be on channel 1 
#         # get frames 
#         subframes = []
#         for _ in range( args.no_samples_per_cmd ):
#             subframes.append( cam_shm.get_data() - dark_current )
#             time.sleep(0.01) #


#         # average 
#         subframe_avg = np.mean( subframes ,axis = 0)
        
#         # get telemetry from reconstructor 

#         # for the signal we use the updated measured N0_runtime_currnt (since we typically update onsky)
#         # but we keep the internal i_setpoint_runtime
#         signal = subframe_avg / N0_runtime_currnt - i_setpoint_runtime

#         if args.signal_space=='dm':
#             e_LO = I2M_LO @ (I2A @ signal)
#             e_HO = I2M_HO @ (I2A @ signal)
#         elif args.signal_space=='pix':
#             e_LO = I2M_LO @ signal
#             e_HO = I2M_HO @ signal

#         # append 
#         pokeramp_list[idx]['imgs'].append( subframe_avg )
#         pokeramp_list[idx]['signal'].append( signal )
#         pokeramp_list[idx]['e_LO'].append( e_LO )
#         pokeramp_list[idx]['e_HO'].append( e_HO )



# # WRITE TO FITS WITH ALL METADATA READ IN FROM TOML CONFIG, +NEW REFERENCES MEASURED (any variable with *_current)
# # ALSO camera_config_current = {k:str(v) for k,v in camclient.config.items()}, and 

# #save 

# # tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# # tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

# # fits_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/pokeramp/{tstamp_rough}/pokeramp_beam{args.beam_id}_{args.phasemask}_{args.basis_name}_{tstamp}.fits"

# # also some summary plots of the response 










