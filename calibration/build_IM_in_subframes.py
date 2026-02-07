#!/usr/bin/env python
import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse
# import subprocess
# import glob

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



# By default HO in this construction of the IM will always contain zonal actuation of each DM actuator.
# Using LO we can also define our Lower order modes on a Zernike basis where LO 
# is the Noll index up to which modes to consider. These LO modes are probed first
# in the IM and then the HO (zonal) modes are probed  


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
    type=lambda s: [int(item) for item in s.split(",")],
    default=[3], # 1, 2, 3, 4],
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
    default='zernike', #"zonal",
    help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
)

# parser.add_argument(
#     "--Nmodes",
#     type=int,
#     default=10,
#     help="number of modes to probe"
# )

parser.add_argument(
    "--poke_amp",
    type=float,
    default=0.05,
    help="amplitude to poke DM modes for building interaction matrix"
)

parser.add_argument(
    "--signal_space",
    type=str,
    default='dm',
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



no_imgs = 10  # how many reference images to take per DM state

I2A_dict = {}
pupil_mask = {}
secondary_mask = {}
exterior_mask = {}
for beam_id in args.beam_id:

    # read in TOML as dictionary for config 
    with open(args.toml_file.replace('#',f'{beam_id}'), "r") as f:
        config_dict = toml.load(f)
        # Baldr pupils from global frame 
        baldr_pupils = config_dict['baldr_pupils']
        I2A_dict[beam_id] = config_dict[f'beam{beam_id}']['I2A']
        
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

        secondary_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

        exterior_mask[beam_id] = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )






# for reading camera config ! 
camclient = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])

# Open SHM once for each subframe 
cam_shm = {b: shm(f"/dev/shm/baldr{b}.im.shm") for b in args.beam_id}


# set up DM SHMs 
print( 'setting up DMs')
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id, main_chn=2 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
    
    ###     UP TO USER TO PUT THE FLAT ON!!!
    # zero all channels
    # dm_shm_dict[beam_id].zero_all()
    
    # if args.DM_flat.lower() == 'factory':
    #     # activate flat (does this on channel 1)
    #     dm_shm_dict[beam_id].activate_flat()
    # elif args.DM_flat.lower() == 'baldr':
    #     # apply dm flat + calibrated offset (does this on channel 1)
    #     dm_shm_dict[beam_id].activate_calibrated_flat()
        
    # else:
    #     print( "Unknow flat option. Valid options are 'factory' or 'baldr'. Using baldr flat as default")
    #     args.DM_flat == 'baldr'
    #     dm_shm_dict[beam_id].activate_calibrated_flat()




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
for beam_id in args.beam_id:
    dark_tmp = []
    for _ in range( 1000 ):
        dark_tmp.append( cam_shm[beam_id].get_data() )
        time.sleep(0.01) #
    dark_dict[beam_id] = np.mean( dark_tmp , axis=0)

send_and_get_response("on SBB")
time.sleep(3)
print("turning back on internal SBB source, check plot that darks are ok")

# check 
util.nice_heatmap_subplots( im_list=[dark_dict[beam_id] for beam_id in args.beam_id], title_list=[f"beam{beam_id}" for beam_id in args.beam_id] )
plt.show() 


print(f"moving to phasemask {args.phasemask} reference position")
# Move to phase mask
for beam_id in args.beam_id:
    message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
    res = send_and_get_response(message)
    print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)

# Get reference pupils (later this can just be a SHM address)
zwfs_pupils = {}
clear_pupils = {}
normalized_pupils = {}
# for new phasemask H band is close to edge so offset MUST be negative direction
rel_offset = -200.0 #um phasemask offset for clear pupil
print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:
    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep( 1 )
    message = f"moverel BMY{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 


time.sleep(0.5)


############
#Clear Pupil
print( 'gettin clear pupils')

inner_pupil_filt = {} # strictly inside (not on boundary)

for beam_id in args.beam_id:

    
    N0s = []
    for _ in range( no_imgs ):
        N0s.append( cam_shm[beam_id].get_data()  - dark_dict[beam_id] )
        time.sleep(0.01) #

    
    clear_pupils[beam_id] = N0s  


    ############
    #Interior pupil filt
    # in baldr python rtc I0_setpoint_runtime is done in build_rt_model and is defined I0 / np.mean( N0[inner_pupil_filt] )
    # so this definition is critical. We try filter edge pupils around pupil perimeter and secondary (for Solarstein source)
    # same convention as BaldrApp.baldr_core.build_IM for inner_pupil_filt
    inner_pupil_filt[beam_id] = binary_erosion( pupil_mask[beam_id]  * (~secondary_mask[beam_id].astype(bool)), structure=np.ones((3, 3), dtype=bool) )
    ## BELOW IS OLD CONVENTION (pixelwise normalized_pupils, with outside pupil set to interior mean) , 
    #  keep for C++ rtc legacy (wrtten. to toml)
    # this is not needed for new python rtc standards    
    pixel_filter = np.array( secondary_mask[beam_id].astype(bool) )  |  np.array( (~(util.remove_boundary(np.array( pupil_mask[beam_id])).astype(bool)) ) ) #| (~bad_pix_mask_tmp )
    normalized_pupils[beam_id] = np.mean( clear_pupils[beam_id] , axis=0) 
    normalized_pupils[beam_id][ pixel_filter ] = np.mean( np.mean(clear_pupils[beam_id],0)[~pixel_filter]  ) # set exterior and boundary pupils to interior mean

    # move phase mask back in
    print( 'Moving FPM back in beam.')
    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(1)
    message = f"moverel BMY{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(3)

#############
# ZWFS Pupil
input("phasemasks aligned? ensure alignment then press enter")


print( 'Getting ZWFS pupils')
for beam_id in args.beam_id:

    
    I0s = []
    for _ in range( no_imgs ):
        I0s.append( cam_shm[beam_id].get_data() - dark_dict[beam_id] )
        time.sleep(0.01) #

    zwfs_pupils[beam_id] = I0s 


#basis_name = args.basis_name #"zonal" #"ZERNIKE"
LO_basis = dmbases.zer_bank(2, args.LO+1 )
if args.basis_name == 'zonal':
    zonal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(140)])
elif args.basis_name == 'zernike':
    zonal_basis = dmbases.zer_bank(4, 143 ) #143 if 
modal_basis = np.array( LO_basis.tolist() +  zonal_basis.tolist() ) 
# should be 144 x 140 (we deal with errors in 140 actuator space (columns), but SHM takes 144 vector as input (rows)) 
# this is why we do transpose 
M2C = modal_basis.copy().reshape(modal_basis.shape[0],-1).T # mode 2 command matrix 


if args.signal_space.lower() not in ["dm", "pixel"] :
    raise UserWarning("signal space must either be 'dm' or 'pixel'")



############
# BUILDING IM 

n_modes = modal_basis.shape[0]
number_of_pokes_per_cmd = 8 
signs = [(-1)**n for n in range(number_of_pokes_per_cmd )]
pos_idx = [k for k,s in enumerate(signs) if s > 0]
neg_idx = [k for k,s in enumerate(signs) if s < 0]
n_plus, n_minus = len(pos_idx), len(neg_idx)



# Infer frame shape once
frame0 = cam_shm[args.beam_id[0]].get_data()
ny, nx = frame0.shape

# Preallocate storage
Iplus_stack  = {b: np.zeros((n_modes, n_plus,  ny, nx), dtype=np.float32) for b in args.beam_id}
Iminus_stack = {b: np.zeros((n_modes, n_minus, ny, nx), dtype=np.float32) for b in args.beam_id}
IM_mat       = {b: np.zeros((n_modes, 140 if args.signal_space=='dm' else ny*nx), dtype=np.float32) for b in args.beam_id}

for i, m in enumerate(modal_basis):
    print(f"executing cmd {i}/{n_modes-1}")

    plus_k = 0
    minus_k = 0

    for s in signs:
        # set all beams once
        cmd = (s * args.poke_amp/2) * m
        for b in args.beam_id:
            dm_shm_dict[b].set_data(cmd)

        # read all beams
        for b in args.beam_id:
            imgs = []
            for _ in range(no_imgs):
                imgs.append(cam_shm[b].get_data() - dark_dict[b])
                time.sleep(0.01)
            img_tmp = np.mean(imgs, axis=0).astype(np.float32)

            if s > 0:
                Iplus_stack[b][i, plus_k] = img_tmp
            else:
                Iminus_stack[b][i, minus_k] = img_tmp

        if s > 0:
            plus_k += 1
        else:
            minus_k += 1

    # build IM row(s) for each beam
    for b in args.beam_id:
        N0_tmp = np.mean(clear_pupils[b], axis=0)
        norm_factor = float(np.mean(N0_tmp[inner_pupil_filt[b]]))

        I_plus  = np.mean(Iplus_stack[b][i],  axis=0).reshape(-1) / norm_factor
        I_minus = np.mean(Iminus_stack[b][i], axis=0).reshape(-1) / norm_factor

        if args.signal_space.lower() == "dm":
            errsig = (I2A_dict[b] @ (I_plus - I_minus)) / args.poke_amp
        else:
            errsig = (I_plus - I_minus) / args.poke_amp

        IM_mat[b][i] = errsig.astype(np.float32).reshape(-1) 


# close camera SHM 
for b in cam_shm :
    cam_shm[b].close(erase_file=False)

for beam_id in args.beam_id:
    # set back to zero on the given probe channel
    dm_shm_dict[beam_id].set_data( 0 * cmd )
    # close
    #dm_shm_dict[beam_id].close(erase_file=False)


# Save components as seperate fits too so we can analyse and keep record  
hdul = fits.HDUList()

# PRIMARY
phdr = fits.Header()
phdr["DATE"]   = datetime.datetime.utcnow().isoformat()
phdr["PHMASK"] = args.phasemask
phdr["POKEAMP"]= float(args.poke_amp)
phdr["LO"]     = int(args.LO)
phdr["SIGSPC"] = args.signal_space.lower()
phdr["NOIMGS"] = int(no_imgs)
phdr["NSIGN"]  = number_of_pokes_per_cmd 
phdr["BEAMS"]  = ",".join(map(str, args.beam_id))
hdul.append(fits.PrimaryHDU(header=phdr))

# Shared basis
hdul.append(fits.ImageHDU(np.asarray(modal_basis, dtype=np.float32), name="MODES"))
hdul.append(fits.ImageHDU(np.asarray(M2C, dtype=np.float32), name="M2C"))

for b in args.beam_id:
    hdul.append(fits.ImageHDU(np.asarray(IM_mat[b], dtype=np.float32), name=f"IM_B{b}"))

    I0 = np.mean(zwfs_pupils[b], axis=0).astype(np.float32)
    N0 = np.mean(clear_pupils[b], axis=0).astype(np.float32)
    hdul.append(fits.ImageHDU(I0, name=f"I0_B{b}"))
    hdul.append(fits.ImageHDU(N0, name=f"N0_B{b}"))
    hdul.append(fits.ImageHDU(I2A_dict[b], name=f"I2A_B{b}"))

    # masks
    hdul.append(fits.ImageHDU(np.asarray(pupil_mask[b], dtype=np.uint8), name=f"PUPIL_B{b}"))
    hdul.append(fits.ImageHDU(np.asarray(secondary_mask[b], dtype=np.uint8), name=f"SEC_B{b}"))
    hdul.append(fits.ImageHDU(np.asarray(exterior_mask[b], dtype=np.uint8), name=f"EXT_B{b}"))
    hdul.append(fits.ImageHDU(np.asarray(inner_pupil_filt[b], dtype=np.uint8), name=f"INNER_B{b}"))

    # compressed stacks (optional but recommended)
    hdul.append(fits.CompImageHDU(np.asarray(Iplus_stack[b], dtype=np.float32), name=f"IPLUS_B{b}", compression_type="RICE_1"))
    hdul.append(fits.CompImageHDU(np.asarray(Iminus_stack[b], dtype=np.float32), name=f"IMINUS_B{b}", compression_type="RICE_1"))
    hdul.append(fits.CompImageHDU(np.asarray(IM_mat[b], dtype=np.float32), name=f"IM_FINAL", compression_type="RICE_1"))

tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

fits_path = f"/home/asg/Progs/repos/asgard-alignment/calibration/reports/IM/{tstamp_rough}/IM_{args.phasemask}_{tstamp}.fits"
os.makedirs(os.path.dirname(fits_path), exist_ok=True)
hdul.writeto(fits_path, overwrite=True)
print( f"saved fits file with IM telemetry {fits_path}")

### Previous non optimized way 
# #############
# # BUILDING IM 
# IM = {beam_id:[] for beam_id in args.beam_id}
# Iplus_all = {beam_id:[] for beam_id in args.beam_id}
# Iminus_all = {beam_id:[] for beam_id in args.beam_id}
# for i,m in enumerate(modal_basis):
#     print(f'executing cmd {i}/{len(modal_basis)}')
#     #if i == args.LO:
#     #    input("close Baldr TT and ensure stable. Then press enter.")
#     I_plus_list = {beam_id:[] for beam_id in args.beam_id}
#     I_minus_list = {beam_id:[] for beam_id in args.beam_id}
#     for sign in [(-1)**n for n in range(4)]: #range(10)]: #[-1,1]:
        
#         for beam_id in args.beam_id:
#             dm_shm_dict[beam_id].set_data(  sign * args.poke_amp/2 * m ) 

#         for beam_id in args.beam_id:
#             # set dm 
#             dm_shm_dict[beam_id].set_data(  sign * args.poke_amp/2 * m ) 
#             # subframe shared memory
#             c = shm( f"/dev/shm/baldr{beam_id}.im.shm")
#             img_list = []
#             for _ in range( no_imgs ):
#                 img_list.append( c.get_data() )
#                 time.sleep(0.01) #
#             img_tmp = np.mean( img_list ,axis = 0 )
            
#             if sign > 0:

#                 I_plus_list[beam_id].append( list( img_tmp ) )

#             if sign < 0:

#                 I_minus_list[beam_id].append( list( img_tmp ) )

#             c.close(erase_file=False)

#     for beam_id in args.beam_id:
#         # normalization by mean signal in the interior of the clear pupil
#         N0_tmp = np.mean( clear_pupils[beam_id],axis=0 ) # average of N measurements of clear pupil
#         norm_factor = np.mean( N0_tmp[inner_pupil_filt[beam_id]] ) # average of the interior pupil pixels

#         I_plus = np.mean( I_plus_list[beam_id], axis = 0).reshape(-1) / norm_factor   #normalized_pupils[beam_id].reshape(-1)
#         I_minus = np.mean( I_minus_list[beam_id], axis = 0).reshape(-1) / norm_factor  #normalized_pupils[beam_id].reshape(-1)

#         if args.signal_space.lower() == 'dm':
            
#             errsig = I2A_dict[beam_id] @   (I_plus - I_minus)  / args.poke_amp  
            
#         elif args.signal_space.lower() == 'pixel':
            
#             errsig =  (I_plus - I_minus)  / args.poke_amp  # 1 / DMcmd * (s * gain)  projected to Pixel space

#         IM[beam_id].append( list(  errsig.reshape(-1) ) ) 




######## WRITE TO TOML 
#  # we store all reference images as flattened array , boolean masks as ints
for beam_id in args.beam_id:
    r1,r2,c1,c2 = baldr_pupils[f'{beam_id}'] 

    dict2write = {f"beam{beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                                    "build_method":"double-sided-poke",
                                                    "DM_flat":args.DM_flat.lower(),
                                                    "signal_space":args.signal_space.lower(),
                                                    "crop_pixels": np.array( baldr_pupils[f"{beam_id}"] ).tolist(), # global corners (r1,r2,c1,c2) of sub pupil cropping region  (local frame)
                                                    "pupil_pixels" : np.where(  np.array( pupil_mask[beam_id] ).reshape(-1) )[0].tolist(),  # pupil pixels in local frame 
                                                    "interior_pixels" : np.where( np.array( inner_pupil_filt[beam_id].reshape(-1) )   )[0].tolist(), # strictly interior pupil pixels in local frame
                                                    "secondary_pixels" : np.where( np.array( secondary_mask[beam_id].reshape(-1) )  )[0].tolist(),   # pixels in secondary obstruction in local frame
                                                    "exterior_pixels" : np.where(  np.array( exterior_mask[beam_id].reshape(-1) )   )[0].tolist(),  # exterior pixels that maximise diffracted light from mask in local frame 
                                                    #"bad_pixels" : np.where( np.array( c.reduction_dict['bad_pixel_mask'][-1])[r1:r2,c1:c2].reshape(-1)   )[0].tolist(),
                                                    "IM": np.array( IM_mat[beam_id] ).tolist(),#np.array( IM[beam_id] ).tolist(),
                                                    "poke_amp":args.poke_amp,
                                                    "LO":args.LO, ## THIS DEFINES WHAT INDEX IN IM WE HAVE LO VS HO MODES , DONE HERE NOW RATHER THAN build_baldr_control_matrix.py.
                                                    "M2C": np.nan_to_num( np.array(M2C), 0 ).tolist(),   # 
                                                    "I0": np.mean( zwfs_pupils[beam_id],axis=0).reshape(-1).tolist(), ## ## post TTonsky  #(float( c.config["fps"] ) / float( c.config["gain"] ) * np.mean( zwfs_pupils[beam_id],axis=0).reshape(-1) ).tolist(),  # ADU / s / gain (flattened)
                                                    "intrn_flx_I0":float(np.sum( np.mean( zwfs_pupils[beam_id],axis=0) ) ),
                                                    "N0": np.mean( clear_pupils[beam_id],axis=0).reshape(-1).tolist(), ## ## post TTonsky #(float( c.config["fps"] ) / float( c.config["gain"] ) * np.mean( clear_pupils[beam_id],axis=0).reshape(-1) ).tolist(), # ADU / s / gain (flattened)
                                                    "norm_pupil": np.array( normalized_pupils[beam_id] ).reshape(-1).tolist(), ## post TTonsky #( float( c.config["fps"] ) / float( c.config["gain"] ) * np.array( normalized_pupils[beam_id] ).reshape(-1) ).tolist(),
                                                    "camera_config" : {k:str(v) for k,v in camclient.config.items()},
                                                    #"bias": np.array(c.reduction_dict["bias"][-1])[r1:r2,c1:c2].reshape(-1).tolist(),
                                                    #"dark": np.array(c.reduction_dict["dark"][-1])[r1:r2,c1:c2].reshape(-1).tolist(),
                                                    #"bad_pixel_mask": np.array(c.reduction_dict['bad_pixel_mask'][-1])[r1:r2,c1:c2].astype(int).reshape(-1).tolist(),
                                                    "pupil": np.array(pupil_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "secondary": np.array(secondary_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "exterior" : np.array(exterior_mask[beam_id]).astype(int).reshape(-1).tolist(),
                                                    "inner_pupil_filt": np.array(inner_pupil_filt[beam_id]).astype(int).reshape(-1).tolist(),
                                                    # !!!! Set these calibration things to zero since they should be dealt with by cred 1 server! 
                                                    "bias" : np.zeros([32,32]).reshape(-1).astype(int).tolist(),
                                                    "dark" : np.zeros([32,32]).reshape(-1).astype(int).tolist(), # just update to a default 1000 adu offset. In rtc this can be updated with dark_update function!
                                                    "bad_pixel_mask" : np.ones([32,32]).reshape(-1).astype(int).tolist(),
                                                    "bad_pixels" : [], 
                                                }
                                                }
                                            }
                                        }

    # Check if file exists; if so, load and update.
    if os.path.exists(args.toml_file.replace('#',f'{beam_id}')):
        try:
            current_data = toml.load(args.toml_file.replace('#',f'{beam_id}'))
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}


    current_data = util.recursive_update(current_data, dict2write)

    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)

    print( f"updated configuration file {args.toml_file.replace('#',f'{beam_id}')}")


## A QUICK LOOK 

for beam_id in args.beam_id:
    U,S,Vt = np.linalg.svd( IM_mat[beam_id], full_matrices=True)
    #singular values
    plt.figure()
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend()
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    if not os.path.exists(args.fig_path):
        print(f"making directory {args.fig_path} for plotting some results.")
        os.makedirs( args.fig_path )
    
    plt.savefig(f'{args.fig_path}' + f'IM_singularvalues_beam{beam_id}.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()


# n_row = round( np.sqrt( M2C.shape[0]) ) - 1

# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( M2C.T @ U.T[i]  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=axplt.tight_layout()
# if save_path is not None:
#     plt.savefig(save_path +  f'cam_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()



# fig,ax = plt.subplots(n_row, n_row, figsize=(15,15))
# plt.subplots_adjust(hspace=0.1,wspace=0.1)
# for i,axx in enumerate(ax.reshape(-1)):
#     axx.imshow( get_DM_command_in_2D( Vt[i] )  )
#     #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
#     axx.text( 1,2,f'{i}',color='w',fontsize=6)
#     axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
#     axx.axis('off')
#     #plt.legend(ax=a

# if save_path is not None:
#     plt.savefig(save_path +  f'DM_eignmodes_{pokeamp}.png',bbox_inches='tight',dpi=200)
# plt.show()



# for beam_id in args.beam_id:

#     ################################
#     # the reference intensities
#     im_list = [ np.mean( zwfs_pupils[beam_id],axis=0), np.mean( clear_pupils[beam_id],axis=0), normalized_pupils[beam_id] ]
#     title_list = ['<I0>','<N0>','normalized pupil']
#     cbar_list = ["UNITLESS"] * len(im_list)
#     util.nice_heatmap_subplots( im_list , title_list=title_list, cbar_label_list=cbar_list) 
#     plt.savefig(f'{args.fig_path}' + f'reference_intensities_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
#     #plt.show()

#     ################################
#     # the interaction signal 
#     modes2look = [0,1,65,67]
#     im_list = [IM_mat[beam_id][m].reshape(12,12) for m in modes2look]

#     title_list = [f'mode {m}' for m in modes2look]
#     cbar_list = ["UNITLESS"] * len(im_list)
#     util.nice_heatmap_subplots( im_list , cbar_label_list=cbar_list, savefig=f'{args.fig_path}' + f'IM_first16modes_beam{beam_id}.png') 
#     plt.savefig(f'{args.fig_path}' + f'IM_some_modes_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
#     #plt.show()

#     ################################
#     # the eigenmodes 
#     U, S, Vt = np.linalg.svd(IM_mat[beam_id], full_matrices=False)  # shapes: (M, M), (min(M,N),), (min(M,N), N)

#     # (a) Plot singular values
#     plt.figure(figsize=(6, 4))
#     plt.semilogy(S, 'o-')
#     plt.title("Singular Values of IM_HO")
#     plt.xlabel("Index")
#     plt.ylabel("Singular value (log scale)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{args.fig_path}" + f'IM_singular_values_beam{beam_id}.png', bbox_inches='tight', dpi=200)

#     # (b) Intensity eigenmodes (Vt)
#     plt.figure(figsize=(15, 3))
#     for i in range(min(5, Vt.shape[0])):
#         ax = plt.subplot(1, 5, i+1)
#         im = ax.imshow(util.get_DM_command_in_2D(Vt[i]), cmap='viridis')
#         ax.set_title(f"Vt[{i}]")
#         plt.colorbar(im, ax=ax)
#     plt.suptitle("First 5 intensity eigenmodes (Vt) mapped to 2D")
#     plt.tight_layout()
#     plt.savefig(f"{args.fig_path}" + f'IM_first5_intensity_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)


#     # (c) System eigenmodes (U)
#     plt.figure(figsize=(15, 3))
#     for i in range(min(5, U.shape[1])):
#         ax = plt.subplot(1, 5, i+1)
#         im = ax.imshow(util.get_DM_command_in_2D(U[:, i]), cmap='plasma')
#         ax.set_title(f"U[:, {i}]")
#         plt.colorbar(im, ax=ax)
#     plt.suptitle("First 5 system eigenmodes (U) mapped to 2D")
#     plt.tight_layout()
#     plt.savefig(f"{args.fig_path}" + f'IM_first5_system_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)
#     #plt.show()


#     plt.close("all")



