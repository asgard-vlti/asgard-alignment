

import common.phasescreens as ps  
import numpy as np
from xaosim.shmlib import shm
import time
import argparse
from astropy.io import fits
import common.phasescreens as ps 
import pyBaldr.utilities as util 
import common.DM_basis_functions as dmbases
from asgard_alignment.DM_shm_ctrl import dmclass
import matplotlib.pyplot as plt
import datetime 
import os
import toml
import zmq

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


tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 


parser = argparse.ArgumentParser(description="build control model")


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
    type=int,
    default=3,
    help="what beam are we considering. Default: %(default)s"
)

parser.add_argument(
    "--phasemask",
    type=str,
    default="H4",
    help="what phasemask was used for building the IM. THis is to search the right entry in the configuration file. Default: %(default)s"
)


args=parser.parse_args()




############ setup

# cam and dm 
cam = shm(f"/dev/shm/baldr{args.beam_id}.im.shm") 
dm = dmclass( beam_id=args.beam_id, main_chn=2 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
    
# mds for taking dark 

MDS_port = 5555
MDS_host = "192.168.100.2" # simmode : "127.0.0.1" #'localhost'
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 5000)
server_address = f"tcp://{MDS_host}:{MDS_port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


# read in baldr confgig 
with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    

    #  read in the current calibrated matricies 
    # pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    # I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    # IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    # M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

   
    inner_pupil_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    # clear pupil 
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)




# hard coded parameters 
opd_per_cmd = 3000

N0_norm = np.mean( N0[inner_pupil_filt.astype(bool)] ) # defined in build_IM_in_subframe (interaction matrix standard)

r0 = 0.1 * (1.65/0.5)**(6/5) #wavelenfgth scaling
L0 = 0.1 #m #doesnt matter here  


# get dark 

print("turning off internal SBB source for bias")
send_and_get_response("off SBB")

time.sleep(10)


dark_tmp = []
for _ in range( 1000 ):
    dark_tmp.append( cam.get_data() )
    time.sleep(0.01) #
dark = np.mean( dark_tmp , axis=0)

send_and_get_response("on SBB")
time.sleep(3)
print("turning back on internal SBB source, check plot that darks are ok")

# check 
util.nice_heatmap_subplots( im_list=[dark] )
plt.show() 


# 1 init DM phase screens 
number_of_screen_initiations = 50
scrn_list = []
for _ in range(number_of_screen_initiations):
    #scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.N, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
    dm_scrn = ps.PhaseScreenKolmogorov(nx_size=24, 
                                       pixel_scale = 1.8 / 24, 
                                       r0=r0, 
                                       L0=L0, 
                                       random_seed=None)
    scrn_list.append( dm_scrn ) 



# 2. init telemetry to build model ()
telem = {
    "N0":N0,
    "N0_norm":N0_norm,
    "i":[],
    "s":[],
    "opd_rms_est":[], # opd
}

# 3. measure telemetry 
scrn_scaling_grid = np.logspace(-1,0.2,5)
for it in range(len(scrn_list)):
    print( f"input aberation {it}/{len(scrn_list)}" )
    # roll screen
    #scrn.add_row()     
    for ph_scale in scrn_scaling_grid: 
        #scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False
        cmd =  util.create_phase_screen_cmd_for_DM(scrn_list[it],  
                                                   scaling_factor=ph_scale , 
                                                   drop_indicies = None, #[0, 11, 11 * 12, -1] , 
                                                   plot_cmd=False) 
        cmd = np.array(cmd).reshape(12,12) 
        opd_est = opd_per_cmd * np.std( cmd )
        dm.set_data( cmd )
        
        time.sleep(0.01)

        i = cam.get_data() - dark

        s = i / N0_norm  # we do like this because its strehl model! 

        #opd_true = np.std( opd_current_dm[zwfs_ns.grid.pupil_mask.astype(bool)] ) # *  2*np.pi / zwfs_ns.optics.wvl0 * (  opd_current_dm  )
        opd_est =  np.std( opd_per_cmd * cmd )
        #plt.figure(); plt.imshow( util.get_DM_command_in_2D( zwfs_ns.dm.opd_per_cmd * np.array( zwfs_ns.dm.current_cmd)));plt.colorbar();plt.show()

        telem['i'].append( i )
        telem['s'].append( s )
        #telem['opd_rms_true'].append( opd_true )
        telem['opd_rms_est'].append( opd_est )

# zero dm 
dm.set_data( 0 * np.array(cmd).reshape(12,12) )

correlation_map = util.compute_correlation_map(np.array( telem['s'] ), np.array( telem['opd_rms_est'] ) )


yy, xx = np.ogrid[:telem['s'][0].shape[0], :telem['s'][0].shape[0]]
snr = (np.mean( np.array( telem['s'] ) , axis =0 ) / np.std(  np.array( telem['s'] ) , axis =0  )) 
radial_constraint = ((xx - telem['s'][0].shape[0]//2)**2 + (yy - telem['s'][0].shape[0]//2)**2 <= 20**2) * ( (xx - telem['s'][0].shape[0]//2)**2 + (yy - telem['s'][0].shape[0]//2)**2 >= 6**2 )
# some criteria to filter (this could be way more advanced if we wanted)
strehl_filt = (correlation_map < -0.7) & (snr > 1.) & radial_constraint
strehl_pixels = np.where( strehl_filt )


util.nice_heatmap_subplots( im_list = [ correlation_map, strehl_filt ] , cbar_label_list = ['Pearson R','filt'] )
#savefig = save_results_path + 'strehl_vs_intensity_pearson_R.png' ) #fig_path + 'strehl_vs_intensity_pearson_R.png' )

plt.figure()
plt.plot( [np.mean( ss[strehl_filt] ) for ss in telem['s']] , np.array( telem['opd_rms_est'] )  ,'.', label='est')
#plt.plot( [np.mean( ss[strehl_filt] ) for ss in telem['s']] , np.array( telem['opd_rms_true'] )  ,'.', label='true')
plt.xlabel('<s>')
plt.ylabel('OPD RMS [nm RMS]')
plt.legend()
plt.show()


filtered_sigs = np.array( [np.mean( ss[strehl_filt] ) for ss in telem['s']] )
opd_nm_est =   np.array( telem['opd_rms_est'] ) 

opd_model_params = util.fit_piecewise_continuous(x=filtered_sigs, y=opd_nm_est, n_grid=80, trim=0.15)

print(f"using util.fit_piecewise_continuous, opd_model_params = {opd_model_params} \ninput these to util.fit_piecewise_continuous")


y = opd_nm_est
x = np.array( [np.mean( ss[strehl_filt] ) for ss in telem['s']] )

opd_pred = util.piecewise_continuous(x ,
                                         interc=opd_model_params['interc'],
                                          slope_1=opd_model_params['slope_1'],
                                           slope_2 = opd_model_params['slope_2'],
                                            x_knee = opd_model_params['x_knee'] )

plt.figure()
plt.scatter( x, y ,label='meas')
plt.scatter( x, opd_pred,label = 'pred' )
plt.xlabel('signal (i_ext / <N0>)')
plt.ylabel('OPD [nm RMS]')
plt.legend()
plt.show()


# write to toml..

# TO DO : FIX M2C PROJECTION ====================
dict2write = {f"beam{args.beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                               "strehl_filter":np.array(strehl_filt).astype(int).reshape(-1).tolist(),
                                                }
                                            }
                                        }
                                    }



# Check if file exists; if so, load and update.
if os.path.exists(args.toml_file.replace('#',f'{args.beam_id}')):
    try:
        current_data = toml.load(args.toml_file.replace('#',f'{args.beam_id}'))
    except Exception as e:
        print(f"Error loading TOML file: {e}")
        current_data = {}
else:
    current_data = {}


current_data = util.recursive_update(current_data, dict2write)

with open(args.toml_file.replace('#',f'{args.beam_id}'), "w") as f:
    toml.dump(current_data, f)

print( f"updated configuration file {args.toml_file.replace('#',f'{args.beam_id}')}")

