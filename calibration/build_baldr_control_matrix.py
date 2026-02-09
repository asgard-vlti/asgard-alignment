#!/usr/bin/env python
import numpy as np 
import zmq
import time
import toml
import os 
import argparse
import matplotlib.pyplot as plt
import argparse
import subprocess
import glob
import argparse
import datetime
import common.DM_basis_functions as dmbases
import pyBaldr.utilities as util 

"""
Here we put together the final control config to be read in by RTC

- invert the interaction matrix by chosen meethod
- project to LO/HO matricies as desired
- write to ctrl key currently calibrated I2A in toml file 
- write to ctrl key currently calibrated strehl modes in toml file
- write to ctrl key desired shapes and states of the control system (default values) 

this is a large non-=human readable toml, in the future could put large matricies to fits files
and just keep the paths here. For now, for simplicity, I like EVERYTHING needed to configure
the RTC in one spot. Right here. 

"""

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
    choices=[f"H{i+1}" for i in range(5)]+[f"J{i+1}" for i in range(5)],
    help="what phasemask was used for building the IM. THis is to search the right entry in the configuration file. Default: %(default)s"
)

parser.add_argument(
    "--signal_space",
    type=str,
    default='dm', # pix | dm 
    choices=['dm','pix'],
    help="what space do we consider the signal on. either dm (uses I2A) or pixel measurement"
)

parser.add_argument(
    "--inverse_method_LO",
    type=str,
    default="pinv",
    help="Method used for inverting interaction matrix for LO to build control (intensity-mode) matrix I2M"
)


parser.add_argument(
    "--inverse_method_HO",
    type=str,
    default="zonal",
    help="Method used for inverting interaction matrix for HO to build control (intensity-mode) matrix I2M"
)


# parser.add_argument("--project_TT_out_HO",
#                     dest="project_TT_out_HO",
#                     action="store_true",
#                     help="Disable projecting TT (or what ever lower order LO is defined as) out of HO (default: enabled)")

parser.add_argument("--no_project_TT_out_HO",
                    dest="project_TT_out_HO",
                    action="store_false",
                    help="Disable projecting TT out of HO (default: enabled)")
parser.set_defaults(project_TT_out_HO=True) 

parser.add_argument("--project_waffle_out_HO",
                    dest="project_waffle_out_HO",
                    action="store_true",
                    help="If set, project out the DM waffle mode from the HO space.")


## NEED TO CHECK THIS AGAIN - BUG
# parser.add_argument("--filter_edge_actuators",
#                     dest="filter_edge_actuators",
#                     action="store_true",
#                     help="Filter actuators that interpolate from edge pixels (default: enabled)")
parser.add_argument("--no_filter_edge_actuators",
                    dest="filter_edge_actuators",
                    action="store_false",
                    help="Disable edge-actuator filtering (default: enabled)")

parser.set_defaults(filter_edge_actuators=True)


parser.add_argument("--fig_path", 
                    type=str, 
                    default=f'/home/asg/ben_bld_data/{tstamp_rough}/', 
                    help="path/to/output/image/ for the saved figures"
                    )



args=parser.parse_args()

# question still what to do with focus with secondary! 
print( "filter_edge_actuators = ",args.filter_edge_actuators)

with open(args.toml_file.replace('#',f'{args.beam_id}'), "r") as f:

    config_dict = toml.load(f)
    
    # Baldr pupils from global frame 
    # baldr_pupils = config_dict['baldr_pupils']
    # I2A = np.array( config_dict[f'beam{beam_id}']['I2A'] )
    
    # # image pixel filters
    # pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    # exter_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) ).astype(bool) # matrix bool
    # secon_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) ).astype(bool) # matrix bool

    #  read in the current calibrated matricies 
    
    pupil_mask = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
    I2A = np.array( config_dict[f'beam{args.beam_id}']['I2A'] )
    IM = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
    M2C = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

    I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("I0", None) )
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)
    norm_pupil = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("norm_pupil", None) )# matrix bool
    intrn_flx_I0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("intrn_flx_I0", None) )# matrix bool
    
    print(f"ENSURE FLUX NORMALIZATION EXISTS AND IS NOT LOW (i.e. ~ 1). intrn_flx_I0={intrn_flx_I0}")
    # also the current calibrated strehl modes 
    I2rms_sec = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", {}).get(f"{args.phasemask}", {}).get("secondary", None)).astype(float)
    I2rms_ext = np.array(config_dict.get(f"beam{args.beam_id}", {}).get("strehl_model", {}).get(f"{args.phasemask}", {}).get("exterior", None)).astype(float)
    
    if not np.isfinite(I2rms_sec):
        print("\n WARNING: No secondary strehl modes found in config file, using 2x2 I matrix instead.")
        I2rms_sec = np.eye(2) #(2, 2))
    if not np.isfinite(I2rms_ext):   
        print("\n WARNING: No exterior strehl modes found in config file, using 2x2 I matrix instead.")
        I2rms_ext = np.eye(2) #((2, 2))
        
    # # define our Tip/Tilt or lower order mode index on zernike DM basis 
    LO = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("LO", None)

    # tight (non-edge) pupil filter
    inside_edge_filt = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
    # clear pupil 
    N0 = np.array(config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)
    # secondary filter
    sec = np.array(config_dict.get(f"beam{args.beam_id}" , {}).get(f"{args.phasemask}", {}).get("ctrl_model",None).get("secondary", None) )
    #norm_pupil =np.array(config_dict.get(f"beam{args.beam_id}" , {}).get(f"{args.phasemask}", {}).get("ctrl_model",None).get("norm_pupil", None) )
    # these are just for testing things 
    poke_amp = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("poke_amp", None)
    camera_config = config_dict.get(f"beam{args.beam_id}", {}).get(f"{args.phasemask}", {}).get("ctrl_model", None).get("camera_config", None)


#util.nice_heatmap_subplots( [ util.get_DM_command_in_2D(a) for a in [IM[65], IM[77] ]],savefig='delme.png')

# define out Tip/Tilt or lower order modes on zernike DM basis
#LO = dmbases.zer_bank(2, LO +1 ) # 12x12 format



if args.signal_space.lower() == "dm":
    print('projecting IM to registered DM space via I2A')
    IM = np.array( [I2A @ ii for ii in IM] )
elif args.signal_space.lower() == "pix":
    print('keeping IM in pixel (measurement) space')
else:
    raise UserWarning('invlaid signal space! --signal_space must be dm | pix')


############ SETUP 

# split LO and HO modes from input LO definition in config file 
IM_LO = IM[:LO]
IM_HO = IM[LO:]


# define effective pupil mask on IM measurement space
pup_im_std = np.std( IM_HO.T, axis=0 )
pup_mask = (pup_im_std - np.min(pup_im_std)) / (np.max( pup_im_std ) - np.min( pup_im_std ))
pup_mask[pup_mask < 0.2] = 0 # normalize 0-1, anything below 0.2 force to 0

# IM is now always in measurement (pixel) space, so we define dm_mask via I2A consistently 
dm_mask = I2A @ pup_mask



########################################
## LO MODES 
########################################
if args.inverse_method_LO.lower() == 'pinv':
    #I2M = np.linalg.pinv( IM )
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    I2M_LO = np.linalg.pinv( IM_LO ) 

elif args.inverse_method_LO.lower() == 'map': # minimum variance of maximum posterior estimator 
    #phase_cov = np.eye( IM.shape[0] )
    #noise_cov = np.eye( IM.shape[1] ) 
    #I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    phase_cov_LO = np.eye( IM_LO.shape[0] )
    noise_cov_LO = np.eye( IM_LO.shape[1] ) 

    I2M_LO = phase_cov_LO @ IM_LO.T @ np.linalg.inv(IM_LO @ phase_cov_LO @ IM_LO.T + noise_cov_LO)



elif 'svd_truncation' in args.inverse_method_LO.lower() :
    k = int( args.inverse_method_LO.split('truncation_')[-1] ) 

    U,S,Vt = np.linalg.svd( IM_LO, full_matrices=True)

    I2M_LO = util.truncated_pseudoinverse(U, S, Vt, k)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
else:
    raise UserWarning('no inverse method provided for LO')

########################################
## HO MODES 
########################################
if args.inverse_method_HO.lower() == 'pinv':
    #I2M = np.linalg.pinv( IM )
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    I2M_HO = np.linalg.pinv( IM_HO )
    
    # for plotting later
    #dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

elif args.inverse_method_HO.lower() == 'map': # minimum variance of maximum posterior estimator 
    #phase_cov = np.eye( IM.shape[0] )
    #noise_cov = np.eye( IM.shape[1] ) 
    #I2M = (phase_cov @ IM @ np.linalg.inv(IM.T @ phase_cov @ IM + noise_cov) ).T #have to transpose to keep convention.. although should be other way round
    #I2M = phase_cov @ IM.T @ np.linalg.inv(IM @ phase_cov @ IM.T + noise_cov)
    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    phase_cov_HO = np.eye( IM_HO.shape[0] )
    noise_cov_HO = np.eye( IM_HO.shape[1] ) 

    I2M_HO = phase_cov_HO @ IM_HO.T @ np.linalg.inv(IM_HO @ phase_cov_HO @ IM_HO.T + noise_cov_HO)

    # for plotting later
    #dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

elif args.inverse_method_HO.lower() == 'zonal':
    # # just literally filter weight the pupil and take inverse of the IM signal on diagonals (dm actuator registered pixels)
    # if args.filter_edge_actuators: # do this in the mode space! 
    #     # only for simulation 
    #     #dm_mask = util.get_circle_DM_command( radius = 4 ) 
    #     # this is good for the real system based on comissioning


    #     # old 
    #     # ##############################################
    #     # # update with simulation mode to also filter secondary obstruction in pixel space
    #     # # Note the original commented out below worked fine on internal source, but this version seems more stable in simulator
    #     # tight_pup_wo_sec_tmp = ~(sec.astype(bool)  | (~inside_edge_filt.astype(bool) ) ) #| (~bad_pix_mask_tmp )
    #     # tight_sec_filter = (N0 < np.min(N0[tight_pup_wo_sec_tmp ])) & inside_edge_filt
    #     # # now get real tight filter 
    #     # tight_pup_wo_sec = (inside_edge_filt - tight_sec_filter).astype(bool)
        
        
    #     # #updated for simulation mode not tested in real system
    #     # dm_mask = I2A @ np.array( tight_pup_wo_sec ).reshape(-1) # I2A @ inside_edge_filt )
    #     # #######################

    #     #original not filtering secondary obstruction 
    #     #dm_mask =I2A @ np.array( inside_edge_filt ).reshape(-1) # I2A @ inside_edge_filt )
    # else:
    #     dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)

    # util.nice_heatmap_subplots(  im_list = [util.get_DM_command_in_2D(dm_mask)], savefig='delme.png' )
    I2M_HO = np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if np.isfinite(1/IM_HO[i][i]) else 0 for i in range(len(IM_HO))]) )
    #I2M_HO = np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if 1/IM_HO[i][i] < 1e3 else 0 for i in range(len(IM_HO))]) )


elif 'svd_truncation' in args.inverse_method_HO.lower() :
    k = int( args.inverse_method_HO.split('truncation_')[-1] ) 
    U,S,Vt = np.linalg.svd( IM_HO, full_matrices=True)

    I2M_HO = util.truncated_pseudoinverse(U, S, Vt, k)

    #I2M_LO , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in LO] )
    # for plotting later
    #dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)



elif "eigen" in args.inverse_method_HO.lower():
    # Parse k from something like "eigen_30"
    try:
        k_eff_req = int(args.inverse_method_HO.lower().split("eigen_")[-1])
    except Exception:
        raise ValueError("For eigen HO inversion use --inverse_method_HO eigen_<k>, e.g. eigen_30")

    eps = 1e-12  # or reuse your eps variable if you have one

    # IM_HO is (Nho, Nmeas) in your convention, so use H_HO = (Nmeas, Nho)
    H_HO = IM_HO.T   # (Nmeas, Nho)

    U, S, Vt = np.linalg.svd(H_HO, full_matrices=False)

    k_eff = min(k_eff_req, S.shape[0])

    U_k  = U[:, :k_eff]        # (Nmeas, k_eff)
    S_k  = S[:k_eff]           # (k_eff,)
    Vt_k = Vt[:k_eff, :]       # (k_eff, Nho)
    V_k  = Vt_k.T              # (Nho, k_eff)

    # Measurement -> eigen coefficients (controller runs on these)
    # This maps y (Nmeas,) -> a (k_eff,)
    I2M_HO = U_k.T              # (k_eff, Nmeas)

    # Eigen coeffs -> HO command basis (whatever basis IM_HO rows represent)
    # This maps a (k_eff,) -> x_HO (Nho,)
    M2C_HO = V_k @ np.diag(1.0 / (S_k + eps))   # (Nho, k_eff)



    # for plotting later if you want
    #dm_mask = I2A @ np.array(pupil_mask).reshape(-1)
    
else:
    raise UserWarning('no inverse method provided for HO')




# # NED TO CHECK THIS AGAIN - BUG, do this in the ZONAL scope of building I2M_HO
# if args.filter_edge_actuators:
#     # tight mask to restrict edge actuators 
#     dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array([int(a) for a in inside_edge_filt]) ) ).reshape(-1)
#     # typically 44 actuators 
# else:
#     # puypil mask
#     dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array( pupil_mask ).reshape(-1) ) ).reshape(-1)
#     # typically 71 actuators 

# filter out exterior actuators in command space (from pupol) - redudant if (args.filter_edge_actuators: # do this in the mode space!)

# updated in simulation mode testing (better handling of secondary pixels!)
# tight_pup_wo_sec_tmp = ~(sec.astype(bool)  | (~inside_edge_filt.astype(bool) ) ) #| (~bad_pix_mask_tmp )
# tight_sec_filter = (N0 < np.min(N0[tight_pup_wo_sec_tmp ])) & inside_edge_filt
# # now get real tight filter 
# tight_pup_wo_sec = (inside_edge_filt - tight_sec_filter).astype(bool)

# dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ tight_pup_wo_sec ) ) 

# original
#dm_mask_144 = np.nan_to_num( util.get_DM_command_in_2D( I2A @ np.array( pupil_mask ).reshape(-1) ) ).reshape(-1)

#util.nice_heatmap_subplots( [dm_mask_144.reshape(12,12),dm_tight_mask_144.reshape(12,12)], savefig='delme.png')



# new
if args.project_TT_out_HO and LO > 0:
    # IM_LO is (LO, Nmeas). We need a projector in measurement space (Nmeas x Nmeas).
    # The LO measurement subspace is spanned by the *rows* of IM_LO, i.e. col(IM_LO.T).
    Umeas, Smeas, _ = np.linalg.svd(IM_LO.T, full_matrices=False)  # (Nmeas, LO) SVD
    r_lo = int(np.sum(Smeas > 1e-12))
    Qlo = Umeas[:, :r_lo]                      # (Nmeas, r_lo)
    P_LO = Qlo @ Qlo.T                         # (Nmeas, Nmeas)

    # sanity checj
    assert P_LO.shape == (IM.shape[1],IM.shape[1])
    assert I2M_HO.shape[1] == IM.shape[1], f"I2M_HO must have Nmeas columns, got {I2M_HO.shape}"
    assert IM_LO.shape[1] == IM.shape[1]

    Btmp = IM_LO.T                      # (Nmeas, LO)
    leak_mat = I2M_HO @ Btmp            # (Nho, LO)

    rel_leak = np.linalg.norm(leak_mat) / (np.linalg.norm(I2M_HO)*np.linalg.norm(Btmp) + 1e-30)
    print("--\nsantiy check project TT out of HO:\nrelative HO response to LO subspace:", rel_leak,'\nif rel_leak ~ 1e-2 or larger => projection not applied in the right space')


    # Make HO reconstructor blind to LO measurement subspace
    # I2M_HO must be (Nho, Nmeas) for this to be valid.
    I2M_HO = I2M_HO @ (np.eye(P_LO.shape[0]) - P_LO)

    # keep M2C_HO, M2C_LO the same (this is measurement-side projection)
    M2C_LO = M2C[:,:LO]
    M2C_HO = M2C[:,LO:]


# # old, doing it in command space , need to update with doing it in intensity space 
# projection_basis = []

# if args.project_TT_out_HO:

#     for t in M2C.T[:LO]:  # TT modes
#         if 'zonal' in args.inverse_method_HO.lower(): # we actively filter actuators with mask
#             projection_basis.append(dm_mask_144 * np.nan_to_num(t, 0))
#         else:
#             projection_basis.append( np.nan_to_num(t, 0))
# if args.project_waffle_out_HO:
#     waffle_mode = util.waffle_mode_2D() #util.convert_12x12_to_140(util.waffle_mode_2D())
#     projection_basis.append(dm_mask_144 * waffle_mode)

# if projection_basis:
#     print("Projecting TT and/or Waffle modes out of HO")
#     proj_mat = np.vstack(projection_basis).reshape(-1, 144)
#     _ , M2C_HO = util.project_matrix(M2C[:,LO:], proj_mat)
#     #M2C_LO , _ = util.project_matrix( M2C[:,:LO], proj_mat)
#     #_ , M2C_HO = util.project_matrix(M2C[:,LO:], proj_mat)
#     #M2C_LO = M2C[:,:LO]

#     M2C_LO = M2C[:, :LO]
#     # M2C_LO_tmp = M2C[:, :LO]  # before projection
#     # overlap = np.dot(proj_mat, M2C_LO_tmp)  # shape (N_proj, LO)
#     # max_overlap = np.max(np.abs(overlap))
#     # if max_overlap > 1e-6:
#     #     print(f"Max overlap between LO and projected modes = {max_overlap:.2e}, re-orthogonalizing LO")
#     #     M2C_LO, _ = util.project_matrix(M2C[:, :LO], proj_mat)
#     # else:
#     #     print(f"LO commands already orthogonal to projection modes (max overlap = {max_overlap:.2e})")
#     #     M2C_LO = M2C[:, :LO]
# else:
#     M2C_LO = M2C[:,:LO]
#     M2C_HO = M2C[:,LO:]

# # project out in command / mode space 
# if args.project_TT_out_HO:
#     print("projecting TT out of HO")
#     #we only need HO and require len 144x 140 (SHM input x number of actuatorss) which projects out the TT 
#     _ , M2C_HO = util.project_matrix( np.nan_to_num( M2C[:,LO:], 0),  (dm_mask_144 * np.nan_to_num(M2C.T[:LO],0) ).reshape(-1,144) )
#     #_ , M2C_HO = util.project_matrix( np.nan_to_num( M2C[:,LO:], 0),  np.nan_to_num(M2C[:,:LO],0).reshape(-1,144) )
#     M2C_LO , _ = util.project_matrix( np.nan_to_num( M2C[:,:LO], 0),  np.nan_to_num(M2C.T[LO:],0).reshape(-1,144) )
# else:
#     M2C_LO = M2C[:,:LO]
#     M2C_HO = M2C[:,LO:]



# bias = np.zeros([32,32]).reshape(-1).astype(int).tolist(),
# dark = np.zeros([32,32]).reshape(-1).astype(int).tolist(),
# bad_pixel_mask = np.ones([32,32]).astype(int).reshape(-1).tolist(),
# bad_pixels = [] # np.where( np.array( c.reduction_dict['bad_pixel_mask'][-1])[r1:r2,c1:c2].reshape(-1)   )[0].tolist(),

# ====================
dict2write = {f"beam{args.beam_id}":{f"{args.phasemask}":{"ctrl_model": {
                                               "inverse_method_LO": args.inverse_method_LO,
                                               "inverse_method_HO": args.inverse_method_HO,
                                               "controller_type":"leaky",
                                               "signal_space":args.signal_space.lower(),
                                               "sza": np.array(M2C).shape[0],
                                               "szm": np.array(M2C).shape[1],
                                               "szp": np.array(I2M_HO).shape[1],
                                               "I2A": np.array(I2A).tolist(), 
                                               #"I2M": np.array(I2M).tolist(),
                                               "I2M_LO": np.array(I2M_LO.T).tolist(),
                                               "I2M_HO": np.array(I2M_HO.T).tolist(),
                                               "M2C_LO" : np.array(M2C_LO).tolist(),
                                               "M2C_HO" : np.array(M2C_HO).tolist(),
                                               "I2rms_sec" : np.array(I2rms_sec).tolist(),
                                               "I2rms_ext" : np.array(I2rms_ext).tolist(),
                                               "telemetry" : 0,  # do we record telem  - need to add to C++ readin
                                               "auto_close" : 0, # automatically close - need to add to C++ readin
                                               "auto_open" : 1, # automatically open - need to add to C++ readin
                                               "auto_tune" : 0, # automatically tune gains  - need to add to C++ readin
                                               "close_on_strehl_limit": 10,
                                               "open_on_strehl_limit": 0,
                                               "open_on_flux_limit": 0,
                                               "open_on_dm_limit"  : 0.3,
                                               "LO_offload_limit"  : 1,
                                                #"dark" : np.zeros([32,32]).reshape(-1).astype(int).tolist(), 
                                                # include temp here 
                                                #### in build_IM.py
                                                # "bias" : np.zeros([32,32]).reshape(-1).astype(int).tolist(),
                                                # "dark" : np.zeros([32,32]).reshape(-1).astype(int).tolist(),
                                                # "bad_pixel_mask" : np.ones([32,32]).reshape(-1).astype(int).tolist(),
                                                # "bad_pixels" : [], 

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




## A QUICK LOOK 
if 0:
    for beam_id in [args.beam_id]:

        ################################
        # the reference intensities
        im_list = [ I0.reshape(32,32), np.array(N0).reshape(32,32), np.array( norm_pupil).reshape(32,32), util.get_DM_command_in_2D(dm_mask) ]
        title_list = ['<I0>','<N0>','normalized pupil','mask']
        cbar_list = ["UNITLESS"] * len(im_list)
        util.nice_heatmap_subplots( im_list , title_list=title_list, cbar_label_list=cbar_list) 
        plt.savefig(f'{args.fig_path}' + f'reference_intensities_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
        plt.show()

        ################################
        # the interaction signal 
        modes2look = [0,1,65,67]
        im_list = [util.get_DM_command_in_2D(IM[m])for m in modes2look]

        title_list = [f'mode {m}' for m in modes2look]
        cbar_list = ["UNITLESS"] * len(im_list)
        util.nice_heatmap_subplots( im_list , cbar_label_list=cbar_list, savefig=f'{args.fig_path}' + f'IM_first16modes_beam{beam_id}.png') 
        plt.savefig(f'{args.fig_path}' + f'IM_some_modes_beam{beam_id}.jpeg', bbox_inches='tight', dpi=200)
        plt.show()

        ################################
        # the eigenmodes 
        U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)  # shapes: (M, M), (min(M,N),), (min(M,N), N)

        # (a) Plot singular values
        plt.figure(figsize=(6, 4))
        plt.semilogy(S, 'o-')
        plt.title("Singular Values of IM_HO")
        plt.xlabel("Index")
        plt.ylabel("Singular value (log scale)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_singular_values_beam{beam_id}.png', bbox_inches='tight', dpi=200)

        # (b) Intensity eigenmodes (Vt)
        plt.figure(figsize=(15, 3))
        for i in range(min(5, Vt.shape[0])):
            ax = plt.subplot(1, 5, i+1)
            im = ax.imshow(util.get_DM_command_in_2D(Vt[i]), cmap='viridis')
            ax.set_title(f"Vt[{i}]")
            plt.colorbar(im, ax=ax)
        plt.suptitle("First 5 intensity eigenmodes (Vt) mapped to 2D")
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_first5_intensity_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)


        # (c) System eigenmodes (U)
        plt.figure(figsize=(15, 3))
        for i in range(min(5, U.shape[1])):
            ax = plt.subplot(1, 5, i+1)
            im = ax.imshow(util.get_DM_command_in_2D(U[:, i]), cmap='plasma')
            ax.set_title(f"U[:, {i}]")
            plt.colorbar(im, ax=ax)
        plt.suptitle("First 5 system eigenmodes (U) mapped to 2D")
        plt.tight_layout()
        plt.savefig(f"{args.fig_path}" + f'IM_first5_system_eigenmodes_beam{beam_id}.png', bbox_inches='tight', dpi=200)
        plt.show()


        plt.close("all")

### test 
test_reco = input("press enter to continue recon tests. 0 to finish ...")

if test_reco != '0':

    import numpy as np
    import os, time, toml, matplotlib.pyplot as plt
    from asgard_alignment.DM_shm_ctrl import dmclass
    #from asgard_alignment import FLI_Cameras as FLI
    from xaosim.shmlib import shm 
    # ---------- configurable test knobs ----------
    TEST_BEAM   = int(args.beam_id)             # use the beam we just wrote
    N_TRIALS    = 40                            # number of random TT trials
    AMP_STD     = 0.05                          # DM units (per-mode stdev)
    CAM_SHM     = f"/dev/shm/baldr{args.beam_id}.im.shm"       # subframe camera SHM
    FIG_DIR     = os.path.expanduser(args.fig_path or "~/Downloads/")
    # --------------------------------------------

    # Load what we just wrote, so the test also works if you re-run later
    with open(args.toml_file.replace('#', f'{TEST_BEAM}'), "r") as f:
        cfg = toml.load(f)

    top      = cfg[f"beam{TEST_BEAM}"]
    ctrl     = top[args.phasemask]["ctrl_model"]
    I2A      = np.array(top["I2A"], dtype=float)                        # (140 x 1024)
    I2M_LO   = np.array(ctrl["I2M_LO"], dtype=float)                    # (LO x P)  (stored transposed)
    M2C_LO   = np.array(ctrl["M2C_LO"], dtype=float)                    # (144 x LO)
    LO_count = int(ctrl.get("LO", 2))                                   # how many LO modes were built
    sigspace = str(ctrl.get("signal_space", "dm")).lower()              # 'dm' or 'piix'
    inner_pupil_filt = np.array(ctrl["inner_pupil_filt"]).astype(bool)
    # Pixel-space references saved by build_IM.py (already normalized)
    I0_flat      = np.array(ctrl["I0"], dtype=float)                     # (1024,)
    N0_flat      = np.array(ctrl["N0"], dtype=float) #np.array(ctrl["norm_pupil"], dtype=float)             # (1024,)
    dark      = np.array(ctrl["dark"], dtype=float)  

    #r1, r2, c1, c2 = map(int, ctrl["crop_pixels"])                       # global crop -> local 32x32

    # Sanity guardrails
    assert LO_count >= 2, "LO must include at least tip & tilt (LO>=2)."
    assert I2M_LO.shape[0] >= 2, "I2M_LO must have at least 2 rows for tip/tilt."

    # Connect camera (global SHM) and determine buffer length (# reads per burst)
    cam = shm(f"/dev/shm/baldr{args.beam_id}.im.shm") #FLI.fli(CAM_SHM, roi=[None, None, None, None])

    # ## NOW WE GET DARKS TO BE CONSISTENT WITH BUILD_IM
    # print("turning off calibration source to get darks ...")
    # cam.build_manual_dark(no_frames = 200 , build_bad_pixel_mask=True, kwargs={'std_threshold':20, 'mean_threshold':6} )
    # print("darks acquired. turn calibration source back on and press enter to continue ...")

    #time.sleep(0.5)
    
    #nrs = cam.mySHM.get_data().shape[0]   # number of reads per buffer (burst)

    # DM control on a side channel (like build_IM)
    dm  = dmclass(beam_id=TEST_BEAM, main_chn=3)

    # Helper: wait for a fresh buffer, then return normalized 32x32 (crop) as 1D (1024,)
    # def grab_norm_frame_flat():
    #     t0 = cam.mySHM.get_counter()
    #     while (cam.mySHM.get_counter() - t0) < 2 * nrs:
    #         time.sleep(1.0 / float(cam.config["fps"]))
    #     frames = cam.get_data(apply_manual_reduction=True)              # (nrs, H, W)
    #     sub    = frames[:, r1:r2, c1:c2].mean(axis=0)                   # 2D mean
    #     sub   /= sub.sum()                                              # post-TTonsky normalization
    #     return sub.reshape(-1)                                          # (1024,)




    # ------------------------------------------------------------
    # ramp Tip and Tilt separately, measure e vs amp
    # Default: amps = linspace(-0.2, +0.2, 20)
    # Produces:
    #   - tip_ramp_response_beamX.png
    #   - tilt_ramp_response_beamX.png
    # ------------------------------------------------------------

    amps = np.linspace(-0.2, 0.2, 20)

    # storage
    tip_true   = []
    tip_rec    = []
    tilt_true  = []
    tilt_rec   = []

    # safety
    zero144 = np.zeros(144, dtype=float)

    # TIP ramp (mode index 0)
    try:
        for a in amps:
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[0] = a

            u_cmd = M2C_LO @ a_lo                      # (144,)
            dm.set_data(u_cmd)
            time.sleep(0.1)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp, axis=0)

            s_pix = (
                I_norm_flat / np.mean(N0_flat[inner_pupil_filt])
                - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            )

            if sigspace == "dm":
                s = I2A @ s_pix                         # (140,)
            else:
                s = s_pix                               # (1024,)

            a_hat_lo = I2M_LO @ s                       # (LO,)
            tip_true.append(a)
            tip_rec.append(a_hat_lo[0])

        dm.set_data(zero144)

    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    tip_true = np.array(tip_true, dtype=float)
    tip_rec  = np.array(tip_rec, dtype=float)
    tip_err  = tip_rec - tip_true

    # Plot TIP response
    plt.figure(figsize=(6, 4))
    plt.plot(tip_true, tip_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(tip_true, tip_true, "--", label="ideal")
    p = np.polyfit(tip_true, tip_rec, 1)
    plt.title(
        f"Tip ramp response (beam{TEST_BEAM}, {args.phasemask})\n"
        f"slope={p[0]:.3g}  offset={p[1]:.3g}  RMSE={np.sqrt(np.mean(tip_err**2)):.3g}"
    )
    plt.xlabel("commanded tip amplitude [DM units]")
    plt.ylabel("reconstructed tip [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_tip = os.path.join(args.fig_path, f"tip_ramp_response_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_tip, dpi=180)
    plt.show()
    plt.close()


    # TILT ramp (mode index 1)
    try:
        for a in amps:
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[1] = a

            u_cmd = M2C_LO @ a_lo                      # (144,)
            dm.set_data(u_cmd)
            time.sleep(0.1)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp, axis=0)

            s_pix = (
                I_norm_flat / np.mean(N0_flat[inner_pupil_filt])
                - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            )

            if sigspace == "dm":
                s = I2A @ s_pix                         # (140,)
            else:
                s = s_pix                               # (1024,)

            a_hat_lo = I2M_LO @ s                       # (LO,)
            tilt_true.append(a)
            tilt_rec.append(a_hat_lo[1])

        dm.set_data(zero144)

    finally:
        try:
            dm.set_data(zero144)
        except Exception:
            pass

    tilt_true = np.array(tilt_true, dtype=float)
    tilt_rec  = np.array(tilt_rec, dtype=float)
    tilt_err  = tilt_rec - tilt_true

    # Plot TILT response
    plt.figure(figsize=(6, 4))
    plt.plot(tilt_true, tilt_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(tilt_true, tilt_true, "--", label="ideal")
    p = np.polyfit(tilt_true, tilt_rec, 1)
    plt.title(
        f"Tilt ramp response (beam{TEST_BEAM}, {args.phasemask})\n"
        f"slope={p[0]:.3g}  offset={p[1]:.3g}  RMSE={np.sqrt(np.mean(tilt_err**2)):.3g}"
    )
    plt.xlabel("commanded tilt amplitude [DM units]")
    plt.ylabel("reconstructed tilt [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_tilt = os.path.join(args.fig_path, f"tilt_ramp_response_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_tilt, dpi=180)
    plt.show()
    plt.close()

    print(f"[OK] Saved TT ramp plots:\n  {out_png_tip}\n  {out_png_tilt}")





    # ------------------------------------------------------------
    # random Tip and Tilt, measure e vs amp
    # ------------------------------------------------------------

    # Run trials
    rng = np.random.default_rng(0)
    true_tt  = []   # shape (N, 2)
    rec_tt   = []   # shape (N, 2)

    # zero command for cleanup
    zero144 = np.zeros(144, dtype=float)

    try:
        for k in range(N_TRIALS):
            # draw random tip/tilt (the first two LO coefficients)
            a_tt = rng.normal(0.0, AMP_STD, size=2)                     # [tip, tilt]
            a_lo = np.zeros(LO_count, dtype=float)
            a_lo[:2] = a_tt

            # command DM in SHM space: u = M2C_LO @ a_lo   (144,)
            u_cmd = M2C_LO @ a_lo
            dm.set_data(u_cmd)
            time.sleep(0.1)
            # acquire normalized pupil and form Baldr signal s = (I - I0)/N0 (pixel-space)
            #I_norm_flat = grab_norm_frame_flat()                        # (1024,)
            img_tmp = []
            for _ in range(10):
                img_tmp.append( cam.get_data().reshape(-1) - dark )
                time.sleep(0.01)
            I_norm_flat = np.mean(img_tmp,axis=0)
            s_pix       = I_norm_flat / np.mean( N0_flat[inner_pupil_filt])   - I0_flat / np.mean( N0_flat[inner_pupil_filt])              # (1024,)

            # map to chosen signal space
            if sigspace == "dm":
                s = I2A @ s_pix                                        # (140,)
            else:
                s = s_pix                                               # (1024,)

            # reconstruct LO coefficients: a_hat = I2M_LO @ s
            a_hat_lo = I2M_LO @ s                                       # (LO,)
            a_hat_tt = a_hat_lo[:2]

            true_tt.append(a_tt)
            rec_tt.append(a_hat_tt)

        # Reset DM shape on exit from loop
        dm.set_data(zero144)

    finally:
        # Ensure DM is cleared even if an exception happens
        try: dm.set_data(zero144)
        except Exception: pass

    true_tt = np.array(true_tt)   # (N,2)
    rec_tt  = np.array(rec_tt)    # (N,2)
    err     = rec_tt - true_tt

    # Per-mode & overall RMSE
    rmse_tip  = np.sqrt(np.mean(err[:,0]**2))
    rmse_tilt = np.sqrt(np.mean(err[:,1]**2))
    rmse_all  = np.sqrt(np.mean(err**2))

    # Simple figure: scatter true vs reconstructed for tip & tilt
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(6,3))
    for i, name in enumerate(["Tip", "Tilt"]):
        plt.subplot(1,2,i+1)
        plt.scatter(true_tt[:,i], rec_tt[:,i], s=18)
        m = max(np.max(np.abs(true_tt[:,i])), np.max(np.abs(rec_tt[:,i]))) * 1.1 + 1e-6
        plt.plot([-m, m], [-m, m], '--', lw=1)
        plt.xlabel(f"True {name} [DM units]")
        plt.ylabel(f"Reconstructed {name} [DM units]")
        plt.title(f"{name}  RMSE={np.sqrt(np.mean(err[:,i]**2)):.3g}")
        plt.axis('equal'); plt.grid(True, alpha=0.3)
    out_png = os.path.join(args.fig_path, f"recon_LO_TT_sanity_beam{TEST_BEAM}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); 
    plt.show()
    plt.close('all')
    # Print summary
    print("\n=== LO (Tip/Tilt) reconstructor sanity test ===")
    print(f"Beam: {TEST_BEAM} | Signal space: {sigspace} | Trials: {N_TRIALS}")
    print(f"Std of commanded TT: {AMP_STD} (per mode)")
    print(f"RMSE Tip : {rmse_tip:.4g}")
    print(f"RMSE Tilt: {rmse_tilt:.4g}")
    print(f"RMSE All : {rmse_all:.4g}")
    print(f"Saved plot: {out_png}")




# ============================================================
# HO single-mode validation (ramp + random), like TT
# ============================================================

def _ensure_I2M_rows_are_modes(I2M_loaded: np.ndarray, n_meas: int) -> np.ndarray:
    """
    Expect I2M to end up (Nmodes, Nmeas).
    Your TOML stores I2M_* as transpose of the raw inverse (so it SHOULD already be (Nmodes, Nmeas)).
    But if it comes back (Nmeas, Nmodes), this flips it.
    """
    I2M_loaded = np.asarray(I2M_loaded, dtype=float)
    if I2M_loaded.ndim != 2:
        raise ValueError(f"I2M must be 2D, got {I2M_loaded.shape}")

    if I2M_loaded.shape[1] == n_meas:
        return I2M_loaded  # already (Nmodes, Nmeas)
    if I2M_loaded.shape[0] == n_meas:
        return I2M_loaded.T  # was (Nmeas, Nmodes)
    raise ValueError(f"I2M shape {I2M_loaded.shape} not compatible with n_meas={n_meas}")

def _ensure_M2C_is_act_by_modes(M2C_loaded: np.ndarray) -> np.ndarray:
    """
    Expect M2C to end up (Nact, Nmodes).
    If it's (Nmodes, Nact), transpose it.
    """
    M2C_loaded = np.asarray(M2C_loaded, dtype=float)
    if M2C_loaded.ndim != 2:
        raise ValueError(f"M2C must be 2D, got {M2C_loaded.shape}")

    # your DM interface uses 144-length vectors, so axis with 144 should be Nact
    if M2C_loaded.shape[0] == 144:
        return M2C_loaded
    if M2C_loaded.shape[1] == 144:
        return M2C_loaded.T
    # fall back: keep as-is
    return M2C_loaded

# ---- Load HO pieces from the written TOML (same style as TT) ----
I2M_HO = np.array(ctrl["I2M_HO"], dtype=float)
M2C_HO = np.array(ctrl["M2C_HO"], dtype=float)

# Determine measurement length in chosen signal space (same as TT section)
# s is either I2A @ s_pix -> length 140 OR s_pix -> length 1024
n_meas = 140 if sigspace == "dm" else int(I0_flat.size)

I2M_HO = _ensure_I2M_rows_are_modes(I2M_HO, n_meas=n_meas)
M2C_HO = _ensure_M2C_is_act_by_modes(M2C_HO)

N_HO_CTRL = int(I2M_HO.shape[0])     # number of HO control modes (could be Nho or k_eff if eigen)
N_HO_CMD  = int(M2C_HO.shape[1])     # number of HO command basis modes

print(f"\nHO sanity: I2M_HO shape = {I2M_HO.shape} (Nmodes_ctrl x Nmeas)")
print(f"HO sanity: M2C_HO shape = {M2C_HO.shape} (Nact x Nmodes_cmd)")

# ---- Ask which HO control-mode index to test ----
ho_in = input(f"\nEnter HO mode index to test [0..{N_HO_CTRL-1}] (or '0' to skip): ").strip()
if ho_in and ho_in != "0":
    ho_idx = int(ho_in)
    if ho_idx < 0 or ho_idx >= N_HO_CTRL:
        raise ValueError(f"HO index {ho_idx} out of range [0..{N_HO_CTRL-1}]")

    # if your HO control modes correspond 1:1 with HO command basis, ho_idx can be used for both
    # if not (e.g. eigen control modes k_eff), command basis index is ambiguous, so we:
    #   - command ONE HO basis mode (ask separately) if sizes differ
    if N_HO_CMD != N_HO_CTRL:
        cmd_in = input(f"HO control Nmodes={N_HO_CTRL} but HO command Nmodes={N_HO_CMD}.\n"
                       f"Enter HO *command-basis* index to excite [0..{N_HO_CMD-1}] (default={ho_idx}): ").strip()
        if cmd_in == "":
            ho_cmd_idx = ho_idx
        else:
            ho_cmd_idx = int(cmd_in)
        if ho_cmd_idx < 0 or ho_cmd_idx >= N_HO_CMD:
            raise ValueError(f"HO command index {ho_cmd_idx} out of range [0..{N_HO_CMD-1}]")
    else:
        ho_cmd_idx = ho_idx

    # ------------ (A) RAMP test on a single HO mode ------------
    amps_ho = np.linspace(-0.2, 0.2, 21)

    ho_true = []
    ho_rec  = []
    lo_leak = []   # track LO leakage magnitude when exciting HO only

    zero144 = np.zeros(144, dtype=float)

    try:
        for a in amps_ho:
            # command a single HO *basis* mode
            a_ho = np.zeros(N_HO_CMD, dtype=float)
            a_ho[ho_cmd_idx] = a
            u_cmd = M2C_HO @ a_ho   # (144,)
            dm.set_data(u_cmd)
            time.sleep(0.12)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_flat = np.mean(img_tmp, axis=0)

            s_pix = (
                I_flat / np.mean(N0_flat[inner_pupil_filt])
                - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            )

            s = (I2A @ s_pix) if (sigspace == "dm") else s_pix

            # HO reconstruct in your chosen HO control basis
            a_hat_ho = I2M_HO @ s   # (N_HO_CTRL,)

            # also check LO leakage if LO exists
            if LO_count > 0:
                a_hat_lo = I2M_LO @ s
                lo_leak.append(np.linalg.norm(a_hat_lo[:min(2, LO_count)]))
            else:
                lo_leak.append(0.0)

            ho_true.append(a)
            ho_rec.append(a_hat_ho[ho_idx])

        dm.set_data(zero144)
    finally:
        try: dm.set_data(zero144)
        except Exception: pass

    ho_true = np.array(ho_true, dtype=float)
    ho_rec  = np.array(ho_rec, dtype=float)
    ho_err  = ho_rec - ho_true
    lo_leak = np.array(lo_leak, dtype=float)

    # plot ramp
    plt.figure(figsize=(6, 4))
    plt.plot(ho_true, ho_rec, marker="o", linestyle="-", label="reconstructed")
    plt.plot(ho_true, ho_true, "--", label="ideal")
    p = np.polyfit(ho_true, ho_rec, 1)
    plt.title(
        f"HO single-mode ramp (beam{TEST_BEAM}, {args.phasemask})\n"
        f"ctrl_idx={ho_idx}, cmd_idx={ho_cmd_idx} | slope={p[0]:.3g} offset={p[1]:.3g} "
        f"RMSE={np.sqrt(np.mean(ho_err**2)):.3g}\n"
        f"median LO leakage={np.median(lo_leak):.3g}"
    )
    plt.xlabel("commanded HO amplitude [DM units]")
    plt.ylabel("reconstructed HO coefficient [DM units]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png_ho_ramp = os.path.join(args.fig_path, f"recon_HO_mode{ho_idx}_ramp_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_ho_ramp, dpi=180)
    plt.show()
    plt.close()

    print(f"[OK] Saved HO ramp plot: {out_png_ho_ramp}")

    # ------------ (B) Random single-mode HO test ------------
    N_TRIALS_HO = 40
    AMP_STD_HO  = 0.05

    rng = np.random.default_rng(1)
    true_ho = []
    rec_ho  = []
    leak_lo = []

    try:
        for k in range(N_TRIALS_HO):
            a = float(rng.normal(0.0, AMP_STD_HO))
            a_ho = np.zeros(N_HO_CMD, dtype=float)
            a_ho[ho_cmd_idx] = a

            u_cmd = M2C_HO @ a_ho
            dm.set_data(u_cmd)
            time.sleep(0.12)

            img_tmp = []
            for _ in range(10):
                img_tmp.append(cam.get_data().reshape(-1) - dark)
                time.sleep(0.01)
            I_flat = np.mean(img_tmp, axis=0)

            s_pix = (
                I_flat / np.mean(N0_flat[inner_pupil_filt])
                - I0_flat / np.mean(N0_flat[inner_pupil_filt])
            )
            s = (I2A @ s_pix) if (sigspace == "dm") else s_pix

            a_hat_ho = I2M_HO @ s
            true_ho.append(a)
            rec_ho.append(a_hat_ho[ho_idx])

            if LO_count > 0:
                a_hat_lo = I2M_LO @ s
                leak_lo.append(np.linalg.norm(a_hat_lo[:min(2, LO_count)]))
            else:
                leak_lo.append(0.0)

        dm.set_data(zero144)
    finally:
        try: dm.set_data(zero144)
        except Exception: pass

    true_ho = np.array(true_ho)
    rec_ho  = np.array(rec_ho)
    err_ho  = rec_ho - true_ho
    rmse_ho = np.sqrt(np.mean(err_ho**2))
    leak_lo = np.array(leak_lo)

    plt.figure(figsize=(5, 4))
    plt.scatter(true_ho, rec_ho, s=18)
    m = max(np.max(np.abs(true_ho)), np.max(np.abs(rec_ho))) * 1.1 + 1e-6
    plt.plot([-m, m], [-m, m], "--", lw=1)
    plt.xlabel("True HO amplitude [DM units]")
    plt.ylabel("Reconstructed HO coefficient [DM units]")
    plt.title(
        f"HO single-mode random (ctrl_idx={ho_idx}, cmd_idx={ho_cmd_idx})\n"
        f"RMSE={rmse_ho:.3g} | median LO leakage={np.median(leak_lo):.3g}"
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    out_png_ho_rand = os.path.join(args.fig_path, f"recon_HO_mode{ho_idx}_random_beam{TEST_BEAM}.png")
    plt.tight_layout()
    plt.savefig(out_png_ho_rand, dpi=180)
    plt.show()
    plt.close()

    print(f"\n=== HO single-mode reconstructor sanity test ===")
    print(f"Beam: {TEST_BEAM} | Signal space: {sigspace}")
    print(f"Control mode idx tested: {ho_idx} | Command-basis idx excited: {ho_cmd_idx}")
    print(f"Trials: {N_TRIALS_HO} | Std commanded: {AMP_STD_HO}")
    print(f"RMSE HO: {rmse_ho:.4g}")
    print(f"Saved plot: {out_png_ho_rand}")






# # #### SOME TESTS FOR THE CURIOUS

# just have a peak at the IM intensities registered 
#plt.figure() ; plt.imshow( util.get_DM_command_in_2D(IM[ 77 ] ) );plt.colorbar(); plt.savefig('delme.png')


# #Perform SVD
# U, S, Vt = np.linalg.svd(IM_HO, full_matrices=False)  # shapes: (M, M), (min(M,N),), (min(M,N), N)

# # (a) Plot singular values
# plt.figure(figsize=(6, 4))
# plt.semilogy(S, 'o-')
# plt.title("Singular Values of IM_HO")
# plt.xlabel("Index")
# plt.ylabel("Singular value (log scale)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('delme.png')

# # (b) Intensity eigenmodes (Vt)
# plt.figure(figsize=(15, 3))
# for i in range(min(5, Vt.shape[0])):
#     ax = plt.subplot(1, 5, i+1)
#     im = ax.imshow(util.get_DM_command_in_2D(Vt[i]), cmap='viridis')
#     ax.set_title(f"Vt[{i}]")
#     plt.colorbar(im, ax=ax)
# plt.suptitle("First 5 intensity eigenmodes (Vt) mapped to 2D")
# plt.tight_layout()
# plt.savefig('delme.png')


# # (c) System eigenmodes (U)
# plt.figure(figsize=(15, 3))
# for i in range(min(5, U.shape[1])):
#     ax = plt.subplot(1, 5, i+1)
#     im = ax.imshow(util.get_DM_command_in_2D(U[:, i]), cmap='plasma')
#     ax.set_title(f"U[:, {i}]")
#     plt.colorbar(im, ax=ax)
# plt.suptitle("First 5 system eigenmodes (U) mapped to 2D")
# plt.tight_layout()
# plt.savefig('delme.png')

# plt.close()

# # look at reconstructors HO 
# I2M_1 = np.linalg.pinv( IM_HO )

# phase_cov = np.eye( IM_HO.shape[0] )
# noise_cov = 10 * np.eye( IM_HO.shape[1] )
# I2M_2 = (phase_cov @ IM_HO @ np.linalg.inv(IM_HO.T @ phase_cov @ IM_HO + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# #dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
# dm_mask = util.get_circle_DM_command( radius = 4 ) 
# I2M_3 = np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if  np.isfinite(1/IM_HO[i][i]) else 0 for i in range(len(IM_HO))]) )
# #np.diag(  np.array( [dm_mask[i]/IM_HO[i][i] if np.isfinite(1/IM_HO[i][i]) else 0 for i in range(len(IM_HO))]) )

# U,S,Vt = np.linalg.svd( IM_HO, full_matrices=True)

# k= 20 # int( 5**2 * np.pi)
# I2M_4 = util.truncated_pseudoinverse(U, S, Vt, k=50)

# act = 65
# im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act], I2M_4@IM[act] ] ]
# titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 



# ## LO 
# # look at reconstructors HO 
# I2M_1 = np.linalg.pinv( IM_LO )

# phase_cov = np.eye( IM_LO.shape[0] )
# noise_cov = 10 * np.eye( IM_LO.shape[1] )
# I2M_2 = (phase_cov @ IM_LO @ np.linalg.inv(IM_LO.T @ phase_cov @ IM_LO + noise_cov) ).T #have to transpose to keep convention.. although should be other way round

# dm_mask = I2A @ np.array( pupil_mask ).reshape(-1)
# I2M_3 = np.diag(  np.array( [dm_mask[i]/IM_LO[i][i] if np.isfinite(1/IM_LO[i][i]) else 0 for i in range(len(IM_LO))]) )

# # U,S,Vt = np.linalg.svd( IM_LO, full_matrices=True)

# # k= 20 # int( 5**2 * np.pi)
# # I2M_4 = util.truncated_pseudoinverse(U, S, Vt, k=50)

# act = 1
# im_list = [util.get_DM_command_in_2D( a) for a in [IM[act], I2M_1@IM[act], I2M_2@IM[act], I2M_3@IM[act] ] ]
# titles = ["real resp.", "pinv", "MAP", "zonal", f"svd trunc. (k={k})"]

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 




# # ## TT projection HO / TT 

# TT = dmbases.zer_bank(2, 3)
# util.nice_heatmap_subplots( im_list= [TT[0],TT[1]], savefig='delme.png' ) 

# sig = dm_mask * ( IM[act] - 0.3*util.convert_12x12_to_140(TT[0]) - 0.1*util.convert_12x12_to_140(TT[1]))

# im_list =  [util.get_DM_command_in_2D( sig )]
# im_TT_list = [util.get_DM_command_in_2D( sig )]
# im_HO_list = [util.get_DM_command_in_2D( sig )]

# for I2M in [I2M_1,I2M_2,I2M_3,I2M_4]:

#     I2M_TT , I2M_HO = util.project_matrix( I2M , [util.convert_12x12_to_140(t) for t in TT] )

#     im_list.append( util.get_DM_command_in_2D(  I2M  @ sig ) )
#     im_TT_list.append( util.get_DM_command_in_2D(  I2M_TT @ sig ) )
#     im_HO_list.append( util.get_DM_command_in_2D(  I2M_HO @ sig ) )

# util.nice_heatmap_subplots(  im_list , title_list=titles, savefig='delme.png' ) 

# # util.nice_heatmap_subplots(  im_TT_list , title_list=["TT reco "+t for t in titles], savefig='delme.png' ) 

# # util.nice_heatmap_subplots(  im_HO_list , title_list=["HO reco "+t for t in titles], savefig='delme.png' ) 



# #### ADDITIONAL PROJECTION TESTS

# ### TEST 
# c0 = 0*M2C.T[0]
# i = 0*IM[0]
# act_list = [0, 65, 43]
# for a in act_list:
#     c0 += poke_amp/2 * M2C.T[a] # original command

#     i +=  IM[a] #+ IM[65] # simulating intensity repsonse

# e_LO = 2 * float(camera_config['gain']) / float(camera_config['fps']) * I2M_LO.T @ i
# e_HO = 2 * float(camera_config['gain']) / float(camera_config['fps']) * I2M_HO.T @ i

# # without projection just using HO (which has full rank)
# c_HO = (M2C[:,LO:] @ e_HO).reshape(12,12)
# res = c_HO - c0.reshape(12,12,)
# im_list = [  c0.reshape(12,12), c_HO, dm_mask_144.reshape(12,12) * res]
# vlims = [[np.min(c0), np.max(c0)] for _ in im_list]
# title_list = [ "disturb",  "c_HO'", "res."]
# cbar_title_list = ["DM UNITS","DM UNITS", "DM UNITS"]
# util.nice_heatmap_subplots( im_list = im_list ,title_list=title_list, vlims = vlims, cbar_label_list=  cbar_title_list, savefig='delme.png')

# # proper projection 
# c_LOg = (M2C_LO @ e_LO).reshape(12,12)
# c_HOg = (M2C_HO @ e_HO).reshape(12,12)

# dcmdg = c_LOg + c_HOg

# resg = dcmdg - c0.reshape(12,12)

# im_list = [  c0.reshape(12,12), c_LOg, c_HOg, dcmdg, dm_mask_144.reshape(12,12) * resg]
# vlims = [[np.min(c0), np.max(c0)] for _ in im_list]
# title_list = [ "disturb", "c_LO", "c_HO'","c_LO + c_HO","res."]
# cbar_title_list = ["DM UNITS","DM UNITS", "DM UNITS","DM UNITS","DM UNITS"]
# util.nice_heatmap_subplots( im_list = im_list ,title_list=title_list, vlims=vlims, cbar_label_list=  cbar_title_list, savefig='delme.png')

# print( np.std( dm_mask_144.reshape(12,12) * res ), np.std( dm_mask_144.reshape(12,12) * resg ))



# # In [8]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['M2C_HO']).shape
# # Out[8]: (144, 142)

# # In [9]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['M2C_LO']).shape
# # Out[9]: (144, 142)

# # In [10]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['I2M_LO']).shape
# # Out[10]: (2, 140)

# # In [11]: np.array(config_dict ["beam2"]["H3"]['ctrl_model']['I2M_HO']).shape
# # Out[11]: (140, 140)
