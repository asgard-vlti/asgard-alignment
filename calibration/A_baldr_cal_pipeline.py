

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,  median_filter
from scipy.optimize import leastsq
import toml  # Make sure to install via `pip install toml` if needed
import argparse
import os
import json
import time
import zmq 
from astropy.io import fits
import matplotlib.gridspec as gridspec

from xaosim.shmlib import shm
from pyBaldr import utilities as util

from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import common.DM_registration as DM_registration

import pyzelda.ztools as ztools
import datetime
from xaosim.shmlib import shm
from asgard_alignment import FLI_Cameras as FLI

from scipy.ndimage import binary_erosion 

#import pyzelda.ztools as ztools





#from asgard_alignment import FLI_Cameras as FLI


# def detect_pupil(image, sigma=2, threshold=0.5, plot=True, savepath=None):
#     """
#     Detects an elliptical pupil (with possible rotation) in a cropped image using edge detection 
#     and least-squares fitting. Returns both the ellipse parameters and a pupil mask.

#     The ellipse is modeled by:

#         ((x - cx)*cos(theta) + (y - cy)*sin(theta))^2 / a^2 +
#         (-(x - cx)*sin(theta) + (y - cy)*cos(theta))^2 / b^2 = 1

#     Parameters:
#         image (2D array): Cropped grayscale image containing a single pupil.
#         sigma (float): Standard deviation for Gaussian smoothing.
#         threshold (float): Threshold factor for edge detection.
#         plot (bool): If True, displays the image with the fitted ellipse overlay.
#         savepath (str): If provided, the plot is saved to this path.

#     Returns:
#         (center_x, center_y, a, b, theta, pupil_mask)
#           where (center_x, center_y) is the ellipse center,
#                 a and b are the semimajor and semiminor axes,
#                 theta is the rotation angle in radians,
#                 pupil_mask is a 2D boolean array (True = inside ellipse).
#     """
#     # Normalize the image
#     image = image / image.max()
    
#     # Smooth the image
#     smoothed_image = gaussian_filter(image, sigma=sigma)
    
#     # Compute gradients (Sobel-like edge detection)
#     grad_x = np.gradient(smoothed_image, axis=1)
#     grad_y = np.gradient(smoothed_image, axis=0)
#     edges = np.sqrt(grad_x**2 + grad_y**2)
    
#     # Threshold edges to create a binary mask
#     binary_edges = edges > (threshold * edges.max())
    
#     # Get edge pixel coordinates
#     y_coords, x_coords = np.nonzero(binary_edges)
    
#     # Initial guess: center from mean, radius from average distance, and theta = 0.
#     def initial_guess(x, y):
#         center_x = np.mean(x)
#         center_y = np.mean(y)
#         r_init = np.sqrt(np.mean((x - center_x)**2 + (y - center_y)**2))
#         return center_x, center_y, r_init, r_init, 0.0  # (cx, cy, a, b, theta)
    
#     # Ellipse model function with rotation.
#     def ellipse_model(params, x, y):
#         cx, cy, a, b, theta = params
#         cos_t = np.cos(theta)
#         sin_t = np.sin(theta)
#         x_shift = x - cx
#         y_shift = y - cy
#         xp =  cos_t * x_shift + sin_t * y_shift
#         yp = -sin_t * x_shift + cos_t * y_shift
#         # Model: xp^2/a^2 + yp^2/b^2 = 1 => residual = sqrt(...) - 1
#         return np.sqrt((xp/a)**2 + (yp/b)**2) - 1.0

#     # Fit via least squares.
#     guess = initial_guess(x_coords, y_coords)
#     result, _ = leastsq(ellipse_model, guess, args=(x_coords, y_coords))
#     center_x, center_y, a, b, theta = result
    
#     # Create a boolean pupil mask for the fitted ellipse
#     yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
#     cos_t = np.cos(theta)
#     sin_t = np.sin(theta)
#     x_shift = xx - center_x
#     y_shift = yy - center_y
#     xp = cos_t * x_shift + sin_t * y_shift
#     yp = -sin_t * x_shift + cos_t * y_shift
#     pupil_mask = (xp/a)**2 + (yp/b)**2 <= 1

#     if plot:
#         # Overlay for visualization
#         overlay = np.zeros_like(image)
#         overlay[pupil_mask] = 1
        
#         plt.figure(figsize=(6, 6))
#         plt.imshow(image, cmap="gray", origin="upper")
#         plt.contour(binary_edges, colors="cyan", linewidths=1)
#         plt.contour(overlay, colors="red", linewidths=1)
#         plt.scatter(center_x, center_y, color="blue", marker="+")
#         plt.title("Detected Pupil with Fitted Ellipse")
#         if savepath is not None:
#             plt.savefig(savepath)
#         plt.show()
    
#     return center_x, center_y, a, b, theta, pupil_mask


# def compute_affine_from_ellipse(ell1, ell2):
#     """
#     Computes an affine transformation that maps points from frame 1 to frame 2
#     using the ellipse parameters from each frame.

#     ell1, ell2: (cx, cy, a, b, theta, pupil_mask) or (cx, cy, a, b, theta) 
#                 The pupil_mask is ignored here; only the numeric parameters are used.

#     Returns:
#       T (ndarray): 3x3 affine transformation matrix mapping frame 1 -> frame 2.
#       T_inv (ndarray): 3x3 inverse transformation matrix.
#     """
#     # Unpack numeric ellipse parameters
#     cx1, cy1, a1, b1, theta1 = ell1[:5]
#     cx2, cy2, a2, b2, theta2 = ell2[:5]
    
#     # Rotation matrices
#     R1 = np.array([[np.cos(theta1), -np.sin(theta1)],
#                    [np.sin(theta1),  np.cos(theta1)]])
#     R2 = np.array([[np.cos(theta2), -np.sin(theta2)],
#                    [np.sin(theta2),  np.cos(theta2)]])
    
#     # Relative scaling matrix
#     S = np.diag([a2/a1, b2/b1])
    
#     # Linear part of the transform
#     A = R2 @ S @ np.linalg.inv(R1)
    
#     # Translation: map center of ellipse 1 to center of ellipse 2
#     c1 = np.array([cx1, cy1])
#     c2 = np.array([cx2, cy2])
#     t = c2 - A @ c1
    
#     # Build full homogeneous 3x3 matrix
#     T = np.array([
#         [A[0,0], A[0,1], t[0]],
#         [A[1,0], A[1,1], t[1]],
#         [0,      0,      1     ]
#     ])
#     T_inv = np.linalg.inv(T)
    
#     return T, T_inv


# def warp_image_manual(image_in, T, output_shape=None, method='nearest'):
#     """
#     Manually warp an image using a 3x3 affine transform matrix T that maps
#     input (x_in, y_in) -> output (x_out, y_out).
#     """
#     if output_shape is None:
#         output_shape = image_in.shape

#     T_inv = np.linalg.inv(T)
#     out_height, out_width = output_shape
#     image_out = np.zeros((out_height, out_width), dtype=image_in.dtype)

#     for y_out in range(out_height):
#         for x_out in range(out_width):
#             p_out = np.array([x_out, y_out, 1.0])
#             p_in = T_inv @ p_out
#             x_in, y_in = p_in[0], p_in[1]

#             if method == 'nearest':
#                 x_nn = int(round(x_in))
#                 y_nn = int(round(y_in))
#                 if (0 <= x_nn < image_in.shape[1]) and (0 <= y_nn < image_in.shape[0]):
#                     image_out[y_out, x_out] = image_in[y_nn, x_nn]

#             elif method == 'bilinear':
#                 x0 = int(np.floor(x_in))
#                 y0 = int(np.floor(y_in))
#                 dx = x_in - x0
#                 dy = y_in - y0
#                 if (0 <= x0 < image_in.shape[1]-1) and (0 <= y0 < image_in.shape[0]-1):
#                     I00 = image_in[y0,   x0  ]
#                     I01 = image_in[y0,   x0+1]
#                     I10 = image_in[y0+1, x0  ]
#                     I11 = image_in[y0+1, x0+1]
#                     Ixy = (1 - dx)*(1 - dy)*I00 + dx*(1 - dy)*I01 \
#                           + (1 - dx)*dy*I10     + dx*dy*I11
#                     image_out[y_out, x_out] = Ixy

#     return image_out





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



def get_bad_pixel_indicies( imgs, std_threshold = 20, mean_threshold=6):
    # To get bad pixels we just take a bunch of images and look at pixel variance and mean

    ## Identify bad pixels
    mean_frame = np.mean(imgs, axis=0)
    std_frame = np.std(imgs, axis=0)

    global_mean = np.mean(mean_frame)
    global_std = np.std(mean_frame)
    bad_pixel_map = (np.abs(mean_frame - global_mean) > mean_threshold * global_std) | (std_frame > std_threshold * np.median(std_frame))

    return bad_pixel_map


def interpolate_bad_pixels(img, bad_pixel_map):
    filtered_image = img.copy()
    filtered_image[bad_pixel_map] = median_filter(img, size=3)[bad_pixel_map]
    return filtered_image






parser = argparse.ArgumentParser(description="Baldr Calibration pipeline to IM.")

# TOML file path; default is relative to the current file's directory.
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
parser.add_argument(
    "--toml_file",
    type=str,
    default=default_toml,
    help="TOML file pattern (replace # with beam) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
)



# setting up socket to ZMQ communication to multi device server
parser.add_argument("--host", type=str, default="192.168.100.2", help="Server host") # "localhost"
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=5000, help="Response timeout in milliseconds"
)

# Beam ids: provided as a comma-separated string and converted to a list of ints.
parser.add_argument(
    "--beam_id",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[3],#[1, 2, 3, 4],
    help="Comma-separated beam IDs. Default: 1,2,3,4"
)

#Camera shared memory path (used to get camera settings )
parser.add_argument(
    "--global_camera_shm",
    type=str,
    default="/dev/shm/cred1.im.shm",
    help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
)

# Plot: default is True, with an option to disable.
parser.add_argument(
    "--plot", 
    dest="plot",
    action="store_true",
    default=True,
    help="Enable plotting (default: True)"
)

parser.add_argument("--fig_path", 
                    type=str, 
                    default='/home/asg/Progs/repos/asgard-alignment/calibration/reports/', 
                    help="path/to/output/image/ for where the saved figures are (DM_registration_in_pixel_space.png)")


# Strehl pixel and IM specific 
parser.add_argument(
    "--phasemask",
    type=str,
    default="H4",
    help="phasemask to move to. Try use a reasonable size one like H3 (default)"
)


parser.add_argument(
    "--mode",
    type=str,
    default='bright',
    help="which baldr mode, bright (12x12 pixels) or faint (6x6 pixels). Default: %(default)s"
)

parser.add_argument("--lobe_threshold",
                    type=float, 
                    default=0.03, 
                    help="threshold for pupil side lobes to define a Strehl proxy pixels. \
                        These are generally where |I0 - N0| > lobe_threshold * <N0[pupil]>,\
                            in addition to some other radii criteria.  Default: Default: %(default)s")        



# IM Specific 


parser.add_argument(
    "--LO",
    type=int,
    default=2,
    help="Up to what zernike order do we consider Low Order (LO). 2 is for tip/tilt, 3 would be tip,tilt,focus etc). Default: %(default)s"
)


# parser.add_argument(
#     "--basis_name",
#     type=str,
#     default="zonal",
#     help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
# )

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





args=parser.parse_args()


# set up context and state dict to move motors 
context = zmq.Context()
context.socket(zmq.REQ)
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, args.timeout)
server_address = f"tcp://{args.host}:{args.port}"
socket.connect(server_address)
state_dict = {"message_history": [], "socket": socket}


# for reading camera config ! 
camclient = FLI.fli(args.global_camera_shm, roi = [None,None,None,None])

# Open SHM once for each subframe 
cam_shm = {b: shm(f"/dev/shm/baldr{b}.im.shm") for b in args.beam_id}

no_imgs = 10 # number of images to average (on internal source 10 is fine for detector purposes)

# BMX, BMY phasemask relative offsets for going between ZWFS and clear pupil
rel_offset = -200.0 #um phasemask offset for clear pupil

############################################
########## 1. register baldr pupil 




print(f"\n======================\nmoving to phasemask {args.phasemask} reference position")
# Move to phase mask
for beam_id in args.beam_id:
    message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
    res = send_and_get_response(message)
    print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)

print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:
    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep( 1 )
    message = f"moverel BMY{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 

# detect them 
for beam_id in args.beam_id:
    

    # mask 
    img_list_tmp = []
    for _ in range( no_imgs ):
        img_list_tmp.append( cam_shm[beam_id].get_data() )
        time.sleep(0.01) #
    cropped_img = np.mean( img_list_tmp, axis=0 )

    if args.fig_path is None:
        savepath=f"delme{beam_id}.png"
    else: # we save with default name at fig path 
        savepath=args.fig_path + f'pupil_reg_beam{beam_id}'

    ell1 = util.detect_pupil(cropped_img, sigma=2, threshold=0.5, plot=args.plot,savepath=savepath)

    #save_pupil_data_toml(beam_id=beam_id, ellipse_params=ell1, toml_path=args.toml_file.replace('#',f'{beam_id}'))



    cx, cy, a, b, theta, pupil_mask = ell1

    # Convert the boolean mask to a nested list of booleans
    mask_list = pupil_mask.tolist()  # shape => Nx x Ny of True/False

    # calculate exterior (Strehl proxy) pixel mask
    rad_est = np.sqrt( 1/np.pi * np.sum( pupil_mask ) )
    exterior = util.filter_exterior_annulus( pupil_mask, inner_radius = rad_est+1, outer_radius = rad_est+5)
    exterior_list = exterior.tolist()

    # calculate secondary pixel mask 
    secondary = np.zeros_like(pupil_mask)
    y_indices, x_indices = np.where(pupil_mask)
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)
    secondary[round(center_x), round(center_y)] = True
    secondary_list = secondary.tolist()
    
    ### NOTE : exterior and secondary filters should be updated more precisely in asgard-alignment/calibration/strehl_filter_registration.py
    new_data = {
        f"beam{beam_id}": {
            "pupil_ellipse_fit": {
                "center_x": float(cx),
                "center_y": float(cy),
                "a": float(a),
                "b": float(b),
                "theta": float(theta),
            },
            "pupil_mask": {
                "mask": mask_list,
                "exterior": exterior_list,
                "secondary":secondary_list
            }
        }
    }

    toml_path = args.toml_file.replace('#',f'{beam_id}')
    # Check if file exists; if so, load and update.
    if os.path.exists(toml_path):
        try:
            current_data = toml.load(toml_path)
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            current_data = {}
    else:
        current_data = {}

    # Update current data with new_data (beam specific)
    #current_data.update(new_data)
    current_data = util.recursive_update(current_data, new_data)


    # Write the updated data back to the TOML file.
    with open(toml_path, "w") as f:
        toml.dump(current_data, f)




    print( f"pupil detection for beam {beam_id} finished")




############################################
########## 2. register DM 


print(f"\n======================\nmoving to phasemask {args.phasemask} reference position")
# Move to phase mask
for beam_id in args.beam_id:
    message = f"fpm_movetomask phasemask{beam_id} {args.phasemask}"
    res = send_and_get_response(message)
    print(f"moved to phasemask {args.phasemask} with response: {res}")

time.sleep(1)

input("\n======================\npress enter when ready to register DM (ensure phasemask is aligned!)")




#################################
# This part is to calibrate the DM actuator registration in pixel space. 
# It generates a bilinear interpolation matrix for each beam to project 
# intensities on rach DM actuator. 
# calibration is done by applying push pull commands on DM corner actuators,
# fitting an interpolated gaussian to the mean region of peak influence in the 
# image for each actuator, and then finding the intersection between the 
# imterpolated image peaks and solving the affine transform matrix.
# By temporal modulation this method can also be used on sky.








# inputs 
number_of_pokes = 8 
poke_amplitude = 0.05
sleeptime = 0.2 #10 is very safe
dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 
dm_turbulence = False # roll phasescreen on DM?

assert hasattr(args.beam_id , "__len__")
assert len(args.beam_id) <= 4
assert max(args.beam_id) <= 4
assert min(args.beam_id) >= 1 

pupil_mask = {}
for beam_id in args.beam_id:
    with open(args.toml_file.replace('#',f'{beam_id}') ) as file:
        config_dict = toml.load(file)

        # get the pupil mask (we only consider pixels within here for the DM calibration)
        pupil_mask[beam_id] = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)



# set up DM shared memories
dm_shm_dict = {}
for beam_id in args.beam_id:
    dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )



# incase we want to test this with dynamic dm cmds (e.g phasescreen)
current_cmd_list = [np.zeros(144)  for _ in args.beam_id]
img_4_corners  = [[] for _ in args.beam_id] 
transform_dicts = []
bilin_interp_matricies = []




print(f'GOING VERY SLOW ({sleeptime}s delays) DUE TO SHM DELAY DM')
for act in dm_4_corners: # 4 corner indicies are in 140 length vector (not 144 2D map)
    print(f"actuator {act}")
    img_list_push = [[] for _ in args.beam_id]
    img_list_pull = [[] for _ in args.beam_id]
    poke_vector = np.zeros(140) # 4 corner indicies are in 140 length vector (not 144 2D map)
    for nn in range(number_of_pokes):
        print( f'poke {nn}')
        poke_vector[act] = (-1)**nn * poke_amplitude
        
        
        # send DM commands 
        for ii, beam_id in enumerate( args.beam_id):
            dm_shm_dict[beam_id].set_data( dm_shm_dict[beam_id].cmd_2_map2D(poke_vector, fill=0) ) 
            ## Try without #DM_flat_offset[beam_id]  )
            
            img_list_tmp = []
            for _ in range( 10 ):
                img_list_tmp.append( cam_shm[beam_id].get_data() )
                time.sleep(0.01) #
            cropped_img = np.mean( img_list_tmp, axis=0 )


            if np.mod(nn,2):
                img_list_push[ii].append(  cropped_img  )
            else:
                img_list_pull[ii].append( cropped_img )
            
            # zero the poke
            dm_shm_dict[beam_id].set_data( dm_shm_dict[beam_id].cmd_2_map2D( 0 * poke_vector, fill=0) ) 
        


        time.sleep(sleeptime)

    for ii, beam_id in enumerate( args.beam_id):
        delta_img = abs( np.mean(img_list_push[ii],axis=0) - np.mean(img_list_pull[ii],axis=0) )
        # the mean difference in images from push/pulls on the current actuator
        img_4_corners[ii].append( np.array( pupil_mask[beam_id] ).astype(float) * delta_img ) #  We multiply by the pupil mask to ignore all external pixels! These can be troublesome with hot pixels etc 


##
## now analyse data and write files
 
## we do beam specific directory from fig_path
if not os.path.exists( args.fig_path + f"beam{beam_id}/"):
    os.makedirs(  args.fig_path + f"beam{beam_id}/" )

# Calibrating coordinate transforms 
dict2write={}
for ii, beam_id in enumerate( args.beam_id ):

    #calibraate the affine transform between DM and camera pixel coordinate systems
    if args.fig_path is not None:
        savefig = args.fig_path #+ 'DM_registration_in_pixel_space.png'
    else:
        savefig = os.path.expanduser('~/Downloads')

    if not os.path.exists( savefig + f"beam{beam_id}/"):
        os.mkdir( savefig + f"beam{beam_id}/" )

    plt.close() # close any open figiures
    transform_dicts.append( DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners[ii] , debug=True, fig_path = savefig + f"beam{beam_id}/"  ) )
    plt.close() # close any open figiures

    # From affine transform construct bilinear interpolation matrix on registered DM actuator positions
    #(image -> actuator transform)
    img = img_4_corners[ii][0].copy()
    x_target = np.array( [x for x,_ in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    y_target = np.array( [y for _,y in transform_dicts[ii]['actuator_coord_list_pixel_space']] )
    x_grid = np.arange(img.shape[0])
    y_grid = np.arange(img.shape[1])
    M = DM_registration.construct_bilinear_interpolation_matrix(image_shape=img.shape, 
                                            x_grid=x_grid, 
                                            y_grid=y_grid, 
                                            x_target=x_target,
                                            y_target=y_target)

    try:
        M @ img.reshape(-1)
    except:
        raise UserWarning("matrix dimensions don't match! ")
    bilin_interp_matricies.append( M )

    # update I2A instead of I2M
    dict2write[f"beam{beam_id}"] = {"I2A":M.tolist()}

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




## write the json file to keep record of stability 
for ii, beam_id in enumerate( args.beam_id ):

    tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path_tmp = f"/home/asg/Progs/repos/asgard-alignment/calibration/cal_data/dm_registration/beam{beam_id}/"
    if not os.path.exists(path_tmp):
        os.makedirs( path_tmp )

    file_tmp = f"dm_reg_beam{beam_id}_{tstamp}.json"
    with open(path_tmp + file_tmp, "w") as json_file:
        json.dump(util.convert_to_serializable(transform_dicts[ii]), json_file)
    print( f"saved dm registration json : {path_tmp + file_tmp}")



############################################
########## 3. register Strehl pixels 
input("\n======================\npress enter when ready to register Strehl pixels")





def plot_strehl_pixel_registration(data , exterior_filter, secondary_filter, savefig = None):

    label = "I0-N0"
    fs = 18
    if np.sum( exterior_filter ):
        # Exterior filter boundaries (red)
        ext_x_min, ext_x_max = 0.5 + np.min(np.where(np.abs(np.diff(exterior_filter, axis=1)) > 0)[1]), \
                            0.5 + np.max(np.where(np.abs(np.diff(exterior_filter, axis=1)) > 0)[1])
        ext_y_min, ext_y_max = 0.5+ np.min(np.where(np.abs(np.diff(exterior_filter, axis=0)) > 0)[0]), \
                            0.5 + np.max(np.where(np.abs(np.diff(exterior_filter, axis=0)) > 0)[0])
    
    if np.sum( secondary_filter ):   
        # Secondary filter boundaries (blue)
        sec_x_min, sec_x_max =  0.5 + np.min( np.where( abs(np.diff( secondary_filter, axis=1  )) > 0)[1] ), \
                                0.5 + np.max( np.where( abs(np.diff( secondary_filter, axis=1  )) > 0)[1] )
        sec_y_min, sec_y_max =  0.5 + np.min( np.where( abs(np.diff( secondary_filter, axis=0   )) > 0)[0] ), \
                                0.5 + np.max( np.where( abs(np.diff( secondary_filter, axis=0  )) > 0)[0] )

    # Create figure and gridspec for joint plot
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                        wspace=0.05, hspace=0.05)

    # Axes: Main heatmap, top x histogram, right y histogram
    ax_main = plt.subplot(gs[1, 0])
    ax_xhist = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_yhist = plt.subplot(gs[1, 1], sharey=ax_main)

    
    # Plot the imag (main axis)

    im = ax_main.imshow(data, aspect='auto', origin='lower', interpolation='nearest')
    ax_main.text(0,0, f"{label}",fontsize=25, color='white')
    ax_main.set_xlabel('X (pixels)',fontsize=fs)
    ax_main.set_ylabel('Y (pixels)',fontsize=fs)

  
    # marginal histograms (for counts)
    # --------------------------
    x_counts = np.sum(data, axis=0)
    y_counts = np.sum(data, axis=1)

    ax_xhist.bar(np.arange(len(x_counts)), x_counts, color='gray', edgecolor='black')
    ax_yhist.barh(np.arange(len(y_counts)), y_counts, color='gray', edgecolor='black')
    ax_yhist.set_xlabel("ADU", fontsize=fs)
    ax_xhist.set_ylabel("ADU", fontsize=fs)
    # Remove tick labels on marginal plots
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)

    # Ensure the histogram axes align with the main heatmap axes:
    ax_xhist.set_xlim(ax_main.get_xlim())
    ax_yhist.set_ylim(ax_main.get_ylim())

    # Draw contours for the filter regions 
    # --------------------------
    # Convert boolean filters to float so that contour finds a level at 0.5.
    if np.sum( exterior_filter ):
        ax_main.contour(exterior_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1]-0.5, -0.5, data.shape[0]-0.5],
                        colors='red', linestyles='-', linewidths=2, origin='lower')

        ex_coords = np.argwhere(exterior_filter)      # shape (N, 2)

        # Plot a cross at each True pixel - to be ABSOLUTTTELY SURE
        # Note: row = y, col = x. So when calling scatter or plot, pass x=col, y=row.
        ax_main.scatter(ex_coords[:,1], ex_coords[:,0],
                        marker='x', color='red', alpha =0.4, label='Exterior Filter')

    if np.sum( secondary_filter ):    
        ax_main.contour(secondary_filter.astype(float), levels=[0.5], extent=[-0.5, data.shape[1]-0.5, -0.5, data.shape[0]-0.5],
                        colors='blue', linestyles='-', linewidths=2, origin='lower')

        sec_coords = np.argwhere(secondary_filter)    # shape (M, 2)

        ax_main.scatter(sec_coords[:,1], sec_coords[:,0],
                        marker='x', color='blue',alpha =0.4, label='Secondary Filter')

    ax_main.legend(fontsize=fs)
    # --------------------------
    # Draw vertical lines on the x-axis (top histogram and main heatmap)

    # Exterior filter (red)
    if np.sum( exterior_filter ):
        ax_xhist.axvline(ext_x_min, color='red', linestyle='--', linewidth=2, label='Exterior Boundary')
        ax_xhist.axvline(ext_x_max, color='red', linestyle='--', linewidth=2)
        ax_main.axvline(ext_x_min, color='red', linestyle='--', linewidth=2)
        ax_main.axvline(ext_x_max, color='red', linestyle='--', linewidth=2)

    # Secondary filter (blue)
    if np.sum( secondary_filter ):  
        ax_xhist.axvline(sec_x_min, color='blue', linestyle='--', linewidth=2, label='Secondary Boundary')
        ax_xhist.axvline(sec_x_max, color='blue', linestyle='--', linewidth=2)
        ax_main.axvline(sec_x_min, color='blue', linestyle='--', linewidth=2)
        ax_main.axvline(sec_x_max, color='blue', linestyle='--', linewidth=2)

    #ax_xhist.legend(loc='upper right', fontsize=fs)

    # Draw horizontal lines on the y-axis (right histogram and main heatmap)
    # --------------------------
    # Exterior filter (red)
    if np.sum( exterior_filter ):
        ax_yhist.axhline(ext_y_min, color='red', linestyle='--', linewidth=2)
        ax_yhist.axhline(ext_y_max, color='red', linestyle='--', linewidth=2)
        ax_main.axhline(ext_y_min, color='red', linestyle='--', linewidth=2)
        ax_main.axhline(ext_y_max, color='red', linestyle='--', linewidth=2)

    # Secondary filter (blue)
    if np.sum( secondary_filter ):  
        ax_yhist.axhline(sec_y_min, color='blue', linestyle='--', linewidth=2)
        ax_yhist.axhline(sec_y_max, color='blue', linestyle='--', linewidth=2)
        ax_main.axhline(sec_y_min, color='blue', linestyle='--', linewidth=2)
        ax_main.axhline(sec_y_max, color='blue', linestyle='--', linewidth=2)


    ax_xhist.tick_params(labelsize=15)
    ax_yhist.tick_params(labelsize=15)
    ax_main.tick_params(labelsize=15)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig( savepath, bbox_inches='tight', dpi=200)
        print( f"saving image {savepath}")
    #plt.show()
    plt.close()





tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")



## Get ZWFS and CLEAR reference pupils 

########____ ASSUME THAT WE HAAVE THINGS ALIGNED WHEN CALLING THIS SCRIPT 
zwfs_pupils = {}
for beam_id in args.beam_id:


    img_list_tmp = []
    for _ in range( 10 ):
        img_list_tmp.append( cam_shm[beam_id].get_data() )
        time.sleep(0.01) #
    img = np.mean( img_list_tmp, axis=0 )


    zwfs_pupils[beam_id] = img
    # close the subframe shared memory 


util.nice_heatmap_subplots( [zz for zz in zwfs_pupils.values()], savefig='delme.png')

# Get reference pupils (later this can just be a SHM address)
clear_pupils = {}
secondary_filter_dict = {}
exterior_filter_dict = {}
#initial_pos = {}

print( 'Moving FPM out to get clear pupils')
for beam_id in args.beam_id:

    message = f"moverel BMX{beam_id} {rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(2)
    print( 'gettin clear pupils')


    img_list_tmp = []
    for _ in range( 10 ):
        img_list_tmp.append( cam_shm[beam_id].get_data() )
        time.sleep(0.01) #
    img = np.mean( img_list_tmp, axis=0 )


    # move back (so we have time buffer while calculating b)
    print( 'Moving FPM back in beam.')
    message = f"moverel BMX{beam_id} {-rel_offset}"
    res = send_and_get_response(message)
    print(res) 
    time.sleep(2)


    #img[img > 0.9e5] = 0
    #img[img < -1e2] = 0
    clear_pupils[beam_id] = img 

    ### DETECT A PUPIL MASK FROM CLEAR MASK 
    center_x, center_y, a, b, theta, pupil_mask = util.detect_pupil(clear_pupils[beam_id], sigma=2, threshold=0.5, plot=False, savepath=None)



    secondary_filter = util.get_secondary_mask(pupil_mask, (center_x, center_y))

    # filter edge of pupil and out radii limit for the strehl mask 
    if args.mode == 'bright':
        pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=7, outer_radius=100) # to limit pupil edge pixels
        pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=11, outer_radius=100) # to limit far out pixel
    elif args.mode == 'faint':
        pupil_edge_filter = util.filter_exterior_annulus(pupil_mask, inner_radius=4, outer_radius=100) # to limit pupil edge pixels
        pupil_limit_filter = ~util.filter_exterior_annulus(pupil_mask, inner_radius=8, outer_radius=100) # to limit far out pixel
    else:
        raise UserWarning("invalid mode. Must be either 'bright' or 'faint'")
    #lobe_threshold = 0.1 # percentage of mean clear pupil interior. Absolute values above this in the exterior pixels are candidates for Strehl pixels 
    #exterior_filter =  ( abs( I0  - N0 )  > lobe_threshold * np.mean( N0[pupil_mask] )  ) * (~pupil_mask) * pupil_edge_filter 
    
    # to be more aggressive we can remove ~pupil_mask in filter
    exterior_filter =  ( abs( zwfs_pupils[beam_id]  - clear_pupils[beam_id] ) > args.lobe_threshold  * np.mean( clear_pupils[beam_id][pupil_mask] ) ) * (~pupil_mask) * pupil_edge_filter * pupil_limit_filter

    exterior_filter_dict[beam_id] = exterior_filter 
    secondary_filter_dict[beam_id] = secondary_filter
    # write to toml 
    ## Eventually this exterior filter should be phasemask dependant (maybe).. lets see how operates! 
    # Note we also define this roughly in pupil_registration script 
    # We do not make these pixels phasemask specific!!!
    new_data = {
            f"beam{beam_id}": {
                "pupil_mask": {
                    "exterior": exterior_filter.astype(int).tolist(),
                    "secondary": secondary_filter.astype(int).tolist(), 
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

    # Update current data with new_data (beam specific)
    #current_data.update(new_data)
    current_data = util.recursive_update(current_data, new_data)

    # Write the updated data back to the TOML file.
    with open(args.toml_file.replace('#',f'{beam_id}'), "w") as f:
        toml.dump(current_data, f)


print('saving output figure')

try:
    for beam_id in args.beam_id:
        if args.fig_path is None:
            savepath=f"delme{beam_id}.png"
        else: # we save with default name at fig path 
            savepath=args.fig_path + f'strehl_pixel_filter{beam_id}.png'

        print(f"saving figure at : {savepath}")
        
        plot_strehl_pixel_registration( data = np.array( zwfs_pupils[beam_id] ) - np.array( clear_pupils[beam_id] ),  
                                       exterior_filter=exterior_filter_dict[beam_id], 
                                       secondary_filter=secondary_filter_dict[beam_id], 
                                       savefig = savepath )

        plt.close("all")



except Exception as e:
    print(f"failed to produce plots : {e}")





############################################
########## 4. Build IM 
input("\n======================\npress enter when ready to start calibrating the Interaction Matrix for Baldr")


# By default HO in this construction of the IM will always contain zonal actuation of each DM actuator.
# Using LO we can also define our Lower order modes on a Zernike basis where LO 
# is the Noll index up to which modes to consider. These LO modes are probed first
# in the IM and then the HO (zonal) modes are probed  




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
util.nice_heatmap_subplots( im_list=[dark_dict[beam_id] for beam_id in args.beam_id], title_list=[f"beam{beam_id} dark" for beam_id in args.beam_id] )
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
    inner_pupil_filt[beam_id] = binary_erosion( pupil_mask[beam_id] * (~secondary_mask[beam_id].astype(bool)), structure=np.ones((3, 3), dtype=bool) )
    ## BELOW IS OLD CONVENTION (pixelwise normalized_pupils, with outside pupil set to interior mean) , 
    #  keep for C++ rtc legacy (wrtten. to toml)
    # this is not needed for new python rtc standards    
    pixel_filter = secondary_mask[beam_id].astype(bool)  | (~(util.remove_boundary(pupil_mask[beam_id]).astype(bool)) ) #| (~bad_pix_mask_tmp )
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
input("\n======================\nphasemasks aligned? ensure alignment then press enter")


print( 'Getting ZWFS pupils')
for beam_id in args.beam_id:

    
    I0s = []
    for _ in range( no_imgs ):
        I0s.append( cam_shm[beam_id].get_data() - dark_dict[beam_id] )
        time.sleep(0.01) #

    zwfs_pupils[beam_id] = I0s 


#basis_name = args.basis_name #"zonal" #"ZERNIKE"
LO_basis = dmbases.zer_bank(2, args.LO+1 )
zonal_basis = np.array([dm_shm_dict[beam_id].cmd_2_map2D(ii) for ii in np.eye(140)]) 
#zonal_basis = dmbases.zer_bank(4, 143 )
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
                                                    "dark" : np.array(dark_dict[beam_id]).astype(int).reshape(-1).tolist(), #np.zeros([32,32]).reshape(-1).astype(int).tolist(), # just update to a default 1000 adu offset. In rtc this can be updated with dark_update function!
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
