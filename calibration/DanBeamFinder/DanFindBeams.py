import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import GeneralCameraClass as CamForm
import GeneralStageClass as StageForm
# import AlignmentRoutine as AlignForm

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 

# def convert_to_serializable(obj):
#     """
#     Recursively converts NumPy arrays and other non-serializable objects to serializable forms.
#     Also converts dictionary keys to standard types (str, int, float).
#     """
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()  # Convert NumPy arrays to lists
#     elif isinstance(obj, np.integer):
#         return int(obj)  # Convert NumPy integers to Python int
#     elif isinstance(obj, np.floating):
#         return float(obj)  # Convert NumPy floats to Python float
#     elif isinstance(obj, dict):
#         return {str(key): convert_to_serializable(value) for key, value in obj.items()}  # Ensure keys are strings
#     elif isinstance(obj, list):
#         return [convert_to_serializable(item) for item in obj]
#     else:
#         return obj  # Base case: return the object itself if it doesn't need conversion




# CamObj = CamForm.GeneralCameraObject()
StageObj = StageForm.GeneralStageObject(host="192.168.100.2", port=5555)
beamNum_arr= np.array([1,2,3,4]) # which beams to do the search on (1-4)

for ibeam in beamNum_arr:
    initial_Xpos=StageObj.Get_pos(stage='BMX', beam=ibeam) # just to check connection and print initial positioninitial_Xpos
    intial_Ypos=StageObj.Get_pos(stage='BMY', beam=ibeam) # just to check connection and print initial position
    
    print(f"initial position for beam {ibeam} is x,y = {initial_Xpos}, {intial_Ypos}")
# import cv2
arr=np.ones((5,5))
plt.imshow(arr)
plt.show()
plt.close()
# frame1=CamObj.GetFrame(1)
# frame2=CamObj.GetFrame(2)
# frame3=CamObj.GetFrame(3)
# frame4=CamObj.GetFrame(4)
# plt.figure(figsize=(8, 8))
# plt.subplot(2,2,1)
# plt.imshow(frame1, cmap='gray')
# plt.title('Frame from Beam 1')
# plt.subplot(2,2,2)
# plt.imshow(frame2, cmap='gray')
# plt.title('Frame from Beam 2')
# plt.subplot(2,2,3)
# plt.imshow(frame3, cmap='gray') 
# plt.title('Frame from Beam 3')
# plt.subplot(2,2,4)
# plt.imshow(frame4, cmap='gray')
# plt.title('Frame from Beam 4')
# plt.savefig('delme.png')

# plt.show()
# # plt.show()
# plt.close() 


# AlignObj=AlignForm.AlginmentObj([StageObj], [CamObj])

CamROI_xCenter=150
CamROI_yCenter=120
CamROI_xhalfwidth=128
CamROI_yhalfwidth=128
# AlignObj.MultiDimAlignmentOfStage(CamObjsIdx=0,StageAlignObjIdx=[0],
#                                Optimiser='Nelder-Mead',
#                                GoalMetric='Pwr',
#                                PropertiesToAlign=None,
#                                InitialStepSizes=None,
#                                f_Tol=1e-7,
#                                x_Tol=1,
#                                maxAttempts=100,
#                                populationSize=None,
#                                simga0=0.2,
#                                ixCamCenter=CamROI_xCenter,
#                                     iyCamCenter=CamROI_yCenter,
#                                     x_half_width=CamROI_xhalfwidth,
#                                     y_half_width=CamROI_yhalfwidth )




