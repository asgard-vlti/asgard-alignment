import numpy as np
import os
import time
import matplotlib.pyplot as plt
import libs.GeneralCameraClass as CamForm
import libs.GeneralStageClass as StageForm
import libs.plotingFunction as pltfunc
import ipywidgets
# %matplotlib
# import AlignmentRoutine as AlignForm

# if server is stuck 
# sudo lsof -i :5555 then kill the PID 




CamObj = CamForm.GeneralCameraObject()
StageObj = StageForm.GeneralStageObject(host="192.168.100.2", port=5555)
beamNum_arr= np.array([1,2,3,4]) # which beams to do the search on (1-4)

for ibeam in beamNum_arr:
    initial_Xpos=StageObj.Get_pos(stage='BMX', beam=ibeam) # just to check connection and print initial positioninitial_Xpos
    intial_Ypos=StageObj.Get_pos(stage='BMY', beam=ibeam) # just to check connection and print initial position
    
    print(f"initial position for beam {ibeam} is x,y = {initial_Xpos}, {intial_Ypos}")

frame1=CamObj.GetFrame(1)
frame2=CamObj.GetFrame(2)
frame3=CamObj.GetFrame(3)
frame4=CamObj.GetFrame(4)

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

# plt.show(block=False)
# # plt.show()
# plt.close() 



# appxcenters=CamObj.FindMaxValueOnFrame(frame1)
# relpwr=CamObj.GetRelativePower(
#         frame=frame4,
#         centre=[idx[0],idx[1]],
#         x_half_width=5,
#         y_half_width=5,
#         show_plot=False,
#         avgCount=1
#     )
# ibeam=1
# print(relpwr)
# print("ibeam",ibeam)
# start= time.time()
# initial_Xpos=StageObj.Get_pos(stage='BMX', beam=ibeam) # just to check connection and print initial positioninitial_Xpos

# postomove=8159.972249999996#initial_Xpos+2
# print(postomove)
# StageObj.Set_pos(stage='BMX', beam=ibeam,pos=postomove) # just to check connection and print initial positioninitial_Xpos
# end=time.time()
# print(f"time for move {end-start:.6f} sec")

# aftermove_Xpos=StageObj.Get_pos(stage='BMX', beam=ibeam) # just to check connection and print initial positioninitial_Xpos
# print("after move ",aftermove_Xpos," diff= ",aftermove_Xpos-postomove )





ibeam=1
StartX=StageObj.Get_pos(stage='BMX', beam=ibeam)
StartY=StageObj.Get_pos(stage='BMY', beam=ibeam)
StepAwayFromStartX =600
StepAwayFromStartY =600
StepCountX = 20
StepCountY = 20
# make a snake like grid pattern
grid_points = StageObj.rasterScanSnakePattern(StartX,StartY,StepAwayFromStartX,StepAwayFromStartY,StepCountX, StepCountY)

# get a frame that you know the beam is not on a masks
MountPosNoFeatureY=1000
MountPosNoFeaturex=1000
StageObj.Set_pos(stage='BMY', beam=ibeam,pos=MountPosNoFeatureY)   
StageObj.Set_pos(stage='BMX', beam=ibeam,pos=MountPosNoFeaturex)
Ref_frame=CamObj.GetFrame(beam=ibeam)
appxcenters_Ref_frame=CamObj.FindMaxValueOnFrame(Ref_frame)
ref_flux=CamObj.GetRelativePower(
        frame=Ref_frame,centre=[appxcenters_Ref_frame[0],appxcenters_Ref_frame[1]],
        x_half_width=5,y_half_width=5)

# do the scan 
TotalscanScount=0   
ix=0
for iy in range(StepCountY):
    StageObj.Set_pos(stage='BMY', beam=ibeam,pos=grid_points[iy,ix,1])

    for _ in range(StepCountX):
    
        StageObj.Set_pos(stage='BMX', beam=ibeam,pos=grid_points[iy,ix,0])
        ix+=1
        
        frame = CamObj.GetFrame(beam=ibeam)
        
        flux = CamObj.GetRelativePower(
        frame=frame,centre=[appxcenters_Ref_frame[0],appxcenters_Ref_frame[1]],
        x_half_width=5,y_half_width=5)
        TotalscanScount+=1

ipywidgets.interact(pltfunc.PlotResults,iscan=(0,TotalscanScount,1),Ref_frame=ipywidgets.fixed(Ref_frame),AllFrames=ipywidgets.fixed(AllFrames),gridpoints=ipywidgets.fixed(gridpoints),MetricMatrix=ipywidgets.fixed(MetricMatrix))






    








# AlignObj=AlignForm.AlginmentObj([StageObj], [CamObj])

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




