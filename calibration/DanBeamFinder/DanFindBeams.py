import numpy as np
import os
import time
import matplotlib.pyplot as plt
import libs.GeneralCameraClass as CamForm
import libs.GeneralStageClass as StageForm
import libs.plotingFunction as pltfunc
# import ipywidgets
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


StageObj.Set_pos(stage='BMX', beam=ibeam,pos=5299.995749999997)
StageObj.Set_pos(stage='BMY', beam=ibeam,pos=4354.996687499998)

print(StageObj.Get_pos(stage='BMX', beam=ibeam))
print(StageObj.Get_pos(stage='BMY', beam=ibeam))
print("test")


frame1=CamObj.GetFrame(1)
frame2=CamObj.GetFrame(2)
frame3=CamObj.GetFrame(3)
frame4=CamObj.GetFrame(4)


ibeam=1
StartX=StageObj.Get_pos(stage='BMX', beam=ibeam)
StartY=StageObj.Get_pos(stage='BMY', beam=ibeam)
StepAwayFromStartX = 200
StepAwayFromStartY = 200
StepCountX = 10
StepCountY = 10
# make a snake like grid pattern
grid_points = StageObj.rasterScanSnakePattern(StartX,StartY,StepAwayFromStartX,StepAwayFromStartY,StepCountX, StepCountY)

# get a frame that you know the beam is not on a masks
MountPosNoFeatureY=4700.2
MountPosNoFeaturex=4498.2
StageObj.Set_pos(stage='BMY', beam=ibeam,pos=MountPosNoFeatureY)   
StageObj.Set_pos(stage='BMX', beam=ibeam,pos=MountPosNoFeaturex)
Ref_frame=CamObj.GetFrame(ibeam)
appxcenters_Ref_frame=CamObj.FindMaxValueOnFrame(Ref_frame)
ref_flux=CamObj.GetRelativePower(
        frame=Ref_frame,centre=[appxcenters_Ref_frame[0],appxcenters_Ref_frame[1]],
        x_half_width=5,y_half_width=5)
ref_corr_temp=Ref_frame.astype(float).ravel()
ref_corr_temp = ref_corr_temp-np.mean(ref_corr_temp)

# do the scan 
AllFrames=np.zeros((StepCountY*StepCountX,Ref_frame.shape[0],Ref_frame.shape[1]))
MetricMatrix_flux=np.zeros((StepCountY,StepCountX))
MetricMatrix_corr=np.zeros((StepCountY,StepCountX))

TotalscanScount=StepCountY*StepCountX
icount=0  
ixCounting=0

for iy in range(StepCountY):
    StageObj.Set_pos(stage='BMY', beam=ibeam,pos=grid_points[iy,0,1])
    print(iy)
    for ix in range(StepCountX):
    
        StageObj.Set_pos(stage='BMX', beam=ibeam,pos=grid_points[iy,ix,0])
        # ix+=1
        
        frame = CamObj.GetFrame(ibeam=ibeam)
        AllFrames[icount]=frame

        flux = CamObj.GetRelativePower(frame=frame,centre=[appxcenters_Ref_frame[0],appxcenters_Ref_frame[1]],
        x_half_width=5,y_half_width=5)
        
        #spatial correlation
        frame_corr_temp=frame.astype(float).ravel()
        frame_corr_temp = frame_corr_temp-np.mean(frame_corr_temp)
        corr = np.sum(frame_corr_temp*ref_corr_temp)/np.sqrt(np.sum(frame_corr_temp**2)*np.sum(ref_corr_temp**2))


        MetricMatrix_corr[iy,ix]=corr
        MetricMatrix_flux[iy,ix]=flux
        pltfunc.PlotResults(icount,Ref_frame=Ref_frame,AllFrames=AllFrames,gridpoints=grid_points,MetricMatrix=MetricMatrix_corr)

        icount+=1


np.save("Data/Allframes.npy",AllFrames)
np.save("Data/MetrixMatrix_flux.npy",MetricMatrix_flux)
np.save("Data/MetrixMatrix_Corr.npy",MetricMatrix_corr)







    








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




