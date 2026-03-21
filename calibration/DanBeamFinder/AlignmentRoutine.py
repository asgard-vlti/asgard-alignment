from Lab_Equipment.Config import config

# import tomography.standard as standard
# import tomography.masks as masks

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import ctypes
import copy
from IPython.display import display, clear_output
# import cma
import ipywidgets
import multiprocessing
import time
import scipy.io

from scipy import io, integrate, linalg, signal
from scipy.io import savemat, loadmat
from scipy.fft import fft, fftfreq, fftshift,ifftshift, fft2,ifft2,rfft2,irfft2
from scipy.signal import find_peaks
from scipy.optimize import minimize

# Defult Pploting properties 
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [5,5]

# from script_functions import start_worker
# import CameraWindowForm as CamForm
import GeneralCameraClass as CamForm
import GeneralStageClass as StageForm

# Power Meter Libs
# import  Lab_Equipment.PowerMeter.PowerMeterObject as PMLib
# import Lab_Equipment.PowerMeter.PowerMeter_Thorlabs_lib as pwrMeter


# Alginment Functions
import  AlignmentFunctions as AlignFunc
from typing import List




class AlginmentObj():
    def __init__(self,
                StageObj: List[StageForm.GeneralStageObject],
                CamObjs: List[CamForm.GeneralCameraObject],):
        super().__init__()
        
        # Store lists of devices
        self.StageObjs = StageObj
        self.CamObjs = CamObjs
       
        # Ensure equal lengths
        assert len(StageObj) == len(CamObjs), \
            "slmObjects, camObjs, and digiholoObjs must have the same length"
        self.ObjCount = len(StageObj)
        print(self.ObjCount)
        # Default to first channel
        # Initial properties

            
        # self.channel = Channel
        # self.pol = pol
        # self.ApplyZernike = ApplyZernike
        # self.imask = 0
        # self.PixelsCountFromCenters = 50
        # self.AvgFrameCount = 30
        # self.PlotTracking = True
        # self.MaskSize = [256,256]
        # Build reference field
        # self.MakeReferenceField()

        
        
    def __del__(self):
        print("Cleaning up AlginmentObj_Stage_PwrMeter")




    def MultiDimAlignmentOfStage(self,CamObjsIdx=0,StageAlignObjIdx=[0],ibeam=1,
                               Optimiser='Nealder-Mead',
                               GoalMetric='Pwr',
                               PropertiesToAlign=None,
                               InitialStepSizes=None,
                               f_Tol=1e-7,
                               x_Tol=1,
                               maxAttempts=100,
                               populationSize=None,
                               simga0=0.2,
                               ixCamCenter=None,
                                    iyCamCenter=None,
                                    x_half_width=None,
                                    y_half_width=None ):
        # if channel is None:#if no channel is passed in then use the first active channel on the SLM
        #     channel=self.slmObjs[SLMObjIdx].ActiveRGBChannels[0]
         # Need to set up self variables for the the function to be passed to the golden search function
        # self.channel=channel
        # self.pol=pol
        if not isinstance(StageAlignObjIdx, list):
                raise TypeError(f"Expected a list, got {type(StageAlignObjIdx).__name__!r}")
                return
        self.StageAlignObjIdx=StageAlignObjIdx
        self.CamObjIdx=CamObjsIdx
        self.ixCamCenter=ixCamCenter
        self.iyCamCenter=iyCamCenter
        self.x_half_width=x_half_width
        self.y_half_width=y_half_width
        self.ibeam=ibeam
       
        self.GoalMetric=GoalMetric
        
        
        if PropertiesToAlign is None:
            self.PropertiesToAlign = [{
                "AlignBMX": False,
                "AlignBMY": False
            }]
            print("You need to make a dict that follows the below format were you set the values you want to be aligned: ")
            print("PropertiesToAlign = {")
            for key, value in self.PropertiesToAlign[0].items():
                print(f'    "{key}": {value},')
            print("}")
            return
        else: 
            self.PropertiesToAlign=PropertiesToAlign
        if not isinstance(PropertiesToAlign, list):
                raise TypeError(f"Expected a list, got {type(PropertiesToAlign).__name__!r}")
                return
            
        if InitialStepSizes is None:
            self.InitialStepSizes = [{
                "d_X": 5,
                "d_Y": 5
                
            }]
            print("Initial step sizes have been auto set to the below values. If you wanted to change it you need to make a dict of that fromat and pass it in to function: ")
            print("InitialStepSizes = {")
            for key, value in self.InitialStepSizes[0].items():
                print(f'    "{key}": {value},')
            print("}")
        else:
            self.InitialStepSizes = InitialStepSizes
        if not isinstance(InitialStepSizes, list):
                raise TypeError(f"Expected a list, got {type(InitialStepSizes).__name__!r}")
                return
            
        StepArray,InitalPhysical=self.GetInitialVerticeForStageAlignment()
        print(type(StepArray))
        print(type(InitalPhysical))
        
        self.LowerPhysicalBounds,self.UpperPhysicalBounds=AlignFunc.MakeBoundsFromCentre(InitalPhysical,StepArray)
        
        InitalNorm=AlignFunc.physical_to_normalised(InitalPhysical,self.LowerPhysicalBounds,self.UpperPhysicalBounds)

        x_Tol_norm_arr=AlignFunc.physical_to_normalised(InitalPhysical,self.LowerPhysicalBounds,self.UpperPhysicalBounds)+AlignFunc.physical_to_normalised(InitalPhysical+x_Tol,self.LowerPhysicalBounds,self.UpperPhysicalBounds)
        print('test')
        print(x_Tol_norm_arr)
        x_Tol_norm=x_Tol_norm_arr[0]
        #this is the scipy minimisation function might be better then my one that i wrote
     
        self.counter = 0
        self.bestPhysicalVetex = None
        self.BestMetric = np.inf

        # self.CamObjs[self.CamObjIdx].SetSingleFrameCapMode()

        if Optimiser != 'CMA-ES':
            try:
                if Optimiser == 'Nelder-Mead':
                    intial_simplex = AlignFunc.MakeIntialSimplex(InitalPhysical, StepArray,self.LowerPhysicalBounds,self.UpperPhysicalBounds)
                    PhysicalVertex=AlignFunc.normalised_to_physical(intial_simplex,self.LowerPhysicalBounds,self.UpperPhysicalBounds)
                    print(PhysicalVertex)
                    result = minimize(
                        self.UpdateVertex_PwrReading,
                        InitalNorm,
                        method=Optimiser,
                        options={
                            'disp': True,
                            'initial_simplex': intial_simplex,
                            'xatol': x_Tol_norm,
                            'fatol': f_Tol,
                            'maxiter': maxAttempts
                        }
                    )
                else:
                    result = minimize(
                        self.UpdateVertex_PwrReading,
                        InitalNorm,
                        method=Optimiser,
                        bounds=[(-1, 1)] * InitalNorm.size,
                        options={
                            'disp': True,
                            'xtol': x_Tol_norm,
                            'ftol': f_Tol,
                            'maxiter': maxAttempts
                        }
                    )
            except RuntimeError as e:
                print(f"\nOptimisation stopped: {e}")
                print(f"Best-so-far: {self.BestMetric} at x = {self.bestPhysicalVetex}")
            else:
                print("\nOptimisation completed.")
                print(f"Result: {result.fun} at x = {result.x}")
                print(f"Best-so-far: {self.BestMetric} at x = {self.bestPhysicalVetex}")

        else:
            try:
                if populationSize is None:
                    populationSize = 4 + (3 * np.log10(InitalNorm.size))
                lower_bounds = np.array([-1.0] * len(InitalNorm))
                upper_bounds = np.array([1.0] * len(InitalNorm))
                # result = cma.fmin(
                #     objective_function=self.UpdateVertex_PwrReading,
                #     x0=InitalNorm,
                #     sigma0=simga0,
                #     options={
                #         'bounds': [lower_bounds, upper_bounds],
                #         'popsize': populationSize,
                #         'maxiter': maxAttempts,
                #         'verb_disp': 1
                #     }
                # )
            except RuntimeError as e:
                print(f"\nOptimisation stopped: {e}")
                print(f"Best-so-far: {self.BestMetric} at x = {self.bestPhysicalVetex}")
            else:
                print("\nOptimisation completed.")
                print(f"Result: {result[1]} at x = {result[0]}")
                print(f"Best-so-far: {self.BestMetric} at x = {self.bestPhysicalVetex}")


        # self.CamObjs[CamObjIdx].SetContinousFrameCapMode()
       
        print("Updating the stage to have the best properties")
        self.UpdateVerticesForStageAlignment(self.bestPhysicalVetex)
        
        # result.x

        # AlignFunc.NelderMead(StepArray,InitalxVertex,ErrTol,maxAttempts,self.UpdateVertex_TakeDigholoBatch)
        AlignFunc.ChangeFileForStopAliginment(0)
        # self.CamObjs[self.CamObjIdx].SetContinousFrameCapMode()

        
        return 
    
    # def print_callback(self):
    #     x, y = params
    #     dErr = np.std(funcVertex);
    #     print(attemptCount,' Function Value= ',funcVertex[0],' Error Accros Values= ',dErr, ' Verterx Value= ',xVertex[:,0])
    #     print(funcVertex[:])
    #     print(f"Callback: x={x:.3f}, y={y:.3f}")

    def UpdateVertex_PwrReading(self,xVertexSingle):
        self.counter=self.counter+1
        if AlignFunc.CheckFileForStopAliginment():
            raise RuntimeError("Optimisation manually terminated.")
        PhysicalVertex=AlignFunc.normalised_to_physical(xVertexSingle,self.LowerPhysicalBounds,self.UpperPhysicalBounds)

        self.UpdateVerticesForStageAlignment(PhysicalVertex)
        # Frames=np.zeros((self.batchCount,self.CamObj.Nx,self.CamObj.Ny))\
        # MetricVaule=self.CamObjs[self.CamObjIdx].GetRelativePower()
       
        MetricVaule=self.CamObjs[self.CamObjIdx].GetRelativePower(
            centre=[self.ixCamCenter, self.iyCamCenter],
            x_half_width=self.x_half_width,y_half_width=self.y_half_width)
        MetricVaule=np.log10(MetricVaule)

        # print(Metrics)
        # MetricVaule=Metrics[self.GoalMetric,0]
        # print(MetricVaule)
        # print(xVertexSingle)
        # if self.GoalMetric==digholoMod.digholoMetrics.MDL:
        #     MetricVaule=-MetricVaule

        # return -MetricVaule,xVertexSingle
        print("Func Evals: "+str(self.counter) + " Metric: "+ str(MetricVaule))
        # print(f"x values = {PhysicalVertex}")
        # Update best result so far
        if -MetricVaule < self.BestMetric:
            self.BestMetric =-MetricVaule
            self.bestPhysicalVetex= PhysicalVertex.copy()

        return -MetricVaule
    
       
    def UpdateVerticesForStageAlignment(self,VertexArr):
        vertexIdx=0  
        for StageObjIdx in (self.StageAlignObjIdx):
            if (self.PropertiesToAlign[StageObjIdx]["AlignBMX"]):  
                # self.StageObjs[StageObjIdx].Set_Single_Stage_State_abs(stagelib.Axes.X,VertexArr[vertexIdx])
                self.StageObjs[StageObjIdx].Set_pos(stage="BMX",beam=self.ibeam,position=VertexArr[vertexIdx])
                VertexArr[vertexIdx]=self.StageObjs[StageObjIdx].Get_pos(stage='BMX',beam=self.ibeam)
                vertexIdx=vertexIdx+1
                
            if (self.PropertiesToAlign[StageObjIdx]["AlignBMY"]):    
                # self.StageObjs[StageObjIdx].Set_Single_Stage_State_abs(stagelib.Axes.Y,VertexArr[vertexIdx])
                self.StageObjs[StageObjIdx].Set_pos(stage="BMY",beam=self.ibeam,position=VertexArr[vertexIdx])
                VertexArr[vertexIdx]=self.StageObjs[StageObjIdx].Get_pos(stage='BMY',beam=self.ibeam)
                vertexIdx=vertexIdx+1
        
        return VertexArr
    
    
    def GetInitialVerticeForStageAlignment(self):
        VertexArr=np.empty(0)
        stepSizeVertexArr=np.empty(0)
        for StageObjIdx in (self.StageAlignObjIdx): 
            if (self.PropertiesToAlign[StageObjIdx]["AlignBMX"]):    
                VertexArr=np.append(VertexArr,self.StageObjs[StageObjIdx].Get_pos(stage='BMX',beam=self.ibeam))
                stepSizeVertexArr=np.append(stepSizeVertexArr,self.InitialStepSizes[StageObjIdx]["d_X"])
            if (self.PropertiesToAlign[StageObjIdx]["AlignBMY"]):    
                VertexArr=np.append(VertexArr,self.StageObjs[StageObjIdx].Get_pos(stage='BMY',beam=self.ibeam))
                stepSizeVertexArr=np.append(stepSizeVertexArr,self.InitialStepSizes[StageObjIdx]["d_Y"])
            
        self.TotalDims=VertexArr.shape
            
        return stepSizeVertexArr,VertexArr
    
