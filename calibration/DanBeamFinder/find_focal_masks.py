import numpy as np
import os
import time
import matplotlib.pyplot as plt
import libs.GeneralCameraClass as CamForm
import libs.GeneralStageClass as StageForm
import libs.plotingFunction as pltfunc
import datetime
import pathlib


class FPMFinder:
    def __init__(self, host="mimir", port=5555):
        self.CamObj = CamForm.GeneralCameraObject()
        self.StageObj = StageForm.GeneralStageObject(host=host, port=port)


"""
python find_focal_mask.py

Usage: Find the focal plane mask. 


--beam [int] : which beam to do the search on (1-4). must be provided.
--start-center [x,y] | "current" : the centre of the search area. Can be "current" to 
    use the current position of the stage as the center, or a list of [x,y] coordinates.
--step-size [float] : the step size in microns for the search grid. Default is 20 microns.
--search-width [float] : the width of the search area in microns. Equal to the height. Default is 200 microns.
--dot-spacing [float] : the spacing of the dots on the focal plane mask in microns. Default is 1000 microns.
--save-path [str] : the path to save the results. Default is "Data/{date}/Scan_{beam}_{current_datetime}".
--n-dots [int] : the number of dots to search for in the focal plane mask. Default is 5.
--detection-threshold [float] : the threshold for detecting the dots in the camera images. No mask is ~1.0, thresholds must be <1.0. Default 0.9.

This script finds a line of phase mask dots without moving through the whole focal plane. First,
it does a scan around the starting center. If a 

"""
