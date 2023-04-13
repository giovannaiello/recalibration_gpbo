#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location and Threshold Optimization:
By using a Gaussian-Process based Bayesian Optimization, this script 
supports the methodological background of the paper.
Here we report the code to run for one patient. The same procedure can be repeated for all patients.
"""

from AS_resorting import * 
from GPBO_location import *
from GPBO_threshold import * 

#%% Loading toy dataset for location
raw_data_location = pd.read_pickle('ToyData/toy_dataset_location.pkl')  

#%%  Processing raw data for location
sbjTime= 20
sbjTimeRemove = 11
processed_data_location = orderAccordingToCentroid(raw_data_location,sbjTime,sbjTimeRemove)

#%% Running Random Search, reset-GPBO and GPBO
targetPatterns = [0,0.5,1,1.5]
results_loc = run_all_models(processed_data_location,targetPatterns)

#%%  Loading toy dataset for location
raw_data_threshold = pd.read_pickle('ToyData/toy_dataset_threshold.pkl')  

#%% Processing raw data for threshold
sbjTime= 20
sbjTimeRemove = 11
processed_data_threshold = orderAccordingToCentroid(raw_data_threshold,sbjTime,sbjTimeRemove)

#%% Running model for threshold
results_threshold = run_threshold_GPBO(processed_data_threshold)