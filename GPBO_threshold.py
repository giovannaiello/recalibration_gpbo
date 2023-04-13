#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These functions are needed to run the GPBO in threshold search.
"""

import numpy as np
from natsort import natsorted
import GPyOpt
import GPy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(10)

from numpy.random import seed
seed(12345)

def run_threshold_GPBO(dataFrame_all):
    # This function computes the iterations needed by the gpbo method to find the threshold and represents it (day by day and in average) 
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        
    # OUTPUT: LIST_ALL_VALUES: DataFrame containining search info
    
    
    LIST_ALL_VALUES = gpbo_threshold(dataFrame_all)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3))
    sns.boxplot(ax=ax1,y='Nb Iter',data=LIST_ALL_VALUES,showfliers = False)
    ax1.set_ylabel('Mean Number Iterations',fontsize=14)
    
    days=[]
    for day in set(LIST_ALL_VALUES['Day']) :
        if np.std(LIST_ALL_VALUES['Nb Iter'].loc[LIST_ALL_VALUES['Day']==day]) != 0:
            days.append(day)
    idxToSave = []
    for idx in LIST_ALL_VALUES.index:
        if LIST_ALL_VALUES['Day'][idx] in days:
            idxToSave.append(idx)
            
    LIST_ALL_VALUES_2 = LIST_ALL_VALUES.loc[idxToSave]
        
    #fig,ax=plt.subplots(1,figsize=(6,3))
    sns.boxplot(ax=ax2,x="Day",y="Nb Iter",showfliers = False,data=LIST_ALL_VALUES_2,palette="Blues")
    ax2.set_ylabel('Mean Number Iterations/Day',fontsize=14)
    ax2.set_xlabel('Day',fontsize=14)
    plt.savefig('Figures/Results_Threshold.png')
    plt.show()
    
    return LIST_ALL_VALUES
def gpbo_threshold(dataFrame_all):
    # This function applies the GPBO algorithm and outputs the research for all the AS and the mapping days 
    # INPUT: dataFrame_all: DataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    # OUTPUT: LIST_ALL_VALUES: DataFrame containing all the information deriving from the gpbo-driven research for each repetition, AS and mapping day 
    
    couplesEL_AS = define_AS(dataFrame_all)
    
    ALL_ITERATIONS=[]
    LIST_ALL_VALUES=[]
    DIFF_THRESHOLD=[]
    for rep in np.arange(5): # repetitions for robustness 
        for AS in couplesEL_AS:
        
            current_AS_data = dataFrame_all.loc[(dataFrame_all['EL']==AS[0]) & (dataFrame_all['AS']==AS[1])]
            
            all_Perceptual_Thresholds, all_Observations_DayByDay = define_all_PT(current_AS_data)
                        
            X = np.array([all_Perceptual_Thresholds[0]]).reshape(-1, 1)
            ITERATIONS_PER_AS=[]
            
            for CD,thisThreshold,thisObservations,n in zip(natsorted(list(set(current_AS_data['DAY'])))[1:],all_Perceptual_Thresholds[1:],all_Observations_DayByDay[1:],np.arange(len(all_Observations_DayByDay[1:]))):
               
                realIns=[]
                realOuts=[]
                Qtarget = all_Perceptual_Thresholds[n+1]
                X = np.array([all_Perceptual_Thresholds[n]]).reshape(-1, 1)
                
                case = 0
                if all_Perceptual_Thresholds[n] - Qtarget > 0 : # if previous threshold is too strong
                    X = np.vstack((X,0.5*all_Perceptual_Thresholds[n]))
                    case = 1
                elif all_Perceptual_Thresholds[n] - Qtarget < 0 : # if previous threshold can't be felt (too low)
                    case = 2
                    X = np.vstack((X,2.5*all_Perceptual_Thresholds[n]))
                
                y = []
                for i in X:
                    if i - Qtarget >= 0 : 
                        y.append(np.abs(i-Qtarget)[0])
                        
                    elif i - Qtarget < 0 :
                        y.append(0)
                        
                y = np.array(y).reshape(-1, 1)
                
                kernel = GPy.kern.RBF(input_dim=1, variance=120, lengthscale=30)
    
                if obj_func(actualval(all_Perceptual_Thresholds[n],Qtarget)) == -1: # if the PT of the day before works already
                    nStepsNeeded = 1
                    diffFromX = all_Perceptual_Thresholds[n]-Qtarget
                    realXs = all_Perceptual_Thresholds[n]
                else: 
                    if case == 2: # previous threshold is too low 
                        if y[1][0] == 0: #also 2.5*PT is too low 
                            actual_domain = np.array(np.arange(all_Perceptual_Thresholds[n],120))
                            newPoint=120
                        else: # 2.5*PT is strong enough to be felt
                            actual_domain = np.array(np.arange(all_Perceptual_Thresholds[n],np.min([120,2.5*all_Perceptual_Thresholds[n]])))
                            newPoint=np.min([2.5*all_Perceptual_Thresholds[n],120])
                        
                    if case == 1:
                        if y[1][0] == 0: # 0.5*PT cannot be felt (too low)
                            actual_domain = np.array(np.arange(0.5*all_Perceptual_Thresholds[n],all_Perceptual_Thresholds[n]))
                            newPoint=0.5*all_Perceptual_Thresholds[n] 
                        else:
                            actual_domain = np.array(np.arange(1,all_Perceptual_Thresholds[n]))
                            newPoint=1 
                            
                    m1 = define_model(X,y,kernel,1.20)
                    actual_domain2 = [actualval(i,Qtarget) for i in actual_domain]
                    domain = [{'name': 'Charge','type':'discrete','domain':np.arange(np.min(actual_domain2),np.max(actual_domain2)),'dimensionality':1}]
                    BOStep = GPyOpt.methods.BayesianOptimization(f=obj_func,domain=domain,acquisition_type='EI',kernel = kernel, model=m1)
                    
                    realOuts, realIns = run_BO(BOStep)
                    
                    nStepsNeeded = (np.where(realIns==-1)[0])[0]+3 #2 initial stimulations, + 0-index = 3
                    diffFromX = realOuts[np.where(realIns==-1)[0][0]]
                    realXs = np.concatenate((np.array([all_Perceptual_Thresholds[n],newPoint]),np.array(realOuts + Qtarget)))
                    final_domain = actual_domain
                    changeDomain = False  
    
    
                ITERATIONS_PER_AS.append(nStepsNeeded)
                DIFF_THRESHOLD.append(diffFromX)
    
                LIST_ALL_VALUES.append(tuple((rep,AS[0],AS[1],CD,Qtarget,Qtarget+diffFromX,nStepsNeeded,realXs)))
                
            
        ALL_ITERATIONS.append(ITERATIONS_PER_AS)
        
    LIST_ALL_VALUES = pd.DataFrame(LIST_ALL_VALUES,columns=['Repetition','Electrode','AS','Day','PT Target','PT Chosen','Nb Iter','X_Tried'])
        
    return LIST_ALL_VALUES


def define_all_PT(AS_data):
    # This function applies defines the thresholds for each mapping day and the difference of each threshold with respect to the threshold of the following mapping
    # INPUT: AS_data: DataFrame, subset of the whole dataset, containing all the information (from all mappings) for a certain AS 
    # OUTPUT: all_Perceptual_Thresholds: list containing the PT for each mapping day 
    #         all_Observations_DayByDay: list containing the difference of the PT of one mapping with respect to the other mappings
    
    
    all_Perceptual_Thresholds=[]
    for CD in natsorted(list(set(AS_data['DAY']))):
        current_DAY = AS_data.loc[AS_data['DAY']==CD]
        current_TargetQ = int(current_DAY['Charge'])
        all_Perceptual_Thresholds.append(current_TargetQ)
    
    all_Observations_DayByDay=[]
    for i,k in zip(all_Perceptual_Thresholds,np.arange(len(all_Perceptual_Thresholds))):
        current_Target = i
        all_Observations_DayByDay.append([np.abs(j-current_Target) for j in all_Perceptual_Thresholds[:k+1]])
        

    return all_Perceptual_Thresholds, all_Observations_DayByDay

def define_AS(dataFrame_all):
    # This function defines all the AS used for a certain patient
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    # OUTPUT: couplesEL_AS: list of tuples in the shape of for ex. ('electrode_1','AS_3')
    
    couplesEL_AS=[]
    for idx in dataFrame_all.index:
        couplesEL_AS.append(tuple((dataFrame_all['EL'][idx],dataFrame_all['AS'][idx])))
            
    couplesEL_AS= list(set(couplesEL_AS))
    
    return couplesEL_AS

def actualval(x_next,targetQ):
    # This function computes the difference between the charge tried (x_next) and the actual target
    # INPUT: x_next: float indicating the actual charge value to try 
    #        targetQ: float indicating target perceptual threshold charge
    # OUTPUT: y_next: vector containing the number of iterations needed to find each target pattern throughout all the days (concatenated)
    
    
    y_next  = x_next-targetQ
            
    return y_next


def run_BO(BOStep):
    # This function applies the GPBO algorithm and computes the needed iterations to find the target patterns
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        targetPatterns: vector containing the target patterns (as centroid on the y-axis)
    # OUTPUT: iterations_gpbo: vector containing the number of iterations needed to find each target pattern throughout all the days (concatenated)
    #         results_gpbo: DataFrame containing for each day the number of queries needed to find each target pattern
    
    
    BOStep.run_optimization(max_iter=20)
    ins = BOStep.get_evaluations()[1].flatten()
    outs = BOStep.get_evaluations()[0].flatten()
    
    kk = 0
    realOuts = [outs[0]]
    realIns=[ins[0]]
    for outV,inV in  zip(outs[1:],ins[1:]):
        if outV not in outs[:kk+1]:
            realOuts.append(outV)
            realIns.append(inV)
        kk = kk+1
        
    realOuts = np.array(realOuts)
    realIns = np.array(realIns)
    
    return realOuts, realIns

def obj_func(x):
    # This function computes the y-value (i.e. output) based on x, the input (intended as the difference with respect to the perceptual threshold)
    # INPUT: x: float, numeric value indicating the difference between the actual tried charge and the threshold
    # OUTPUT: out: float, numeric value indicating the output 

    if x < -5:
        out = 0
    elif x <= 5 and x >= -5:
        out = -1 
    else:
        out= np.abs(x)
        
    return(out)


def define_model(X,y,kernel,gaussian_noise):
    # This function defines the gaussian process prior based on a regression on the existing data (x,y)
    # INPUT: X: numpy array containing the x-values (i.e., the charges )
    #        y:numpy array containing the y-values (i.e., the reported sensation based on the objective function )
    #       kernel: kernel of the gaussian prior distribution
    #       gaussian_noise: noise variance 
    # OUTPUT: m1 : instance of models.gp_regression.GPRegression
    
    
    m1 = GPy.models.GPRegression(X,y, kernel)
    m1.Gaussian_noise[:] = gaussian_noise

    return m1