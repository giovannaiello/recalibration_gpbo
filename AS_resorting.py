#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These functions are needed to resort the AS according to the location they elicited
on the first mapping.
"""
import pandas as pd
import numpy as np
from natsort import natsorted



def orderAccordingToCentroid(raw_data,sbjTime,sbjTimeRemove):
    # This function re-sorts the AS as explained in the main text of the paper.
    # INPUT: raw_data; DataFrame containing the fields present in the toy dataset "raw_data_toy".
    #        sbjTime;  numeric value indicating the day before which the dataset represents the first mapping.
    #                   Indeed, the first mapping might not have been concluded on one day and ended on the day after, for instance.
    # OUTPUT: processed_data; DataFrame containing the fields present in the toy dataset "processed_data_toy".
    
    whereRemove = raw_data.loc[raw_data['Day_Nb']<sbjTimeRemove]
    X = raw_data.drop(whereRemove.index)
    

    raw_data_centroid = defineCentroids(X) # defining the centroids 
    raw_data_firstMapping = defineFirstMapping(raw_data_centroid,sbjTime) # defining the first mapping data
    
    raw_data_firstMapping = raw_data_firstMapping.sort_values('Cy', axis=0, ignore_index=True) # sorting the data from the first mapping according to the elicited location (on the y-axis)

    couplesEL_AS=[] # defining all the AS used in the first mapping, and giving them an absolute ordinal index based on which location they elicit
    k = 0
    ordinal_AS_list=[]
    for idx in raw_data_firstMapping.index:
        couplesEL_AS.append(tuple((raw_data_firstMapping['EL'][idx],raw_data_firstMapping['AS'][idx]))) # in the form ('electrode_1','AS_3')
        ordinal_AS_list.append(k)
        k = k+1
        
    raw_data_firstMapping['AS_ABS']=ordinal_AS_list # adding a new column based on the AS re-sorting
    
    # Based on the sorting made during the first mapping, a new absolute index is given to the AS throughout the follow-up period.
    raw_data_followUp = raw_data_centroid.loc[raw_data_centroid['Day_Nb']>sbjTime]
    
    
    # Similarly to what was done for the first mapping, we average stimulations made with the same AS on the same day
    data_followUp=[]
    for day in natsorted(list(set(raw_data_followUp['Day_Nb']))):
        currentDay = raw_data_followUp.loc[raw_data_followUp['Day_Nb']==day]
        
        couplesEL_AS=[]
        for idx in currentDay.index:
            couplesEL_AS.append(tuple((currentDay['Electrode'][idx],currentDay['AS'][idx])))
                
        couplesEL_AS= list(set(couplesEL_AS))
        
        # One dataframe for each day
        X_thisDay = pd.DataFrame(index=np.arange(len(couplesEL_AS)),columns=['DAY','EL','AS','Charge','Cy'])
        
        idx=0
        for element in couplesEL_AS:
            current_AS_df = currentDay.loc[(currentDay['Electrode']==element[0])&(currentDay['AS']==element[1])]
            meanCharge = np.mean(current_AS_df['Charge'])
            meanCy = np.mean(current_AS_df['Cy'])
            
            X_thisDay['DAY'][idx]=day
            X_thisDay['EL'][idx]=element[0]
            X_thisDay['AS'][idx]=element[1]
            X_thisDay['Charge'][idx]=np.round(meanCharge)
            X_thisDay['Cy'][idx]=meanCy
            idx = idx + 1
            
        data_followUp.append(X_thisDay) 
    
    data_followUp = pd.concat(data_followUp,ignore_index=True) # concatenating all days

    
    # At this point we also add the absolute AS ordinal index, based on the first mapping
    AS_ABS=[]
    data_followUp['AS_ABS']=200*np.ones(len(data_followUp)) # we initialize it as a vector of 200
    for idx in data_followUp.index:
        currentEL = data_followUp['EL'][idx]
        currentAS = data_followUp['AS'][idx]
                
        for idx2 in raw_data_firstMapping.index:
            if (raw_data_firstMapping['EL'][idx2] == currentEL) and (raw_data_firstMapping['AS'][idx2]==currentAS):
                AS_ABS.append(raw_data_firstMapping['AS_ABS'][idx2])
    
                data_followUp['AS_ABS'][idx]=raw_data_firstMapping['AS_ABS'][idx2]
                
    data_followUp = data_followUp[data_followUp.AS_ABS != 200]   # removing those AS that were not present in the first characterization
    data_followUp = data_followUp.reset_index()
    
    # finally, we concatenate the data from the first mapping with the data from the follow up
    processed_data = pd.concat([raw_data_firstMapping,data_followUp],ignore_index=True)
    
    return processed_data

#%%

def defineCentroids(raw_data):
    # This function defines the centroid of the elicited location based on what is indicated in the raw data.
    # More specifically, based on which areas were selected by the subject, computes the average position
    # on the x-axis and the y-axis. Please notice that, for the leg, no centroid was indicated, thus the position was set at 1.5.
    # Also, everything was normalized, based on the size of the image. In this way, for instance, the y-axis centroid goes from 0 to 1,
    # indicating respectively the front to the heel of the foot. The x-axis centroid goes from 0 to 1, indicating
    # the lateral to medial part of the foot.
    # INPUT: raw_data; DataFrame containing the fields present in the toy dataset "raw_data_toy".
    # OUTPUT: raw_data; DataFrame containing additionally the centroid in the x-axis and the y-axis
    
    centroid_x=[]
    centroid_y=[]
    for idx in raw_data.index:
        if not np.isnan(raw_data['Centroid'][idx]).all(): # foot sensation contains specific centroid information
            centroid_x.append(np.mean([raw_data['Centroid'][idx][i][0] for i in np.arange(len(raw_data['Centroid'][idx]))])/250)
            centroid_y.append(np.mean([raw_data['Centroid'][idx][i][1] for i in np.arange(len(raw_data['Centroid'][idx]))])/450)
        else: # leg information does not include centroid info
            centroid_x.append(1.50)
            centroid_y.append(1.50)
          
    raw_data['Cx']=centroid_x
    raw_data['Cy']=centroid_y
    
    
    return raw_data

#%%

def defineFirstMapping(raw_data, sbjTime):
    # This function is needed in order to resort the AS, which are based on the locations elicited
    # on the first mapping. 
    # The first mapping was defined empirically, on a subject-dependent level. Indeed, each subject performed such mapping on a different day post-implantation.
    
    # INPUT: raw_data; DataFrame containing the fields present in the toy dataset "raw_data_toy".
    #        sbjTime;  numeric value indicating the day before which the dataset represents the first mapping.
    #                   Indeed, the first mapping might not have been concluded on one day and ended on the day after, for instance.
    # OUTPUT: raw_data_firstMapping; DataFrame containing a subset of the whole dataset referring to the first mapping. Only the fields of interest (below) are reported. 
    
    raw_data_subset = raw_data.loc[(raw_data['Day_Nb']<sbjTime)]
    
    couplesEL_AS=[] # ideally, all the AS should have been tried on the first mapping
    for idx in raw_data_subset.index:
        couplesEL_AS.append(tuple((raw_data_subset['Electrode'][idx],raw_data_subset['AS'][idx])))       
    couplesEL_AS= list(set(couplesEL_AS)) # list of tuples, for instance ('electrode_1','AS_3')
    
    # We create at this point a new dataframe containing only one information per AS.
    # For instance, if during the first mapping, an AS was activated more times (which is usually the procedure to
    # guarantee robustness of the output sensation), we average across these outputs to obtain only one value.
    # Also, the new DataFrame only contains the important information (Day, Electrode, AS, Charge, Centroid on the y-axis)
    raw_data_firstMapping = pd.DataFrame(index=np.arange(len(couplesEL_AS)),columns=['DAY','EL','AS','Charge','Cy'])
    
    idx=0
    for el in couplesEL_AS:
        current_AS_df = raw_data_subset.loc[(raw_data_subset['Electrode']==el[0])&(raw_data_subset['AS']==el[1])] # all the stimulations made with that specific AS
        
        meanCy = np.mean(current_AS_df['Cy']) # average location elicited by that AS in the first mapping
        meanCharge = np.mean(current_AS_df['Charge']) # average charge needed for the perceptual threshold
        
        raw_data_firstMapping['DAY'][idx]=1 # Regardless of which day it is, we always set it as 1 (first mapping day)
        raw_data_firstMapping['EL'][idx]=el[0]
        raw_data_firstMapping['AS'][idx]=el[1]
        raw_data_firstMapping['Charge'][idx]=np.round(meanCharge)
        raw_data_firstMapping['Cy'][idx]=meanCy
        idx = idx + 1
        
    return raw_data_firstMapping