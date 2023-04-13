#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
These functions are needed to run the GPBO and the reset-GPBO algorithm, and compare them to a random search in location search.
"""


import GPyOpt
import random
import numpy as np
import pandas as pd
import GPy
from natsort import natsorted
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannot import add_stat_annotation
import seaborn as sns
from matplotlib.gridspec import GridSpec



def run_all_models(dataFrame_all,targetPatterns):
    # This function applies the three models (Random Search, reset-GPBO), show results and compares them statistically
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        targetPatterns: vector containing the target patterns (as centroid on the y-axis)
    # OUTPUT: results_struct: dictionary containing results for each model
    
    
    results_struct={}
    iterations_gpbo,results_gpbo,AS_Chosen = GPBO_loc(dataFrame_all,targetPatterns)
    iterations_reset_gpbo,results_reset_gpbo = resetGPBO_loc(dataFrame_all,targetPatterns)
    iterations_random_search,results_random_search = randomSearch_loc(dataFrame_all, targetPatterns)
    
    results_struct['Random Search'] = iterations_random_search
    results_struct['reset-GPBO'] = iterations_reset_gpbo
    results_struct['GPBO'] = iterations_gpbo
    
    dataFrame_all_mappings = dataFrame_all.loc[dataFrame_all['DAY']>1]
    days = natsorted(list(set(dataFrame_all_mappings['DAY'])))
    # 30 repetitions, 3 methods 
    k = max(results_random_search['Repetition'])
    dataFrameToTest = pd.DataFrame({'Day': np.repeat(days, k*3), 'Method': np.tile(['Random Search','reset-GPBO','GPBO'], k*len(days)),'Iterations': [results_struct[method][r*len(days)+i]  for i in np.arange(len(days)) for r in np.arange(k) for method in ['Random Search','reset-GPBO','GPBO']]})
    TKHSD = pairwise_tukeyhsd(dataFrameToTest['Iterations'], dataFrameToTest['Method'])

    TKHSD_LOC = TKHSD
    
    plotResults(days,dataFrameToTest,results_struct,TKHSD_LOC,AS_Chosen,results_gpbo,results_reset_gpbo,results_random_search)

    
    return results_struct

def plotResults(days,dataFrameToTest,results_struct,TKHSD_LOC,AS_Chosen,results_gpbo,results_reset_gpbo,results_random_search):
    # This function plots panel 4F, 4E and 4G. 
   
    
    df_to_plot = pd.DataFrame([results_struct['Random Search'],results_struct['reset-GPBO'],results_struct['GPBO']]).transpose()
    df_to_plot = df_to_plot.rename(columns={0:'Random Search', 1:'reset-GPBO',2:'GPBO'})
    
    fig = plt.figure(figsize=(14,10))
    fig.tight_layout()
    gs=GridSpec(2,4) # 2 rows, 3 columns
    ax1=fig.add_subplot(gs[0,:4]) # First row, first column
    ax2=fig.add_subplot(gs[1,:2]) # First row, second column
    ax3=fig.add_subplot(gs[1,2:4]) # First row, third column


    #fig,ax=plt.subplots(1,figsize=(5,10))
    sns.boxplot(ax=ax2,x="variable", y="value", data=pd.melt(df_to_plot),palette="BuGn")
    ax2.set_xlabel(' ',fontsize=15)
    ax2.set_ylabel('Number of Query Points',fontsize=15)
    couples=[('Random Search','reset-GPBO'),('Random Search','GPBO'),('reset-GPBO','GPBO')]
    ax2.set_xticklabels(['Random Search','reset-GPBO','GPBO'],fontsize=16,rotation=45)
    add_stat_annotation(ax2, data=pd.melt(df_to_plot), x='variable', y='value',
                        box_pairs=couples,
                        perform_stat_test=False, pvalues=TKHSD_LOC.pvalues, text_format='star',
                        verbose=2,comparisons_correction='bonferroni',fontsize='x-large',linewidth=3)
    
    ALL_AS=np.zeros((len([0,0.5,1,1.5]),len(set(results_gpbo['Day']))))

    for targetPattern,i in zip([0,0.5,1,1.5],np.arange(len([0,0.5,1,1.5]))):
        for day,j in zip(natsorted(list(set(results_gpbo['Day']))),np.arange(len(set(results_gpbo['Day'])))):
            current = results_gpbo.loc[(results_gpbo['Day']==day)&(results_gpbo['TargetPattern']==targetPattern)&(results_gpbo['Repetition']==1)]
            
            if len(current)>0:
                ALL_AS[i,j]=current['AS']
            else:
                ALL_AS[i,j]=np.nan
    ASChosen_Array = np.vstack(AS_Chosen)
    final_AS_chosen = np.zeros((len(days),len(np.arange(ALL_AS.shape[0]))))
    for i in np.arange(ALL_AS.shape[0]):
        current_loc = ASChosen_Array[:,i]
        days = natsorted(list(set(results_gpbo['Day'])))
        for j in np.arange(len(days)):
            AS_chosen = (current_loc[range(j, int(len(current_loc)/2), 2)])
            
            AS_chosen = [int(k) for k in AS_chosen]
            counts = np.bincount(AS_chosen)
            
            final_AS_chosen[j,i] = np.argmax(counts)

    colors = ['limegreen','seagreen','palegreen','olivedrab']
    for i,lab,c in zip(np.arange(ALL_AS.shape[0]),['Front','Mid','Heel','Leg'],colors):
        ax3.plot(final_AS_chosen[:,i],'o-',c=c,label=lab)
        ax3.legend()
        #plt.yticks([0,5,10,15,20,25,30,35,40])
        ax3.set_xlabel('Characterization Days',fontsize=14)
        ax3.set_ylabel('Chosen AS',fontsize=14)
        ax3.set_yticks([0,5,10,15,20,25,30,35,40])
        ax3.set_xticks(np.arange(len(set(results_gpbo['Day']))))
        ax3.set_xticklabels(set(results_gpbo['Day']),fontsize=14)
        
    ax3.set_title('Chosen AS for Each Pattern on different CD',fontsize=16)
    
    
    sns.barplot(ax=ax1,x="Day",y="Iterations",hue="Method",data=dataFrameToTest,palette="BuGn")
    ax1.set_ylabel('Number of Query Points',fontsize=15)
    ax1.set_xlabel(' ',fontsize=15)
    ax1.set_xticklabels(set(results_gpbo['Day']),fontsize=15)
    #plt.show()
    plt.savefig('Figures/Results_Location.png')
    plt.show()
def GPBO_loc(dataFrame_all,targetPatterns):
    # This function applies the GPBO algorithm and computes the needed iterations to find the target patterns
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        targetPatterns: vector containing the target patterns (as centroid on the y-axis)
    # OUTPUT: iterations_gpbo: vector containing the number of iterations needed to find each target pattern throughout all the days (concatenated)
    #         results_gpbo: DataFrame containing for each day the number of queries needed to find each target pattern
    
    iterations_gpbo = []
    allresults=[]
    AS_Chosen=[]

    for rep in np.arange(30): # repetitions for robustness 
        X_firstMapping = dataFrame_all.loc[dataFrame_all['DAY']==1] # selecting first mapping as collected data points 
        firstMapping_AS = X_firstMapping[['AS_ABS']]
        stimsToTrain = list(X_firstMapping.index) # all stims used in first mapping
       
        X_step = firstMapping_AS.to_numpy() #AS to consider
        
        for day in natsorted(list(set(dataFrame_all['DAY'])))[1:]: #for each day following the first mapping (i.e., follow up)
            thisDayData = dataFrame_all.loc[dataFrame_all['DAY']==day]
            thisDay_AS = thisDayData[['AS_ABS']]
            
            X_test = thisDay_AS
            thisDayStimsToTrain=[] 
            thisDayAS=[]
            nbIterationsEachWeek= 0
            AS_Chosen_This_Day=[]
            for targetPattern in targetPatterns:
                currentf = objFunc(np.array(dataFrame_all['Cy']),targetPattern)
                fThisDay = list(currentf.loc[thisDayData.index])
                f = currentf - np.min(fThisDay)
                fToTest = currentf - np.min(fThisDay)
                
                f_step_df = f.loc[stimsToTrain]
                f_test_df = fToTest.loc[X_test.index]
                
                f_already_tried = list(f.loc[thisDayStimsToTrain])
                f_step = list(f_step_df)
                f_test = list(f_test_df)
              
                count_iter=0
                max_iter = 20
                
                AS_tried=[]
                all_stims=[]
                if all(i >0.15 for i in f_already_tried):
                    if len(f_test_df) > 0:
                        percentage_successful_stim = f_test.count(0)*100/len(f_test_df)
                        if percentage_successful_stim > 0:
                            yvalues=[]
                            f_next = 1
                            while (count_iter < max_iter) and f_next>0.15: 
                                x_next,f_next,what_stim,BO_Step = perform_search(X_test,f_test_df,X_step,f_step)
                                yvalues.append(f_next)
                                stimsToTrain.append(what_stim)
                                thisDayStimsToTrain.append(what_stim)
                                
                                
                                X_step = np.vstack((X_step,x_next))
                                f_step.append(f_next)
                                whatAS = x_next[0]
                                thisDayAS.append(whatAS)

                                X_test = X_test.drop(what_stim)
                                f_test_df = f_test_df.drop(what_stim)
                                
                                AS_tried.append(whatAS)
                                all_stims.append(what_stim)
                                count_iter = count_iter +1

                            nbIterationsEachWeek = nbIterationsEachWeek + count_iter
                            allresults.append(tuple((rep,'GPBO',day,targetPattern,x_next[0],count_iter)))
                            AS_Chosen_This_Day.append(x_next[0])
                else: # if that location was already found while looking for another location
                    nbIterationsEachWeek = nbIterationsEachWeek + 0
                    
                    AS_Chosen_This_Day.append(thisDayAS[np.where(np.array(f_already_tried)<0.15)[0][0]])
                
            AS_Chosen.append(AS_Chosen_This_Day)
            iterations_gpbo.append(nbIterationsEachWeek)
    results_gpbo = pd.DataFrame(allresults,columns=['Repetition','Method','Day','TargetPattern','AS','NbQueries'])
    
    return iterations_gpbo,results_gpbo,AS_Chosen





def resetGPBO_loc(dataFrame_all,targetPatterns):
    # This function applies the GPBO algorithm and computes the needed iterations to find the target patterns
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        targetPatterns: vector containing the target patterns (as centroid on the y-axis)
    # OUTPUT: iterations_reset_gpbo: vector containing the number of iterations needed to find each target pattern throughout all the days (concatenated)
    #         results_reset_gpbo: DataFrame containing for each day the number of queries needed to find each target pattern
    
    df_mappings = dataFrame_all.loc[dataFrame_all['DAY']>1]
    df_mappings_reset = df_mappings.reset_index()
    iterations_reset_gpbo = []    
    allresults=[]
    random.seed(999)
    for rep in np.arange(30):
        
        for day in natsorted(list(set(df_mappings_reset['DAY']))):
            
            thisWeekStimsToTrain=[]
            
            thisWeekData = df_mappings_reset.loc[df_mappings_reset['DAY']==day]
            thisWeekData_dropCol = thisWeekData[['AS_ABS']]
            
            randomInitialPoints = random.sample(list(thisWeekData_dropCol.index), 3)

            stimsToTrain = randomInitialPoints.copy()
            
            X_step_df = thisWeekData_dropCol.loc[randomInitialPoints]
            X_step = X_step_df.to_numpy() 
            X_test = thisWeekData_dropCol.drop(X_step_df.index) 
                
            nbIterationsEachWeek= 3 # three stimulations were performed randomly to initiate the gaussian process
            for targetPattern in targetPatterns:
                currentf = objFunc(np.array(df_mappings_reset['Cy']),targetPattern)
                fOfThisWeek = list(currentf.reindex(index=thisWeekData.index))
                f = currentf - np.min(fOfThisWeek)
                fToTest = currentf - np.min(fOfThisWeek)
                f_step_df = f.reindex(index=stimsToTrain)
                f_test_df = fToTest.reindex(index=X_test.index)
                
                f_already_tried = list(f.loc[thisWeekStimsToTrain])
                f_step = list(f_step_df)
                f_test = list(f_test_df)  
              
                count_iter=0
                max_iter = len(f_test_df)
                
                
                if all(i >0.15 for i in f_already_tried):
                    if len(f_test_df) > 0:
                        percentage_successful_stim = f_test.count(0)*100/len(f_test_df)
                        if percentage_successful_stim > 0:
                        
                            yvalues=[]
                            f_next = 1
                            while (count_iter < max_iter) and f_next>0.15 : 
                                x_next,f_next,what_stim,BOStep = perform_search(X_test,f_test_df,X_step,f_step)
                              
                                yvalues.append(f_next)
                                stimsToTrain.append(what_stim)
                                thisWeekStimsToTrain.append(what_stim)
                    
                                
                                X_step = np.vstack((X_step,x_next))
                                f_step.append(f_next)
                                
                                X_test = X_test.drop(what_stim)
                                f_test_df = f_test_df.drop(what_stim)
                                
                                
                                count_iter = count_iter +1

                                
                            nbIterationsEachWeek = nbIterationsEachWeek + count_iter
                            allresults.append(tuple((rep,'reset-GPBO',day,targetPattern,count_iter)))
                else:
                    nbIterationsEachWeek = nbIterationsEachWeek + 0
            
            iterations_reset_gpbo.append(nbIterationsEachWeek)
        
                
    results_reset_gpbo = pd.DataFrame(allresults,columns=['Repetition','Method','Day','TargetPattern','NbQueries'])
    
    return iterations_reset_gpbo,results_reset_gpbo




def randomSearch_loc(dataFrame_all, targetPatterns):

    # This function computes the objective function to minimize.
    # INPUT: dataFrame_all: dataFrame containing all the mappings. As in the toy dataset "processed_data_toy"
    #        targetPatterns: vector containing the target patterns (as centroid on the y-axis)
    # OUTPUT: iterations_random_search: vector containing the number of iterations needed to find each target pattern throughout all the days (concatenated)
    #         results_random_search: DataFrame containing for each day the number of queries needed to find each target pattern
    df_mappings = dataFrame_all.loc[dataFrame_all['DAY']>1]
    df_mappings_reset = df_mappings.reset_index()
    list_all_values=[]
    iterations_random_search= []
    all_results=[]
    random.seed(999)
    for rep in np.arange(30): # repeating the process 30 times for robustness
        for day in natsorted(list(set(df_mappings_reset['DAY']))): # for each mapping day 
            
            nbIterationsToFindBothPatterns=0 # we start at zero iterations
            thisDayStimsToTrain=[]
            
            thisDayData = df_mappings_reset.loc[df_mappings_reset['DAY']==day] #reducing to each mapping day
            thisDayData_dropCol = thisDayData[['AS_ABS']]

            stimsToTrain=[]
            X_step_df = df_mappings_reset.loc[stimsToTrain] 
            X_step_df = X_step_df[['AS_ABS']]
            X_step = X_step_df.to_numpy()
            X_test = thisDayData_dropCol
            
            for targetPattern in targetPatterns: # for each target pattern
                
                
                currentf = objFunc(np.array(df_mappings_reset['Cy']),targetPattern) # objective function for each points 
                f_thisDay = list(currentf.reindex(index=thisDayData.index))
                f = currentf - np.min(f_thisDay) # normalizing with respect to the min possible 
                fToTest = currentf - np.min(f_thisDay) 
                f_step_df = f.loc[stimsToTrain]
                f_test_df = fToTest.reindex(index=X_test.index)
                
                f_already_tried = list(f.reindex(index=thisDayStimsToTrain))
                f_step = list(f_step_df)
                f_test = list(f_test_df)        
              
                count_iter=0
                max_iter = 20 
                
                if all(i >0.15 for i in f_already_tried): # tolerance of 0.15 distance in centroid with respect to the optimal value for that day 
                    percentage_successful_stim = f_test.count(0)*100/len(f_test_df)
                    if percentage_successful_stim > 0: # if there is at least one value that is optimal
                    
                        yvalues=[]
                        f_next = 1
                        while count_iter < max_iter and f_next>0.15 : #Maximum 30 iterations
                            
                            what_stim = random.sample(list(X_test.index),1)[0] # randomly taking one stimulation out of the test set 
                            x_next = X_test.loc[what_stim] # next stim  (AS) to try 
                            f_next = f_test_df[what_stim] # objective func of that stim 
                          
                            yvalues.append(f_next) # concatenating to the tried ones 
                            
                            stimsToTrain.append(what_stim)
                            thisDayStimsToTrain.append(what_stim)
                            X_step = np.vstack((X_step,x_next)) # step is now made of previous step and current one (i.e., tried ones)
                            f_step.append(f_next)
                            X_test = X_test.drop(what_stim) # removing ones already tried from the space of possible choices 
                            f_test_df = f_test_df.drop(what_stim)
                            
                            
                            count_iter = count_iter+1 # increasing the count of needed stims 
                            list_all_values.append([rep,day,targetPattern,x_next[0],f_next])
                            
                        nbIterationsToFindBothPatterns = nbIterationsToFindBothPatterns + count_iter # they add up for all the targets 
                        all_results.append(tuple((rep,'RandomSearch',day,targetPattern,count_iter)))
                        
                else: # if that location was already found in the already tried stimulations (for another location, for instance)
                    nbIterationsToFindBothPatterns = nbIterationsToFindBothPatterns + 0 

            iterations_random_search.append(nbIterationsToFindBothPatterns)
                    
    results_random_search = pd.DataFrame(all_results,columns=['Repetition','Method','Day','TargetPattern','NbQueries'])
    
    return iterations_random_search,results_random_search


def perform_search(X_test,y_test_df,X_step,y_step):
    actual_domain = np.array([(X_test['AS_ABS'][i]) for i in X_test.index])
    domain = [{'name': 'stimParams','type':'discrete','domain':actual_domain,'dimensionality':1}]
    
    k = GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 10)
    m1 = GPy.models.GPRegression(X_step,np.array(y_step)[:,np.newaxis], k)
    m1.Gaussian_noise[:] = 0.5
    
    BOStep = GPyOpt.methods.BayesianOptimization(f=None,domain=domain,X=X_step,Y=np.array(y_step)[:,np.newaxis],acquisition_type='EI',kernel = k, model=m1,exploration_weight=2)
    x_next = BOStep.suggest_next_locations()[0]

    stim = X_test.loc[(X_test['AS_ABS']==x_next[0])]
    what_stim = stim.index[0]
    y_next = y_test_df[what_stim]
    
    return x_next,y_next,what_stim,BOStep


def objFunc(y,targetLoc): 
    # This function computes the objective function to minimize.
    # INPUT: y: vector, elicited locations (in y-axis centroid), of the tried AS
    #        targetLoc:  target centroid value based on the target patterns
    # OUTPUT: f: pandas Series with objective function value for each stimulation
    
    f=[]
    for element in y:
        f.append(np.abs(element-targetLoc))
    
    f = pd.Series(f)
    return f

    