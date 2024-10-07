import numpy as np
import scipy.io
import sys
import h5py

def mainWorkflow_Data_generation1():
    #===================================================================================================================================
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 11 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow_Data_generation1
    #  Subject      : allows to generate data for the Benchmark4_Direct_NoPartition_CondStat_ConfRegion
    #
    #
    #--- OUTPUT (data that must be generated, which are application dependent) 
    #
    #          n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q     
    #          Indq_real(nqreal)      : contains the nqreal component numbers of QQ, which are real (positive, negative, or zero) 
    #                                   with 0 <= nqreal <=  n_q and for which a "standard scaling" will be used
    #          Indq_pos(nqpos)        : contains the nqpos component numbers of QQ, which are strictly positive a "specific scaling"
    #                                   with  0 <= nqpos <=  n_q  and for which the scaling is {log + "standard scaling"}
    #                                   --- we must have n_q = nqreal + nqpos  
    #
    #          n_w                    : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
    #          Indw_real(nwreal)      : contains the nwreal component numbers of WW, which are real (positive, negative, or zero) 
    #                                   with 0 <= nwreal <=  n_w and for which a "standard scaling" will be used
    #          Indw_pos(nwpos)        : contains the nwpos component numbers of WW, which are strictly positive a "specific scaling"
    #                                   with  0 <= nwpos <=  n_w  and for which the scaling is {log + "standard scaling"}
    #                                   --- we must have n_w = nwreal + nwpos 
    #
    #          Indq_obs(nq_obs)       : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
    #          Indw_obs(nw_obs)       : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
    #                                   
    #                     WARNING: if Inverse Analysis is considered (ind_workflow = 3) with the option
    #                              ind_type_targ = 2 or 3, then all the components of XX and XX_targ
    #                              must be considered as real even if some components are positive. When thus must have:
    #                                    nqreal     = n_q and nwreal     = n_w
    #                                    nqpos      = 0   and nwpos      = 0
    #                                    nqpos_targ = 0   and nwpos_targ = 0
    #
    #                     WARNING: For the analysis of the conditional statistics of Step4, the organization of the components of the 
    #                              QQ vector of the quantity of interest QoI is as follows (this organization must be planned from the 
    #                              creation of the data in this function "mainWorkflow_Data_generation1.m" and  also in
    #                              "mainWorkflow_Data_generation2.m" .
    #
    #                              If the QoI depends on the sampling in nbParam points of a physical system parameter
    #                              (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
    #                              f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
    #                              as follows: 
    #                              [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
    #
    #                     WARNING: If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
    #                              of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be 
    #                              an integer, if not there is an error in the given value of nbParam of in the Data generation in 
    #                              "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 
    #
    #                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the 
    #                              information structure must be consistent with the case nbParam > 1.  
    #
    #          n_d                   : number of points in the training set for XX_d and X_d
    #          MatRxx_d(n_x,n_d)     : n_d realizations of random vector XX_d (unscale) with dimension n_x = n_q + n_w
    #
    #                                  WARNING: Matrix MatRxx_d(n_x,n_d) musy be coherent with the following construction 
    #                                           of the training dataset:
    #                                                                   MatRxx_d = [MatRqq_d     % MatRqq_d(n_q,n_d) 
    #                                                                               MatRww_d];   % MatRww_d(n_w,n_d) 
    #======================================================================================================================================
    
    #--- load data
    indv73 = 2  # 1: Case of a matlab file.mat saved using -v7.3 option 
                # 2: Case of a matlab file.mat saved without using -v7.3 option 

    if indv73 == 1: # Case of a matlab file.mat saved with -v7.3 option         
        with h5py.File('FileMatRxxData.mat', 'r') as file:     # Load MatRxx_data from a file.mat saved with -v7.3 option 
            MatRqq_data= file['MatRqq_data'][:].T
            MatRww_data= file['MatRww_data'][:].T

    if indv73 == 2: # Case of a matlab file.mat saved without using -v7.3 option 
        data        = scipy.io.loadmat('FileMatRxxData.mat')  # Assuming the file has a .mat extension
        MatRqq_data = data['MatRqq_data']
        MatRww_data = data['MatRww_data']   # Loading the MatRqq_data and MatRwwdata from the file

    # Name               Size       Bytes  
    # MatRqq_data      400x200     640000             
    # MatRww_data        2x200       3200 

    n_d = 200
    n_q = 400
    n_w = 2
    n_x = n_q + n_w  # n_x = 402

    #--- loading MatRqq_d(n_q,n_d) and MatRww_d(n_w,n_d)
    
    MatRqq_d = MatRqq_data   # MatRqq_d(n_q,n_d)
    MatRww_d = MatRww_data   # MatRww_d(n_w,n_d)
    
    #--- loading MatRxx_d(n_x,n_d) for PLoM that is constructed with the convention RXX = (RQQ,RWW)
    MatRxx_d = np.vstack([MatRqq_d,MatRww_d])  # MatRqq_d(n_q,n_d) and MatRww_d(n_w,n_d)
    
    # --- loading Indq_real(nqreal,1) and Indq_pos(nqpos,1)
    Indq_real = np.arange(1,401)                          # loading the integers: 1,2, ...,400 in a 1D array
    Indq_pos  = np.array([])                              # if no positive-valued components, enter an empty 1D array

    # --- loading Indw_real(nwreal,1) and Indw_pos(nwpos,1)
    Indw_real = np.array([])                             # if no real-valued components, enter an empty 1D array
    Indw_pos  = np.arange(1,3)                           # loading the integers: 1,2 in a 1D array

    # --- loading Indq_obs(nq_obs,1) and Indw_obs(nw_obs,1)
    Indq_obs = np.concatenate([np.arange(1, 101), np.arange(201, 301)])  # nq_obs component numbers of QQ that are observed, 1 <= nq_obs <= n_q
                                 # loading the integers 1, 2, 3, ..., 100, 201, 202, ..., 300
                                #  if no components of q are observed, enter the empty 1D array  Indq_obs = np.array([])      
    Indw_obs = np.arange(1,3)   # nw_obs component numbers of WW that are observed, 1 <= nw_obs <= n_w
                                 # loading the integers: 1,2 in a 1D array
                                 #  if no components of w are observed, enter the empty 1D array  Indw_obs = np.array([])                       
   
    return n_q, Indq_real, Indq_pos, n_w, Indw_real, Indw_pos, Indq_obs, Indw_obs, n_d, MatRxx_d
