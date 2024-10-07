import numpy as np
import scipy.io
import h5py

def mainWorkflow_Data_generation1():
    #===================================================================================================================================
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 15 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow_Data_generation1
    #  Subject      : allows to generate data for the Benchmark3_Direct_NoPartition_PCE_Zwiener
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
            MatRxx_data= file['MatRxx_data'][:].T

    if indv73 == 2: # Case of a matlab file.mat saved without using -v7.3 option 
        data = scipy.io.loadmat('FileMatRxxData.mat')                  # Load MatRxx_data from a file.mat saved without using -v7.3 option   
        MatRxx_data = data['MatRxx_data']  

    # Name               Size       Bytes  
    # MatRxx_data      40x400       128000  

    n_d = 400
    n_q = 35
    n_w = 5
    n_x = n_q + n_w  # n_x = 40

    #--- loading MatRqq_d(n_q,n_d) and MatRww_d(n_w,n_d)
    MatRqq_d = MatRxx_data[0:35,:]   # MatRqq_d(n_q,n_d)
    MatRww_d = MatRxx_data[35:40,:]  # MatRww_d(n_w,n_d)

    #--- loading MatRxx_d(n_x,n_d) for PLoM that is constructed with the convention RXX = (RQQ,RWW)
    MatRxx_d = np.vstack([MatRqq_d,MatRww_d])  # MatRqq_d(n_q,n_d) and MatRww_d(n_w,n_d)

    #--- loading Indq_real(nqreal) and Indq_pos(nqpos)
    Indq_real = np.arange(1,36)                # loading the integers: 1,2, ...,35
    Indq_pos  = np.array([])

    #--- loading Indw_real(nwreal) and Indw_pos(nwpos)
    Indw_real = np.arange(1,6)                 # loading the integers: 1,2,3,4,5
    Indw_pos  = np.array([])

    #--- loading Indq_obs(nq_obs) and Indw_obs(nw_obs)
    Indq_obs = np.array([1,2,3,4,35])          # nq_obs component numbers 1,2,3,4,35 of QQ that are observed , 1 <= nq_obs <= n_q
    Indw_obs = np.array([1,5])                 # nw_obs component numbers 1,5 of WW that are observed,  1 <= nw_obs <= n_w
    
    return n_q, Indq_real, Indq_pos, n_w, Indw_real, Indw_pos, Indq_obs, Indw_obs, n_d, MatRxx_d
