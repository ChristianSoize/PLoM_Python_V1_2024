import numpy as np
import scipy.io as sio
import h5py

def mainWorkflow_Data_generation1():
    #===================================================================================================================================
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 14 September 2024 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM)
    #  Function name: mainWorkflow_Data_generation1
    #  Subject      : This function allows to generate data for the Benchmark2_Direct_NoPartition_PCE_QWU
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
    #                                  --- we must have n_w = nwreal + nwpos
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
    #                              creation of the data in this function "mainWorkflow_Data_generation1.m" and also in
    #                              "mainWorkflow_Data_generation2.m".
    #
    #                              If the QoI depends on the sampling in nbParam points of a physical system parameter
    #                              (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if
    #                              f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized
    #                              as follows:
    #                              [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
    #
    #                     WARNING: If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
    #                              of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be
    #                              an integer, if not there is an error in the Data generation in "mainWorkflow_Data_generation1.m" and
    #                              "mainWorkflow_Data_generation2.m"
    #
    #                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the
    #                              information structure must be consistent with the case nbParam > 1.
    #
    #
    #          n_d                     : number of points in the training set for XX_d and X_d
    #          MatRxx_d(n_x,n_d)       : n_d realizations of random vector XX_d (unscale) with dimension n_x = n_q + n_w
    #
    #                                  WARNING: Matrix MatRxx_d(n_x,n_d) must be coherent with the following construction
    #                                           of the training dataset:
    #                                                                   MatRxx_d = [MatRqq_d     % MatRqq_d(n_q,n_d)
    #                                                                               MatRww_d];   % MatRww_d(n_w,n_d)
    #======================================================================================================================================

    #--- load data generated on 17 Fev 2023 and corresponding to the paper:
    #    C. Soize, Q-D. To, Polynomial-chaos-based conditional statistics for probabilistic learning with heterogeneous
    #                       data applied to atomic collisions of Helium on graphite substrate, Journal of Computational Physics,
    #                       doi:10.1016/j.jcp.2023.112582, 496, 112582, pp.1-20 (2024),
    #    Warning: the parameterization and the PCE algorithm are not exactly the same as in the code used for generating the results
    #             of the reference above. Consequently, there are some small differences if results are compared
    #
    
    indv73 = 1  # 1: Case of a matlab file.mat saved using -v7.3 option 
                # 2: Case of a matlab file.mat saved without using -v7.3 option 

    if indv73 == 1: # Case of a matlab file.mat saved with -v7.3 option         
        with h5py.File('FileMatRxxData.mat', 'r') as file:     # Load MatRxx_temp_DATA from a file.mat saved with -v7.3 option 
            MatRxx_temp_DATA = file['MatRxx_temp_DATA'][:].T

    if indv73 == 2: # Case of a matlab file.mat saved without using -v7.3 option 
        data = sio.loadmat('FileMatRxxData.mat')                  # Load MatRxx_temp_DATA from a file.mat saved without using -v7.3 option   
        MatRxx_temp_DATA = data['MatRxx_temp_DATA']  
    
    # Name                Size        Bytes    
    # MatRxx_temp_DATA    11x27909    2455992  

    #--- load data parameters
    #    RX_temp_DATA = (Timein,Vxin,Vyin,Vzin,Timeout,Vxout,Vyout,Vzout,Depth,Dx,Dy)
    #    TEMPORARY LOADING FOR RX_temp because PLoM is constructed with the convention RXX = (RQQ,RWW)
    
    n_d_DATA = 1000  # Temporary number of realizations considered for constructing the training dataset
    n_x      = 9

    #--- constructing the training dataset MatRxx_temp_LOG(n_x,n_d) with n_d = 2*n_d_DATA for time reversal properties
    #    RX_temp_DATA  = (Timein,Vxin,Vyin,Vzin,Timeout,Vxout,Vyout,Vzout,Depth,Dx,Dy)
    #    RX_temp       = (Vxin,Vyin,Vzin,Vxout,Vyout,Vzout,Delta,Dx,Dy) with Delta = Timeout - Timein

    n_d         = 2*n_d_DATA           # Twice n_d_DATA due to the time reversibility property
    MatRxx_temp = np.zeros((n_x,n_d))  # MatRxx_temp(n_x,n_d)

    MatRxx_temp[0,:] = np.concatenate([MatRxx_temp_DATA[1,:n_d_DATA], -MatRxx_temp_DATA[5,:n_d_DATA]])  # Vxin -> -Vxout
    MatRxx_temp[1,:] = np.concatenate([MatRxx_temp_DATA[2,:n_d_DATA], -MatRxx_temp_DATA[6,:n_d_DATA]])  # Vyin -> -Vyout
    MatRxx_temp[2,:] = np.concatenate([MatRxx_temp_DATA[3,:n_d_DATA], MatRxx_temp_DATA[7,:n_d_DATA]])   # Vzin -> +Vzout

    MatRxx_temp[3,:] = np.concatenate([MatRxx_temp_DATA[5,:n_d_DATA], -MatRxx_temp_DATA[1,:n_d_DATA]])  # Vxout -> -Vxin
    MatRxx_temp[4,:] = np.concatenate([MatRxx_temp_DATA[6,:n_d_DATA], -MatRxx_temp_DATA[2,:n_d_DATA]])  # Vyout -> -Vyin
    MatRxx_temp[5,:] = np.concatenate([MatRxx_temp_DATA[7,:n_d_DATA], MatRxx_temp_DATA[3,:n_d_DATA]])   # Vzout -> +Vzin

    MatRDelta        = np.zeros((1,n_d_DATA))                                                           # Ensure MatRDelta is a 2D array like in MATLAB
    MatRDelta[0,:]   = np.log(MatRxx_temp_DATA[4,:n_d_DATA] - MatRxx_temp_DATA[0,:n_d_DATA])             # Log(Delta)
    MatRxx_temp[6,:] = np.concatenate([MatRDelta[0,:], MatRDelta[0,:]])                                  # Delta -> Delta
    MatRxx_temp[7,:] = np.concatenate([MatRxx_temp_DATA[9,:n_d_DATA], -MatRxx_temp_DATA[9,:n_d_DATA]])   # Dx -> -Dx
    MatRxx_temp[8,:] = np.concatenate([MatRxx_temp_DATA[10,:n_d_DATA], -MatRxx_temp_DATA[10,:n_d_DATA]]) # Dy -> -Dy

    #--- loading MatRww_d(n_w,n_d) and MatRqq_d(n_q,n_d)
    n_w = 3
    MatRww_d = MatRxx_temp[:3,:]    # MatRww_d(n_w,n_d)
    n_q = 6
    MatRqq_d = MatRxx_temp[3:9, :]  # MatRqq_d(n_q,n_d)

    #--- loading MatRxx_d(n_x,n_d) for PLoM that is constructed with the convention RXX = (RQQ,RWW)
    MatRxx_d = np.vstack((MatRqq_d, MatRww_d))  # MatRxx_d(n_x,n_d)
   
    #--- loading Indq_real(nqreal) and Indq_pos(nqpos)
    #    real     components of QQ = (Vxout,Vyout,Vzout,log(Delta),Dx,Dy): [1 2 4 5 6]
    #    positive components of QQ = (Vxout,Vyout,Vzout,log(Delta),Dx,Dy): [3]
    Indq_real = np.array([1, 2, 4, 5, 6])  # Real components of QQ
    Indq_pos  = np.array([3])              # Positive components of QQ

    #--- loading Indw_real(nwreal) and Indw_pos(nwpos)
    #    real     components of RWW = (Vxin,Vyin,Vzin): [1 2]
    #    positive components of RWW = (Vxin,Vyin,Vzin): [3]
    Indw_real = np.array([1, 2])  # Real components of RWW
    Indw_pos  = np.array([3])     # Positive components of RWW

    #--- loading Indq_obs(nq_obs) and Indw_obs(nw_obs)
    Indq_obs = np.array([1, 2, 3, 4, 5, 6])  # nq_obs component numbers of QQ that are observed
    Indw_obs = np.array([1, 2, 3])           # nw_obs component numbers of WW that are observed
    
    return n_q, Indq_real, Indq_pos, n_w, Indw_real, Indw_pos, Indq_obs, Indw_obs, n_d, MatRxx_d
