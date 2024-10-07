import numpy as np
import scipy.io
import h5py

def mainWorkflow_Data_generation2():
    #===================================================================================================================================
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 26 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow_Data_generation2
    #  Subject      : This function allows to generate data for Benchmark7_Inverse_TargType1_CondStat_ConfRegion
    #
    #--- OUTPUT (data that must be generated for ind_workflow = 3) ---------------------------------------------------------
    #
    #   ind_type_targ  : = 1, targets defined by giving N_r realizations
    #                  : = 2, targets defined by giving the target mean value of the unscaled XX_targ 
    #                  : = 3, targets defined by giving the target mean value and the target covariance matrix of the unscaled XX_targ 
    #
    #--- WARNING 1: all the following parameters and arrays: ind_type_targ, Indq_targ_real, Indq_targ_pos, Indw_targ_real, Indw_targ_pos, N_r, 
    #               MatRxx_targ_real, MatRxx_targ_pos, Rmeanxx_targ, MatRcovxx_targ, must be defined as explained below.
    #
    #    WARNING 2: if an ArrayName is not used or empty, for the case considered, ArrayName = []
    #
    #    WARNING 3: about the organization of data:
    #               Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ) 1D array
    #                                 n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
    #
    #               Indx_targ_pos  = [Indq_targ_pos            % Indx_targ_pos(nbpos_targ) 1D array
    #                                 n_q + Indw_targ_pos];    % nbpos_targ component numbers of XX, 
    #                                                          % which are strictly positive
    #               nx_targ        = nbreal_targ + nbpos_targ; % dimension of random vector XX_targ = (QQ_targ, WW_targ)
    #               Indx_targ      = [Indx_targ_real           % 1D array, nx_targ component numbers of XX_targ 
    #                                 Indx_targ_pos];          % for which a target is given 
    #
    #    WARNING 4: For the analysis of the conditional statistics in Step4, the organization of the components of the 
    #               QQ vector of the quantity of interest QoI is as follows: this organization must be planned from the 
    #               creation of the data, not only in this function "mainWorkflow_Data_generation2.m but also in
    #               "mainWorkflow_Data_generation1.m".
    #
    #               If the QoI depends on the sampling in nbParam points of a physical system parameter
    #               (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
    #               f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
    #               as follows: 
    #                          [(QoI_1, f_1), (QoI_1, f_2), ... , (QoI_1, f_nbParam), (QoI_2, f_1), (QoI_2, f_2), ... , (QoI_2, f_nbParam), ... ]'.
    #
    #               If nbParam > 1, this means that nq_obs is equal to nqobsPhys * nbParam, in which nqobsPhys is the number
    #                               of the components of the state variables that are observed. Consequently, nq_obs / nbParam must be 
    #                               an integer, if not there is an error in the given value of nbParam in the Data generation in 
    #                               "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 
    #
    #               NOTE THAT if such a physical system parameter is not considered, then nbParam = 1, but the 
    #                         information structure must be consistent with the case nbParam > 1.  
    #
    #--- DATA FOR THE CASE ind_type_targ = 1 --------------------------------------------------------------------------------------------------
    #
    #          ind_type_targ  = 1;
    #
    #          Indq_targ_real(nqreal_targ): 1D array, nqreal_targ component numbers of QQ for which a target is real, 0 <= nqreal_targ <= n_q
    #          Indq_targ_pos(nqpos_targ)  : 1D array, nqpos_targ  component numbers of QQ for which a target is positive, 0 <= nqpos_targ <= n_q
    #
    #          Indw_targ_real(nwreal_targ): 1D array, nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
    #          Indw_targ_pos(nwpos_targ)  : 1D array, nwpos_targ  component numbers of WW for which a target is positive, 0 <= nwpos_targ <= n_w
    #
    #          N_r  : number of target realizations  
    #                                                         
    #          MatRxx_targ_real(nbreal_targ,N_r) : 2D array, N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
    #          MatRxx_targ_pos(nbpos_targ,N_r)   : 2D array, N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
    # 
    #          Rmeanxx_targ   = [];
    #          MatRcovxx_targ = [];
    #
    #--- DATA FOR THE CASE ind_type_targ = 2 --------------------------------------------------------------------------------------------------
    #
    #          ind_type_targ  = 2;
    #
    #          Indq_targ_real(nqreal_targ): 1D array, nqreal_targ component numbers of QQ for which a target is real, 1 <= nqreal_targ <= n_q
    #          Indq_targ_pos(nqpos_targ)  = [];
    #
    #          Indw_targ_real(nwreal_targ): 1D array, nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
    #          Indw_targ_pos(nwpos_targ)  = [];
    #
    #          N_r = 0;  
    #                                                         
    #          MatRxx_targ_real(nbreal_targ, N_r) = [];
    #          MatRxx_targ_pos(nbpos_targ, N_r)   = [];
    # 
    #          Rmeanxx_targ(nx_targ, 1) : target mean value of the unscaled XX_targ
    #          MatRcovxx_targ = [];
    #
    #--- DATA FOR THE CASE ind_type_targ = 3 --------------------------------------------------------------------------------------------------
    #
    #          ind_type_targ  = 3;
    #
    #          Indq_targ_real(nqreal_targ): 1D array, qreal_targ component numbers of QQ for which a target is real, 1 <= nqreal_targ <= n_q
    #          Indq_targ_pos(nqpos_targ)  = [];
    #
    #          Indw_targ_real(nwreal_targ): 1D array, nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
    #          Indw_targ_pos(nwpos_targ)  = [];
    #
    #          N_r = 0;  
    #                                                         
    #          MatRxx_targ_real(nbreal_targ, N_r) = [];
    #          MatRxx_targ_pos(nbpos_targ, N_r)   = [];
    # 
    #          Rmeanxx_targ(nx_targ, 1)         : target mean value of the unscaled XX_targ
    #          MatRcovxx_targ(nx_targ, nx_targ) : target covariance matrix of the unscaled XX_targ
    #======================================================================================================================================

    ind_type_targ = 1;   # = 1, targets defined by giving N_r realizations
                         # = 2, targets defined by giving the target mean value of the unscaled XX_tar 
                         # = 3, targets defined by giving the target mean value and the target covariance matrix of the unscaled XX_tar 
    #--- load data
    indv73 = 2  # 1: Case of a matlab file.mat saved using -v7.3 option 
                # 2: Case of a matlab file.mat saved without using -v7.3 option 

    if indv73 == 1: # Case of a matlab file.mat saved with -v7.3 option         
        with h5py.File('FileMatRxxTarget1', 'r') as file:     # Load FileMatRxxTarget1 from a file.mat saved with -v7.3 option 
            MatRqq_target_data = data['MatRqq_target_data'][:].T
            MatRww_target_data = data['MatRww_target_data'][:].T

    if indv73 == 2: # Case of a matlab file.mat saved without using -v7.3 option 
        data = scipy.io.loadmat('FileMatRxxTarget1')          # Load FileMatRxxTarget1 from a file.mat saved without using -v7.3 option   
        MatRqq_target_data = data['MatRqq_target_data']  
        MatRww_target_data = data['MatRww_target_data'] 
                                    #  Name                Size      Bytes   
                                    # MatRqq_target_data   400x200   640000       
                                    # MatRww_target_data   2x200     3200 
    # NOTE: 
    #        1) For generating the training of WW = (WW_1,WW_2), 
    #        WW_1 is uniform on [0.9,1.1]
    #        WW_2 is uniform on [0.003,0.007] 
    #        2) The observations considered in mainWorkflow_Data_generation1.m sont QoI_1 and QoI_3
    #        For the updating (Inverse problem) the targets are given for QoI_2 and QoI_4 in order to not observe the QoI for which
    #        the target is given

    #--- Loading Indq_targ_real(nqreal_targ,1) and Indq_targ_pos(nqpos_targ,1) for QoI_2 and QoI_4
    startQoI2 = 101
    endQoI2   = 200
    startQoI4 = 301
    endQoI4   = 400
    Indq_targ_real = np.concatenate((np.arange(startQoI2, endQoI2 + 1), np.arange(startQoI4, endQoI4 + 1)))  # Indq_targ_real(nqreal_targ) is a 1D array
    Indq_targ_pos  = np.array([])                                                                            # Indq_targ_pos(nqpos_targ)

    #--- Loading Indw_targ_real(nwreal_targ) and Indw_targ_pos(nwpos_targ)
    Indw_targ_real = np.array([])     # Indw_targ_real(nwreal_targ)
    Indw_targ_pos  = np.array([1,2])  # Indw_targ_pos(nwpos_targ)
    
    #--- Loading the targets MatRqq_targ_real, MatRqq_targ_pos, MatRww_targ_real, and MatRww_targ_pos
    N_r = 100

    # There is no subset of positive values for qq_targ, consequenly, MatRqq_targ_pos = np.array([])  is not used
    # In addition, only a part of the qq_target are used. Therefore: 
    MatRqq_targ_real = MatRqq_target_data[Indq_targ_real-1,:N_r]  # MatRqq_targ_real(nqreal_targ,N_r)
    
    # There is no subset of real values for ww_targ, consequenly, MatRww_targ_real = np.array([])  is not used:    
    MatRww_targ_pos = MatRww_target_data[:,:N_r]  # MatRww_targ_pos(nwpos_targ, N_r)

    #--- Loading MatRxx_targ_real and MatRxx_targ_pos
    MatRxx_targ_real = MatRqq_targ_real
    MatRxx_targ_pos  = MatRww_targ_pos
       
    Rmeanxx_targ   = np.array([])
    MatRcovxx_targ = np.array([])
    
    return ind_type_targ, Indq_targ_real, Indq_targ_pos, Indw_targ_real, Indw_targ_pos, N_r, MatRxx_targ_real, MatRxx_targ_pos, Rmeanxx_targ, MatRcovxx_targ
