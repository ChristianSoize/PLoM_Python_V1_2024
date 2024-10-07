import numpy as np
import scipy.io
import h5py

def mainWorkflow_Data_generation2():
    #===================================================================================================================================
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow_Data_generation2
    #  Subject      : This function allows to generate data for Benchmark9_Inverse_TargType3_CondStat_ConfRegion
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

    ind_type_targ = 3;   # = 1, targets defined by giving N_r realizations
                         # = 2, targets defined by giving the target mean value of the unscaled XX_targ 
                         # = 3, targets defined by giving the target mean value and the traget covariance matrix of the unscaled XX_targ 
    #--- load data
    indv73 = 1  # 1: Case of a matlab file.mat saved using -v7.3 option 
                # 2: Case of a matlab file.mat saved without using -v7.3 option 

    if indv73 == 1: # Case of a matlab file.mat saved with -v7.3 option   
        with h5py.File('FileMatRxxTarget3.mat', 'r') as f:
            # Access the data (adjust 'dataset_name' to the actual variable name in the .mat file)
            Rmeanxx_targ_data = f['Rmeanxx_targ_data'][()].T
            MatRcovxx_targ_data = f['MatRcovxx_targ_data'][()].T

    if indv73 == 2: # Case of a matlab file.mat saved without using -v7.3 option 
        data = scipy.io.loadmat('FileMatRxxTarget3')          # Load FileMatRxxTarget3 from a file.mat saved without using -v7.3 option  
        MatRcovxx_targ_data = data['MatRcovxx_targ_data']  
        Rmeanxx_targ_data   = data['Rmeanxx_targ_data']  
                                # Name                  Size     Bytes  
                                # MatRcovxx_targ_data   402x402  1292832  
                                # Rmeanxx_targ_data     402x1    3216  
    Rmeanxx_targ_data = Rmeanxx_targ_data.ravel()  

    # NOTE: 
    #        1) For generating the training of WW = (WW_1,WW_2), 
    #        WW_1 is uniform on [0.9,1.1]
    #        WW_2 is uniform on [0.003,0.007] 
    #        2) The observations considered in mainWorkflow_Data_generation1.m sont QoI_1 and QoI_3
    #        For the updating (Inverse problem) the targets are given for QoI_2 and QoI_4 in order to not observe the QoI for which
    #        the target is given
    
    n_q  = 400;    # n_q = n_QoI*nbParam, nbParam = 100 and n_QoI = 4  

    #--- Loading Indq_targ_real(nqreal_targ,1) and Indq_targ_pos(nqpos_targ,1) for QoI_2 and QoI_4
    startQoI2 = 101
    endQoI2   = 200
    startQoI4 = 301
    endQoI4   = 400
    Indq_targ_real = np.concatenate((np.arange(startQoI2, endQoI2 + 1), np.arange(startQoI4, endQoI4 + 1)))  # Indq_targ_real(nqreal_targ) is a 1D array
    Indq_targ_pos  = np.array([])                                                                            # Indq_targ_pos(nqpos_targ)

    #--- Loading Indw_targ_real(nwreal_targ) and Indw_targ_pos(nwpos_targ)
    Indw_targ_real = np.array([1,2])     # Indw_targ_real(nwreal_targ)
    Indw_targ_pos  = np.array([])  # Indw_targ_pos(nwpos_targ)
    
    #--- Loading Loading Indx_targ(nx_targ)
    Indq_targ = Indq_targ_real # np.vstack((Indq_targ_real, Indq_targ_pos))
    Indw_targ = Indw_targ_real  # np.vstack((Indw_targ_real, Indw_targ_pos))
    Indx_targ = np.concatenate((Indq_targ, n_q + Indw_targ))
  
    #--- Loading the targets MatRxx_targ_real and MatRxx_targ_pos
    N_r = 0
    MatRxx_targ_real = np.array([])
    MatRxx_targ_pos  = np.array([])
    
    #--- ind_type_targ = 3: targets defined by giving the traget mean value and the tragetcovariance matrix of unscaled XX_targ 
    # Extract Rmeanxx_targ using the Indx_targ indices for the first column
    Rmeanxx_targ   = Rmeanxx_targ_data[Indx_targ-1]                         # Rmeanxx_targ(nx_targ)   
    MatRcovxx_temp = MatRcovxx_targ_data[np.ix_(Indx_targ-1,Indx_targ-1)]   # MatRcovxx_temp(nx_targ,nx_targ)    
    MatRcovxx_temp = 0.5*(MatRcovxx_temp + MatRcovxx_temp.T)                # Symmetrization   
    tol_rel        = 1e-10 * np.max(np.diag(MatRcovxx_temp))                # Calculate tolerance relative to the maximum diagonal value
    MatRcovxx_targ = MatRcovxx_temp + tol_rel*np.eye(202)                 # Add tolerance to the diagonal

    return ind_type_targ, Indq_targ_real, Indq_targ_pos, Indw_targ_real, Indw_targ_pos, N_r, MatRxx_targ_real, MatRxx_targ_pos, Rmeanxx_targ, MatRcovxx_targ
