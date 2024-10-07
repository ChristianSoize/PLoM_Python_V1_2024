import numpy as np
import time
import sys

def sub_data_structure_and_check(n_q, Indq_real, Indq_pos, n_w, Indw_real, Indw_pos, n_d, MatRxx_d, Indq_obs, Indw_obs,
                                 ind_display_screen, ind_print, ind_exec_solver, Indq_targ_real, Indq_targ_pos,
                                 Indw_targ_real, Indw_targ_pos, ind_type_targ, N_r, MatRxx_targ_real, MatRxx_targ_pos,
                                 Rmeanxx_targ, MatRcovxx_targ):
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 01 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM)
    #  Function name: sub_data_structure_and_check
    #  Subject      : for XX = (QQ,WW) and the associated observation XX_obs = (QQ_obs,WW_obs), construction of the information structure
    #                 that is used for
    #                 - the training dataset with n_d realizations
    #                 - the target datasets with given second-order moments or N_r realizations
    #                 - the learned dataset with n_ar realizations  (with or without targets)
    #
    #  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold,
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science,
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020).
    #                [3] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds,
    #                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
    #                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations
    #                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777,
    #                       380, 113777 (2021).
    #                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for
    #                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #                [6] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler
    #                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
    #                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022).
    #                [7] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of
    #                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
    #                [8] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets
    #                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering,
    #                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
    #                [9] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds,
    #                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
    #
    #--- INPUTS
    #          n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q
    #          Indq_real(nqreal)      : contains the nqreal component numbers of QQ, which are real (positive, negative, or zero)
    #                                   with 0 <= nqreal <=  n_q and for which a "standard scaling" will be used
    #          Indq_pos(nqpos)        : contains the nqpos component numbers of QQ, which are strictly positive a "specific scaling"
    #                                   with  0 <= nqpos <=  n_q  and for which the scaling is {log + "standard scaling"}
    #                                   --- we must have n_q = nqreal + nqpos
    #
    #          n_w                    : dimension of random vector WW (unscaled control variable) with 0 <= n_w.
    #                                   n_w = 0: unsupervised case, n_w >= 1: supervised case
    #          Indw_real(nwreal)      : contains the nwreal component numbers of WW, which are real (positive, negative, or zero)
    #                                   with 0 <= nwreal <=  n_w and for which a "standard scaling" will be used
    #          Indw_pos(nwpos)        : contains the nwpos component numbers of WW, which are strictly positive a "specific scaling"
    #                                   with  0 <= nwpos <=  n_w  and for which the scaling is {log + "standard scaling"}
    #                                   --- we must have n_w = nwreal + nwpos
    #
    #          n_d                    : number of points in the training set for XX_d and X_d
    #          MatRxx_d(n_x,n_d)      : n_d realizations of random vector XX_d (unscale) with dimension n_x = n_q + n_w
    #
    #          Indq_obs(nq_obs)       : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
    #          Indw_obs(nw_obs)       : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
    #                                 --- we must have nx_obs = nq_obs + nw_obs <= n_x
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
    #                              an integer, if not there is an error in the Data generation in "mainWorkflow_Data_generation1.m" and
    #                              "mainWorkflow_Data_generation2.m"
    #
    #                     WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the
    #                              information structure must be consistent with the case nbParam > 1.
    #
    #
    #          ind_display_screen    : = 0 no display,            = 1 display
    #          ind_print             : = 0 no print,              = 1 print
    #
    #          ind_exec_solver       : = 1 Direct analysis : giving a training dataset, generating a learned dataset
    #                                : = 2 Inverse analysis: giving a training dataset and a target dataset, generating a learned dataset
    #
    #                                --- data structure for the target datasets used if ind_exec_solver = 2:  QQ_targ = (QQ_targ,WW_targ)
    #          ind_type_targ         : = 1, targets defined by giving N_r realizations
    #                                : = 2, targets defined by giving target mean-values
    #                                : = 3, targets defined by giving target mean-values and target variance-values
    #
    #          Indq_targ_real(nqreal_targ): nqreal_targ component numbers of QQ for which a target is real, 0 <= nqreal_targ <= n_q
    #          Indq_targ_pos(nqpos_targ)  : nqpos_targ component numbers of QQ for which a target is positive, 0 <= nqpos_targ <= n_q
    #
    #          Indw_targ_real(nwreal_targ): nwreal_targ component numbers of WW for which a target is real, 0 <= nwreal_targ <= n_w
    #          Indw_targ_pos(nwpos_targ)  : nwpos_targ  component numbers of WW for which a target is positive, 0 <= nwpos_targ <= n_w
    #
    #                                         Indx_targ_real = [Indq_targ_real           % Indx_targ_real(nbreal_targ)
    #                                                           n_q + Indw_targ_real];   % nbreal_targ component numbers of XX, which are real
    #                                         Indx_targ_pos  = [Indq_targ_pos            % Indx_targ_pos(nbpos_targ)
    #                                                           n_q + Indw_targ_pos];    % nbpos_targ component numbers of XX,
    #                                                                                    % which are strictly positive
    #                                         nx_targ        = nbreal_targ + nbpos_targ; % dimension of random vector XX_targ = (QQ_targ,WW_targ)
    #                                         Indx_targ      = [Indx_targ_real           % nx_targ component numbers of XX_targ
    #                                                           Indx_targ_pos];          % for which a target is given
    #
    #                                     WARNING: if ind_exec_solver = 2 and ind_type_targ = 2 or 3, all the components of XX and XX_targ
    #                                              must be considered as real even if some components are positive. When thus must have:
    #                                              nqreal     = n_q and nwreal     = n_w
    #                                              nqpos      = 0   and nwpos      = 0
    #                                              nqpos_targ = 0   and nwpos_targ = 0
    #
    #          N_r                               : number of target realizations
    #
    #                                            --- ind_type_targ = 1: targets defined by giving N_r realizations
    #          MatRxx_targ_real(nbreal_targ,N_r) : N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
    #          MatRxx_targ_pos(nbpos_targ,N_r)   : N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
    #
    #                                            --- ind_type_targ = 2 or 3: targets defined by giving the mean value of unscaled XX_targ
    #          Rmeanxx_targ(nx_targ)             : nx_targ components of mean value E{XX_targ} of vector-valued random target XX_targ
    #
    #                                            --- ind_type_targ = 3: targets defined by giving the covariance matrix of unscaled XX_targ
    #          MatRcovxx_targ(nx_targ,nx_targ)   : covariance matrix of XX_targ
    #
    #--- OUTPUTS
    #          n_x                      : dimension of random vector XX = (QQ,WW) (unscaled), n_x = n_q + n_w
    #                                     We must have n_x = nbreal + nbpos
    #          Indx_real(nbreal)        : contains the nbreal component numbers of XX, which are real (positive, negative, or zero)
    #                                     with 0 <= nbreal <=  n_x and for which a "standard scaling" will be used
    #          Indx_pos(nbpos)          : contains the nbpos component numbers of XX, which are strictly positive a "specific scaling"
    #                                     with  0 <= nbpos <=  n_x  and for which the scaling is {log + "standard scaling"}
    #
    #          nx_obs                   : number of components of XX that are observed with nx_obs = nq_obs + nw_obs <= n_x
    #          Indx_obs(nx_obs)         : nx_obs component numbers of XX that are observed
    #
    #                                          --- if ind_exec_solver = 2:  XX_targ = (XX_targ_real,XX_targ_pos)
    #                                              we must have nx_targ = nbreal_targ + nbpos_targ
    #          Indx_targ_real(nbreal_targ)     : contains the nbreal_targ component numbers of XX, which are real
    #                                            with 0 <= nbreal_targ <= n_x and for which a "standard scaling" will be used
    #          Indx_targ_pos(nbpos_targ)       : contains the nbpos_targ component numbers of XX, which are strictly positive (specific scaling)
    #                                            with 0 <= nbpos_targ <= n_x  and for which the scaling is {log + "standard scaling"}
    #          nx_targ                         : number of components of XX with targets such that nx_targ = nbreal_targ + nbpos_targ <= n_x
    #          Indx_targ(nx_targ)              : nx_targ component numbers of XX with targets
    #
    #--- COMMENTS
    #          If ind_exec_solver = 2 and ind_type_targ = 2 or 3,
    #          all the components of XX and XX_targ are considered as real even if some components are positive.
    #          When thus have:  nx_targ = nbreal_targ, nbpos_targ = 0, and Indx_targ(nx_targ) = Indx_targ_real(nbreal_targ)

    if ind_display_screen == 1:
        print('--- beginning Task1_DataStructureCheck')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ------ Task1_DataStructureCheck \n ')
            fidlisting.write('      \n ')

    TimeStartDataCheck = time.time()

    # Checking ind_type_targ
    if ind_exec_solver == 2 and (ind_type_targ != 1 and ind_type_targ != 2 and ind_type_targ != 3):
        raise ValueError('STOP1 in sub_data_structure_and_check: for an inverse problem, ind_type_targ  must be equal to 1, 2 or 3')

    #--- Checking n_q and n_w
    if n_q <= 0 or n_w < 0:
        raise ValueError('STOP2 in sub_data_structure_and_check: n_q <= 0 or n_w < 0')

    #--- Loading dimensions nq_obs, nqreal, and nqpos of Indq_obs(nq_obs), Indq_real(nqreal), and Indq_pos(nqpos)
    if Indq_obs.size >=1:           # checking if not empty
        if Indq_obs.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP3 in sub_data_structure_and_check: Indq_obs must be a 1D array')
        
    if Indq_real.size >=1:          # checking if not empty
        if Indq_real.ndim != 1:     # ckecking if it is a 1D array 
            raise ValueError('STOP4 in sub_data_structure_and_check: Indq_real must be a 1D array')  
          
    if Indq_pos.size >=1:           # checking if not empty
        if Indq_pos.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP5 in sub_data_structure_and_check: Indq_pos must be a 1D array')  
     
    nq_obs = len(Indq_obs)   # Indq_obs(nq_obs)
    nqreal = len(Indq_real)  # Indq_real(nqreal)
    nqpos  = len(Indq_pos)   # Indq_pos(nqpos)

    #--- Checking input data and parameters of Indq_obs(nq_obs), Indq_real(nqreal) and Indq_pos(nqpos)
    if nq_obs < 1 or nq_obs > n_q:
        raise ValueError('STOP6 in sub_data_structure_and_check: nq_obs < 1 or nq_obs > n_q')
    if len(np.unique(Indq_obs)) != len(Indq_obs):
        raise ValueError('STOP7 in sub_data_structure_and_check: there are repetitions in Indq_obs')
    if any(Indq_obs < 1) or any(Indq_obs > n_q):
        raise ValueError('STOP8 in sub_data_structure_and_check: at least one integer in Indq_obs is not  within range [1,n_q]')
    
    if nqreal + nqpos != n_q:
        raise ValueError('STOP9 in sub_data_structure_and_check: nqreal + nqpos != n_q')
    
    if nqreal >= 1:
        if len(np.unique(Indq_real)) != len(Indq_real):
            raise ValueError('STOP10 in sub_data_structure_and_check: there are repetitions in Indq_real')
        if any(Indq_real < 1) or any(Indq_real > n_q):
            raise ValueError('STOP11 in sub_data_structure_and_check: at least one  integer in Indq_real is not within range [1,n_q]')
        
    if nqpos >= 1:
        if ind_exec_solver == 2 and (ind_type_targ == 2 or ind_type_targ == 3):
            raise ValueError('STOP12 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nqpos = 0')
        if len(np.unique(Indq_pos)) != len(Indq_pos):
            raise ValueError('STOP13 in sub_data_structure_and_check: there are repetitions in Indq_pos')
        if any(Indq_pos < 1) or any(Indq_pos > n_q):
            raise ValueError('STOP14 in sub_data_structure_and_check: at least one  integer in Indq_pos is not within range [1,n_q]')
        
    if nqreal >= 1 and nqpos >= 1:  # Check that all integers in Indq_real are different from those in Indq_pos
        if not np.intersect1d(Indq_real, Indq_pos).size:
            combined_list = np.concatenate((Indq_real, Indq_pos))  # Check that the union of both lists is exactly 1:n_q without missing or repetition
            if len(combined_list) != n_q or not np.array_equal(np.sort(combined_list), np.arange(1, n_q + 1)):
                raise ValueError('STOP15 in sub_data_structure_and_check: the union of Indq_real with Indq_pos is not equal to the set (1:n_q)')
        else:
            raise ValueError('STOP16 in sub_data_structure_and_check: there are common integers in Indq_real and Indq_pos')

    #--- Loading dimensions nw_obs, nwreal, and nwpos of Indw_obs(nw_obs), Indw_real(nwreal), and Indw_pos(nwpos)
    if Indw_obs.size >=1:           # checking if not empty
        if Indw_obs.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP17 in sub_data_structure_and_check: Indw_obs must be a 1D array')
        
    if Indw_real.size >=1:          # checking if not empty
        if Indw_real.ndim != 1:     # ckecking if it is a 1D array 
            raise ValueError('STOP18 in sub_data_structure_and_check: Indw_real must be a 1D array')  
          
    if Indw_pos.size >=1:           # checking if not empty
        if Indw_pos.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP19 in sub_data_structure_and_check: Indw_pos must be a 1D array')  
     
    nw_obs = len(Indw_obs)   # Indw_obs(nw_obs)
    nwreal = len(Indw_real)  # Indw_real(nwreal)
    nwpos  = len(Indw_pos)   # Indw_pos(nwpos)

    #--- Checking input data and parameters of Indw_obs(nw_obs), Indw_real(nwreal) and Indw_pos(nwpos)
    if n_w >= 1:  # Supervised case
        if nw_obs < 1 or nw_obs > n_w:
            raise ValueError('STOP20 in sub_data_structure_and_check: nw_obs < 1 or nw_obs > n_w')   
             
        if len(np.unique(Indw_obs)) != len(Indw_obs):
            raise ValueError('STOP21 in sub_data_structure_and_check: there are repetitions in Indw_obs')
        if any(Indw_obs < 1) or any(Indw_obs > n_w):
            raise ValueError('STOP22 in sub_data_structure_and_check: at least one integer in Indw_obs is not within range [1,n_w]')
        
        if nwreal + nwpos != n_w:
            raise ValueError('STOP23 in sub_data_structure_and_check: nwreal + nwpos != n_w')
        
        if nwreal >= 1:
            if len(np.unique(Indw_real)) != len(Indw_real):
                raise ValueError('STOP24 in sub_data_structure_and_check: there are repetitions in Indw_real')
            if any(Indw_real < 1) or any(Indw_real > n_w):
                raise ValueError('STOP25 in sub_data_structure_and_check: at least one  integer in Indw_real is not within range [1,n_w]')
            
        if nwpos >= 1:
            if ind_exec_solver == 2 and (ind_type_targ == 2 or ind_type_targ == 3):
                raise ValueError('STOP26 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nwpos = 0')
            if len(np.unique(Indw_pos)) != len(Indw_pos):
                raise ValueError('STOP27 in sub_data_structure_and_check: there are repetitions in Indw_pos')
            if any(Indw_pos < 1) or any(Indw_pos > n_w):
                raise ValueError('STOP28 in sub_data_structure_and_check: at least one  integer in Indw_pos is not within range [1,n_w]')
            
        if nwreal >= 1 and nwpos >= 1:  # Check that all integers in Indw_real are different from those in Indw_pos
            if not np.intersect1d(Indw_real, Indw_pos).size:
                combined_list = np.concatenate((Indw_real, Indw_pos)) # Check that the union of both lists is exactly 1:n_w without missing or repetition
                if len(combined_list) != n_w or not np.array_equal(np.sort(combined_list), np.arange(1, n_w + 1)):
                    raise ValueError('STOP29 in sub_data_structure_and_check: the union of Indw_real with Indw_pos is not equal to the set (1:n_w)')
            else:
                raise ValueError('STOP30 in sub_data_structure_and_check: there are common integers in Indw_real and Indw_pos')

    if ind_exec_solver != 1 and ind_exec_solver != 2:
        raise ValueError('STOP31 in sub_data_structure_and_check: ind_exec_solver must be equal to 1 or equal to 2')

    #--- Loading dimensions  nqreal_targ, nqpos_targ, nwreal_targ, nwpos_targ
    #    if ind_exec_solver = 1 (no targets) matrices Indq_targ_real,Indq_targ_pos,Indw_targ_real,Indw_targ_pos must be equal to []
    #    if ind_exec_solver = 2 and ind_type_targ = 2 or 3, Indq_targ_pos and Indw_targ_pos must be equal to []

    if ind_exec_solver == 2:    
        if Indq_targ_real.size >=1:          # checking if not empty
            if Indq_targ_real.ndim != 1:     # ckecking if it is a 1D array 
                raise ValueError('STOP32 in sub_data_structure_and_check: Indq_targ_real must be a 1D array')  
          
        if Indq_targ_pos.size >=1:           # checking if not empty
            if Indq_targ_pos.ndim != 1:      # ckecking if it is a 1D array
                raise ValueError('STOP33 in sub_data_structure_and_check: Indq_targ_pos must be a 1D array')  
        
        if Indw_targ_real.size >=1:          # checking if not empty
            if Indw_targ_real.ndim != 1:     # ckecking if it is a 1D array 
                raise ValueError('STOP34 in sub_data_structure_and_check: Indw_targ_real must be a 1D array')  
          
        if Indw_targ_pos.size >=1:           # checking if not empty
            if Indw_targ_pos.ndim != 1:      # ckecking if it is a 1D array
                raise ValueError('STOP35 in sub_data_structure_and_check: Indw_targ_pos must be a 1D array')      

    nqreal_targ = len(Indq_targ_real)  # Indq_targ_real(nqreal_targ)
    nqpos_targ  = len(Indq_targ_pos)   # Indq_targ_pos(nqpos_targ)
    nwreal_targ = len(Indw_targ_real)  # Indw_targ_real(nwreal_targ)
    nwpos_targ  = len(Indw_targ_pos)   # Indw_targ_pos(nwpos_targ)

    #--- Checking data for ind_exec_solver = 2 and ind_type_targ = 2 or = 3: inverse problem with given targets defined by moments.
    #    For such a scale all the components must be considered as real even if there are positive-valued components.
    if ind_exec_solver == 2 and (ind_type_targ == 2 or ind_type_targ == 3):
        if nqpos_targ != 0 and nwpos_targ != 0:
            raise ValueError('STOP36 in sub_data_structure_and_check: for an inverse problem and \
                             if ind_type_targ = 2 or 3, then we must have nqpos_targ = nwpos_targ = 0')

    #--- Checking data for ind_exec_solver = 2 (inverse problem with given targets)
    if ind_exec_solver == 2:
        nbreal_targ = nqreal_targ + nwreal_targ
        nbpos_targ  = nqpos_targ  + nwpos_targ
        ntemp       = nbreal_targ + nbpos_targ
        if ntemp == 0:
            raise ValueError('STOP37 in sub_data_structure_and_check: for an inverse problem, at least 1 component of XX must have a target')

        #--- Checking input data and parameters of  Indq_targ_real(nqreal_targ,1) and Indq_targ_pos(nqpos_targ,1)
        if nqreal_targ >= 1:
            if len(np.unique(Indq_targ_real)) != len(Indq_targ_real):
                raise ValueError('STOP38 in sub_data_structure_and_check: there are repetitions in Indq_targ_real')
            if any(Indq_targ_real < 1) or any(Indq_targ_real > n_q):
                raise ValueError('STOP39 in sub_data_structure_and_check: at least one  integer in Indq_targ_real is not within range [1,n_q]')
            is_subset = all(np.isin(Indq_targ_real, Indq_real))
            is_equal  = np.array_equal(Indq_targ_real, Indq_real)
            if not is_equal and not is_subset:
                raise ValueError('STOP40 in sub_data_structure_and_check: Indq_targ_real is neither a subset of Indq_real nor equal to Indq_real')
            
        if nqpos_targ >= 1:
            if ind_exec_solver == 2 and (ind_type_targ == 2 or ind_type_targ == 3):
                raise ValueError('STOP41 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, \
                                 we must have nqpos_targ = 0')
            if len(np.unique(Indq_targ_pos)) != len(Indq_targ_pos):
                raise ValueError('STOP42 in sub_data_structure_and_check: there are repetitions in Indq_targ_pos')
            if any(Indq_targ_pos < 1) or any(Indq_targ_pos > n_q):
                raise ValueError('STOP43 in sub_data_structure_and_check: at least one  integer in Indq_targ_pos is not within range [1,n_q]')
            is_subset = all(np.isin(Indq_targ_pos, Indq_pos))
            is_equal = np.array_equal(Indq_targ_pos, Indq_pos)
            if not is_equal and not is_subset:
                raise ValueError('STOP44 in sub_data_structure_and_check: Indq_targ_pos is neither a subset of Indq_pos nor equal to Indq_pos')
            
        if nqreal_targ >= 1 and nqpos_targ >= 1:  # Check the coherence
            if not np.intersect1d(Indq_targ_real, Indq_targ_pos).size:
                combined_list = np.concatenate((Indq_targ_real, Indq_targ_pos))
                if len(combined_list) <= n_q:
                    ind_error = 0  # All integers in Indq_targ_real are different from those in Indq_targ_pos, and the length of union is <= n_q
                else:
                    ind_error = 1  # The length of union of Indq_targ_real and Indq_targ_pos is not <= n_q
            else:
                ind_error = 2  # There are common integers in Indq_targ_real and Indq_targ_pos
            if ind_error == 1:
                raise ValueError('STOP45 in sub_data_structure_and_check: at least one integer in Indq_targ_real is equal to \
                                 an integer in Indq_targ_pos')
            if ind_error == 2:
                raise ValueError('STOP46 in sub_data_structure_and_check: there are common integers in Indq_targ_real and Indq_targ_pos')

        #--- Checking input data and parameters of  Indw_targ_real(nwreal_real,1) and Indw_targ_pos(nwpos_targ,1)
        if nwreal_targ >= 1:
            if len(np.unique(Indw_targ_real)) != len(Indw_targ_real):
                raise ValueError('STOP47 in sub_data_structure_and_check: there are repetitions in Indw_targ_real')
            if any(Indw_targ_real < 1) or any(Indw_targ_real > n_w):
                raise ValueError('STOP48 in sub_data_structure_and_check: at least one integer in Indw_targ_real is not within range [1,n_w]')
            is_subset = all(np.isin(Indw_targ_real, Indw_real))
            is_equal = np.array_equal(Indw_targ_real, Indw_real)
            if not is_equal and not is_subset:
                raise ValueError('STOP49 in sub_data_structure_and_check: Indw_targ_real is neither a subset of Indw_real nor equal to Indw_real')
            
        if nwpos_targ >= 1:
            if ind_exec_solver == 2 and (ind_type_targ == 2 or ind_type_targ == 3):
                raise ValueError('STOP50 in sub_data_structure_and_check: for ind_exec_solver = 2 and ind_type_targ = 2 or 3, we must have nwpos_targ = 0')
            if len(np.unique(Indw_targ_pos)) != len(Indw_targ_pos):
                raise ValueError('STOP51 in sub_data_structure_and_check: there are repetitions in Indw_targ_pos')
            if any(Indw_targ_pos < 1) or any(Indw_targ_pos > n_w):
                raise ValueError('STOP52 in sub_data_structure_and_check: at least one  integer in Indw_targ_pos is not within the valid range')
            is_subset = all(np.isin(Indw_targ_pos, Indw_pos))
            is_equal = np.array_equal(Indw_targ_pos, Indw_pos)
            if not is_equal and not is_subset:
                raise ValueError('STOP53 in sub_data_structure_and_check: Indw_targ_pos is neither a subset of Indw_pos nor equal to Indw_pos')
            
        if nwreal_targ >= 1 and nwpos_targ >= 1:  # Check the coherence
            if not np.intersect1d(Indw_targ_real, Indw_targ_pos).size:
                combined_list = np.concatenate((Indw_targ_real, Indw_targ_pos))
                if len(combined_list) <= n_w:
                    ind_error = 0  # All integers in Indw_targ_real are different from those in Indw_targ_pos, and the length of union is <= n_w
                else:
                    ind_error = 1  # The length of union of Indw_targ_real and Indw_targ_pos is not <= n_w
            else:
                ind_error = 2  # There are common integers in Indw_targ_real and Indw_targ_pos
            if ind_error == 1:
                raise ValueError('STOP54 in sub_data_structure_and_check: at least one integer in Indw_targ_real is equal to \
                                 an integer in Indw_targ_pos')
            if ind_error == 2:
                raise ValueError('STOP55 in sub_data_structure_and_check: there are common integers in Indw_targ_real and Indw_targ_pos')

    #--- Construction of Indx_obs(nx_obs) : nx_obs component numbers of XX = (QQ,WW) that are observed
    n_x      = n_q + n_w                                   # dimension of random vector XX = (QQ,WW)
    nx_obs   = nq_obs + nw_obs                             # number of components of XX that are observed
    Indx_obs = np.concatenate((Indq_obs, n_q + Indw_obs))  # Indq_obs(nq_obs): nq_obs component numbers of QQ that are observed

    #--- Checking n_d and MatRxx_d(n_x,n_d)
    if n_d <= 0:
        raise ValueError('STOP56 in sub_data_structure_and_check: n_d must be greater than or equal to 1')    
    if MatRxx_d.shape != (n_x, n_d):
        raise ValueError('STOP57 in sub_data_structure_and_check: dimension error in matrix MatRxx_d(n_x,n_d)')
    MatRqq_d = MatRxx_d[:n_q, :]           # MatRqq_d(n_q,n_d)
    MatRww_d = MatRxx_d[n_q:n_q + n_w, :]  # MatRww_d(n_w,n_d)
    if nqpos >= 1:
        MatRqq_pos = MatRqq_d[Indq_pos - 1, :]  # MatRqq_pos(nqpos,:), Indq_pos(nqpos)
        if any(MatRqq_pos.ravel() <= 0):
            raise ValueError('STOP58 in sub_data_structure_and_check: all values in MatRqq_pos(nqpos,n_d) must be strictly positive')
    if nwpos >= 1:
        MatRww_pos = MatRww_d[Indw_pos - 1, :]  # MatRww_pos(nwpos,:), Indw_pos(nwpos)
        if any(MatRww_pos.ravel() <= 0):
            raise ValueError('STOP59 in sub_data_structure_and_check: all values in MatRww_pos(nwpos,n_d) must be strictly positive')

    #--- construction of :
    #                     Indx_real(nbreal) that contains the nbreal component numbers of XX, which are real (positive, negative, or zero)
    #                                         with 0 <= nbreal <=  n_x and for which a "standard scaling" will be used
    #                     Indx_pos(nbpos)   that contains the nbpos component numbers of XX, which are strictly positive a "specific scaling"
    #                                         with  0 <= nbpos <=  n_x  and for which the scaling is {log + "standard scaling"}
    nbreal    = nqreal + nwreal
    nbpos     = nqpos + nwpos
    Indx_real = np.concatenate((Indq_real, n_q + Indw_real))  # nbreal component numbers of XX, which are real
    Indx_pos  = np.concatenate((Indq_pos, n_q + Indw_pos))    # nbpos component numbers of XX, which are strictly positive
    
    #--- construction of :
    #      Indx_targ_real(nbreal_targ) that contains the nbreal_targ component numbers of XX, which are real
    #                                    with 0 <= nbreal_targ <=  n_x and for which a "standard scaling" will be used
    #      Indx_targ_pos(nbpos_targ)   that contains the nbpos_targ component numbers of XX, which are strictly positive a "specific scaling"
    #                                    with  0 <= nbpos_targ <=  n_x  and for which the scaling is {log + "standard scaling"}
    #                                    nx_targ = nbreal_targ + nbpos_targ <= n_x
    if ind_exec_solver == 1:
        Indx_targ_real = np.array([])
        Indx_targ_pos  = np.array([])
        nx_targ        = 0
        Indx_targ      = np.array([])

    if ind_exec_solver == 2:
        if ind_type_targ == 1:
            Indx_targ_real = np.concatenate((Indq_targ_real, n_q + Indw_targ_real))   # Indx_targ_real(nbreal_targ)
            Indx_targ_pos  = np.concatenate((Indq_targ_pos, n_q + Indw_targ_pos))     # Indx_targ_pos(nbpos_targ,1)
            nx_targ   = nbreal_targ + nbpos_targ                                      # dimension of random vector XX_targ = (QQ_targ,WW_targ)
            Indx_targ = np.concatenate((Indx_targ_real, Indx_targ_pos))               # nx_targ components of XX_targ for which a target is given

        if ind_type_targ == 2 or ind_type_targ == 3:
            Indx_targ_real = np.concatenate((Indq_targ_real, n_q + Indw_targ_real))   # Indx_targ_real(nbreal_targ)
            Indx_targ_pos  = np.array([])                                             # Indx_targ_pos(nbpos_targ)
            nx_targ        = nbreal_targ                                              # dimension of random vector XX_targ = (QQ_targ,WW_targ)
            Indx_targ      = Indx_targ_real                                           # nx_targ components of XX_targ for which a target is given

    #--- Checking data related to target dataset
    if ind_exec_solver == 2:
        if ind_type_targ == 1:
            if N_r <= 0:
                raise ValueError('STOP60 in sub_data_structure_and_check: for an inverse problem, if ind_type_targ = 1, then N_r must \
                                 be greater than or equal to 1')
            if nbreal_targ >= 1:
                if MatRxx_targ_real.shape != (nbreal_targ, N_r):
                    raise ValueError('STOP61 in sub_data_structure_and_check: dimension error in matrix MatRxx_targ_real(nbreal_targ,N_r)')
            if nbpos_targ >= 1:
                if MatRxx_targ_pos.shape != (nbpos_targ, N_r):
                    raise ValueError('STOP62 in sub_data_structure_and_check: dimension error in matrix MatRxx_targ_pos(nbpos_targ,N_r)')
                if any(MatRxx_targ_pos.ravel() <= 0):
                    raise ValueError('STOP63 in sub_data_structure_and_check: all values in MatRxx_targ_pos(nbpos_targ,N_r) must be strictly positive')
                
        if ind_type_targ == 2 or ind_type_targ == 3:
            if Rmeanxx_targ.size >=1:          # checking if not empty
                if Rmeanxx_targ.ndim != 1:     # ckecking if it is a 1D array 
                    raise ValueError('STOP64 in sub_data_structure_and_check: Rmeanxx_targ must be a 1D array')  
           
        if ind_type_targ == 3:
            if MatRcovxx_targ.shape != (nx_targ, nx_targ):
                raise ValueError('STOP65 in sub_data_structure_and_check: dimension error in matrix MatRcovxx_targ(nx_targ,nx_targ)')
            if not np.allclose(MatRcovxx_targ, MatRcovxx_targ.T):
                raise ValueError('STOP66 in sub_data_structure_and_check: matrix MatRcovxx_targ(nx_targ,nx_targ) must be symmetric')
            Reigenvalues = np.linalg.eigvals(MatRcovxx_targ)
            max_eigenvalue = np.max(Reigenvalues)
            if any(Reigenvalues <= 1e-12 * max_eigenvalue):
                raise ValueError('STOP67 in sub_data_structure_and_check: matrix MatRcovxx_targ(nx_targ,nx_targ) must be positive definite')

    #--- Display
    if ind_display_screen == 1:
        if ind_exec_solver == 1:
            print(f'ind_exec_solver = {ind_exec_solver}, Direct Solver used.')
        if ind_exec_solver == 2:
            print(f'ind_exec_solver = {ind_exec_solver}, Inverse Solver used.')
        print(' ')
        print(f'n_q    = {n_q}')
        print(f'n_w    = {n_w}')
        print(f'n_x    = {n_x}')
        print(' ')
        print(f'nqreal = {nqreal}')
        print(f'nwreal = {nwreal}')
        print(f'nbreal = {nbreal}')
        print(' ')
        print(f'nqpos  = {nqpos}')
        print(f'nwpos  = {nwpos}')
        print(f'nbpos  = {nbpos}')
        print(' ')
        print(f'nq_obs    = {nq_obs}')
        print(f'nw_obs    = {nw_obs}')
        print(f'nx_obs    = {nx_obs}')
        print(' ')
        if ind_exec_solver == 2:
            print(' ')
            print(f'ind_type_targ     = {ind_type_targ}')
            print(' ')
            print(f'     nqreal_targ  = {nqreal_targ}')
            print(f'     nwreal_targ  = {nwreal_targ}')
            print(f'     nbreal_targ  = {nbreal_targ}')
            print(' ')
            print(f'     nqpos_targ   = {nqpos_targ}')
            print(f'     nwpos_targ   = {nwpos_targ}')
            print(f'     nbpos_targ   = {nbpos_targ}')
            print(' ')
            print(f'     nx_targ      = {nx_targ}')
            if ind_type_targ == 1:
                print(' ')
                print(f'     N_r          = {N_r}')

    #--- Print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            if ind_exec_solver == 1:
                fidlisting.write(f'ind_exec_solver = {ind_exec_solver}, Direct Solver used \n ')
            if ind_exec_solver == 2:
                fidlisting.write(f'ind_exec_solver = {ind_exec_solver}, Inverse Solver used \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'n_q     = {n_q:9d} \n ')
            fidlisting.write(f'n_w     = {n_w:9d} \n ')
            fidlisting.write(f'n_x     = {n_x:9d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'nqreal  = {nqreal:9d} \n ')
            fidlisting.write(f'nwreal  = {nwreal:9d} \n ')
            fidlisting.write(f'nbreal  = {nbreal:9d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'nqpos   = {nqpos:9d} \n ')
            fidlisting.write(f'nwpos   = {nwpos:9d} \n ')
            fidlisting.write(f'nbpos   = {nbpos:9d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'nq_obs  = {nq_obs:9d} \n ')
            fidlisting.write(f'nw_obs  = {nw_obs:9d} \n ')
            fidlisting.write(f'nx_obs  = {nx_obs:9d} \n ')
            fidlisting.write('      \n ')
            if ind_exec_solver == 2:
                fidlisting.write('      \n ')
                fidlisting.write(f'ind_type_targ     = {ind_type_targ:9d} \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f'     nqreal_targ      = {nqreal_targ:9d} \n ')
                fidlisting.write(f'     nwreal_targ      = {nwreal_targ:9d} \n ')
                fidlisting.write(f'     nbreal_targ      = {nbreal_targ:9d} \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f'     nqpos_targ       = {nqpos_targ:9d} \n ')
                fidlisting.write(f'     nwpos_targ       = {nwpos_targ:9d} \n ')
                fidlisting.write(f'     nbpos_targ       = {nbpos_targ:9d} \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f'     nx_targ          = {nx_targ:9d} \n ')
                if ind_type_targ == 1:
                    fidlisting.write('      \n ')
                    fidlisting.write(f'     N_r              = {N_r:9d} \n ')
                fidlisting.write('      \n ')

    ElapsedTimeDataCheck = time.time() - TimeStartDataCheck

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ----- Elapsed time for Task1_DataStructureCheck \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' Elapsed Time   =  {ElapsedTimeDataCheck:10.2f}\n')
            fidlisting.write('      \n ')

    if ind_display_screen == 1:
        print('--- end Task1_DataStructureCheck')

    return n_x, Indx_real, Indx_pos, nx_obs, Indx_obs, Indx_targ_real, Indx_targ_pos, nx_targ, Indx_targ

