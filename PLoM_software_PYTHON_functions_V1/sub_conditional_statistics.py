import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import time
import sys
from sub_conditional_PCAback import sub_conditional_PCAback 
from sub_conditional_scalingBack import sub_conditional_scalingBack
from sub_conditional_mean import sub_conditional_mean
from sub_conditional_second_order_moment import sub_conditional_second_order_moment
from sub_conditional_pdf import sub_conditional_pdf
from sub_conditional_confidence_interval import sub_conditional_confidence_interval

def sub_conditional_statistics(ind_mean, ind_mean_som, ind_pdf, ind_confregion,
                               n_x, n_q, nbParam, n_w, n_d, n_ar, nbMC, nu, MatRx_d, MatRxx_d, MatReta_ar, RmuPCA,
                               MatRVectPCA, Indx_real, Indx_pos, Indq_obs, Indw_obs, nx_obs, Indx_obs, ind_scaling,
                               Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log, Ralpha_scale_log,
                               nbw0_obs, MatRww0_obs, Ind_Qcomp, nbpoint_pdf, pc_confregion, ind_display_screen, ind_print):

    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 26 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_conditional_statistics
    #  Subject      : computation of conditional statistics:
    #                - Estimation of the conditional mean
    #                - Estimation of the conditional mean and second-order moment
    #                - Estimation of the conditional pdf of component jcomponent <= nq_obs
    #                - Estimation of the conditional confidence region
    #
    #  Publications [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                         American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020).   
    #               [3] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
    #                         for nonlinear dynamical systems, Computer Methods in Applied Mechanics and Engineering, 
    #                         doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024). 
    #               [ ] For the conditional statistics formula, see the Appendix of paper [3]
    #
    #--- INPUTS    
    # 
    #     ind_mean               : = 0  No estimation of the conditional mean
    #                              = 1     estimation of the conditional mean
    #     ind_mean_som           : = 0 No estimation of the conditional mean and second-order moment
    #                              = 1    estimation of the conditional mean and second-order moment
    #     ind_pdf                : = 0 No estimation of the conditional pdf of component jcomponent <= nq_obs
    #                              = 1    estimation of the conditional pdf of component jcomponent <= nq_obs
    #     ind_confregion         : = 0 No estimation of the conditional confidence region
    #                              = 1    estimation of the conditional confidence region
    #
    #     n_x                    : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
    #     n_q                    : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q    
    #     nbParam                : number of sampling point of the physical parameters 1 <= nbParam <= n_q
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
    #     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
    #     n_d                         : number of points in the training set for XX_d and X_d  
    #     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
    #     nbMC                        : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]    
    #     nu                          : order of the PCA reduction, which is the dimension of H_ar 
    #     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
    #     MatRxx_d(n_x,n_d)           : n_d realizations of XX_d (unscaled)
    #     MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar 
    #     RmuPCA(nu)                  : vector of PCA eigenvalues in descending order
    #     MatRVectPCA(n_x,nu)         : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA   
    #     Indx_real(nbreal)           : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
    #     Indx_pos(nbpos)             : nbpos component numbers of XX_ar that are strictly positive 
    #     Indq_obs(nq_obs)            : nq_obs component numbers of QQ that are observed , 1 <= nq_obs <= n_q
    #     Indw_obs(nw_obs)            : nw_obs component numbers of WW that are observed,  1 <= nw_obs <= n_w
    #     nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled) (extracted from X_ar)  
    #     Indx_obs(nx_obs)            : nx_obs component numbers of X_ar and XX_ar that are observed with nx_obs <= n_x  
    #     ind_scaling                 : = 0 no scaling
    #                                 : = 1    scaling
    #     Rbeta_scale_real(nbreal)    : loaded if nbreal >= 1 or = [] if nbreal  = 0               
    #     Ralpha_scale_real(nbreal)   : loaded if nbreal >= 1 or = [] if nbreal  = 0    
    #     Rbeta_scale_log(nbpos)      : loaded if nbpos >= 1  or = [] if nbpos = 0                 
    #     Ralpha_scale_log(nbpos)     : loaded if nbpos >= 1  or = [] if nbpos = 0  
    #
    #     nbw0_obs                     : number of vectors Rww0_obs de WW_obs  used for conditional statistics  Q_obs | WW_obs = Rww0_obs
    #     MatRww0_obs(nw_obs,nbw0_obs) : MatRww0_obs(:,kw0) = Rww0_obs_kw0(nw_obs,1)
    #                                  --- for ind_pdf = 1
    #     Ind_Qcomp(nbQcomp)           :   pdf of Q_obs(k) | WW_obs = Rww0_obs for k = Ind_Qcomp(kcomp), kcomp = 1,...,nbQcomp
    #                                      where 1 <= k < = nq_obs is such that MatRqq_obs(k,:) are the n_ar realizations of Q_obs(k)
    #                                      note that for ind_pdf = 0, Ind_Qcomp = [] 
    #     nbpoint_pdf                  :   number of points in which the pdf is computed
    # 
    #     pc_confregion                : only used if ind_confregion (example pc_confregion =  0.98)
    #
    #     ind_display_screen  : = 0 no display,  = 1 display
    #     ind_print           : = 0 no print,    = 1 print
    #
    #--- OUPUTS  
    #     []

    if ind_display_screen == 1:
        print('--- beginning Task13_ConditionalStatistics')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write(' ------ Task13_ConditionalStatistics \n')
            fidlisting.write('\n')

    TimeStartCondStats = time.time()
    numfig = 0

    #-------------------------------------------------------------------------------------------------------------------------------------   
    #             Checking parameters and data, and loading MatRqq_obs(nq_obs,n_ar) and MatRww_obs(nw_obs,n_ar) 
    #-------------------------------------------------------------------------------------------------------------------------------------
 
    #--- Checking parameters and data
    if n_x <= 0:
        raise ValueError('STOP1 in sub_conditional_statistics: n_x <= 0')
    if n_q <= 0 or n_w <= 0:
        raise ValueError('STOP2 in sub_conditional_statistics: n_q <= 0 or n_w <= 0')
    nxtemp = n_q + n_w  # dimension of random vector XX = (QQ,WW)
    if nxtemp != n_x:
        raise ValueError('STOP3 in sub_conditional_statistics: n_x not equal to n_q + n_w')
    if nbParam <= 0 or nbParam > n_q:
        raise ValueError('STOP4 in sub_conditional_statistics: integer nbParam must be in interval [1,n_q]')
    if n_d <= 0:
        raise ValueError('STOP5 in sub_conditional_statistics: n_d <= 0')
    if n_ar <= 0:
        raise ValueError('STOP6 in sub_conditional_statistics: n_ar <= 0')
    if nbMC <= 0:
        raise ValueError('STOP7 in sub_conditional_statistics: nbMC <= 0')
    if nu <= 0 or nu >= n_d:
        raise ValueError('STOP8 in sub_conditional_statistics: nu <= 0 or nu >= n_d')

    (n1temp, n2temp) = MatRx_d.shape          # MatRx_d(n_x,n_d)
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP9 in sub_conditional_statistics: dimension error in matrix MatRx_d(n_x,n_d)')
    (n1temp, n2temp) = MatRxx_d.shape         # MatRxx_d(n_x,n_d)
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP10 in sub_conditional_statistics: dimension error in matrix MatRxx_d(n_x,n_d)')
    (n1temp, n2temp) = MatReta_ar.shape        # MatReta_ar(nu,n_ar)
    if n1temp != nu or n2temp != n_ar:
        raise ValueError('STOP11 in sub_conditional_statistics: dimension error in matrix MatReta_ar(nu,n_ar)')
    n1temp = len(RmuPCA)                       # RmuPCA(nu)
    if n1temp != nu: 
        raise ValueError('STOP12 in sub_conditional_statistics: dimension error in matrix RmuPCA(nu)')
    (n1temp, n2temp) = MatRVectPCA.shape       # MatRVectPCA(n_x,nu)
    if n1temp != n_x or n2temp != nu:
        raise ValueError('STOP13 in sub_conditional_statistics: dimension error in matrix MatRVectPCA(n_x,nu)')
    
    nbreal = len(Indx_real)                              # Indx_real(nbreal)
    if nbreal >= 1:
      ntemp = Indx_real.ndim                  
      if ntemp != 1:
        raise ValueError('STOP14 in sub_conditional_statistics: dimension error in 1D array Indx_real(nbreal)')

    nbpos = len(Indx_pos)                                # Indx_pos(nbpos)
    if nbpos >= 1:
      ntemp= Indx_pos.ndim                  
      if ntemp != 1:
        raise ValueError('STOP15 in sub_conditional_statistics: dimension error in 1D array Indx_pos(nbpos)')
    
    nxtemp = nbreal + nbpos
    if nxtemp != n_x:
        raise ValueError('STOP16 in sub_conditional_statistics: n_x not equal to nreal + nbpos')

    # Loading dimension nq_obs of Indq_obs(nq_obs)
    nq_obs = len(Indq_obs)                          # Indq_obs(nq_obs)
   
    # Checking input data and parameters of Indq_obs(nq_obs)
    if nq_obs < 1 or nq_obs > n_q:
        raise ValueError('STOP17 in sub_conditional_statistics: nq_obs < 1 or nq_obs > n_q')
   
    if len(Indq_obs) != len(np.unique(Indq_obs)):
        raise ValueError('STOP19 in sub_conditional_statistics: there are repetitions in Indq_obs')
    if np.any(Indq_obs < 1) or np.any(Indq_obs > n_q):
        raise ValueError('STOP20 in sub_conditional_statistics: at least one integer in Indq_obs is not within the valid range [1 , n_q]')

    # Loading dimension nw_obs of Indw_obs(nw_obs)
    nw_obs = len(Indw_obs)  # Indw_obs(nw_obs)

    # Checking input data and parameters of Indw_obs(nw_obs)
    if nw_obs < 1 or nw_obs > n_w:
        raise ValueError('STOP21 in sub_conditional_statistics: nw_obs < 1 or nw_obs > n_w')
   
    if len(Indw_obs) != len(np.unique(Indw_obs)):
        raise ValueError('STOP23 in sub_conditional_statistics: there are repetitions in Indw_obs')
    if np.any(Indw_obs < 1) or np.any(Indw_obs > n_w):
        raise ValueError('STOP24 in sub_conditional_statistics: at least one integer in Indw_obs is not within the valid range [1 , n_w]')

    if nx_obs <= 0:
        raise ValueError('STOP25 in sub_conditional_statistics: nx_obs <= 0')
    
    n1temp = len(Indx_obs)  # Indx_obs(nx_obs)
    if n1temp != nx_obs:
        raise ValueError('STOP26 in sub_conditional_statistics: dimension error in matrix Indx_obs(nx_obs,1)')
    
    if ind_scaling != 0 and ind_scaling != 1:
        raise ValueError('STOP27 in sub_conditional_statistics: ind_scaling must be equal to 0 or to 1')
    
    if nbreal >= 1:
        n1temp = len(Rbeta_scale_real) # Rbeta_scale_real(nbreal)
        if n1temp != nbreal:
            raise ValueError('STOP28 in sub_conditional_statistics: dimension error in matrix Rbeta_scale_real(nbreal)')
        n1temp = len(Ralpha_scale_real)  # Ralpha_scale_real(nbreal)
        if n1temp != nbreal:
            raise ValueError('STOP29 in sub_conditional_statistics: dimension error in matrix Ralpha_scale_real(nbreal)')
        
    if nbpos >= 1:
        n1temp = len(Rbeta_scale_log)  # Rbeta_scale_log(nbpos)
        if n1temp != nbpos:
            raise ValueError('STOP30 in sub_conditional_statistics: dimension error in matrix Rbeta_scale_log(nbpos)')
        n1temp = len(Ralpha_scale_log)  # Ralpha_scale_log(nbpos)
        if n1temp != nbpos:
            raise ValueError('STOP31 in sub_conditional_statistics: dimension error in matrix Ralpha_scale_log(nbpos)')
        
    if nbw0_obs <= 0:
        raise ValueError('STOP32 in sub_conditional_statistics: nbw0_obs must be greater than or equal to 1')
    
    (n1temp, n2temp) = MatRww0_obs.shape                        # MatRww0_obs(nw_obs,nbw0_obs)
    if n1temp != nw_obs or n2temp != nbw0_obs:
        raise ValueError('STOP33 in sub_conditional_statistics: dimension error in matrix MatRww0_obs(nw_obs,nbw0_obs)')
    
    if ind_pdf == 0:                                            # Check if matrix Ind_Qcomp is empty
        if Ind_Qcomp.size != 0:
            raise ValueError('STOP34 in sub_conditional_statistics: for ind_pdf = 0, matrix Ind_Qcomp must be empty')
                
    if ind_pdf == 1:
        nbQcomp = len(Ind_Qcomp)                                # Ind_Qcomp(nbQcomp)        
        for kcomp in range(nbQcomp):
            kk = Ind_Qcomp[kcomp]                               # Ind_Qcomp(nbQcomp)
            ind_error = 0
            for iqobs in range(nq_obs):
                if kk == Indq_obs[iqobs]:                       # Indq_obs(nq_obs)
                    ind_error = 1
                    break
            if ind_error == 0:
                raise ValueError('STOP37 in sub_conditional_statistics: some values in Ind_Qcomp(nbQcomp) are not in Indq_obs(nq_obs)')
        if nbpoint_pdf < 1:
            raise ValueError('STOP38 in sub_conditional_statistics: for ind_pdf = 1, nbpoint_pdf must be greater than of equal to 1')
        
    if ind_confregion == 1:
        if pc_confregion <= 0 or pc_confregion >= 1:
            raise ValueError('STOP39 in sub_conditional_statistics: for ind_confregion = 1, pc_confregion must be in the real interval ]0,1[')

    # Case nbParam = 1
    if nbParam == 1:
        nqobsPhys = nq_obs

    # Checking that, if nbParam > 1, nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number of the components of
    # the state variables that are observed. Consequently, nq_obs/nbParam must be an integer, if not there is an error in the given
    # value of nbParam of in the Data generation in "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m"

    if (nbParam > 1) and (ind_mean == 1 or ind_mean_som == 1 or ind_confregion == 1):
        # calculate nqobsPhys and check if it is an integer
        nqobsPhys = int(nq_obs / nbParam)
        # test if nqobsPhys is an integer
        if nq_obs % nbParam != 0:                      # matlab equiavalence: if mod(nq_obs, nbParam) ~= 0
            raise ValueError('STOP40 in sub_conditional_statistics: nq_obs divided by nbParam is not an integer. \
                             There is either an error in the given value of nbParam, or in the data generation in \
                             function mainWorkflow_Data_generation1.m and function mainWorkflow_Data_generation2.m')

    #--- PCA back: MatRx_obs(nx_obs,n_ar)
    MatRx_obs = sub_conditional_PCAback(n_x,n_d,nu,n_ar,nx_obs,MatRx_d,MatReta_ar,Indx_obs,RmuPCA,MatRVectPCA,ind_display_screen,ind_print)

    #--- Scaling back: MatRxx_obs(nx_obs,n_ar)
    MatRxx_obs = sub_conditional_scalingBack(nx_obs, n_x, n_ar, MatRx_obs, Indx_real, Indx_pos, Indx_obs, Rbeta_scale_real, Ralpha_scale_real,
                                             Rbeta_scale_log, Ralpha_scale_log, ind_display_screen, ind_print, ind_scaling)
    del MatRx_obs

    #--- Loading MatRqq_obs(nq_obs,n_ar) and MatRww_obs(nw_obs,n_ar) from MatRxx_obs(nx_obs,n_ar)
    MatRqq_obs = MatRxx_obs[:nq_obs, :n_ar]               # MatRqq_obs(nq_obs,n_ar),MatRxx_obs(nx_obs,n_ar)
    MatRww_obs = MatRxx_obs[nq_obs:nq_obs+nw_obs, :n_ar]  # MatRww_obs(nw_obs,n_ar),MatRxx_obs(nx_obs,n_ar)
    del MatRxx_obs

    #--- Construction of MatRqq_d_obs(nq_obs,n_d) and MatRww_d_obs(nw_obs,n_d) from MatRxx_d(n_x,n_d)
    MatRxx_d_obs = MatRxx_d[Indx_obs - 1, :]                     # MatRxx_d_obs(nx_obs,n_d),MatRxx_d(n_x,n_d),Indx_obs(nx_obs)
    MatRqq_d_obs = MatRxx_d_obs[:nq_obs, :n_d]                   # MatRqq_d_obs(nq_obs,n_d),MatRxx_d_obs(nx_obs,n_d)
    MatRww_d_obs = MatRxx_d_obs[nq_obs:nq_obs+nw_obs, :n_d]      # MatRww_d_obs(nw_obs,n_d),MatRxx_d_obs(nx_obs,n_d)
    del MatRxx_d_obs

    #--- print  
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write(f' ind_mean      = {ind_mean:9d} \n')
            fidlisting.write(f' ind_mean_som  = {ind_mean_som:9d} \n')
            fidlisting.write(f' ind_pdf       = {ind_pdf:9d} \n')
            fidlisting.write(f' ind_confregion= {ind_confregion:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f' n_q           = {n_q:9d} \n')
            fidlisting.write(f' nbParam       = {nbParam:9d} \n')
            fidlisting.write(f' n_w           = {n_w:9d} \n')
            fidlisting.write(f' n_x           = {n_x:9d} \n')
            fidlisting.write(f' nbreal        = {nbreal:9d} \n')
            fidlisting.write(f' nbpos         = {nbpos:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f' nq_obs        = {nq_obs:9d} \n')           
            fidlisting.write(f' nbParam       = {nbParam:9d} \n')
            fidlisting.write(f' nqobsPhys     = {nqobsPhys:9d} \n')
            fidlisting.write(f' nw_obs        = {nw_obs:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f' ind_scaling   = {ind_scaling:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f' n_d           = {n_d:9d} \n')
            fidlisting.write(f' nbMC          = {nbMC:9d} \n')
            fidlisting.write(f' n_ar          = {n_ar:9d} \n')
            fidlisting.write(f' nu            = {nu:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f' nbw0_obs      = {nbw0_obs:9d} \n')
            fidlisting.write('\n')
            if ind_pdf == 1:
                fidlisting.write(f' nbQcomp       = {nbQcomp:9d} \n')
                fidlisting.write(f' nbpoint_pdf   = {nbpoint_pdf:9d} \n')
            if ind_confregion == 1:
                fidlisting.write(f' pc_confregion = {pc_confregion:9.2f} \n')
            fidlisting.write('\n')
            fidlisting.write(f' ind_display_screen = {ind_display_screen:1d} \n')
            fidlisting.write(f' ind_print          = {ind_print:1d} \n')
            fidlisting.write('\n')
            fidlisting.write('\n')

    #-------------------------------------------------------------------------------------------------------------------------------------   
    #                             Computing and plot of conditional statistics
    #-------------------------------------------------------------------------------------------------------------------------------------
    # nqobsPhys = nq_obs/nbParam
    if nbParam == 1:
        mobsPhys = 1
        mq       = nq_obs
        ArrayMatRqq            = np.zeros((mq, n_ar, 1))
        ArrayMatRqq_d          = np.zeros((mq, n_d, 1))
        ArrayMatRqq[:, :, 0]   = MatRqq_obs                 # ArrayMatRqq(mq,n_ar,1),MatRqq_obs(nq_obs,n_ar)
        ArrayMatRqq_d[:, :, 0] = MatRqq_d_obs               # ArrayMatRqq_d(mq,n_d,1),MatRqq_d_obs(nq_obs,n_d)
    if nbParam > 1:
        mobsPhys      = nqobsPhys
        mq            = nbParam
        ArrayMatRqq   = np.zeros((mq, n_ar, nqobsPhys))
        ArrayMatRqq_d = np.zeros((mq, n_d, nqobsPhys))
        istart = 1
        for iobsPhys in range(nqobsPhys):
            iend = istart + nbParam - 1
            ArrayMatRqq[:, :, iobsPhys]   = MatRqq_obs[istart-1:iend, :]
            ArrayMatRqq_d[:, :, iobsPhys] = MatRqq_d_obs[istart-1:iend, :]
            istart = iend + 1

    for kw0 in range(nbw0_obs):         # nbw0_obs: number of Rww0_obs for WW_obs used for conditional statistics Q_obs | WW_obs = Rww0_obs 
                                        #           and Q_d_obs | WW_obs = Rww0_obs
        Rww0_obs = MatRww0_obs[:, kw0]  # Rww0_obs(nw_obs),MatRww0_obs(nw_obs,nw0_obs)

        #--- Estimation of the conditional mean
        if ind_mean == 1:
            for iobsPhys in range(mobsPhys):
                MatRqq     = ArrayMatRqq[:, :, iobsPhys]
                MatRqq_d   = ArrayMatRqq_d[:, :, iobsPhys]
                REqq_ww0   = sub_conditional_mean(nw_obs, mq, n_ar, MatRqq, MatRww_obs, Rww0_obs)
                REqq_d_ww0 = sub_conditional_mean(nw_obs, mq, n_d, MatRqq_d, MatRww_d_obs, Rww0_obs)
                # Plot
                Rk = np.arange(1, mq + 1)                        # generate: 1,2,...,mq
                plt.figure()
                plt.plot(Rk, REqq_ww0, linestyle='-', linewidth=1, color='b')
                plt.plot(Rk, REqq_d_ww0,linestyle='-', linewidth=0.5, color='k')
                plt.xlabel(r'$k$', fontsize=16)
                plt.ylabel(r'$E\{ Q_k \vert W = w_0(:,' + str(kw0+1) + r')\}$', fontsize=16)
                plt.title(r'$E\{ Q_k$ (blue thick) and ${Q_d}_k$ (black thin) $\vert W = w_0(:,' + str(kw0+1) + r')\}$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_CondStats{numfig}_mean_w0_{kw0+1}_obsPhys_{iobsPhys+1}.png')
                plt.close()

        #--- Estimation of the conditional mean and second-order moment
        if ind_mean_som == 1:
            for iobsPhys in range(mobsPhys):
                MatRqq    = ArrayMatRqq[:, :, iobsPhys]
                MatRqq_d  = ArrayMatRqq_d[:, :, iobsPhys]
                (REqq_ww0, REqq2_ww0)     = sub_conditional_second_order_moment(nw_obs, mq, n_ar, MatRqq, MatRww_obs, Rww0_obs)
                (REqq_d_ww0, REqq2_d_ww0) = sub_conditional_second_order_moment(nw_obs, mq, n_d, MatRqq_d, MatRww_d_obs, Rww0_obs)

                # Plot mean value
                Rk = np.arange(1, mq + 1)
                plt.figure()
                plt.plot(Rk, REqq_ww0,linestyle='-',linewidth=1,color='b')
                plt.plot(Rk, REqq_d_ww0,linestyle='-',linewidth=0.5,color='k')
                plt.xlabel(r'$k$', fontsize=16)
                plt.ylabel(r'$E\{ Q_k \vert W = w_0(:, ' + str(kw0+1) + r')\}$', fontsize=16)
                plt.title(r'$E\{ Q_k$ (blue thick) and ${Q_d}_k$ (black thin) $\vert W = w_0(:,' + str(kw0+1) + r')\}$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_CondStats{numfig}_mean_w0_{kw0+1}_obsPhys_{iobsPhys+1}.png')
                plt.close()

                # Plot second-order moment
                plt.figure()
                plt.plot(Rk, REqq2_ww0,linestyle='-',linewidth=1,color='b')
                plt.plot(Rk, REqq2_d_ww0,linestyle='-',linewidth=0.5,color='k')
                plt.xlabel(r'$k$', fontsize=16)
                plt.ylabel(r'$E\{ Q_k^2 \vert W = w_0(:,' + str(kw0+1) + r')\}$', fontsize=16)
                plt.title(r'$E\{ Q_k^2$ (blue thick) and ${Q_d^2}_k$ (black thin) $\vert W = w_0(:,' + str(kw0+1) + r')\}$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_CondStats{numfig}_mean_som_w0_{kw0+1}_obsPhys_{iobsPhys+1}.png')
                plt.close()

        #--- Estimation of the conditional pdf of component k <= nq_obs
        if ind_pdf == 1:
            for kcomp in range(nbQcomp):
                kk = Ind_Qcomp[kcomp]                       # Ind_Qcomp(nbQcomp)
                for iqobs in range(nq_obs):
                    if kk == Indq_obs[iqobs]:
                        k = iqobs
                        break
                MatRqq_obs_k = MatRqq_obs[k, :]            # MatRqq_obs_k(n_ar)
                MatRqq_d_obs_k = MatRqq_d_obs[k, :]        # MatRqq_d_obs_k(1,n_d)
                (_, Rq, Rpdfqq_ww0) = sub_conditional_pdf(nw_obs, n_ar, MatRqq_obs_k, MatRww_obs, Rww0_obs, nbpoint_pdf)
                (_, Rq_d, Rpdfqq_d_ww0) = sub_conditional_pdf(nw_obs, n_d, MatRqq_d_obs_k, MatRww_d_obs, Rww0_obs, nbpoint_pdf)

                # Plot
                plt.figure()
                plt.plot(Rq, Rpdfqq_ww0,linestyle='-',linewidth=1,color='b')
                plt.plot(Rq_d, Rpdfqq_d_ww0,linestyle='-',linewidth=0.5,color='k')
                plt.xlabel(r'$q_{' + str(kk) + '}$', fontsize=16)
                plt.ylabel(r'$p_{{Q_{{{}}} \mid W}}(q_{{{}}})$'.format(kk, kk), fontsize=16)
                plt.title(r'pdf of $Q_{' + str(kk) + '}$ (blue thick) and of \n ${Q_d}_{' + str(kk) + r'}$ (black thin) $\mid W = w_0(:, ' + str(kw0+1) + r')$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_CondStats{numfig}_pdfQ{kk}_w0_{kw0+1}.png')
                plt.close()

        #--- Estimation of the conditional confidence region
        if ind_confregion == 1:
            for iobsPhys in range(mobsPhys):
                MatRqq = ArrayMatRqq[:, :, iobsPhys]

                # Confidence region computation
                (RqqLower_ww0, RqqUpper_ww0) = sub_conditional_confidence_interval(nw_obs, mq, n_ar, MatRqq, MatRww_obs, Rww0_obs, pc_confregion)
                RDplus = RqqUpper_ww0.T
                RDmoins = RqqLower_ww0.T
                Rkp = np.arange(1, mq + 1)               # np.arange(start, stop, step), stop is not included, and step = 1
                Rkm = np.arange(mq, 0, -1)               # np.arange(start, stop, step), stop is not included
                RDmoinsinv = RDmoins[Rkm - 1]
                Rpm = np.concatenate([Rkp, Rkm])
                RDrc = np.concatenate([RDplus.T, RDmoinsinv.T])
                # Mean value computation of the conditional mean with the lean dataset and the training dataset
                MatRqq_d   = ArrayMatRqq_d[:, :, iobsPhys]
                REqq_ww0   = sub_conditional_mean(nw_obs, mq, n_ar, MatRqq, MatRww_obs, Rww0_obs)
                REqq_d_ww0 = sub_conditional_mean(nw_obs, mq, n_d, MatRqq_d, MatRww_d_obs, Rww0_obs)

                # Plot
                Rk = np.arange(1, mq + 1)
                h = plt.figure()
                plt.fill(Rpm,RDrc,'y',linewidth=1,facecolor=[1, 1, 0],edgecolor=[0.850980392156863, 0.325490196078431, 0.0980392156862745])
                plt.plot(Rk,REqq_ww0,linestyle='-',linewidth=1,color='b')
                plt.plot(Rk,REqq_d_ww0,linestyle='-',linewidth=0.5,color='k')
                plt.xlabel(r'$k$',fontsize=16)
                plt.ylabel(r'$\{{ Q_k \vert W = w_0(:,{}))\}}$'.format(kw0+1), fontsize=16)    
                plt.title(r'Confidence region of $\{{ Q_k \vert W = w_0(:,{}\}}$ and conditional'.format(kw0+1) + 
                          '\n' +  r'mean value of $Q_k$ (blue thick) and ${Q_d}_k$ (black thin)', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_CondStats{numfig}_confregion_w0_{kw0+1}_obsPhys_{iobsPhys+1}.png')
                plt.close(h)

    ElapsedTimeCondStats = time.time() - TimeStartCondStats

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write('\n')
            fidlisting.write(' ----- Elapsed time for Task13_ConditionalStatistics \n')
            fidlisting.write('\n')
            fidlisting.write(f' Elapsed Time   =  {ElapsedTimeCondStats:10.2f}\n')
            fidlisting.write('\n')
            fidlisting.write('\n')
    
    if ind_display_screen == 1:
        print('--- end Task13_ConditionalStatistics')

    return