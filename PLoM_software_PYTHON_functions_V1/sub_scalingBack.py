import numpy as np
import time
from sub_scalingBack_standard import sub_scalingBack_standard

def sub_scalingBack(nx_obs, n_x, n_ar, MatRx_obs, Indx_real, Indx_pos, Indx_obs, Rbeta_scale_real, Ralpha_scale_real,
                    Rbeta_scale_log, Ralpha_scale_log, ind_display_screen, ind_print, ind_scaling):
    
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_scalingBack
    #  Subject      : constructing the n_ar realizations MatRxx_obs(nx_obs,n_ar) of the nx_obs <= n_x unscaled observations XX_obs from the 
    #                 n_ar realizations MatRx_obs(nx_obs,n_ar) of the nx_obs scaled observations X_obs 
    #                 This is the inverse transform of the scaling done by the function sub_scaling1_main, restricted to the observations
    #
    #  Publications: 
    #               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #
    #  Function definition: 
    #                     Indx_real(nbreal) : contains the nbreal component numbers of X_ar and XX_ar that are real (positive, negative, 
    #                                          or zero) with 0 <= nbreal <=  n_x, and for which a "standard scaling" is used.
    #                     Indx_pos(nbpos)   : contains the nbpos component numbers of X_ar and XX_ar that are strictly positive, 
    #                                          with  0 <= nbpos <=  n_x , and for which the scaling {log + "standard scaling"} is used.
    #                     Indx_obs(nx_obs)  : contains the nx_obs components of XX_ar (unscaled) and X_ar (scaled) that are observed 
    #                                          with nx_obs <= n_x
    #
    #--- INPUTS
    #          nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled)
    #          n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)
    #          n_ar                        : number of points in the learning set for X_obs and XX_obs
    #          MatRx_obs(nx_obs,n_ar)      : n_ar realizations of X_obs
    #          Indx_real(nbreal)           : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
    #          Indx_pos(nbpos)             : nbpos component numbers of XX_ar that are strictly positive 
    #          Indx_obs(nx_obs)            : nx_obs component numbers of XX_ar that are observed with nx_obs <= n_x
    #          Rbeta_scale_real(nbreal)    : loaded if nbreal >= 1 or = [] if nbreal  = 0               
    #          Ralpha_scale_real(nbreal)   : loaded if nbreal >= 1 or = [] if nbreal  = 0    
    #          Rbeta_scale_log(nbpos)      : loaded if nbpos >= 1  or = [] if nbpos = 0                 
    #          Ralpha_scale_log(nbpos)     : loaded if nbpos >= 1  or = [] if nbpos = 0   
    #          ind_display_screen          : = 0 no display, = 1 display
    #          ind_print                   : = 0 no print,   = 1 print
    #          ind_scaling                 : = 0 no scaling
    #                                      : = 1    scaling
    #
    #--- OUTPUTS
    #          MatRxx_obs(nx_obs,n_ar)      : n_ar realizations of XX_obs
      
    if ind_display_screen == 1:
        print('--- beginning Task11_ScalingBack')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write(' ------ Task11_ScalingBack \n')
            fidlisting.write('\n')

    TimeStartScale = time.time()
     
    #--- Checking input data and parameters concerning Indx_obs(nx_obs,1)
    if nx_obs > n_x:
        raise ValueError('STOP1 in sub_scalingBack: nx_obs > n_x')
    nobstemp, nartemp = MatRx_obs.shape  # MatRx_obs(nx_obs,n_ar)
    if nobstemp != nx_obs or nartemp != n_ar:
        raise ValueError('STOP2 in sub_scalingBack: dimension errors in MatRx_obs(nx_obs,n_ar)')
    if len(Indx_obs) != len(np.unique(Indx_obs)):
        raise ValueError('STOP3 in sub_scalingBack: there are repetitions in Indx_obs')  # There are repetitions in Indx_obs
    if np.any(Indx_obs < 1) or np.any(Indx_obs > n_x):
        raise ValueError('STOP4 in sub_scalingBack: at least one integer in Indx_obs is not within the valid range')  # At least one integer in Indx_obs is not within the valid range.
 
    #--- Loading
    nbreal = len(Indx_real)  # Indx_real(nbreal)
    nbpos  = len(Indx_pos)    # Indx_pos(nbpos)

    #--- Initialization
    nbpos_obs  = 0
    nbreal_obs = 0
   
    if nbpos >= 1:  # List_pos(nbpos_obs,1) contains the positive-valued observation numbers
        List_pos             = np.zeros(nbpos, dtype=int)  # nbpos_obs <= nbpos, nbpos_obs is unknown, then initialization: List_pos(nbpos)
        Rbeta_scale_log_obs  = np.zeros(nbpos)             # Rbeta_scale_log_obs(nbpos_obs)
        Ralpha_scale_log_obs = np.zeros(nbpos)             # Ralpha_scale_log_obs(nbpos_obs)
        for ipos in range(nbpos):
            jXpos = Indx_pos[ipos]
            for iobs in range(nx_obs):
                if jXpos == Indx_obs[iobs]:
                    nbpos_obs = nbpos_obs + 1
                    List_pos[nbpos_obs - 1] = iobs + 1
                    Rbeta_scale_log_obs[nbpos_obs - 1]  = Rbeta_scale_log[ipos]
                    Ralpha_scale_log_obs[nbpos_obs - 1] = Ralpha_scale_log[ipos]
        if nbpos_obs < nbpos:            
            List_pos             = List_pos[:nbpos_obs]
            Rbeta_scale_log_obs  = Rbeta_scale_log_obs[:nbpos_obs]
            Ralpha_scale_log_obs = Ralpha_scale_log_obs[:nbpos_obs]
        MatRx_obs_log = MatRx_obs[List_pos-1, :]  # MatRx_obs_log(nbpos_obs,n_ar), MatRx_obs(nx_obs,n_ar), List_pos(nbpos_obs,1)
    
    if nbreal >= 1:                                          # List_real(nbreal_obs,1) contains the real-valued observation numbers
        List_real             = np.zeros(nbreal, dtype=int)  # nbreal_obs <= nbreal, nbreal_obs is unknown, then initialization: List_real(nbreal)
        Rbeta_scale_real_obs  = np.zeros(nbreal)             # Rbeta_scale_real_obs(nbreal_obs,1)
        Ralpha_scale_real_obs = np.zeros(nbreal)             # Ralpha_scale_real_obs(nbreal_obs,1)
        for ireal in range(nbreal):
            jXreal = Indx_real[ireal]
            for iobs in range(nx_obs):
                if jXreal == Indx_obs[iobs]:
                    nbreal_obs = nbreal_obs + 1
                    List_real[nbreal_obs - 1] = iobs + 1
                    Rbeta_scale_real_obs[nbreal_obs - 1]  = Rbeta_scale_real[ireal]
                    Ralpha_scale_real_obs[nbreal_obs - 1] = Ralpha_scale_real[ireal]
        if nbreal_obs < nbreal:            
            List_real             = List_real[:nbreal_obs]
            Rbeta_scale_real_obs  = Rbeta_scale_real_obs[:nbreal_obs]
            Ralpha_scale_real_obs = Ralpha_scale_real_obs[:nbreal_obs]
        MatRx_obs_real = MatRx_obs[List_real-1, :]  # MatRx_obs_real(nbreal_obs,n_ar), MatRx_obs(nx_obs,n_ar), List_real(nbreal_obs)
            
    #--- Print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write('\n')
            fidlisting.write(f'n_x        = {n_x:9d} \n')
            fidlisting.write(f'nbreal     = {nbreal:9d} \n')
            fidlisting.write(f'nbpos      = {nbpos:9d} \n')
            fidlisting.write('\n')
            fidlisting.write(f'nx_obs      = {nx_obs:9d} \n')
            fidlisting.write(f'nbreal_obs = {nbreal_obs:9d} \n')
            fidlisting.write(f'nbpos_obs  = {nbpos_obs:9d} \n')
            fidlisting.write('\n')
    
    if nbreal_obs + nbpos_obs != nx_obs:
        raise ValueError('STOP5 in sub_scalingBack1')

    #--- Scaling 
    if nbreal >= 1:
        if ind_scaling == 0:
            MatRxx_obs_real = MatRx_obs_real
        if ind_scaling == 1:
            MatRxx_obs_real = sub_scalingBack_standard(nbreal_obs, n_ar, MatRx_obs_real, Rbeta_scale_real_obs, Ralpha_scale_real_obs)
    
    if nbpos >= 1:
        if ind_scaling == 0:
            MatRxx_obs_log = MatRx_obs_log
        if ind_scaling == 1:
            MatRxx_obs_log = sub_scalingBack_standard(nbpos_obs, n_ar, MatRx_obs_log, Rbeta_scale_log_obs, Ralpha_scale_log_obs)
        MatRxx_obs_pos = np.exp(MatRxx_obs_log)  # MatRxx_obs_pos(nbpos_obs,n_ar), MatRxx_obs_log(nbpos_obs,n_ar)

    #--- Construction of MatRxx_obs(nx_obs,n_ar)
    MatRxx_obs = np.zeros((nx_obs, n_ar))
    if nbreal_obs >= 1:
        MatRxx_obs[List_real-1, :] = MatRxx_obs_real  # MatRxx_obs(nx_obs,n_ar), MatRxx_obs_real(nbreal_obs,n_ar), List_real(nbreal_obs,1)
    if nbpos_obs >= 1:
        MatRxx_obs[List_pos-1, :] = MatRxx_obs_pos  # MatRxx_obs(nx_obs,n_ar), MatRxx_obs_pos(nbpos_obs,n_ar), List_pos(nbpos_obs,1)

    ElapsedTimeScale = time.time() - TimeStartScale

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write('\n')
            fidlisting.write(' ------ Elapsed time for Task11_ScalingBack \n')
            fidlisting.write('\n')
            fidlisting.write(f' Elapsed Time   =  {ElapsedTimeScale:10.2f}\n')
            fidlisting.write('\n')

    if ind_display_screen == 1:
        print('--- end Task11_ScalingBack')

    return MatRxx_obs
