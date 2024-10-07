import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
import time
import random
import math
import sys

from sub_polynomialChaosQWU_PCAback import sub_polynomialChaosQWU_PCAback
from sub_polynomialChaosQWU_scalingBack import sub_polynomialChaosQWU_scalingBack
from sub_polynomialChaosQWU_chaos0 import sub_polynomialChaosQWU_chaos0
from sub_polynomialChaosQWU_chaosU import sub_polynomialChaosQWU_chaosU
from sub_polynomialChaosQWU_Jcost import sub_polynomialChaosQWU_Jcost
from sub_polynomialChaosQWU_surrogate import sub_polynomialChaosQWU_surrogate
from sub_polynomialChaosQWU_OVL import sub_polynomialChaosQWU_OVL
from sub_polynomialChaosQWU_print_plot import sub_polynomialChaosQWU_print_plot

def sub_polynomialChaosQWU(n_x, n_q, n_w, n_d, n_ar, nbMC, nu, MatRx_d, MatReta_ar, RmuPCA, 
                           MatRVectPCA, Indx_real, Indx_pos, Indq_obs, Indw_obs, nx_obs, Indx_obs, 
                           ind_scaling, Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log, 
                           Ralpha_scale_log, nbMC_PCE, Ng, Ndeg, ng, ndeg, MaxIter, 
                           SAVERANDstartPolynomialChaosQWU, ind_display_screen, ind_print, ind_plot, 
                           ind_parallel):
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 23 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_polynomialChaosQWU
    #  Subject      : Post processing allowing the construction of the polynomial chaos expansion (PCE) 
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
    #               [10] C. Soize, R. Ghanem, Physical systems with random uncertainties: Chaos representation with arbitrary probability 
    #                       measure, SIAM Journal on Scientific Computing, doi:10.1137/S1064827503424505, 26}2), 395-410 (2004).
    #               [11] C. Soize, C. Desceliers, Computational aspects for constructing realizations of polynomial chaos in high 
    #                       dimension}, SIAM Journal On Scientific Computing, doi:10.1137/100787830, 32(5), 2820-2831 (2010).
    #               [12] G. Perrin, C. Soize, D. Duhamel, C. Funfschilling, Identification of polynomial chaos representations in 
    #                       high dimension from a set of realizations, SIAM Journal on Scientific Computing, doi:10.1137/11084950X, 
    #                       34(6), A2917-A2945 (2012).
    #               [13] C. Soize, Q-D. To, Polynomial-chaos-based conditional statistics for probabilistic learning with heterogeneous
    #                       data applied to atomic collisions of Helium on graphite substrate, Journal of Computational Physics,
    #                       doi:10.1016/j.jcp.2023.112582, 496, 112582, pp.1-20 (2024).
    # 
    #--- Algebraic representation of the polynomial chaos for RQ^0 = sum_alpha Rgamma_alpha Psi_alpha(Xi) values in RR^n_q with
    #       RQ^0         = random vector with values in  RR^n_q  
    #       Rgamma_alpha = coefficient with values in RR^n_q
    #       Psi_alpha(Xi)= real-valued polynomial chaos 
    #       Xi           = (Xi_1, ... , Xi_Ng) random vector for the germ
    #       Ng           = dimension of the germ Xi = (Xi_1, ... , Xi_Ng) 
    #       alpha        = (alpha_1,...,alpha_Ng) multi-index of length Ng   
    #       Ndeg         = max degree of the polynomial chaos                                      
    #       K0           = factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg)) number of terms
    #                      including alpha^0 = (0, ... ,0) for which psi_alpha^0(Xi) = 1
    #
    #--- Algebraic representation of the polynomial chaos for [Gamma_r] = sum_a [g_a] phi_a(U) values in the set of (n_q x n_q) matrices
    #       [Gamma_r]    = random matrix with values in  (n_q x n_q) matrix
    #       [g_a]        = coefficient with values in (n_q x n_q) matrices 
    #       phi_a(U)     = real-valued normalized Hermite polynomial chaos 
    #       U            = (U_1, ... , U_ng) random vector for normalized Gaussian germ
    #       ng           = dimension of the germ U = (U_1, ... , U_ng) 
    #       a            = (a_1,...,a_ng) multi index of length ng   
    #       ndeg         = max degree of the polynomial chaos                                      
    #       KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
    #                      of terms NOT including a^0 = (0, ... ,0) for which phi_a^0(U) = 1
    #
    #--- INPUTS 
    #
    #     n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
    #     n_q                         : dimension of random vector QQ (unscaled quantitity of interest)  1 <= n_q    
    #     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
    #     n_d                         : number of points in the training set for XX_d and X_d  
    #     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
    #     nbMC                        : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]    
    #     nu                          : order of the PCA reduction, which is the dimension of H_ar 
    #     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
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
    #     nbMC_PCE                    : number of learned realizations used for PCE is nar_PCE= nbMC_PCE x n_d 
    #                                   (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible) 
    #     Ng                          : dimension of the germ for polynomial chaos Psi_alpha that is such that Ng = nw_obs   
    #     Ndeg                        : max degree of the polynomial chaos Psi_alpha with Ndeg >= 1  
    #     ng                          : dimension of the germ for polynomial chaos phi_a that is such that ng >= 1  
    #     ndeg                        : max degree of the polynomial chaos phi_a with ndeg >= 0 (if ndeg = 0, then KU = 1)  
    #     MaxIter                     : maximum number of iteration used by the quasi-Newton optimization algorithm (example 400)
    #  
    #     SAVERANDstartPolynomialChaosQWU : state of the random generator at the beginning
    #     ind_display_screen              : = 0 no display,              = 1 display
    #     ind_print                       : = 0 no print,                = 1 print
    #     ind_plot                        : = 0 no plot,                 = 1 plot
    #     ind_parallel                    : = 0 no parallel computation, = 1 parallel computation
    #
    #---- EXAMPLE OF DATA FOR EXPLAINING THE DATA STRUCTURE USING MATLAB INSTRUCTION
    #     component numbers of qq     = 1:100
    #     component numbers of ww     = 1:20
    #     Indq_obs(nq_obs,1)          = [2 4 6 8 80 98]', nq_obs = 6 
    #     Indw_obs(nw_obs,1)          = [1 3 8 15 17]'  , nw_obs = 5
    #     nx_obs                      = 6 + 5 = 11
    #     Indx_obs    = [Indq_obs                     
    #                    n_q + Indw_obs] =  [2 4 6 8 80 98  101 103 108 115 117]'  
    #--- OUTPUT
    #          SAVERANDendPolynomialChaosQWU: state of the random generator at the end of the function
    #
    #--- REUSING THE CONSTRUCTED PCE ----------------------------------------------------------------------------------------------------
               #   Example of script for reusing the PCE with function sub_polynomialChaosQWU_realization_chaos.m
               # ind_display_screen = 1;
               # ind_print          = 1;
               # ind_plot           = 1;
               # 
               # --- Load filePolynomialChaosQWU_for_realization.mat
               # Define the file name
               # fileName = 'filePolynomialChaosQWU_for_realization.npz'  # Use .npz for NumPy's compressed save format
               # Check if the file exists
               # if os.path.isfile(fileName):
               #     print(f'The file "{fileName}" exists.')
               #     data = np.load(fileName, allow_pickle=True)
               #     Extract variables from the loaded data
               #     nw_obs = data['nw_obs']
               #     nq_obs = data['nq_obs']
               #     Indw_obs = data['Indw_obs']
               #     Indq_obs = data['Indq_obs']
               #     nar_PCE = data['nar_PCE']
               #     MatRww_ar0 = data['MatRww_ar0']
               #     MatRqq_ar0 = data['MatRqq_ar0']
               #     Ralpham1_scale_chaos = data['Ralpham1_scale_chaos']
               #     Rbeta_scale_chaos = data['Rbeta_scale_chaos']
               #     Ng = data['Ng']
               #     K0 = data['K0']
               #     MatPower0 = data['MatPower0']
               #     MatRa0 = data['MatRa0']
               #     ng = data['ng']
               #     KU = data['KU']
               #     MatPowerU = data['MatPowerU']
               #     MatRaU = data['MatRaU']
               #     Jmax = data['Jmax']
               #     n_y = data['n_y']
               #     MatRgamma_opt = data['MatRgamma_opt']
               #     Indm = data['Indm']
               #     Indk = data['Indk']
               #     Ralpha_scale_yy = data['Ralpha_scale_yy']
               #     RQQmean = data['RQQmean']
               #     MatRVectEig1s2 = data['MatRVectEig1s2']
               #     print(f'The file "{fileName}" has been loaded.')
               # else:
               #     print(f'STOP-ERROR: the file "{fileName}" does not exist.')
               #
               # MatRqq_chaos = np.zeros((nq_obs, nar_PCE))
               # for ell in range(nar_PCE):  # Loop on the control variable
               #     Rww_ell = MatRww_ar0[:, ell]
               #     Rqq_chaos_ell = sub_polynomialChaosQWU_realization_chaos(nw_obs, Rww_ell, Ralpham1_scale_chaos, Rbeta_scale_chaos,
               #     Ng, K0, MatPower0, MatRa0, ng, KU, MatPowerU, MatRaU,Jmax, n_y, MatRgamma_opt, Indm, Indk, Ralpha_scale_yy,
               #     RQQmean, MatRVectEig1s2)
               #     MatRqq_chaos[:, ell] = Rqq_chaos_ell
               # Print and plot
               # sub_polynomialChaosQWU_print_plot(nq_obs, nar_PCE, Indq_obs, MatRqq_ar0, MatRqq_chaos, ind_display_screen, ind_print, ind_plot, 3)
    #----------------------------------------------------------------------------------------------------------------------------------------------

    if ind_display_screen == 1:                             
        print('--- beginning Task15_PolynomialChaosQWU')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ') 
            fidlisting.write(' ------ Task15_PolynomialChaosQWU \n ')
            fidlisting.write('      \n ')  

    np.random.set_state(SAVERANDstartPolynomialChaosQWU)
    TimeStartPolynomialChaosQWU = time.time()

    #-------------------------------------------------------------------------------------------------------------------------------------   
    #             Loading MatRxx_obs(nx_obs,n_ar)  and checking parameters and data
    #-------------------------------------------------------------------------------------------------------------------------------------

    #--- Checking parameters and data
    if n_x <= 0:
        raise ValueError('STOP1 in sub_polynomialChaosQWU: n_x <= 0')
    if n_q <= 0 or n_w <= 0:
        raise ValueError('STOP2 in sub_polynomialChaosQWU: n_q <= 0 or n_w <= 0')

    nxtemp = n_q + n_w  # dimension of random vector XX = (QQ,WW)
    if nxtemp != n_x:
        raise ValueError('STOP3 in sub_polynomialChaosQWU: n_x not equal to n_q + n_w')
    
    if n_d <= 0:
        raise ValueError('STOP4 in sub_polynomialChaosQWU: n_d <= 0')
    if n_ar <= 0:
        raise ValueError('STOP5 in sub_polynomialChaosQWU: n_ar <= 0')
    if nbMC <= 0:
        raise ValueError('STOP6 in sub_polynomialChaosQWU: nbMC <= 0')
    if nu <= 0 or nu >= n_d:
        raise ValueError('STOP7 in sub_polynomialChaosQWU: nu <= 0 or nu >= n_d')

    n1temp, n2temp = MatRx_d.shape  # MatRx_d(n_x,n_d)
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP8 in sub_polynomialChaosQWU: dimension error in matrix MatRx_d(n_x,n_d)')
    
    n1temp, n2temp = MatReta_ar.shape  # MatReta_ar(nu,n_ar)
    if n1temp != nu or n2temp != n_ar:
        raise ValueError('STOP9 in sub_polynomialChaosQWU: dimension error in matrix MatReta_ar(nu,n_ar)')
    
    if RmuPCA.size >=1:               # checking if not empty # RmuPCA(nu)
        if RmuPCA.ndim !=1:           # ckecking if it is a 1D array
            raise ValueError('STOP10 in sub_polynomialChaosQWU:  RmuPCA(nu) must be a 1D array')
    
    n1temp, n2temp = MatRVectPCA.shape  # MatRVectPCA(n_x,nu)
    if n1temp != n_x or n2temp != nu:
        raise ValueError('STOP11 in sub_polynomialChaosQWU: dimension error in matrix MatRVectPCA(n_x,nu)')
    
    if Indx_real.size >=1:                  # checking if not empty # Indx_real(nbreal)
        if Indx_real.ndim !=1:              # ckecking if it is a 1D array
            raise ValueError('STOP12 in sub_polynomialChaosQWU:  Indx_real(nbreal) must be a 1D array')
    nbreal = len(Indx_real)                 # Indx_real(nbreal)

    if Indx_pos.size >=1:                  # checking if not empty # Indx_pos(nbpos)
        if Indx_pos.ndim !=1:              # ckecking if it is a 1D array
            raise ValueError('STOP13 in sub_polynomialChaosQWU:  Indx_real(nbpos) must be a 1D array')
    nbpos = len(Indx_pos)                  # Indx_pos(nbpos)

    nxtemp = nbreal + nbpos
    if nxtemp != n_x:
        raise ValueError('STOP14 in sub_polynomialChaosQWU: n_x not equal to nreal + nbpos')

    # Loading dimension nq_obs of Indq_obs(nq_obs)
    if Indq_obs.size >=1:                  # checking if not empty 
        if Indq_obs.ndim !=1:              # ckecking if it is a 1D array
            raise ValueError('STOP15 in sub_polynomialChaosQWU:  Indq_obs(nq_obs) must be a 1D array')
    nq_obs = len(Indq_obs)                 # Indq_obs(nq_obs)

    # Checking input data and parameters of Indq_obs(nq_obs)
    if nq_obs < 1 or nq_obs > n_q:
        raise ValueError('STOP16 in sub_polynomialChaosQWU: nq_obs < 1 or nq_obs > n_q')
    if len(Indq_obs) != len(set(Indq_obs)):
        raise ValueError('STOP17 in sub_polynomialChaosQWU: there are repetitions in Indq_obs')
    if np.any(Indq_obs < 1) or np.any(Indq_obs > n_q):
        raise ValueError('STOP18 in sub_polynomialChaosQWU: at least one integer in Indq_obs is not in range [1,n_q]')

    # Loading dimension nw_obs of Indw_obs(nw_obs)
    if Indw_obs.size >=1:                  # checking if not empty 
        if Indw_obs.ndim !=1:              # ckecking if it is a 1D array
            raise ValueError('STOP19 in sub_polynomialChaosQWU:  Indw_obs(nw_obs) must be a 1D array')
    nw_obs = len(Indw_obs)                 # Indw_obs(nw_obs)

    # Checking input data and parameters of Indw_obs(nw_obs,1)
    if nw_obs < 1 or nw_obs > n_w:
        raise ValueError('STOP20 in sub_polynomialChaosQWU: nw_obs < 1 or nw_obs > n_w')    
    if len(Indw_obs) != len(set(Indw_obs)):
        raise ValueError('STOP21 in sub_polynomialChaosQWU: there are repetitions in Indw_obs')
    if np.any(Indw_obs < 1) or np.any(Indw_obs > n_w):
        raise ValueError('STOP22 in sub_polynomialChaosQWU: at least one integer in Indw_obs is not within range [1,n_w]')

    if nx_obs <= 0:
        raise ValueError('STOP23 in sub_polynomialChaosQWU: nx_obs <= 0')
    if Indx_obs.size >=1:                  # checking if not empty 
        if Indx_obs.ndim !=1:              # ckecking if it is a 1D array
            raise ValueError('STOP24 in sub_polynomialChaosQWU:  Indx_obs(nx_obs) must be a 1D array')

    if ind_scaling != 0 and ind_scaling != 1:
        raise ValueError('STOP25 in sub_polynomialChaosQWU: ind_scaling must be equal to 0 or to 1')
    
    if nbreal >= 1:
        if Rbeta_scale_real.size >= 1:                    # checking if not empty 
            if Rbeta_scale_real.ndim != 1:                # ckecking if it is a 1D array
                raise ValueError('STOP26 in sub_polynomialChaosQWU:  Rbeta_scale_real(nbreal) must be a 1D array')
        n1temp = len(Rbeta_scale_real)                    # Rbeta_scale_real(nbreal)
        if n1temp != nbreal:
            raise ValueError('STOP27 in sub_polynomialChaosQWU: dimension error in matrix Rbeta_scale_real(nbreal)')
        
        if Ralpha_scale_real.size >= 1:                   # checking if not empty 
            if Ralpha_scale_real.ndim != 1:               # ckecking if it is a 1D array
                raise ValueError('STOP28 in sub_polynomialChaosQWU:  Ralpha_scale_real(nbreal) must be a 1D array')
        n1temp = len(Ralpha_scale_real)                    # Ralpha_scale_real(nbreal)
        if n1temp != nbreal:
            raise ValueError('STOP29 in sub_polynomialChaosQWU: dimension error in matrix Ralpha_scale_real(nbreal)')
   
    if nbpos >= 1:
        if Rbeta_scale_log.size >= 1:                    # checking if not empty 
            if Rbeta_scale_log.ndim != 1:                # ckecking if it is a 1D array
                raise ValueError('STOP30 in sub_polynomialChaosQWU:  Rbeta_scale_log(nbpos) must be a 1D array')
        n1temp = len(Rbeta_scale_log)                    # Rbeta_scale_log(nbpos)
        if n1temp != nbpos:
            raise ValueError('STOP31 in sub_polynomialChaosQWU: dimension error in matrix Rbeta_scale_log(nbpos)')
        
        if Ralpha_scale_log.size >= 1:                   # checking if not empty 
            if Ralpha_scale_log.ndim != 1:               # ckecking if it is a 1D array
                raise ValueError('STOP32 in sub_polynomialChaosQWU:  Ralpha_scale_log(nbpos) must be a 1D array')
        n1temp = len(Ralpha_scale_log)                   # Ralpha_scale_log(nbpos)
        if n1temp != nbpos:
            raise ValueError('STOP33 in sub_polynomialChaosQWU: dimension error in matrix Ralpha_scale_log(nbpos)')
    
    # Number of learned realizations used for the PCE expansion
    if nbMC_PCE <= 0 or nbMC_PCE > nbMC:
        raise ValueError('STOP34 in sub_polynomialChaosQWU: nbMC_PCE <= 0 or nbMC_PCE > nbMC')
    nar_PCE = n_d*nbMC_PCE

    #--- PCA back: MatRx_obs(nx_obs,n_ar)
    MatRx_obs = sub_polynomialChaosQWU_PCAback(n_x, n_d, nu, n_ar, nx_obs, MatRx_d, MatReta_ar, Indx_obs, RmuPCA, MatRVectPCA, 
                                               ind_display_screen, ind_print)
    
    #--- Scaling back: MatRxx_obs(nx_obs,n_ar)
    MatRxx_obs = sub_polynomialChaosQWU_scalingBack(nx_obs, n_x, n_ar, MatRx_obs, Indx_real, Indx_pos, Indx_obs, Rbeta_scale_real, 
                                                    Ralpha_scale_real, Rbeta_scale_log, Ralpha_scale_log, ind_display_screen, ind_print, ind_scaling)
    del MatRx_obs

    #-------------------------------------------------------------------------------------------------------------------------------------   
    #             Loading MatRqq_ar0(nq_obs,nar_PCE) and MatRww_ar0(nw_obs,nar_PCE)
    #-------------------------------------------------------------------------------------------------------------------------------------

    # Loading MatRqq_ar0(nq_obs,nar_PCE) and MatRww_ar0(nw_obs,nar_PCE) from MatRxx_obs(nx_obs,n_ar)
    MatRqq_ar0 = MatRxx_obs[:nq_obs, :nar_PCE]                 # MatRqq_ar0(nq_obs,nar_PCE),MatRxx_obs(nx_obs,n_ar)
    MatRww_ar0 = MatRxx_obs[nq_obs:nq_obs + nw_obs, :nar_PCE]  # MatRww_ar0(nw_obs,nar_PCE),MatRxx_obs(nx_obs,n_ar)
    del MatRxx_obs

    #-------------------------------------------------------------------------------------------------------------------------------------   
    #             Checking information related to the PCE and checking the control parameters
    #-------------------------------------------------------------------------------------------------------------------------------------

    #     Ng                          : dimension of the germ for polynomial chaos Psi_alpha that is such that Ng = n_w   
    #     Ndeg                        : max degree of the polynomial chaos Psi_alpha with Ndeg >= 1  
    #     ng                          : dimension of the germ for polynomial chaos phi_a that is such that ng >= 1  
    #     ndeg                        : max degree of the polynomial chaos phi_a with ndeg >= 0 (if ndeg = 0, then KU = 1)  

    if Ng != nw_obs:
        raise ValueError('STOP35 in sub_polynomialChaosQWU: Ng must be equal to nw_obs')
    if Ndeg <= 0:
        raise ValueError('STOP36 in sub_polynomialChaosQWU: Ndeg must be greater than or equal to 1')
    if ng <= 0:
        raise ValueError('STOP37 in sub_polynomialChaosQWU: ng must be greater than or equal to 1')
    if ndeg < 0:
        raise ValueError('STOP38 in sub_polynomialChaosQWU: ndeg must be greater than or equal to 0')
    if MaxIter <= 0:
        raise ValueError('STOP39 in sub_polynomialChaosQWU: MaxIter must be greater than or equal to 1')

    if ind_display_screen != 0 and ind_display_screen != 1:
        raise ValueError('STOP40 in sub_polynomialChaosQWU: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:
        raise ValueError('STOP41 in sub_polynomialChaosQWU: ind_print must be equal to 0 or equal to 1')
    if ind_plot != 0 and ind_plot != 1:
        raise ValueError('STOP42 in sub_polynomialChaosQWU: ind_plot must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:
        raise ValueError('STOP43 in sub_polynomialChaosQWU: ind_parallel must be equal to 0 or equal to 1')

    #--- Computation of K0 and KU and checking the consistency
    K0 = int(1e-12 + math.factorial(Ng + Ndeg) / (math.factorial(Ng) * math.factorial(Ndeg)))
    KU = int(1e-12 + math.factorial(ng + ndeg) / (math.factorial(ng) * math.factorial(ndeg)))

    if K0 >= nar_PCE:
        print(f'K0 = {K0}')
        print(f'nar_PCE = {nar_PCE}')
        raise ValueError('STOP40 in sub_polynomialChaosQWU: K0 must be less than nar_PCE')

    if KU >= nar_PCE:
        print(f'KU = {KU}')
        print(f'nar_PCE = {nar_PCE}')
        raise ValueError('STOP41 in sub_polynomialChaosQWU: KU must be less than nar_PCE')

    #--- Components of QQ_obs that are selected for the cost function. In this code version all the components are kept
    Ind_qqC = np.arange(1,nq_obs+1)   # Ind_qqC(nbqqC): components of QQ_obs that are selected for the cost function
    nbqqC = nq_obs

    #--- display
    if ind_display_screen == 1:
        print('    [Ng   Ndeg  K0  ]')
        print([Ng, Ndeg, K0])

        print('    [ng   ndeg  KU  ]')
        print([ng, ndeg, KU])

    #--- print  
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('                                \n ')
            fidlisting.write(f' n_q           = {n_q:9d} \n ')
            fidlisting.write(f' n_w           = {n_w:9d} \n ')
            fidlisting.write(f' n_x           = {n_x:9d} \n ')
            fidlisting.write(f' nbreal        = {nbreal:9d} \n ')
            fidlisting.write(f' nbpos         = {nbpos:9d} \n ')
            fidlisting.write(f' nx_obs        = {nx_obs:9d} \n ')
            fidlisting.write(f' ind_scaling   = {ind_scaling:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' n_d           = {n_d:9d} \n ')
            fidlisting.write(f' nbMC          = {nbMC:9d} \n ')
            fidlisting.write(f' n_ar          = {n_ar:9d} \n ')
            fidlisting.write(f' nu            = {nu:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' nbMC_PCE      = {nbMC_PCE:9d} \n ')
            fidlisting.write(f' nar_PCE       = {nar_PCE:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' Ng            = {Ng:9d} \n ')
            fidlisting.write(f' Ndeg          = {Ndeg:9d} \n ')
            fidlisting.write(f' K0            = {K0:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' ng            = {ng:9d} \n ')
            fidlisting.write(f' ndeg          = {ndeg:9d} \n ')
            fidlisting.write(f' KU            = {KU:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' MaxIter       = {MaxIter:9d} \n ')
            fidlisting.write('                                \n ')
            fidlisting.write(f' ind_display_screen = {ind_display_screen:1d} \n ')
            fidlisting.write(f' ind_print          = {ind_print:1d} \n ')
            fidlisting.write(f' ind_plot           = {ind_plot:1d} \n ')
            fidlisting.write(f' ind_parallel       = {ind_parallel:1d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
    
    #--- construction of MatRxipointOVL(nq_obs,nbpoint) of the points used by ksdensity in OVL
    nbpoint = 200
    MatRxipointOVL = np.zeros((nq_obs,nbpoint))                            # MatRxipointOVL(nq_obs,nbpoint)
    for k in range(nq_obs):
        maxqk = np.max(MatRqq_ar0[k,:])                                    # MatRqq_ar0(nq_obs,nar_PCE)
        minqk = np.min(MatRqq_ar0[k,:])
        MatRxipointOVL[k,:] = np.linspace(minqk,maxqk,nbpoint)             # MatRxipointOVL(nq_obs,nbpoint)

    #--- construction of RbwOVL(nq_obs) for the bandwidth used by ksdensity in OVL
    RbwOVL = np.zeros((nq_obs))                                            # RbwOVL(nq_obs)
    for k in range(nq_obs):                                                # MatRqq_ar0(nq_obs,nar_PCE)
        Rqq_k   = MatRqq_ar0[k,:].T
        Rqq_k   = np.squeeze(Rqq_k)
        sigma_k = np.std(Rqq_k,ddof=1) 
        #--- For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
        RbwOVL[k]  = 1.0592*sigma_k*nar_PCE**(-1/5)                        # Silverman bandwidth in 1D 
        

    #--- computing second-order moment using the learned dataset for QQ_ar0
    #    MatRmom2QQ_ar0  = (MatRqq_ar0*(MatRqq_ar0'))/(nar_PCE-1);        # MatRmom2QQ_ar0(nq_obs,nq_obs)

    #--- Construction of MatRxiNg0(Ng,nar_PCE) (we have Ng = nw_obs) by scaling MatRww_ar0(Ng,nar_PCE) between cmin and cmax 
    cmax = 1.0
    cmin = -1.0
    Rmax = np.max(MatRww_ar0, axis=1)    # MatRww_ar0(Ng,nar_PCE), Ng = nw_obs
    Rmin = np.min(MatRww_ar0, axis=1)
    Rbeta_scale_chaos    = np.zeros(Ng)  # Rbeta_scale_chaos(Ng)
    Ralpha_scale_chaos   = np.zeros(Ng)  # Ralpha_scale_chaos(Ng)
    Ralpham1_scale_chaos = np.zeros(Ng)  # Ralpham1_scale_chaos(Ng)
    for k in range(Ng):
        A = (Rmax[k] - Rmin[k])/(cmax - cmin)
        B = Rmin[k] - A*cmin
        Rbeta_scale_chaos[k]    = B
        Ralpha_scale_chaos[k]   = A
        Ralpham1_scale_chaos[k] = 1/Ralpha_scale_chaos[k]
    del cmax,cmin,Rmax,Rmin
    MatRxiNg0 = Ralpham1_scale_chaos[:,np.newaxis] * (MatRww_ar0 - Rbeta_scale_chaos[:,np.newaxis]) # MatRxiNg0(Ng,nar_PCE), MatRww_ar0(Ng,nar_PCE), Ng = nw_obs

    #--- Normalization of QQ_ar0 using a PCA: MatRqq_ar0  = repmat(RQQmean,1,nar_PCE) + MatRVectEig1s2*MatRyy_ar0  
    RQQmean            = np.mean(MatRqq_ar0,axis=1)              # RQQmean(nq_obs),MatRqq_ar0(nq_obs,nar_PCE)
    MatRQQcov          = np.cov(MatRqq_ar0)                      # MatRQQcov(nq_obs,nq_obs), MatRqq_ar0(nq_obs,nar_PCE)
    MatRQQcov          = 0.5*(MatRQQcov + MatRQQcov.T)
    (REig,MatRVect)    = np.linalg.eigh(MatRQQcov)               # MatRVect(nq_obs,nq_obs),REig(nq_obs)
    MatRVectEig1s2     = MatRVect @ (np.diag(np.sqrt(REig)))     # MatRVectEig1s2(nq_obs,nq_obs)

    n_y = nq_obs  # no reduction applied
                                                                 # MatRyy_ar0(n_y,nar_PCE),MatRVect(n_y,n_y),MatRqq_ar0(n_y,nar_PCE)  
    MatRyy_ar0 = np.sqrt(1 / REig[:, np.newaxis]) * (MatRVect.T @ (MatRqq_ar0 - RQQmean[:, np.newaxis]))  
    del MatRQQcov, MatRVect, REig
     
    #--- scaling MatRyy_ar0 in MatRy_ar0  in order that  the maximum of Y on the realizations be normalized to 1
    Rmax              = np.max(np.abs(MatRyy_ar0), axis=1)             # Rmax(n_y)
    Ralpha_scale_yy   = Rmax                                           # Ralpha_scale_yy(n_y)
    Ralpham1_scale_yy = 1. / Ralpha_scale_yy                           # Ralpham1_scale_yy(n_y)
    MatRy_ar0         = Ralpham1_scale_yy[:, np.newaxis] * MatRyy_ar0  # MatRy_ar0(n_y,nar_PCE),MatRyy_ar0(n_y,nar_PCE) with n_y = nq_obs
    
    #--- Computing matrices: MatRPsi0(K0,nar_PCE),MatPower0(K0,Ng),MatRa0(K0,K0)
    #                        of the polynomial chaos Psi_{alpha^(k)}(Xi) with alpha^(k) = (alpha_1^(k),...,alpha_Ng^(k)) in R^Ng with k=1,...,K0
    #                        MatRxiNg0(Ng,nar_PCE) 
    (MatRPsi0,MatPower0,MatRa0) = sub_polynomialChaosQWU_chaos0(K0,Ndeg,Ng,nar_PCE,MatRxiNg0)
    del MatRxiNg0
    
    #--- Computing matrices: MatRphiU(KU,nar_PCE),MatPowerU(KU,Ng),MatRaU(KU,KU)
    #                        of the polynomial chaos phi_{a^(m)}(U) with a^(m) = (a_1^(m),...,a_ng^(m)) in R^ng with m=1,...,KU    
    MatRU                       = np.random.randn(ng, nar_PCE)                                 # MatRU(ng,nar_PCE)
    (MatRphiU,MatPowerU,MatRaU) = sub_polynomialChaosQWU_chaosU(KU,ndeg,ng,nar_PCE,MatRU)      # MatRphiU(KU,nar_PCE)
    del MatRU
     
    #--- Computing the global index j = (m,k), m = 1,...,KU and k = 1,...,K0, 
    #                               j = 1,..., Jmax  with Jmax = KU*K0
    Jmax = KU * K0
                                # Scalar Matlab sequence
                                # Indm = zeros(Jmax,1);   m = Indm(j)
                                # Indk = zeros(Jmax,1);   k = Indk(j)
                                # j = 0;
                                # for m = 1:KU
                                #     for k = 1:K0
                                #         j = j+1;
                                #         Indm(j) = m;
                                #         Indk(j) = k;
                                #     end
                                # end

    Indm = np.repeat(np.arange(1,KU+1),K0)
    Indk = np.tile(np.arange(1,K0+1),KU)

    if n_y > Jmax:
        raise ValueError('STOP42 in sub_polynomialChaosQWU: Jmax must be greater than or equal to n_y; increase Ndeg and/or ng and/or ndeg')

    #--- construction of MatRb(Jmax,nar_PCE) such that MatRb(j,ell) = MatRphiU(m,ell)*MatRPsi0(k,ell)
    MatRb = np.zeros((Jmax,nar_PCE))
    for j in range(Jmax):
        m = Indm[j]
        k = Indk[j]
        MatRb[j,:] = MatRphiU[m-1,:]*MatRPsi0[k-1,:]    # MatRb(Jmax,nar_PCE)

    #--- Computing MatRMrondY(n_y,n_y)
    MatRMrondY = MatRy_ar0 @ MatRy_ar0.T / (nar_PCE - 1)  # MatRy_ar0(n_y,nar_PCE)

    #--- Computing an initial value MatRgamma_bar of MatRgamma for fmincon 
    MatRgamma_bar = MatRy_ar0 @ MatRb.T / (nar_PCE - 1)  # MatRgamma_bar(n_y,Jmax)

    #--- Pre-computation 
    MatRchol = np.linalg.cholesky(MatRMrondY)  # MatRchol(n_y,n_y), MatRMrondY(n_y,n_y)
    MatRones = np.ones((n_y,Jmax))             # MatRones(n_y,Jmax)
    
    #--- computing MatRsample_opt(n_y, Jmax) using parallel computation in sub_polynomialChaosQWU_OVL independently
    #    from the value of ind_parallel

    # Initialize MatRsample0 and flatten it
    MatRsample0      = np.zeros((n_y,Jmax))  # MatRsample0(n_y, Jmax): initial value
    MatRsample0_flat = MatRsample0.flatten()  # Flatten MatRsample0 for the optimizer

    # Define the cost function
    def J(MatRsample_flat):
        # Reshape the flattened array back to its original shape
        MatRsample = MatRsample_flat.reshape((n_y,Jmax))
        # Compute the cost function value
        fxk = sub_polynomialChaosQWU_Jcost(MatRsample,n_y,Jmax,nar_PCE,nq_obs,nbqqC,nbpoint,MatRgamma_bar,MatRchol, 
                                           MatRb,Ralpha_scale_yy,RQQmean,MatRVectEig1s2,Ind_qqC,MatRqq_ar0,MatRxipointOVL, 
                                           RbwOVL,MatRones)  
        return fxk
    
    # Parallelized numerical gradient calculation using central differences
    def compute_gradient(J,MatRsample_flat,i,eps=1e-5):
        # Calculate relative eps based on the current parameter value
        eps_relative = eps*max(np.abs(MatRsample_flat[i]), 1e-2)

        MatRsample_flat_pos = MatRsample_flat.copy()
        MatRsample_flat_neg = MatRsample_flat.copy()        

        # Perturb the i-th variable
        MatRsample_flat_pos[i] += eps_relative
        MatRsample_flat_neg[i] -= eps_relative
        
        # Central difference approximation for the gradient
        grad_i = (J(MatRsample_flat_pos) - J(MatRsample_flat_neg)) / (2*eps_relative)
        return grad_i

    # Compute all gradients in parallel
    def compute_gradients_in_parallel(J, MatRsample_flat, n_jobs=-1, eps=1e-5):
        gradients = Parallel(n_jobs=-1)(delayed(compute_gradient)(J,MatRsample_flat,i,eps) for i in range(len(MatRsample_flat)))
        return np.array(gradients)

    # Define a function to create a callback with an iteration counter
    def make_callback(fidlisting):
        iteration = [0]  # Use a list to create a mutable counter
        def callback(xk):
            iteration[0] += 1          
            fxk = J(xk)                                                    # Recompute the function value at xk to ensure correctness              
            print(f"Iteration {iteration[0]}, f(x) = {fxk}")               # Print and log the current iteration and function value
            fidlisting.write(f"Iteration {iteration[0]}, f(x) = {fxk}\n")
            sys.stdout.flush()
        return callback    
    
    fidlisting = open('listing.txt', 'a+')                                 # Open the file once before the optimization
    callback_func = make_callback(fidlisting)                              # Create the callback function  

    # BFGS algorithm using the computed (and not analytical gradient) 
    options = {                     # Define the options for the optimizer with adjusted tolerances
        'maxiter': MaxIter,         # Maximum number of iterations allowed
        'disp': True,               # Display detailed iteration information
        'gtol': 1e-6,               # Termination tolerance on the gradient (similar to TolFun)
        'return_all': True,         # Returns all intermediate results
        'eps': 1e-2                 # Increase step size for numerical differentiation of gradients (the default value for eps is 1e-8)
    }
    # Call the optimizer with the 'BFGS' method 
    result = minimize(J,MatRsample0_flat,method='BFGS',
                      jac=lambda x: compute_gradients_in_parallel(J,x,n_jobs=-1),  # Parallel gradient computation
                      options=options,callback=callback_func,tol=1e-6)
   
    # Handle the case where the maximum iterations were reached without an error
    if not result.success:
        print(f"MaxIter reached. Increase the value of Maxiter or decrease the tolerance") 
        print(f"-------------------------------------------------------") 
        print(f"  ") 
    
    fidlisting.close()                                # Close the file after the optimization
    MatRsample_opt = result.x.reshape((n_y,Jmax))     # Reshape the result back to the original shape
        
    #--- computing MatRgamma_opt(n_y,Jmax) for the polynomial chaos expansion  
    MatRtilde     = MatRgamma_bar * (np.ones((n_y, Jmax)) + MatRsample_opt)  # MatRtilde(n_y,Jmax), MatRgamma_bar(n_y,Jmax)
    MatRFtemp     = np.linalg.cholesky(MatRtilde @ MatRtilde.T)              # MatRFtemp(n_y,n_y)
    MatRAtemp     = np.linalg.inv(MatRFtemp).T                               # MatRAtemp(n_y,n_y)
    MatRhat       = MatRAtemp @ MatRtilde                                    # MatRhat(n_y,Jmax), MatRtilde(n_y,Jmax)
    MatRgamma_opt = MatRchol.T @ MatRhat                                     # MatRgamma(n_y,Jmax)
    del MatRtilde, MatRFtemp, MatRAtemp, MatRhat

    #-----------------------------------------------------------------------------------------------------------------------
    #                      Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) 
    #-----------------------------------------------------------------------------------------------------------------------

    seeds_temp = np.random.randint(0, 2**32, size=1, dtype='uint32')  # Random initialization of the generator with the seed 
    np.random.seed(seeds_temp)

    MatRU = np.random.randn(ng,nar_PCE)  # MatRU(ng,nar_PCE)
    MatRy_PolChaos_ar0 = sub_polynomialChaosQWU_surrogate(nar_PCE, nw_obs, n_y, MatRww_ar0, MatRU, Ralpham1_scale_chaos,
                                                          Rbeta_scale_chaos, Ng, K0, MatPower0, MatRa0, ng, KU, MatPowerU,
                                                          MatRaU, Jmax, MatRgamma_opt, Indm, Indk)
    MatRyy_PolChaos_ar0 = Ralpha_scale_yy[:, np.newaxis] * MatRy_PolChaos_ar0                           # MatRy_PolChaos_ar0(n_y,nar_PCE)
    MatRqq_PolChaos_ar0 = RQQmean[:, np.newaxis] + MatRVectEig1s2 @ MatRyy_PolChaos_ar0                 # MatRqq_PolChaos_ar0(nq_obs,nar_PCE)
    # MatRmom2QQ_PolChaos_ar0 = MatRqq_PolChaos_ar0 @ MatRqq_PolChaos_ar0.T / (nar_PCE - 1);            # MatRmom2QQ_PolChaos_ar0(nq_obs,nq_obs)
    RerrorOVL = sub_polynomialChaosQWU_OVL(nbqqC, nar_PCE, MatRqq_ar0[Ind_qqC-1,:], nar_PCE, MatRqq_PolChaos_ar0[Ind_qqC-1, :],
                                           nbpoint, MatRxipointOVL[Ind_qqC-1,:], RbwOVL[Ind_qqC-1])
    errorOVL_PolChaos_ar0C = np.sum(RerrorOVL) / nbqqC

    #--- display
    if ind_display_screen == 1:
        print(f'errorOVL_PolChaos_ar0C = {errorOVL_PolChaos_ar0C}')

    #--- print
    if ind_print == 1:
        with open('listing.txt', 'a') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' errorOVL_PolChaos_ar0C =  =  {errorOVL_PolChaos_ar0C:.7e} \n')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')

    #--- debug sequence
    ind_plot_debug = 0
    if ind_plot_debug == 1:
        for i in range(nq_obs):
            plt.figure()
            plt.plot(np.arange(1, nar_PCE + 1), MatRqq_PolChaos_ar0[i, :], '-')
            plt.title(f'$\\ell\\mapsto qq_{{{i+1},{{\\rm{{chaos,ar0s}}}}}}^\\ell$', fontsize=16)
            plt.xlabel('$\\ell$', fontsize=16)
            plt.ylabel(f'$qq_{{{i+1},{{\\rm{{chaos,ar0s}}}}}}^\\ell$', fontsize=16)
            plt.savefig(f'figure_sub_polynomialChaosQWU_qq_{i+1}_chaos_ar0.png')
            plt.close()

    #--- print and plot
    sub_polynomialChaosQWU_print_plot(nq_obs, nar_PCE, Indq_obs, MatRqq_ar0, MatRqq_PolChaos_ar0, ind_display_screen, ind_print, ind_plot, 1)

    #-----------------------------------------------------------------------------------------------------------------------
    #                          Polynomial-chaos validation for MatRww_o(nw_obs,nar_PCE)
    #-----------------------------------------------------------------------------------------------------------------------

    MatRww_o = MatRww_ar0  # MatRww_o(nw_obs,nar_PCE), MatRww_ar0(nw_obs,nar_PCE)

    seeds_temp = np.random.randint(0, 2**32, size=1, dtype='uint32')  # Random initialization of the generator with the seed 
    np.random.seed(seeds_temp)    
    MatRqq_PolChaos_o = np.zeros((nq_obs,nar_PCE))   # MatRqq_PolChaos_o(nq_obs,nar_PCE)
    for kno in range(nar_PCE):                       #... loop on the control variable
        Rww_kno = MatRww_o[:, kno]
        RU_kno = np.random.randn(ng)
        Ry_PolChaos_o = sub_polynomialChaosQWU_surrogate(1, nw_obs, n_y, Rww_kno, RU_kno, Ralpham1_scale_chaos, Rbeta_scale_chaos,
                                                         Ng, K0, MatPower0, MatRa0, ng, KU, MatPowerU, MatRaU, Jmax, MatRgamma_opt, Indm, Indk)
        Ry_PolChaos_o  = Ry_PolChaos_o.flatten()
        Ryy_PolChaos_o = Ralpha_scale_yy * Ry_PolChaos_o            # Ryy_PolChaos_o(n_y)
        Rqq_PolChaos_o = RQQmean + MatRVectEig1s2 @ Ryy_PolChaos_o  # Rqq_PolChaos_o(nq_obs)
        MatRqq_PolChaos_o[:, kno] = Rqq_PolChaos_o                  # MatRqq_PolChaos_o(nq_obs,nar_PCE), Rqq_PolChaos_o(nq_obs)

    #--- print and plot
    sub_polynomialChaosQWU_print_plot(nq_obs, nar_PCE, Indq_obs, MatRqq_ar0, MatRqq_PolChaos_ar0, ind_display_screen, ind_print, ind_plot, 2)

    #--- save the file.mat:  "filePolynomialChaosQWU_for_realization"

    # Define the file name
    fileName = 'filePolynomialChaosQWU_for_realization.mat'

    # Save on filename.mat file
    np.savez_compressed(fileName, nw_obs=nw_obs, nq_obs=nq_obs, Indw_obs=Indw_obs, Indq_obs=Indq_obs, nar_PCE=nar_PCE, MatRww_ar0=MatRww_ar0, 
                        MatRqq_ar0=MatRqq_ar0, Ralpham1_scale_chaos=Ralpham1_scale_chaos, Rbeta_scale_chaos=Rbeta_scale_chaos, Ng=Ng, K0=K0, 
                        MatPower0=MatPower0, MatRa0=MatRa0, ng=ng, KU=KU, MatPowerU=MatPowerU, MatRaU=MatRaU, Jmax=Jmax, n_y=n_y, 
                        MatRgamma_opt=MatRgamma_opt, Indm=Indm, Indk=Indk, Ralpha_scale_yy=Ralpha_scale_yy, RQQmean=RQQmean, MatRVectEig1s2=MatRVectEig1s2)
    print(f'The file "{fileName}" has been saved.')

    #----------------------------------------------------------------------------------------------------------------------------------
    #                                                           end
    #----------------------------------------------------------------------------------------------------------------------------------
     
    SAVERANDendPolynomialChaosQWU = np.random.get_state()
    ElapsedPolynomialChaosQWU     = time.time() - TimeStartPolynomialChaosQWU

    if ind_print == 1:
        with open('listing.txt', 'a') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write('-------   Elapsed time for Task15_PolynomialChaosQWU \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'Elapsed Time   =  {ElapsedPolynomialChaosQWU:10.2f}\n')
            fidlisting.write('      \n ')

    if ind_display_screen == 1:
        print('--- end Task15_PolynomialChaosQWU')
       
    return SAVERANDendPolynomialChaosQWU