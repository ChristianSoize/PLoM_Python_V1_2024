import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import time
import random
from scipy.special import factorial
from joblib import Parallel, delayed
from sub_polynomial_chaosZWiener_plot_Har_HPCE import sub_polynomial_chaosZWiener_plot_Har_HPCE
from sub_polynomial_chaosZWiener_PCE import sub_polynomial_chaosZWiener_PCE


def sub_polynomial_chaosZWiener(nu, n_d, nbMC, nbmDMAP, MatRg, MatRa, n_ar, MatReta_ar, ArrayZ_ar, 
                                ArrayWienner, icorrectif, coeffDeltar, ind_PCE_ident, ind_PCE_compt, 
                                nbMC_PCE, Rmu, RM, mu_PCE, M_PCE, SAVERANDstartPCE, ind_display_screen, 
                                ind_print, ind_plot, ind_parallel, MatRplotHsamples, MatRplotHClouds, 
                                MatRplotHpdf, MatRplotHpdf2D):
    
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_polynomial_chaosZWiener
    #  Subject      : Polynomial chaos expansion (PCE) [Z_PCE] of random matrix [Z_ar] (manifold), whose learned realizations are 
    #                 ArrayZ_ar(nu,nbmDMAP,nbMC). The learned realizations of [H_ar] = [Z_ar] [g]' (reshaped) are MatReta_ar(nu,n_ar)
    #                 with n_ar = n_d x nbMC. The PCE [H_PCE] = [Z_PCE][g]' of [H_ar] is constructed using the learned realizations 
    #                 ArrayZ_ar(nu,nbmDMAP,nbMC) of [H_ar], and the germ of the PCE is made up of an extraction of the Wiener realizations
    #                 ArrayWienner(nu,n_d,nbMC), which are used by the reduced-order ISDE for computing ArrayZ_ar(nu,nbmDMAP,nbMC). 
    #                 The nar_PCE = n_d x nbMC_PCE  <= n_ar realizations MatReta_PCE(nu,nar_PCE) of H_PCE are the reshaping of the realizations 
    #                 of [H_PCE]. The used theory is presented in [2]
    #
    #  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [2] C. Soize, R. Ghanem, Polynomial chaos representation of databases on manifolds, Journal of Computational Physics, 
    #                       doi: 10.1016/j.jcp.2017.01.031, 335, 201-221 (2017).
    #                [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
    #                       Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
    #                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #                [5] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
    #                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
    #                [6] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
    #                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777, 
    #                       380, 113777 (2021).
    #                [7] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #                [8] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler 
    #                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
    #                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022). 
    #                [9] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of 
    #                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
    #               [10] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
    #                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering, 
    #                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
    #               [11] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
    #                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
    #               
    #--- INPUTS 
    #
    #--- parameters related to the learning
    #    nu                         : dimension of random vector H_d, H_ar, and H_PCE
    #    n_d                        : number of points in the training set for H_d
    #    nbMC                       : number of learned realizations of (nu,n_d)-valued random matrix [H_ar]   
    #    nbmDMAP                    : dimension of the ISDE-projection basis
    #    MatRg(n_d,nbmDMAP)         : matrix of the ISDE-projection basis 
    #    MatRa(n_d,nbmDMAP)         : MatRa = MatRg*(MatRg'*MatRg)^{-1} 
    #    n_ar                       : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #    MatReta_ar(nu,n_ar)        : n_ar realizations of H_ar  
    #
    #    ArrayZ_ar(nu,nbmDMAP,nbMC) : nbMC learned realizations of the (nu,nbmDMAP)-matrix valued random variable [H_ar]
    #    ArrayWienner(nu,n_d,nbMC)  : nbMC realizations of the (nu,n_d)-matrix-valued of the Wiener process used by the 
    #                                 reduced-order ISDE for computing ArrayZ_ar(nu,nbmDMAP,nbMC). 
    #    icorrectif                  = 0: usual Silverman-bandwidth formulation, the normalization conditions are not exactly satisfied
    #                                = 1: modified Silverman bandwidth, for any value of nu, the normalization conditions are verified  
    #    coeffDeltar                : coefficient > 0 (usual value is 20) for calculating Deltar for ISDE solver
    #
    #    ind_PCE_ident              = 0 : no identification of the PCE 
    #                               = 1 :    identification of the PCE 
    #    ind_PCE_compt              = 0 : no computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M
    #                               = 1:     computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M
    #    nbMC_PCE                   : number of realizations generated for [Z_PCE] and consequently for [H_PCE] with nbMC_PCE <= nbMC
    #                                 (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible)
    #--- parameters for PCE identification (ind_PCE_ident = 1):
    #    Rmu(nbmu)                  : 1D array of the nbmu values of the dimension mu of the germ (Xi_1,...,Xi_mu) of the PCE with 1 <= mu <= nu, 
    #                                 explored to find the optimal value muopt
    #    RM(nbM)                    : 1D array of the nbM values of the maximum degree M of the PCE with 0 <= M, explored to find the optimal 
    #                                 value Mopt
    #                                 M = 0 : multi-index  (0,0,...,0) 
    #                                 M = 1 : multi-indices (0,...,0) and (1,0,...,0), (0,1,...,0),..., (0,...,1): Gaussian representation
    #
    #--- parameters for computing the PCE for given values mu = mu_PCE  and M = M_PCE (ind_PCE_compt = 1):
    #    mu_PCE                      : value of mu for computing the realizations with the PCE with mu >= 1
    #    M_PCE                       : value of M  for computing the realizations with the PCE with M  >= 0 
    #
    #--- Parameters controlling random generator, print, plot, and parallel computation
    #    SAVERANDstartPCE    : state of the random generator at the start
    #    ind_display_screen  : = 0 no display,              = 1 display
    #    ind_print           : = 0 no print,                = 1 print
    #    ind_plot            : = 0 no plot,                 = 1 plot
    #    ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #
    #--- if ind_PCE_compt = 1 and ind_plot = 1, parameters and data controlling plots for H_ar and H_PCE
    #    in the example below, nu >=  9
    #    MatRplotHsamples = np.array([3, 7, 8])  1D array of the components numbers of H_ar for which the realizations are plotted 
    #                                               Example 1: plot components 3, 7, and 8. 
    #                                                          nbplotHsamples = 3
    #                                               Example 2: MatRplotHsamples = np.array([]), no plot, nbplotHsamples = 0
    #    MatRplotHClouds = np.array([            2D array containing the 3 components numbers of H_ar for which the clouds are plotted 
    #                              [2, 4, 6],       Example 1: plot of components 2, 4, and 6, 
    #                               [3, 4, 8]                  plot of components 3, 4, and 8. 
    #                               ])                         nbplotHClouds  = 2                                                               
    #                                               Example 2: MatRplotHClouds = np.array([]); no plot, nbplotHClouds = 0
    #    MatRplotHpdf = np.array([3, 5, 7, 9])   1D array containing the components numbers of H_ar for which the pdfs are plotted 
    #                                               Example 1: plot of components 3, 5, 7, and 9. 
    #                                                          nbplotHpdf = 4
    #                                               Example 2: MatRplotHpdf = np.array([]), no plot, nbplotHpdf = 0
    #    MatRplotHpdf2D = np.array([             2D array containing the 2 components numbers of H_ar for which the joint pdfs are plotted 
    #                               [2, 4],         Example 1: plot for the components 2 and 4
    #                               [3, 4]                     plot for the components 3 and 4.
    #                               ])                         nbplotHpdf2D = 2 
    #                                               Example 2: MatRplotHpdf2D = np.array([]), no plot, nbplotHpdf2D = 0
    #
    #--- OUTPUT:
    #          nar_PCE                 : nar_PCE = n_d*nbMC_PCE realizations of H_PCE as the reshaping of the nbMC_PCE realizations of [H_PCE]
    #          MatReta_PCE(nu,nar_PCE) : nar_PCE of H_PCE as the reshaping of the nbMC_PCE realizations of [H_PCE]
    #          SAVERANDendPCE          : state of the random generator at the end of the function
    
    TimeStartPCE = time.time()
    numfig = 0
    nar_PCE = n_d * nbMC_PCE  # number of realizations MatReta_PCE(nu,nar_PCE) of H_PCE as the reshaping of 
                              # the nbMC_PCE realizations of [H_PCE]
    
    if MatRplotHsamples.size >=1:           # checking if not empty
        if MatRplotHsamples.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP1 in sub_polynomial_chaosZWiener: MatRplotHsamples must be a 1D array')
    if MatRplotHClouds.size >=1:           # checking if not empty
        if MatRplotHClouds.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP2 in sub_polynomial_chaosZWiener: MatRplotHClouds must be a 2D array')    
    if MatRplotHpdf.size >=1:             # checking if not empty
        if MatRplotHpdf.ndim != 1:        # ckecking if it is a 1D array
            raise ValueError('STOP3 in sub_polynomial_chaosZWiener: MatRplotHpdf must be a 1D array') 
    if MatRplotHpdf2D.size >=1:           # checking if not empty
        if MatRplotHpdf2D.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP4 in sub_polynomial_chaosZWiener: MatRplotHpdf2D must be a 2D array')           

    #--- initializing the random generator
    np.random.set_state(SAVERANDstartPCE) 

    nbplotHsamples = len(MatRplotHsamples)       # MatRplotHsamples(nbplotHsamples)
    nbplotHClouds  = MatRplotHClouds.shape[0]    # MatRplotHClouds(nbplotHClouds,3)
    nbplotHpdf     = len(MatRplotHpdf)           # MatRplotHpdf(nbplotHpdf)
    nbplotHpdf2D   = MatRplotHpdf2D.shape[0]     # MatRplotHpdf2D(nbplotHpdf2D,2)

    if ind_display_screen == 1:
        print(' ')
        print('--- beginning Task14_PolynomialChaosZwiener')
        print(' ')
        if ind_PCE_ident == 1:
            print(' ind_PCE_ident  = 1: identification of the optimal values Mopt of M, for each given value of mu = Rmu(imu)')
            print('                     of the PCE H_PCE of H_ar, using the learned realizations MatReta_ar(nu,n_ar) of H_ar ')
        if ind_PCE_compt == 1:
            print(' ind_PCE_compt  = 1: computation of the realizations MatReta_PCE(nu,nar_PCE) of the PCE representation    ')
            print('                     H_PCE of H_ar  for given mu = mu_PCE and M = M_PCE                                   ')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ------ Task14_PolynomialChaosZwiener \n ')
            fidlisting.write('                         \n ')
            if ind_PCE_ident == 1:
                fidlisting.write(' ind_PCE_ident  = 1: identification of the optimal values Mopt of M, for each given value of mu = Rmu(imu), \n ')
                fidlisting.write('                     of the PCE H_PCE of H_ar, using the learned realizations MatReta_ar(nu,n_ar) of H_ar   \n ')
            if ind_PCE_compt == 1:
                fidlisting.write(' ind_PCE_compt  = 1: computation of the realizations MatReta_PCE(nu,nar_PCE) of the PCE representation      \n ')
                fidlisting.write('                     H_PCE of H_ar  for given mu = mu_PCE and M = M_PCE                                     \n ')
            fidlisting.write('      \n ')
  
    #-----------------------------------------------------------------------------------------------------------------------------------                           
    #       Checking input parameters and input data
    #-----------------------------------------------------------------------------------------------------------------------------------  

    #--- Check parameters and data related to the learning step
    if nu > n_d or nu < 1 or n_d < 1:
        raise ValueError('STOP5 in sub_polynomial_chaosZWiener: nu > n_d or nu < 1 or n_d < 1')  
    if nbMC < 1: 
        raise ValueError('STOP6 in sub_polynomial_chaosZWiener: nbMC < 1')
    if nbmDMAP < 1 or nbmDMAP > n_d:
        raise ValueError('STOP7 in sub_polynomial_chaosZWiener: nbmDMAP < 1 or nbmDMAP > n_d')
    if MatRg.shape != (n_d, nbmDMAP):
        raise ValueError('STOP8 in sub_polynomial_chaosZWiener: the dimensions of MatRg are not consistent with n_d and nbmDMAP')
    if MatRa.shape != (n_d, nbmDMAP):
        raise ValueError('STOP9 in sub_polynomial_chaosZWiener: the dimensions of MatRa are not consistent with n_d and nbmDMAP')
    if n_ar != n_d * nbMC:
        raise ValueError('STOP10 in sub_polynomial_chaosZWiener: n_ar must be equal to n_d * nbMC')
    if MatReta_ar.shape != (nu, n_ar):
        raise ValueError('STOP11 in sub_polynomial_chaosZWiener: the dimensions of MatReta_ar are not consistent with nu and n_ar')
    if ArrayZ_ar.shape != (nu, nbmDMAP, nbMC):
        raise ValueError('STOP12 in sub_polynomial_chaosZWiener: the dimensions of ArrayZ_ar are not consistent with nu, nbmDMAP, and nbMC')
    if ArrayWienner.shape != (nu, n_d, nbMC):
        raise ValueError('STOP13 in sub_polynomial_chaosZWiener: the dimensions of ArrayWienner are not consistent with nu, n_d, and nbMC')
    if icorrectif != 0 and icorrectif != 1:  
        raise ValueError('STOP14 in sub_polynomial_chaosZWiener: icorrectif must be equal to 0 or equal to 1')  
    if coeffDeltar < 1:
        raise ValueError('STOP15 in sub_polynomial_chaosZWiener: coeffDeltar must be greater than or equal to 1')

    #--- Check parameters controlling the type of exec (PCE identification or computation of PCE)
    if ind_PCE_ident != 0 and ind_PCE_ident != 1:  
        raise ValueError('STOP16 in sub_polynomial_chaosZWiener: ind_PCE_ident must be equal to 0 or equal to 1')  
    if ind_PCE_compt != 0 and ind_PCE_compt != 1:  
        raise ValueError('STOP17 in sub_polynomial_chaosZWiener: ind_PCE_compt must be equal to 0 or equal to 1')  
    if nbMC_PCE < 1 or nbMC_PCE > nbMC:
        raise ValueError('STOP18 in sub_polynomial_chaosZWiener: nbMC_PCE < 1 or nbMC_PCE > nbMC')

    #--- Case ind_PCE_ident = 1 (parameters identification of the PCE)
    if ind_PCE_ident == 1:
        if Rmu.size >=1:           # checking if not empty
           if Rmu.ndim != 1:       # ckecking if it is a 1D array
               raise ValueError('STOP19 in sub_polynomial_chaosZWiener: Rmu must be a 1D array')
        nbmu = len(Rmu)
        if nbmu < 1:
            raise ValueError('STOP20 in sub_polynomial_chaosZWiener: the dimension of Rmu must be greater than or equal to 1')
        if len(Rmu) != len(np.unique(Rmu)):
            raise ValueError('STOP21 in sub_polynomial_chaosZWiener: there are repetitions in Rmu')  
        if np.any(Rmu < 1) or np.any(Rmu > nu):
            raise ValueError('STOP22 in sub_polynomial_chaosZWiener: at least one integer in Rmu is not within the valid range [1,nu]')  

        if RM.size >=1:           # checking if not empty
           if RM.ndim != 1:       # ckecking if it is a 1D array
               raise ValueError('STOP23 in sub_polynomial_chaosZWiener: RM must be a 1D array')              
        nbM = len(RM)
        if nbM < 1:
            raise ValueError('STOP24 in sub_polynomial_chaosZWiener: the dimension of RM must be greater than or equal to 1')
        if len(RM) != len(np.unique(RM)):
            raise ValueError('STOP25 in sub_polynomial_chaosZWiener: there are repetitions in RM')  
        if np.any(RM < 0): 
            raise ValueError('STOP26 in sub_polynomial_chaosZWiener: at least one integer in RM is not within the valid range')                

    #--- Case  ind_PCE_compt = 1 (computation of the PCE for given mu = mu_PCE and M = M_PCE)
    if ind_PCE_compt == 1:
        if mu_PCE < 1 or mu_PCE > nu:
            raise ValueError('STOP27 in sub_polynomial_chaosZWiener: integer mu_PCE is not within the valid range [1,nu]')
        if M_PCE < 0:
            raise ValueError('STOP28 in sub_polynomial_chaosZWiener: integer M_PCE must be greater than or equal to 0')

    #--- Parameters controlling display, print, plot, and parallel computation
    if ind_display_screen != 0 and ind_display_screen != 1:       
        raise ValueError('STOP29 in sub_polynomial_chaosZWiener: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:       
        raise ValueError('STOP30 in sub_solverDirect: ind_print must be equal to 0 or equal to 1')
    if ind_plot != 0 and ind_plot != 1:       
        raise ValueError('STOP31 in sub_polynomial_chaosZWiener: ind_plot must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:       
        raise ValueError('STOP32 in sub_polynomial_chaosZWiener: ind_parallel must be equal to 0 or equal to 1')

    #--- if ind_PCE_compt = 1 and ind_plot = 1, parameters and data controlling plots for H_ar and H_PCE
    if ind_PCE_compt == 1 and ind_plot == 1:
        if nbplotHsamples >= 1:  # MatRplotHsamples(nbplotHsamples)
            if np.any(MatRplotHsamples < 1) or np.any(MatRplotHsamples > nu):  # at least one integer is not within the valid range
                raise ValueError('STOP33 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHsamples is not in range [1,nu]') 
        if nbplotHClouds >= 1:  # MatRplotHClouds(nbplotHClouds,3)
            if MatRplotHClouds.shape[1] != 3:
                raise ValueError('STOP34 in sub_polynomial_chaosZWiener: the second dimension of MatRplotHClouds must be equal to 3') 
            if np.any(MatRplotHClouds < 1) or np.any(MatRplotHClouds > nu):  # At least one integer is not within the valid range
                raise ValueError('STOP35 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHClouds is not in range [1,nu]')         
        if nbplotHpdf >= 1:  # MatRplotHpdf(nbplotHpdf)
            if np.any(MatRplotHpdf < 1) or np.any(MatRplotHpdf > nu):  # at least one integer  is not within the valid range
                raise ValueError('STOP36 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHpdf is not in [1,nu]')            
        if nbplotHpdf2D >= 1:  # MatRplotHpdf2D(nbplotHpdf2D,2)
            if MatRplotHpdf2D.shape[1] != 2:
                raise ValueError('STOP37 in sub_polynomial_chaosZWiener: the second dimension of MatRplotHpdf2D must be equal to 2') 
            if np.any(MatRplotHpdf2D < 1) or np.any(MatRplotHpdf2D > nu):  # at least one integer is not within the valid range
                raise ValueError('STOP38 in sub_polynomial_chaosZWiener: at least one integer in MatRplotHpdf2D is not in [1,nu]')       

    #--- for PCE-parameters identification, checking that K < nbMC 
    if ind_PCE_ident == 1: 
        MatRK = np.zeros((nbM, nbmu))
        for imu in range(nbmu):
            mu = Rmu[imu]  # mu = germ dimension, Rmu(nbmu)
            for iM in range(nbM):
                M = RM[iM]  # max degree of chaos polynomials
                K = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE 
                MatRK[iM, imu] = K
                if K >= nbMC:
                    if ind_print == 1:
                        with open('listing.txt', 'a+') as fidlisting:
                            fidlisting.write('      \n ')
                            fidlisting.write('            mu         M        K        nbMC \n ')
                            Rprint = [mu, M, K, nbMC]
                            fidlisting.write('      %7i    %7i  %7i   %7i \n ' % tuple(Rprint))
                            fidlisting.write('      \n ')
                            fidlisting.write(' STOP39 in sub_polynomial_chaosZWiener: K must be smaller than nbMC  \n ')
                            fidlisting.write('      \n ')
                    print(['     mu    M     K    nbMC'])
                    print([mu, M, K, nbMC])
                    print(' K must be smaller than nbMC for the computation of the PCE coefficients')
                    raise ValueError('STOP39 in sub_polynomial_chaosZWiener: K must be smaller than nbMC')

    #--- for PCE computation, checking that K < nbMC 
    if ind_PCE_compt == 1:
        mu = mu_PCE   # mu = germ dimension, Rmu(nbmu,1)            
        M  = M_PCE    # max degree of chaos polynomials
        K  = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE 
        if K >= nbMC:
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write('          mu_PCE     M_PCE    K        nbMC \n ')
                    Rprint = [mu, M, K, nbMC]
                    fidlisting.write('      %7i    %7i  %7i   %7i \n ' % tuple(Rprint))
                    fidlisting.write('      \n ')
                    fidlisting.write(' STOP40 in sub_polynomial_chaosZWiener: K must be smaller than nbMC  \n ')
                    fidlisting.write('      \n ')
            print(['   mu_PCE M_PCE  K    nbMC'])
            print([mu, M, K, nbMC])
            print(' K must be smaller than nbMC for the computation of the PCE coefficients')
            raise ValueError('STOP40 in sub_polynomial_chaosZWiener: K must be smaller than nbMC')

    #--- print parameters and data
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' nu      = %9i \n ' % nu)
            fidlisting.write(' n_d     = %9i \n ' % n_d)
            fidlisting.write(' nbMC    = %9i \n ' % nbMC)
            fidlisting.write(' n_ar    = %9i \n ' % n_ar)
            fidlisting.write(' nbmDMAP = %9i \n ' % nbmDMAP)
            fidlisting.write('      \n ')
            fidlisting.write(' icorrectif    = %1i \n ' % icorrectif)
            fidlisting.write(' coeffDeltar   = %4i   \n ' % coeffDeltar)
            fidlisting.write(' ind_PCE_ident   = %1i \n ' % ind_PCE_ident)
            fidlisting.write(' ind_PCE_compt   = %1i \n ' % ind_PCE_compt)
            fidlisting.write(' nbMC_PCE    = %9i \n ' % nbMC_PCE)
            fidlisting.write('      \n ')
            if ind_PCE_ident == 1:
                fidlisting.write('      \n ')
                fidlisting.write(' nbmu    = %9i \n ' % nbmu)
                fidlisting.write('      \n ')
                fidlisting.write('----- Values of mu (germ dimension) and M (max degree) to search the optimal values muopt and Mopt  \n ')
                fidlisting.write('                \n ')
                fidlisting.write(' Rmu =          \n ')
                for value in Rmu:
                    fidlisting.write(f'{value:3d} ')
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')
                fidlisting.write(' RM =          \n ')
                for value in RM:
                    fidlisting.write(f'{value:3d} ')
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')
                fidlisting.write(' MatRK(nbM,nbmu) =      \n ')
                for i in range(nbM):
                    fidlisting.write(' '.join('%9i' % x for x in MatRK[i, :]))
                    fidlisting.write('\n')
                fidlisting.write('      \n ')
            if ind_PCE_compt == 1:
                fidlisting.write('      \n ')
                fidlisting.write(' mu_PCE    = %9i \n ' % mu_PCE)
                fidlisting.write(' M_PCE     = %9i \n ' % M_PCE)
                fidlisting.write(' K         = %9i \n ' % K)
                fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(' ind_display_screen = %1i \n ' % ind_display_screen)
            fidlisting.write(' ind_print          = %1i \n ' % ind_print)
            fidlisting.write(' ind_plot           = %1i \n ' % ind_plot)
            fidlisting.write(' ind_parallel       = %1i \n ' % ind_parallel)
            fidlisting.write('      \n ')

    #----------------------------------------------------------------------------------------------------------------------------------                           
    #       Construction of the realizations MatRxi(nu,nbMC) of the largest dimension germ RXi = (Xi_1,...Xi_nu) (max Rmu <= nu) 
    #       with RXi Gaussian, centered, and with covariance matrix equal to [I_nu], from ArrayWienner(nu,n_d,nbMC). Therefore 
    #       Xi_1,...,Xi_nu are statistically independent
    #-----------------------------------------------------------------------------------------------------------------------------------
                    
    # MatRxi(nu,nbMC),ArrayWienner(nu,n_d,nbMC),MatRa(n_d,nbmDMAP) 
    RrondA = np.sum(MatRa**2, axis=0)  # RrondA(nbmDMAP),MatRa(n_d,nbmDMAP)
    Rtemp  = 1.0 / np.sqrt(RrondA)     # Rtemp(nbmDMAP)
    Rahat = MatRa @ Rtemp              # Rahat(n_d)

    #--- Computing the step Deltar used for solving the ISDE
    s = ((4 / ((nu + 2) * n_d))**(1 / (nu + 4)))  # usual Silverman bandwidth  
    s2 = s**2   
    if icorrectif == 0:                     
        shss = 1
    elif icorrectif == 1:
        shss = 1 / np.sqrt(s2 + (n_d - 1) / n_d)                          
    sh = s * shss  
    Deltar = 2 * np.pi * sh / coeffDeltar

    #--- Computing MatRxi(nu,nbMC)
    coef = 1 / (np.sqrt(Deltar) * np.linalg.norm(Rahat))
    MatRxi = np.zeros((nu, nbMC))                                 # MatRxi(nu,nbMC)
    for ell in range(nbMC):
        MatRxi[:, ell] = coef * ArrayWienner[:, :, ell] @ Rahat   # MatRxi(nu,nbMC),ArrayWienner(nu,n_d,nbMC),Rahat(n_d,1)
    ArrayZhat = np.transpose(ArrayZ_ar, (0, 2, 1))                # ArrayZhat(nu,nbMC,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)
    
    #--- Computing the matrix-valued mean value [Zbar_ar] of [Z_ar] 
    MatRZbar_ar = np.mean(ArrayZ_ar, axis=2)  # MatRZbar_ar(nu,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)

    #--- Computing the square of the norm : E{||[Z_ar] - [Zbar_ar]||_F^2 } with the nbMC realizations of [Z_ar]
    normMatRZ2_ar = 0
    for ell in range(nbMC):
        MatRtemp_ell = ArrayZ_ar[:, :, ell] - MatRZbar_ar  # MatRtemp_ell(nu,nbmDMAP),MatRZbar_ar(nu,nbmDMAP),ArrayZ_ar(nu,nbmDMAP,nbMC)
        normMatRZ2_ar = normMatRZ2_ar +  np.sum(MatRtemp_ell**2) 
    norm2Zc = normMatRZ2_ar / nbMC
  
    #-------------------------------------------------------------------------------------------------------------------------------                           
    #       ind_PCE_ident = 1  : identification of the optimal values Mopt of M for each considered value of mu = Rmu(imu) and 
    #                            loading the optimal values in RMopt(imu), imu = 1 : nbmu   
    #-------------------------------------------------------------------------------------------------------------------------------

    if ind_PCE_ident == 1:  
        RMopt = np.zeros(nbmu,dtype=int)      # RMopt(nbmu)
        RJopt = np.zeros(nbmu)                # RJopt(nbmu)
        MatRJ = np.zeros((nbM, nbmu))         # MatRJ(nbM,nbmu)
        MatRerrorZL2 = np.zeros((nbM, nbmu))  # MatRerrorZL2(nbM,nbmu)  
        MatReta_PCE = np.zeros((nu, nar_PCE))
        
        for imu in range(nbmu):
            mu = Rmu[imu]                                  # mu = germ dimension, Rmu(nbmu)
            MatRxi_mu = MatRxi[:mu, :]                     # MatRxi_mu(mu,nbMC),MatRxi(nu,nbMC)
            MatRxi_mu_PCE = np.random.randn(mu, nbMC_PCE)  # MatRxi_mu_PCE(mu,nbMC_PCE)      
            RJ_mu = np.zeros(nbM)                          # RJ_mu(nbM)
            
            #--- Vectorized sequence
            if ind_parallel == 0:
                for iM in range(nbM):
                    M = RM[iM]  # max degree of chaos polynomials
                    K = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE 

                    #--- Construction of the realizations MatRPsiK(K,nbMC) of the chaos Psi_{Ralpha^(k)}(Xi) 
                    #
                    #    Xi = (Xi_1,...,Xi_mu)
                    #    Rbeta^(k) = (beta_1^(k) , ... , beta_mu^(k)) in R^mu with k = 1,...,K
                    #    Rbeta^(1) = (     0      , ... ,      0      ) in R^mu 
                    #    0 <= beta_1^(k) + ... + beta_mu^(k) <= M  for k = 1,...,K
                    #    Psi_{Rbeta^(1)}(Xi) = 1
                    #    MatRPsiK(k,ell) =  Psi_{Rbeta^(k)}(xi^ell), xi^ell = MatRxi(1:mu,ell), ell=1:nbMC              
                    #    MatRPsiK(K,nbMC)   
                    #    
                    MatRPsiK = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC, MatRxi_mu)  # MatRPsiK(K,nbMC)  
        
                    #--- Construction of ArrayRyhat(nu,K,nbmDMAP)   
                    ArrayRyhat = np.zeros((nu, K, nbmDMAP))  # ArrayRyhat(nu,K,nbmDMAP)
                    co = 1 / (nbMC - 1)
                    for alpha in range(nbmDMAP):  # ArrayRyhat(nu,K,nbmDMAP),ArrayZhat(nu,nbMC,nbmDMAP),MatRPsiK(K,nbMC)
                        ArrayRyhat[:, :, alpha] = co * ArrayZhat[:, :, alpha] @ MatRPsiK.T
        
                    #--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi) 
                    MatRPsiK_PCE = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC_PCE, MatRxi_mu_PCE)  # MatRPsiK_PCE(K,nbMC_PCE),MatRxi_mu_PCE(mu,nbMC_PCE)
        
                    #--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP) 
                    ArrayZhatPCE_mu = np.zeros((nu, nbMC_PCE, nbmDMAP))  # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
                    for alpha in range(nbmDMAP):                         # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE)
                        ArrayZhatPCE_mu[:, :, alpha] = ArrayRyhat[:, :, alpha] @ MatRPsiK_PCE
        
                    #--- Constructing ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                    ArrayZ_PCE_mu = np.transpose(ArrayZhatPCE_mu, (0, 2, 1))  # ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)  
        
                    #--- Computing MatRerrorZbar(iM,imu)  and RJ1(iM)    
                    MatRZbar_PCE_mu = np.mean(ArrayZ_PCE_mu, axis=2)  # MatRZbar_PCE_mu(nu,nbmDMAP),ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
        
                    #--- Computing MatRerrorZL2(iM,imu)
                    normMatRZ2_PCE_mu = 0
                    for ell in range(nbMC_PCE):                                         # ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                        MatRtemp_ell = ArrayZ_PCE_mu[:, :, ell] - MatRZbar_PCE_mu       # MatRtemp_ell(nu,nbmDMAP)
                        normMatRZ2_PCE_mu = normMatRZ2_PCE_mu + np.linalg.norm(MatRtemp_ell, 'fro')**2
                    norm2Zc_PCE_mu = normMatRZ2_PCE_mu / nbMC_PCE
                    MatRerrorZL2[iM, imu] = np.abs(norm2Zc - norm2Zc_PCE_mu) / norm2Zc  # MatRerrorZL2(nbM,nbmu)  
        
                    #--- Cost function J_mu(iM) with a L2-norm of  ArrayZ_ar(nu,nbmDMAP,nbMC) and ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)                    
                    RJ_mu[iM] = MatRerrorZL2[iM, imu] 
            
            #--- Parallel computation
            if ind_parallel == 1:                
                
                def compute_RJ_mu(iM):
                    M = RM[iM]                                                           # max degree of chaos polynomials
                    K = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE 
                                                                                         # (including index (0,0,...,0) 
                    #--- Construction of the realizations MatRPsiK(K,nbMC) of the chaos Psi_{Ralpha^(k)}(Xi) 
                    #
                    #    Xi = (Xi_1,...,Xi_mu)
                    #    Rbeta^(k) = (beta_1^(k) , ... , beta_mu^(k)) in R^mu with k = 1,...,K
                    #    Rbeta^(1) = (     0      , ... ,      0      ) in R^mu 
                    #    0 <= beta_1^(k) + ... + beta_mu^(k) <= M  for k = 1,...,K
                    #    Psi_{Rbeta^(1)}(Xi) = 1
                    #    MatRPsiK(k,ell) =  Psi_{Rbeta^(k)}(xi^ell), xi^ell = MatRxi(1:mu,ell), ell=1:nbMC              
                    #    MatRPsiK(K,nbMC)   
                    #                                                                         
                    MatRPsiK = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC, MatRxi_mu)  # MatRPsiK(K,nbMC)  

                    #--- Construction of ArrayRyhat(nu,K,nbmDMAP)   
                    ArrayRyhat = np.zeros((nu, K, nbmDMAP))       # ArrayRyhat(nu,K,nbmDMAP)
                    co = 1 / (nbMC - 1)
                    for alpha in range(nbmDMAP):                  # ArrayRyhat(nu,K,nbmDMAP),ArrayZhat(nu,nbMC,nbmDMAP),MatRPsiK(K,nbMC)
                        ArrayRyhat[:, :, alpha] = co * ArrayZhat[:, :, alpha] @ MatRPsiK.T

                    #--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi)     
                    MatRPsiK_PCE = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC_PCE, MatRxi_mu_PCE)  # MatRPsiK_PCE(K,nbMC_PCE),MatRxi_mu_PCE(mu,nbMC_PCE)
                    
                    #--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP) 
                    ArrayZhatPCE_mu = np.zeros((nu, nbMC_PCE, nbmDMAP))      # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
                    for alpha in range(nbmDMAP):                             # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE)
                        ArrayZhatPCE_mu[:, :, alpha] = ArrayRyhat[:, :, alpha] @ MatRPsiK_PCE

                    #---Constructing ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)  
                    ArrayZ_PCE_mu = np.transpose(ArrayZhatPCE_mu, (0, 2, 1))  # ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)  
                    
                    #---Computing MatRerrorZbar(iM,imu)  and RJ1(iM)
                    MatRZbar_PCE_mu = np.mean(ArrayZ_PCE_mu, axis=2)  # MatRZbar_PCE_mu(nu,nbmDMAP),ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)

                    #--- Computing MatRerrorZL2(iM,imu)
                    normMatRZ2_PCE_mu = 0
                    for ell in range(nbMC_PCE):                                    # ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                        MatRtemp_ell = ArrayZ_PCE_mu[:, :, ell] - MatRZbar_PCE_mu  # MatRtemp_ell(nu,nbmDMAP)
                        normMatRZ2_PCE_mu = normMatRZ2_PCE_mu + np.linalg.norm(MatRtemp_ell, 'fro')**2

                    norm2Zc_PCE_mu = normMatRZ2_PCE_mu / nbMC_PCE
                    MatRerrorZL2[iM, imu] = np.abs(norm2Zc - norm2Zc_PCE_mu) / norm2Zc  # MatRerrorZL2(nbM,nbmu)  
                    return MatRerrorZL2[iM, imu]
                
                #--- Cost function J_mu(iM) with a L2-norm of  ArrayZ_ar(nu,nbmDMAP,nbMC) and ArrayZ_PCE_mu(nu,nbmDMAP,nbMC_PCE)
                RJ_mu = Parallel(n_jobs=-1)(delayed(compute_RJ_mu)(iM) for iM in range(nbM))
                MatRerrorZL2[:, imu] = RJ_mu

            MatRJ[:, imu] = RJ_mu  # MatRJ(nbM,nbmu), RJ_mu(nbM)              

            #--- find optimal value Mopt(mu) of M for given mu based of RJ_mu
            Jopt_mu  = np.min(RJ_mu)
            iMopt_mu = np.argmin(RJ_mu)
            RMopt[imu] = RM[iMopt_mu]  # RMopt(nbmu)
            RJopt[imu] = Jopt_mu  # RJopt(nbmu)
        
        #--- Print the cost function and the optimal values based on RJ
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('               \n ')
                fidlisting.write('               \n ')  
                fidlisting.write(' --- Optimal value Mopt of M as a function of mu based of RJ \n ') 
                fidlisting.write('     computed using the L2-norm \n ')  
                fidlisting.write('               \n ')
                fidlisting.write('   mu  Mopt(mu)  Jopt(mu) \n ')  
                fidlisting.write('      \n ') 
                for imu in range(nbmu):
                    Rprint = [Rmu[imu], RMopt[imu], RJopt[imu]]
                    fidlisting.write('  %3i  %3i   %14.7e \n ' % tuple(Rprint))
                fidlisting.write('               \n ')  
                fidlisting.write('               \n ')  
                fidlisting.write(' --- Values of the cost function J(mu,M) (second-order moment) and number of PCE coefficients \n ')  
                fidlisting.write('     computed using the L2-norm \n ')  
                fidlisting.write('               \n ')
                fidlisting.write('  mu     M       J(mu,M)             K \n ') 
                fidlisting.write('      \n ')
                for imu in range(nbmu):
                    mu = Rmu[imu]  # germ dimension
                    for iM in range(nbM):
                        M = RM[iM]  # max degree of chaos polynomials
                        K = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE (including index (0,0,...,0) 
                        J = MatRJ[iM, imu]  # MatRJ(nbM,nbmu)
                        Rprint = [mu, M, J, K]
                        fidlisting.write(' %3i   %3i   %14.7e  %9i \n ' % tuple(Rprint))
                    fidlisting.write('      \n ')
                fidlisting.write('               \n ')  
                fidlisting.write('               \n ')  
                fidlisting.write(' --- Values of the second-order moment of Z_PCE as a function of mu and M  \n ')                        
                fidlisting.write('               \n ')
                fidlisting.write('  mu     M   error_L2(mu,M) \n ') 
                fidlisting.write('      \n ')
                for imu in range(nbmu):
                    mu = Rmu[imu]  # germ dimension
                    for iM in range(nbM):
                        M = RM[iM]  # max degree of chaos polynomials
                        Rprint = [mu, M, MatRerrorZL2[iM, imu]]
                        fidlisting.write(' %3i   %3i   %14.7e \n ' % tuple(Rprint))
                    fidlisting.write('      \n ')
                fidlisting.write('      \n ') 
                fidlisting.write('      \n ')  

        #--- plot the family of curves { M--> J(M,mu) }_mu 
        if ind_plot == 1:
            if nbM >= 2:    
                h = plt.figure()
                legendEntries = []  # Create a list to store legend entries
                for imu in range(nbmu):
                    mu = Rmu[imu]
                    plt.plot(RM, MatRJ[:, imu], '-')  # RM(nbM,1), MatRJ(nbM,nbmu)
                    legendEntries.append(f'$\\mu = {mu}$')  # Store legend entry for this plot
                plt.title(r'Function $M\mapsto J(M,\mu)$ computed with the $L^2$-norm', fontsize=16)
                plt.xlabel(r'$M$', fontsize=16)
                plt.ylabel(r'$J(M,\mu)$', fontsize=16)
                plt.legend(legendEntries, loc='upper right')  # Add legend with entries
                numfig += 1
                plt.savefig(f'figure_PolynomialChaosZWiener_J(M) with L2-norm_{numfig}.png')
                plt.close(h)

    #------------------------------------------------------------------------------------------------------------------------------                           
    #       ind_PCE_compt = 1: PCE computation for a given value  mu_PCE and M_PCE of mu and M 
    #------------------------------------------------------------------------------------------------------------------------------
  
    if ind_PCE_compt == 1:
        mu = mu_PCE
        M = M_PCE
        K = int(1e-12 + factorial(mu + M) / (factorial(mu) * factorial(M)))  # number of coefficients in the PCE (including index (0,0,...,0)
        
        #--- re-building the PCE coefficients for mu_PCE and M_PCE
        MatRxi_mu = MatRxi[:mu, :]                                                     # MatRxi_mu(mu,nbMC), MatRxi_mu(nu,nbMC)
        MatRPsiK = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC, MatRxi_mu)          # MatRPsiK(K,nbMC)
        co = 1 / (nbMC - 1)
        ArrayRyhat = np.zeros((nu, K, nbmDMAP))                                        # ArrayRyhat(nu,K,nbmDMAP)
        for alpha in range(nbmDMAP):                                                   # ArrayRyhat(nu,K,nbmDMAP), ArrayZhat(nu,nbMC,nbmDMAP)
            ArrayRyhat[:, :, alpha] = co * np.dot(ArrayZhat[:, :, alpha], MatRPsiK.T)  # MatRPsiK(K,nbMC)
        del MatRPsiK, MatRxi_mu, ArrayZhat

        #%%%%%%%% SURROGATE MODEL DEFINED BY PCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #--- Construction of nbMC_PCE realizations MatRPsiK_PCE(K,nbMC_PCE) of the chaos Psi_{Ralpha^(k)}(Xi) 
        MatRxi_mu_PCE = np.random.randn(mu, nbMC_PCE)                                      # MatRxi_mu_PCE(mu,nbMC_PCE)      
        MatRPsiK_PCE = sub_polynomial_chaosZWiener_PCE(K, M, mu, nbMC_PCE, MatRxi_mu_PCE)  # MatRPsiK_PCE(K,nbMC_PCE)
        del MatRxi_mu_PCE 

        #--- Construction of ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)    
        ArrayZhatPCE_mu = np.zeros((nu, nbMC_PCE, nbmDMAP))                                 # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP) 
        for alpha in range(nbmDMAP):                                                        # ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
            ArrayZhatPCE_mu[:, :, alpha] = ArrayRyhat[:, :, alpha] @ MatRPsiK_PCE           # ArrayRyhat(nu,K,nbmDMAP),MatRPsiK_PCE(K,nbMC_PCE) 
        del ArrayRyhat, MatRPsiK_PCE

        #--- Constructing ArrayH_PCE(nu,n_d,nbMC_PCE)
        ArrayZ_PCE = np.transpose(ArrayZhatPCE_mu, (0, 2, 1))  # ArrayZ_PCE(nu,nbmDMAP,nbMC_PCE),ArrayZhatPCE_mu(nu,nbMC_PCE,nbmDMAP)
        ArrayH_PCE = np.zeros((nu, n_d, nbMC_PCE))                                           
        for ell in range(nbMC_PCE):
            ArrayH_PCE[:, :, ell] = ArrayZ_PCE[:, :, ell] @ MatRg.T  # ArrayH_PCE(nu,n_d,nbMC_PCE),ArrayZ_PCE(nu,nbmDMAP,nbMC_PCE),MatRg(n_d,nbmDMAP) 
        MatReta_PCE = ArrayH_PCE.reshape(nu, nar_PCE)  # MatReta_PCE(nu,nar_PCE)
        del ArrayZ_PCE,  ArrayZhatPCE_mu,  ArrayH_PCE  

        # %%%%%%%% END SEQUENCE OF THE SURROGATE PCE MODEL

        #--- plot statistics for H_ar from realizations MatReta_ar(nu,n_ar) (learning) and plot statistics  for H_PCE from realizations 
        #    MatReta_PCE(nu,nar_PCE) computed with the PCE H_PCE of H_ar 
        if ind_plot == 1:
            sub_polynomial_chaosZWiener_plot_Har_HPCE(n_ar, nar_PCE, MatReta_ar, MatReta_PCE, nbplotHClouds, nbplotHpdf, nbplotHpdf2D, 
                                                     MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, numfig)
    SAVERANDendPCE = np.random.get_state()         
    ElapsedPCE     = time.time() - TimeStartPCE

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write('-------   Elapsed time for Task14_PolynomialChaosZwiener \n ')
            fidlisting.write('      \n ')
            fidlisting.write('Elapsed Time   =  %10.2f\n' % ElapsedPCE)
            fidlisting.write('      \n ')

    if ind_display_screen == 1:
        print('--- end Task14_PolynomialChaosZwiener')

    return nar_PCE, MatReta_PCE, SAVERANDendPCE
