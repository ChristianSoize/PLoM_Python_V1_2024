import numpy as np
from sub_polynomialChaosQWU_surrogate import sub_polynomialChaosQWU_surrogate

def sub_polynomialChaosQWU_realization_chaos(nw_obs, Rww_ell, Ralpham1_scale_chaos, Rbeta_scale_chaos, 
                                             Ng, K0, MatPower0, MatRa0, ng, KU, MatPowerU, MatRaU, Jmax, n_y, 
                                             MatRgamma_opt, Indm, Indk, Ralpha_scale_yy, RQQmean, MatRVectEig1s2):
    #----------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 23 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_polynomialChaosQWU_realizationPCE
    #  Subject      : This function allows to compute a realization qq_PCE^ell of QQ_PCE such that
    #                 qq_PCE^ell = PCE(ww^ell,u^ell) in which ww^ell is a given value of WW and u^ell is a realization of the
    #                 latent random variable U 
    #  Methodology  : the code is an extraction of the end or the script of sub_polynomialChaosQWU corresponding to
    #                 " Polynomial-chaos validation for MatRww_o(nw_obs,nar_PCE)"
    # 
    #  MANDATORY    : before using this function the following script for a loading file must be done
    #--- Load 
             # Define the file name
             # file_name = 'filePolynomialChaosQWU_for_realization.mat'
             # # Check if the file exists
             # if os.path.isfile(file_name):
             #     print(f'The file "{file_name}" exists.')
             #     data = scipy.io.loadmat(file_name)
             #     print(f'The file "{file_name}" has been loaded.')
             # else:
             #     print(f'STOP-ERROR: the file "{file_name}" does not exist.')

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
    #         nw_obs                      : dimension of WW_obs extracted from WW (unscaled observed control variable) 
    #         Rww_ell(nw_obs)             : value of the vector ww^ell
    #         Ralpham1_scale_chaos(Ng)
    #         Rbeta_scale_chaos(Ng)
    #         Ng,K0
    #         MatPower0(K0,Ng)
    #         MatRa0(K0,K0)
    #         ng,KU
    #         MatPowerU(KU,ng)
    #         MatRaU(KU,KU)
    #         Jmax
    #         n_y
    #         MatRgamma_opt(n_y,Jmax)
    #         Indm(Jmax)
    #         Indk(Jmax)
    #         Ralpha_scale_yy(n_y)
    #         RQQmean(nq_obs)
    #         MatRVectEig1s2(nq_obs,n_y)
    #
    #--- OUTPUT
    #         Rqq_chaos_ell(nq_obs)       

    seeds_temp = np.random.randint(0, 2**32)         # Random initialization of the generator with the seed 
    np.random.seed(seeds_temp)
       
    RU_ell = np.random.randn(ng, 1)
    Ry_chaos_ell = sub_polynomialChaosQWU_surrogate(1, nw_obs, n_y, Rww_ell, RU_ell, Ralpham1_scale_chaos, Rbeta_scale_chaos, 
                                                    Ng, K0, MatPower0, MatRa0, ng, KU, MatPowerU, MatRaU, Jmax, MatRgamma_opt, Indm, Indk)
    Ryy_chaos_ell = Ralpha_scale_yy * Ry_chaos_ell            # Ryy_chaos_ell(n_y)   
    Rqq_chaos_ell = RQQmean + MatRVectEig1s2 @ Ryy_chaos_ell  # Rqq_chaos_ell(nq_obs),RQQmean(nq_obs),MatRVectEig1s2(nq_obs,n_y),Ryy_chaos_ell(n_y)   
    return Rqq_chaos_ell
