import numpy as np
import time
import sys
from sub_solverDirect_parameters import sub_solverDirect_parameters
from sub_solverDirect_constraint0 import sub_solverDirect_constraint0
from sub_solverDirect_constraint123 import sub_solverDirect_constraint123
from sub_solverDirect_Kullback import sub_solverDirect_Kullback
from sub_solverDirect_Entropy import sub_solverDirect_Entropy
from sub_solverDirect_Mutual_Information import sub_solverDirect_Mutual_Information
from sub_solverDirect_plot_Hd_Har import sub_solverDirect_plot_Hd_Har

def sub_solverDirect(nu, n_d, nbMC, MatReta_d, ind_generator, icorrectif, f0_ref, ind_f0, coeffDeltar, M0transient,
                     nbmDMAP, MatRg, MatRa, ind_constraints, ind_coupling, iter_limit, epsc, minVarH, maxVarH, alpha_relax1,
                     iter_relax2, alpha_relax2, SAVERANDstartDirect, ind_display_screen, ind_print, ind_plot, ind_parallel,
                     MatRplotHsamples, MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, ind_Kullback, ind_Entropy, ind_MutualInfo):

    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM)
    #  Function name: sub_solverDirect
    #  Subject      : solver PLoM for direct predictions with or without the constraints of normalization fo H_ar
    #                 computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar
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
    #
    #    nu                  : dimension of random vector H = (H_1, ... H_nu)
    #    n_d                 : number of points in the training set for H
    #    nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar]
    #    MatReta_d(nu,n_d)   : n_d realizations of H
    #    ind_generator:      : 0 generator without using ISDE-projection basis = standard MCMC generator based on Hamiltonian dissipative
    #                        : 1 generator using the ISDE-projection basis
    #    icorrectif          = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
    #                        = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified
    #    f0_ref              : reference value (recommended value f0_ref = 4)
    #    ind_f0              : indicator for generating f0 (recommended value ind_f0 = 0):
    #                          if ind_f0 = 0, then f0 = f0_ref, and if ind_f0 = 1, then f0 = f0_ref/sh
    #    coeffDeltar         : coefficient > 0 (usual value is 20) for calculating Deltar
    #    M0transient         : the end-integration value, M0transient (for instance, 30), at which the stationary response of the ISDE is
    #                          reached, is given by the user. The corresponding final time at which the realization is extrated from
    #                          solverDirect_Verlet is M0transient*Deltar
    #
    #--- parameters related to the ISDE-projection basis
    #    nbmDMAP             : dimension of the ISDE-projection basis
    #    MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #    MatRa(n_d,nbmDMAP)  = MatRg*(MatRg'*MatRg)^{-1}
    #
    #--- parameters for the constraints (ind_constraints >= 1) related to the convergence of the Lagrange-multipliers iteration algorithm
    #    ind_constraints     = 0 : no constraints concerning E{H] = 0 and E{H H'} = [I_nu]
    #                        = 1 : constraints E{H_j^2} = 1 for j =1,...,nu
    #                        = 2 : constraints E{H] = 0 and E{H_j^2} = 1 for j =1,...,nu
    #                        = 3 : constraints E{H] = 0 and E{H H'} = [I_nu]
    #    ind_coupling        = 0 : for ind_constraints = 2 or 3, no coupling in  matrix MatRGammaS_iter (HIGHLY RECOMMENDED)
    #                        = 1 : for ind_constraints = 2 or 3, coupling all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
    #    iter_limit          : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers.
    #    epsc                =   : relative tolerance (for instance 1e-3) for the iteration-algorithm convergence
    #    minVarH                 : minimum imposed on E{H_j^2} with respect to 1 (for instance 0.999)
    #    maxVarH                 : maximum imposed on E{H_j^2} with respect to 1 (for instance 1.001)
    #                          NOTE 5: on the convergence criteria for the iteration algorithm computing the Lagrange multipliers:
    #                               Criterion 1: if iter > 10 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained
    #                                            for iter - 1. Convergence is then assumed to be reached and then, exit from the loop on iter
    #                               Criterion 2: if {minVarH_iter >= minVarH and maxVarH_iter <= maxVarH} or Rerr(iter) <= epsc, the
    #                                            variance of each component is greater than or equal to minVarH and less than or equal
    #                                            to maxVarH, or the relative error of the constraint satisfaction is less than or equal
    #                                            to the tolerance. The convergence is reached, and then exit from the loop on iter.
    #                               Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
    #                                            the convergence is assumed to be reached and then, exit from the loop on iter
    #                          --- Relaxation function  iter --> alpha_relax(iter) controlling convergence of the iterative algorithm
    #                              is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
    #    alpha_relax1        : value of alpha_relax for iter = 1  (for instance 0.001)
    #    iter_relax2         : value of iter (for instance, 20) such that  alpha_relax2 = alpha_relax(iter_relax2)
    #                          if iter_relax2 = 1, then alpha_relax (iter) = alpha_relax2 for all iter >=1
    #    alpha_relax2        : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = apha_relax2
    #                          NOTE 1: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
    #                          NOTE 2: If iter_relax2 >= 2, then
    #                                  for iter >= 1 and for iter < iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
    #                                  for iter >= iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax2
    #                          NOTE 3: for decreasing the error err(iter), increase the value of iter_relax2
    #                          NOTE 4: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2
    #
    #--- parameters and variables controling execution
    #    SAVERANDstartDirect : state of the random generator at the end of the PCA step
    #    ind_display_screen  : = 0 no display,              = 1 display
    #    ind_print           : = 0 no print,                = 1 print
    #    ind_plot            : = 0 no plot,                 = 1 plot
    #    ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #
    #--- data for the plots
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
    #    ind_Kullback                       = 0 : no computation of the Kullback-Leibler divergence of H_ar with respect to H_d
    #                                       = 1 :    computation of the Kullback-Leibler divergence of H_ar with respect to H_d
    #    ind_Entropy                        = 0 : no computation of the Entropy of Hd and Har
    #                                       = 1 :    computation of the Entropy of Hd and Har
    #    ind_MutualInfo                     = 0 : no computation of the Mutual Information iHd and iHar for Hd and Har
    #                                       = 1 :    computation of the Mutual Information iHd and iHar for Hd and Har
    #
    #--- OUTPUTS
    #
    #      n_ar                        : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #      MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar
    #      ArrayZ_ar(nu,nbmDMAP,nbMC)  : n_ar realizations of Z_ar
    #      ArrayWienner(nu,n_d,nbMC)   : ArrayWienner(nu,n_d,nbMC)
    #      SAVERANDendDirect           : state of the random generator at the end of sub_solverDirect
    #      d2mopt_ar                   : concentration of the probability measure of H_ar with respect to H_d in the means-square sense
    #      divKL                       : Kullback-Leibler divergence of H_ar with respect to H_d
    #      iHd                         : Mutual Information iHd for Hd
    #      iHar                        : Mutual Information iHar for Har
    #      entropy_Hd                  : Entropy of Hd
    #      entropy_Har                 : Entropy of Har
    #
    #--- INTERNAL PARAMETERS
    #
    #      s       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset)
    #                of the pdf p_H of H, having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu]
    #      sh      : modified Silver bandwidth for wich the normalization conditions are satisfied for any value of nu >= 1
    #      shss    : = 1 if icorrectif  = 0, and = sh/s if icorrectif = 1
    #      f_0     : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
    #      Deltar  : Stormer-Verlet integration-step of the ISDE
    #      M0estim : estimate of M0transient provided as a reference to the user
    #      Ralpha_relax(iter_limit,1): relaxation function for the iteration algorithm that computes the LaGrange multipliers
    #      n_ar    : number of learned realizations equal to nbMC*n_d:
    #
    #      nbplotHsamples : number >= 0 of the components numbers of H_ar for which the plot of the realizations are made
    #      nbplotHClouds  : number >= 0 of the 3 components numbers of H_ar for which the plot of the clouds are made
    #      nbplotHpdf     : number >= 0 of the components numbers of H_d and H_ar for which the plot of the pdfs are made
    #      nbplotHpdf2D   : number >= 0 of the 2 components numbers of H_d and H_ar for which the plot of the joint pdfs are made

    if ind_display_screen == 1:
        print('--- beginning Task6_SolverDirect')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write(' ------ Task6_SolverDirect \n ')
            fidlisting.write('\n ')

    TimeStartSolverDirect = time.time()
    numfig = 0
    n_ar   = nbMC * n_d
    
    if MatRplotHsamples.size >=1:           # checking if not empty
        if MatRplotHsamples.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP1 in sub_solverDirect: MatRplotHsamples must be a 1D array')
    if MatRplotHClouds.size >=1:           # checking if not empty
        if MatRplotHClouds.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP2 in sub_solverDirect: MatRplotHClouds must be a 2D array')    
    if MatRplotHpdf.size >=1:             # checking if not empty
        if MatRplotHpdf.ndim != 1:        # ckecking if it is a 1D array
            raise ValueError('STOP3 in sub_solverDirect: MatRplotHpdf must be a 1D array') 
    if MatRplotHpdf2D.size >=1:           # checking if not empty
        if MatRplotHpdf2D.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP4 in sub_solverDirect: MatRplotHpdf2D must be a 2D array')        

    nbplotHsamples = len(MatRplotHsamples)       # MatRplotHsamples(nbplotHsamples)
    nbplotHClouds  = MatRplotHClouds.shape[0]    # MatRplotHClouds(nbplotHClouds,3)
    nbplotHpdf     = len(MatRplotHpdf)           # MatRplotHpdf(nbplotHpdf)
    nbplotHpdf2D   = MatRplotHpdf2D.shape[0]     # MatRplotHpdf2D(nbplotHpdf2D,2)
    
    #--- initializing the random generator at the value of the end of the PCA step 
    np.random.set_state(SAVERANDstartDirect)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    #                                    Check data, parameters, and initialization
    #----------------------------------------------------------------------------------------------------------------------------------

    if nu > n_d or nu < 1 or n_d < 1:
        raise ValueError('STOP5 in sub_solverDirect: nu > n_d or nu < 1 or n_d < 1')
    
    nutemp, ndtemp = MatReta_d.shape
    if nutemp != nu or ndtemp != n_d:
        raise ValueError('STOP6 in sub_solverDirect: the dimensions of MatReta_d are not consistent with nu and n_d')
    if ind_generator != 0 and ind_generator != 1:
        raise ValueError('STOP7 in sub_solverDirect: ind_generator must be equal to 0 or equal to 1')
    if icorrectif != 0 and icorrectif != 1:
        raise ValueError('STOP8 in sub_solverDirect: icorrectif must be equal to 0 or equal to 1')
    if f0_ref <= 0:
        raise ValueError('STOP9 in sub_solverDirect: f0_ref must be strictly positif')
    if ind_f0 != 0 and ind_f0 != 1:
        raise ValueError('STOP10 in sub_solverDirect: ind_f0 must be equal to 0 or equal to 1')
    if coeffDeltar < 1:
        raise ValueError('STOP11 in sub_solverDirect: coeffDeltar must be greater than or equal to 1')
    if M0transient < 1:
        raise ValueError('STOP12 in sub_solverInverse: M0transient must be greater than or equal to 1')
    if nbmDMAP < 1 or nbmDMAP > n_d:
        raise ValueError('STOP13 in sub_solverDirect: nbmDMAP < 1 or nbmDMAP > n_d')
    ndtemp, nbmDMAPtemp = MatRg.shape
    if ndtemp != n_d or nbmDMAPtemp != nbmDMAP:
        raise ValueError('STOP14 in sub_solverDirect: the dimensions of MatRg are not consistent with n_d and nbmDMAP')
    ndtemp, nbmDMAPtemp = MatRa.shape
    if ndtemp != n_d or nbmDMAPtemp != nbmDMAP:
        raise ValueError('STOP15 in sub_solverDirect: the dimensions of MatRa are not consistent with n_d and nbmDMAP')
    
    if ind_constraints != 0 and ind_constraints != 1 and ind_constraints != 2 and ind_constraints != 3:
        raise ValueError('STOP16 in sub_solverDirect: ind_constraints must be equal to 0, 1, 2, or 3')
    if ind_constraints == 0:
        iter_limit   = 0
        iter_relax2  = 0
        alpha_relax1 = 0
        alpha_relax2 = 0
    if ind_constraints >= 1:
        if ind_coupling != 0 and ind_coupling != 1:
            raise ValueError('STOP17 in sub_solverDirect: ind_coupling must be equal to 0 or equal to 1')
        if iter_limit < 1:
            raise ValueError('STOP18 in sub_solverDirect: iter_limit must be greater than or equal to 1')
        if epsc < 0 or epsc >= 1:
            raise ValueError('STOP19 in sub_solverDirect: epsc < 0 or epsc >= 1 ')
        if minVarH <= 0 or minVarH >= 1:
            raise ValueError('STOP20 in sub_solverDirect: minVarH <= 0 or minVarH >= 1 ')
        if maxVarH <= minVarH or maxVarH <= 1:
            raise ValueError('STOP21 in sub_solverDirect: maxVarH <= minVarH or maxVarH <= 1 ')
        if alpha_relax1 < 0 or alpha_relax1 > 1:
            raise ValueError('STOP22 in sub_solverDirect: value of alpha_relax1 out the range [0,1]')
        if alpha_relax2 < 0 or alpha_relax2 > 1:
            raise ValueError('STOP23 in sub_solverDirect: value of alpha_relax2 out the range [0,1]')
        if iter_relax2 >= 2 and iter_relax2 <= iter_limit:
            if alpha_relax1 > alpha_relax2:
                raise ValueError('STOP24 in sub_solverDirect: alpha_relax1 must be less than or equal to alpha_relax2')
        if iter_relax2 > iter_limit:
            raise ValueError('STOP25 in sub_solverDirect: iter_relax2 must be less than or equal to iter_limit')

    #--- Parameters controlling display, print, plot, and parallel computation
    if ind_display_screen != 0 and ind_display_screen != 1:       
        raise ValueError('STOP26 in sub_solverDirect: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:       
        raise ValueError('STOP27 in sub_solverDirect: ind_print must be equal to 0 or equal to 1')
    if ind_plot != 0 and ind_plot != 1:       
        raise ValueError('STOP28 in sub_solverDirect: ind_plot must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:       
        raise ValueError('STOP29 in sub_solverDirect: ind_parallel must be equal to 0 or equal to 1')
    
    if nbplotHsamples >= 1:  # MatRplotHsamples(nbplotHsamples)
        if np.any(MatRplotHsamples < 1) or np.any(MatRplotHsamples > nu):  # at least one integer is not within the valid range
            raise ValueError('STOP30 in sub_solverDirect: at least one integer in MatRplotHsamples is not in range [1,nu]') 
    if nbplotHClouds >= 1:  # MatRplotHClouds(nbplotHClouds,3)
        if MatRplotHClouds.shape[1] != 3:
            raise ValueError('STOP31 in sub_solverDirect: the second dimension of MatRplotHClouds must be equal to 3') 
        if np.any(MatRplotHClouds < 1) or np.any(MatRplotHClouds > nu):  # At least one integer is not within the valid range
                raise ValueError('STOP32 in sub_solverDirect: at least one integer in MatRplotHClouds is not in range [1,nu]')         
    if nbplotHpdf >= 1:  # MatRplotHpdf(nbplotHpdf)
        if np.any(MatRplotHpdf < 1) or np.any(MatRplotHpdf > nu):  # at least one integer  is not within the valid range
            raise ValueError('STOP33 in sub_solverDirect: at least one integer in MatRplotHpdf is not in [1,nu]')            
    if nbplotHpdf2D >= 1:  # MatRplotHpdf2D(nbplotHpdf2D,2)
        if MatRplotHpdf2D.shape[1] != 2:
            raise ValueError('STOP34 in sub_solverDirect: the second dimension of MatRplotHpdf2D must be equal to 2') 
        if np.any(MatRplotHpdf2D < 1) or np.any(MatRplotHpdf2D > nu):  # at least one integer is not within the valid range
            raise ValueError('STOP35 in sub_solverDirect: at least one integer in MatRplotHpdf2D is not in [1,nu]') 

    #----------------------------------------------------------------------------------------------------------------------------------
    #                                    Computing the parameters used by solverDirect
    #----------------------------------------------------------------------------------------------------------------------------------

    (s,sh,shss,f0,Deltar,M0estim,Ralpha_relax) = sub_solverDirect_parameters(nu,n_d,icorrectif,f0_ref,ind_f0,
                                                                                 coeffDeltar,ind_constraints,iter_limit,
                                                                                 alpha_relax1,iter_relax2,alpha_relax2)

    #----------------------------------------------------------------------------------------------------------------------------------
    #                                    Print data input for learning
    #----------------------------------------------------------------------------------------------------------------------------------

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(' ---  Parameters for the learning \n ')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(f' nu            = {nu:7d} \n')
            fidlisting.write(f' n_d           = {n_d:7d} \n')
            fidlisting.write(f' nbMC          = {nbMC:7d} \n')
            fidlisting.write(f' n_ar          = {n_ar:7d} \n')
            fidlisting.write('\n ')
            fidlisting.write(f' ind_generator = {ind_generator:1d} \n')
            fidlisting.write(f' icorrectif    = {icorrectif:1d} \n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(f' f0_ref        = {f0_ref:8.4f} \n')
            fidlisting.write(f' ind_f0        = {ind_f0:7d}   \n')
            fidlisting.write(f' f0            = {f0:8.4f} \n')
            fidlisting.write(f' coeffDeltar   = {coeffDeltar:4d}   \n')
            fidlisting.write(f' Deltar        = {Deltar:14.7e}\n')           
            fidlisting.write(f' M0transient   = {M0transient:7d}   \n')
            fidlisting.write(f' M0estim       = {M0estim:8.2f}   \n')
            fidlisting.write('\n ')
            fidlisting.write(f' nbmDMAP       = {nbmDMAP:7d} \n')
            fidlisting.write('\n ')
            fidlisting.write(f' ind_constraints  = {ind_constraints:1d} \n')
            if ind_constraints >= 1:
                fidlisting.write(f'    ind_coupling  = {ind_coupling:1d}    \n')
                fidlisting.write(f'    iter_limit    = {iter_limit:7d}    \n')
                fidlisting.write(f'    epsc          = {epsc:14.7e} \n')
                fidlisting.write(f'    minVarH       = {minVarH:8.6f}  \n')
                fidlisting.write(f'    maxVarH       = {maxVarH:8.6f}  \n')
                fidlisting.write(f'    alpha_relax1  = {alpha_relax1:8.4f}  \n')
                fidlisting.write(f'    iter_relax2   = {iter_relax2:7d}    \n')
                fidlisting.write(f'    alpha_relax2  = {alpha_relax2:8.4f}  \n')
            fidlisting.write('\n ')
            fidlisting.write(f's           = {s:14.7e} \n')
            fidlisting.write(f'shss        = {shss:14.7e} \n')
            fidlisting.write(f'sh          = {sh:14.7e} \n')
            fidlisting.write('\n ')
            fidlisting.write(f' ind_display_screen = {ind_display_screen:1d} \n')
            fidlisting.write(f' ind_print          = {ind_print:1d} \n')
            fidlisting.write(f' ind_plot           = {ind_plot:1d} \n')
            fidlisting.write(f' ind_parallel       = {ind_parallel:1d} \n')
            fidlisting.write('\n ')
            fidlisting.write(f' ind_Kullback       = {ind_Kullback:1d} \n')
            fidlisting.write(f' ind_Entropy        = {ind_Entropy:1d} \n')
            fidlisting.write(f' ind_MutualInfo     = {ind_MutualInfo:1d} \n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    #-----------------------------------------------------------------------------------------------------------------------------------------
    #             Generating NnbMC realizations
    #             For avoiding CPU time generated by " if test" on ind_constraints in sub_solverDirect_Verlet_constraintj.py and
    #             sub_solverDirect_Lrond_constraintj.py, the "if test" is implementent outside
    #-----------------------------------------------------------------------------------------------------------------------------------------

    #--- Generation of random Wienner germs for the Ito equation
    ArrayGauss              = np.random.randn(nu,n_d,nbMC)                 # ArrayGauss(nu,n_d,nbMC)
    ArrayWiennerM0transient = np.random.randn(nu,n_d,M0transient,nbMC)     # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC) 

    #--- Construction/saving of ArrayWienner(nu,n_d,nbMC), which contains of the nbMC realizations of the (nu,n_d) Wienner matrix for the last
    #    integration time M0transient. This array is saved for a possible use in postprocessing for the polynomial chaos expansion (PCE) of Z
    ArrayTemp    = np.transpose(ArrayWiennerM0transient, (0, 1, 3, 2))     # ArrayTemp(nu,n_d,nbMC,M0transient) 
    ArrayWienner = np.sqrt(Deltar) * ArrayTemp[:, :, :, M0transient - 1]   # ArrayWienner(nu,n_d,nbMC)
    del ArrayTemp

    #--- No constraints on H
    if ind_constraints == 0:                                                # MatReta_ar(nu,n_ar),ArrayZ_ar(nu,nbmDMAP,nbMC)  
        (MatReta_ar,ArrayZ_ar) = sub_solverDirect_constraint0(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh,
                                                               MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,
                                                               ind_parallel,ind_print)

    #--- constraints are applied on H
    if ind_constraints >= 1:                                                # MatReta_ar(nu,n_ar),ArrayZ_ar(nu,nbmDMAP,nbMC)
        (MatReta_ar,ArrayZ_ar) = sub_solverDirect_constraint123(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh,
                                                                MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,
                                                                ind_constraints,ind_coupling,epsc,iter_limit,Ralpha_relax,
                                                                minVarH,maxVarH,ind_display_screen,ind_print,ind_parallel,numfig)

    del ArrayGauss, ArrayWiennerM0transient

    #-----------------------------------------------------------------------------------------------------------------------------------
    #                    Estimation of the measure concentration of H_ar with respect to H_d,
    #                    whose realizations are MatReta_ar(nu,n_ar) and MatReta_ar(nu,n_d)
    #-----------------------------------------------------------------------------------------------------------------------------------

    #--- Estimation of the measure concentration d2mopt_ar = E{|| [H_ar] - [eta_d] ||^2} / || [eta_d] ||^2
    d2mopt_ar = 0
    ArrayH_ar = MatReta_ar.reshape((nu,n_d,nbMC))
    for ell in range(nbMC):
        d2mopt_ar = d2mopt_ar + np.linalg.norm(ArrayH_ar[:, :, ell] - MatReta_d,'fro') ** 2  # MatReta_d(nu,n_d)

    deno2 = np.linalg.norm(MatReta_d,'fro') ** 2
    d2mopt_ar = d2mopt_ar/ (nbMC * deno2)

    if ind_display_screen == 1:
        print(' ')
        print('------ Concentration of the measure of H_ar with respect to H_d')
        print(f'       d^2(m_opt)_ar = {d2mopt_ar}')
        print(' ')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(' --- Concentration of the measure of H_ar with respect to H_d \n ')
            fidlisting.write('                                                 \n ')
            fidlisting.write(f'         d^2(m_opt)_ar =  {d2mopt_ar:14.7e} \n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    #----------------------------------------------------------------------------------------------------------------------------------
    #   Estimation of the Kullback divergence between H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
    #----------------------------------------------------------------------------------------------------------------------------------

    divKL = 0
    if ind_Kullback == 1:
        divKL = sub_solverDirect_Kullback(MatReta_ar,MatReta_d,ind_parallel)

        if ind_display_screen == 1:
            print(' ')
            print('------ Kullback-Leibler divergence of H_ar with respect to H_d')
            print(f'       divKL = {divKL}')
            print(' ')

        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(' --- Kullback-Leibler divergence of H_ar with respect to H_d \n ')
                fidlisting.write('                                                 \n ')
                fidlisting.write(f'        divKL =  {divKL:14.7e} \n')
                fidlisting.write('\n ')
                fidlisting.write('\n ')

    #----------------------------------------------------------------------------------------------------------------------------------
    #   Entropy of H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
    #----------------------------------------------------------------------------------------------------------------------------------

    entropy_Hd  = 0
    entropy_Har = 0
    if ind_Entropy == 1:
        entropy_Hd  = sub_solverDirect_Entropy(MatReta_d,ind_parallel)
        entropy_Har = sub_solverDirect_Entropy(MatReta_ar,ind_parallel)

        if ind_display_screen == 1:
            print(' ')
            print('------ Entropy of Hd and Har')
            print(f'       entropy_Hd  = {entropy_Hd}')
            print(f'       entropy_Har = {entropy_Har}')
            print(' ')

        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(' --- Entropy of Hd and Har \n ')
                fidlisting.write('\n ')
                fidlisting.write(f'        entropy_Hd   =  {entropy_Hd:14.7e} \n')
                fidlisting.write(f'         entropy_Har  =  {entropy_Har:14.7e} \n')
                fidlisting.write('\n ')
                fidlisting.write('\n ')

    #----------------------------------------------------------------------------------------------------------------------------------
    #   Statistical dependence of the components of H_d and H_ar whose realizations are MatReta_d(nu,n_d) and MatReta_ar(nu,n_ar)
    #----------------------------------------------------------------------------------------------------------------------------------

    iHd  = 0
    iHar = 0
    if ind_MutualInfo == 1:
        iHd  = sub_solverDirect_Mutual_Information(MatReta_d,ind_parallel)
        iHar = sub_solverDirect_Mutual_Information(MatReta_ar,ind_parallel)

        if ind_display_screen == 1:
            print(' ')
            print('------ Mutual Information iHd and iHar for Hd and Har')
            print(f'       iHd  = {iHd}')
            print(f'       iHar = {iHar}')
            print(' ')

        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(' --- Mutual Information iHd and iHar for Hd and Har \n ')
                fidlisting.write('\n ')
                fidlisting.write(f'        iHd   =  {iHd:14.7e} \n')
                fidlisting.write(f'         iHar  =  {iHar:14.7e} \n')
                fidlisting.write('\n ')
                fidlisting.write('\n ')

    #------------------------------------------------------------------------------------------------------------------------------------------
    #  plot statistics for H_d and H_ar, from realizations MatReta_d(nu,n_d) (training) and learned realizations MatReta_ar(nu,n_ar) (learning)
    #------------------------------------------------------------------------------------------------------------------------------------------

    sub_solverDirect_plot_Hd_Har(n_d,n_ar,MatReta_d,MatReta_ar,nbplotHsamples,nbplotHClouds,nbplotHpdf,nbplotHpdf2D,
                                 MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig)
    
    SAVERANDendDirect   = np.random.get_state()
    ElapsedSolverDirect = time.time() - TimeStartSolverDirect

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('-------   Elapsed time for Task6_SolverDirect \n ')
            fidlisting.write('\n ')
            fidlisting.write(f'Elapsed Time   =  {ElapsedSolverDirect:10.2f}\n')
            fidlisting.write('\n ')

    if ind_display_screen == 1:
        print('--- end Task6_SolverDirect')

    return (n_ar, MatReta_ar, ArrayZ_ar, ArrayWienner, SAVERANDendDirect, d2mopt_ar, divKL, iHd, iHar, entropy_Hd, entropy_Har)
