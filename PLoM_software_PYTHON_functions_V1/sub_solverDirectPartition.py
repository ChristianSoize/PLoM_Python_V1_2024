import numpy as np
import time
import random
from sub_solverDirectPartition_parameters_groupj import sub_solverDirectPartition_parameters_groupj
from sub_solverDirectPartition_isotropic_kernel_groupj import sub_solverDirectPartition_isotropic_kernel_groupj
from sub_solverDirectPartition_constraint0 import sub_solverDirectPartition_constraint0
from sub_solverDirectPartition_constraint0mj1 import sub_solverDirectPartition_constraint0mj1
from sub_solverDirectPartition_constraint123 import sub_solverDirectPartition_constraint123
from sub_solverDirectPartition_parameters_Ralpha import sub_solverDirectPartition_parameters_Ralpha
from sub_solverDirect_Kullback import sub_solverDirect_Kullback
from sub_solverDirect_Entropy import sub_solverDirect_Entropy
from sub_solverDirect_Mutual_Information import sub_solverDirect_Mutual_Information
from sub_solverDirectPartition_plot_Hd_Har import sub_solverDirectPartition_plot_Hd_Har

def sub_solverDirectPartition(nu, n_d, nbMC, MatReta_d, ind_generator, icorrectif, f0_ref, ind_f0, coeffDeltar, M0transient,
                              epsilonDIFFmin, step_epsilonDIFF, iterlimit_epsilonDIFF, comp_ref, ind_constraints, ind_coupling,
                              iter_limit, epsc, minVarH, maxVarH, alpha_relax1, iter_relax2, alpha_relax2, ngroup, Igroup,
                              MatIgroup, SAVERANDstartDirectPartition, ind_display_screen, ind_print, ind_plot, ind_parallel,
                              MatRplotHsamples, MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, ind_Kullback, ind_Entropy,
                              ind_MutualInfo):

    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 01 July 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirectPartitionPartition
    #  Subject      : solver PLoM for direct predictions with the partition, and with or without constraints of normalization for H_ar
    #                 computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar using partition
    #                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
    #                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
    #                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
    #
    #  Publications: [1] C. Soize, Optimal partition in terms of independent random vectors of any non-Gaussian vector defined by a set of
    #                       realizations,SIAM-ASA Journal on Uncertainty Quantification,doi: 10.1137/16M1062223, 5(1), 176-211 (2017).
    #                [2] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [3] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #                [4] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
    #                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).   
    #                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).  
    #                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
    #                [6] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
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
    #                          reached, is given by the user. The corresponding final time at which the realization is extracted from 
    #                          solverDirect_Verlet is M0transient*Deltar 
    #
    #--- parameters for computing epsilonDIFF for each group j
    #    epsilonDIFFmin        : epsilonDIFF is searched in interval [epsilonDIFFmin , +infty[                                    
    #    step_epsilonDIFF      : step for searching the optimal value epsilonDIFF starting from epsilonDIFFmin
    #    iterlimit_epsilonDIFF : maximum number of the iteration algorithm for computing epsilonDIFF                              
    #    comp_ref              : value in  [ 0.1 , 0.5 [  used for stopping the iteration algorithm.
    #                            if comp =  Rlambda(nbmDMAP+1)/Rlambda(nbmDMAP) <= comp_ref, then algorithm is stopped
    #                            The standard value for comp_ref is 0.2 
    #
    #--- parameters for the constraints (ind_constraints >= 1) related to the convergence of the Lagrange-multipliers iteration algorithm 
    #    these constraints are applied to each group of the partition  Y^j  = (Y^j_1,...Y^j_mj) with dimension mj
    #    ind_constraints     = 0 : no constraints concerning E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj]
    #                        = 1 : constraints E{(Y^j_k)^2} = 1 for k =1,...,mj   
    #                        = 2 : constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
    #                        = 3 : constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
    #    ind_coupling        = 0 : for ind_constraints = 2 or 3, no coupling in  matrix MatRGammaS_iter (HIGHLY RECOMMENDED)
    #                        = 1 : for ind_constraints = 2 or 3, coupling all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
    #    iter_limit          : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
    #    epsc                =   : relative tolerance (for instance 1e-3) for the iteration-algorithm convergence 
    #    minVarH                 : minimum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 0.999) 
    #    maxVarH                 : maximum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 1.001) 
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
    #    alpha_relax2        : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = alpha_relax2
    #                          NOTE 1: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
    #                          NOTE 2: If iter_relax2 >= 2, then  
    #                                  for iter >= 1 and for iter < iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
    #                                  for iter >= iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax2
    #                          NOTE 3: for decreasing the error err(iter), increase the value of iter_relax2
    #                          NOTE 4: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2  
    #
    #--- information concerning the partition 
    #    ngroup                 : number of constructed independent groups  
    #    Igroup(ngroup)         : vector Igroup(ngroup), mj = Igroup(j),  mj is the dimension of Y^j = (Y^j_1,...,Y^j_mj) = (H_jr1,... ,H_jrmj)  
    #    MatIgroup(ngroup,mmax) : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj 
    #                             with mmax = max_j mj for j = 1, ... , ngroup
    #
    #--- parameters and variables controling execution
    #    SAVERANDstartDirectPartition : state of the random generator at the end of the PCA step
    #    ind_display_screen           : = 0 no display,              = 1 display
    #    ind_print                    : = 0 no print,                = 1 print
    #    ind_plot                     : = 0 no plot,                 = 1 plot
    #    ind_parallel                 : = 0 no parallel computation, = 1 parallel computation
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
    #      SAVERANDendDirectPartition  : state of the random generator at the end of sub_solverDirectPartition
    #      d2mopt_ar                   : concentration of the probability measure of H_ar with respect to H_d in the means-square sense
    #      divKL                       : Kullback-Leibler divergence of H_ar with respect to H_d
    #      iHd                         : Mutual Information iHd for Hd 
    #      iHar                        : Mutual Information iHar for Har
    #      entropy_Hd                  : Entropy of Hd
    #      entropy_Har                 : Entropy of Har
    #
    #--- INTERNAL PARAMETERS
    #      For each group j:
    #      sj       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset) 
    #                 of the pdf p_{Y^j} of Y^j, having to satisfy the normalization condition E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj] 
    #      shj      : modified Silver bandwidth for which the normalization conditions are satisfied for any value of mj >= 1 
    #      shssj    : = 1 if icorrectif  = 0, and = shj/sj if icorrectif = 1
    #      f0j      : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
    #      Deltarj  : Stormer-Verlet integration-step of the ISDE        
    #      M0estimj : estimate of M0transient provided as a reference to the user
    #
    #      For all groups (independent of j)
    #      Ralpha_relax(iter_limit,1): relaxation function for the iteration algorithm that computes the LaGrange multipliers
    #      n_ar    : number of learned realizations equal to nbMC*n_d
    #      nbplotHsamples : number >= 0 of the components numbers of H_ar for which the plot of the realizations are made   
    #      nbplotHClouds  : number >= 0 of the 3 components numbers of H_ar for which the plot of the clouds are made
    #      nbplotHpdf     : number >= 0 of the components numbers of H_d and H_ar for which the plot of the pdfs are made   
    #      nbplotHpdf2D   : number >= 0 of the 2 components numbers of H_d and H_ar for which the plot of the joint pdfs are made

    if ind_display_screen == 1:
        print('--- beginning Task7_SolverDirectPartition')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write(' ------ Task7_SolverDirectPartition \n ')
            fidlisting.write('\n ')

    import time
    TimeStartSolverDirectPartition = time.time()
    n_ar = nbMC * n_d

    if MatRplotHsamples.size >=1:           # checking if not empty
        if MatRplotHsamples.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP1 in sub_solverDirectPartition: MatRplotHsamples must be a 1D array')
    if MatRplotHClouds.size >=1:           # checking if not empty
        if MatRplotHClouds.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP2 in sub_solverDirectPartition: MatRplotHClouds must be a 2D array')    
    if MatRplotHpdf.size >=1:             # checking if not empty
        if MatRplotHpdf.ndim != 1:        # ckecking if it is a 1D array
            raise ValueError('STOP3 in sub_solverDirectPartition: MatRplotHpdf must be a 1D array') 
    if MatRplotHpdf2D.size >=1:           # checking if not empty
        if MatRplotHpdf2D.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP4 in sub_solverDirectPartition: MatRplotHpdf2D must be a 2D array')        

    nbplotHsamples = len(MatRplotHsamples)       # MatRplotHsamples(nbplotHsamples)
    nbplotHClouds  = MatRplotHClouds.shape[0]    # MatRplotHClouds(nbplotHClouds,3)
    nbplotHpdf     = len(MatRplotHpdf)           # MatRplotHpdf(nbplotHpdf)
    nbplotHpdf2D   = MatRplotHpdf2D.shape[0]     # MatRplotHpdf2D(nbplotHpdf2D,2)

    #--- initializing the random generator at the value of the end of the PCA step 
    np.random.set_state(SAVERANDstartDirectPartition)
 
    #----------------------------------------------------------------------------------------------------------------------------------
    #                                    Check data, parameters, and initialization
    #---------------------------------------------------------------------------------------------------------------------------------- 

    if nu > n_d or nu < 1 or n_d < 1:
        raise ValueError('STOP5 in sub_solverDirectPartition: nu > n_d or nu < 1 or n_d < 1')
    nutemp, ndtemp = MatReta_d.shape  # MatReta_d(nu,n_d) 
    if nutemp != nu or ndtemp != n_d:
        raise ValueError('STOP6 in sub_solverDirectPartition: the dimensions of MatReta_d are not consistent with nu and n_d')
    if ind_generator != 0 and ind_generator != 1:
        raise ValueError('STOP7 in sub_solverDirectPartition: ind_generator must be equal to 0 or equal to 1')
    if icorrectif != 0 and icorrectif != 1:
        raise ValueError('STOP8 in sub_solverDirectPartition: icorrectif must be equal to 0 or equal to 1')

    #--- Checking parameters controlling the time integration scheme of the ISDE
    if f0_ref <= 0:
        raise ValueError('STOP9 in sub_solverDirectPartition: f0_ref must be strictly positive')
    if ind_f0 != 0 and ind_f0 != 1:
        raise ValueError('STOP10 in sub_solverDirectPartition: ind_f0 must be equal to 0 or equal to 1')
    if coeffDeltar < 1:
        raise ValueError('STOP11 in sub_solverDirectPartition: coeffDeltar must be greater than or equal to 1')
    if M0transient < 1:
        raise ValueError('STOP12 in sub_solverDirectPartition: M0transient must be greater than or equal to 1')

    #--- Parameters controlling the computation of epsilonDIFFj for each group j
    if ind_generator == 0:
        comp_ref              = 0
        epsilonDIFFmin        = 0
        step_epsilonDIFF      = 0
        iterlimit_epsilonDIFF = 0

    if ind_generator == 1:
        if 0.1 > comp_ref or comp_ref > 0.5:  # comp_ref given by the user in [ 0.1 , 0.5 [
            raise ValueError('STOP13 in sub_solverDirectPartition: for ind_basis_type = 2, comp_ref must be given by the user between 0.1 and 0.5')
        if epsilonDIFFmin <= 0:
            raise ValueError('STOP14 in sub_solverDirectPartition: for ind_basis_type = 2, epsilonDIFFmin must be given by the user as a strictly positive real number')
        if step_epsilonDIFF <= 0:
            raise ValueError('STOP15 in sub_solverDirectPartition: for ind_basis_type = 2, step_epsilonDIFF must be given by the user as a strictly positive real number')
        if iterlimit_epsilonDIFF < 1:
            raise ValueError('STOP16 in sub_solverDirectPartition: for ind_basis_type = 2, iterlimit_epsilonDIFF must be given by the user as an integer larger than or equal to 1')

    #--- Checking data controlling constraints
    if ind_constraints != 0 and ind_constraints != 1 and ind_constraints != 2 and ind_constraints != 3:
        raise ValueError('STOP17 in sub_solverDirectPartition: ind_constraints must be equal to 0, 1, 2, or 3')

    if ind_constraints == 0:
        iter_limit   = 0
        iter_relax2  = 0
        alpha_relax1 = 0
        alpha_relax2 = 0

    if ind_constraints >= 1:
        if ind_coupling != 0 and ind_coupling != 1:
            raise ValueError('STOP18 in sub_solverDirectPartition: ind_coupling must be equal to 0 or equal to 1')
        if iter_limit < 1:
            raise ValueError('STOP19 in sub_solverDirectPartition: iter_limit must be greater than or equal to 1')
        if epsc < 0 or epsc >= 1:
            raise ValueError('STOP20 in sub_solverDirectPartition: epsc < 0 or epsc >= 1 ')
        if minVarH <= 0 or minVarH >= 1:
            raise ValueError('STOP21 in sub_solverDirectPartition: minVarH <= 0 or minVarH >= 1 ')
        if maxVarH <= minVarH or maxVarH <= 1:
            raise ValueError('STOP22 in sub_solverDirectPartition: maxVarH <= minVarH or maxVarH <= 1 ')
        if alpha_relax1 < 0 or alpha_relax1 > 1:
            raise ValueError('STOP23 in sub_solverDirectPartition: value of alpha_relax1 out the range [0,1]')
        if alpha_relax2 < 0 or alpha_relax2 > 1:
            raise ValueError('STOP24 in sub_solverDirectPartition: value of alpha_relax2 out the range [0,1]')
        if iter_relax2 >= 2 and iter_relax2 <= iter_limit:
            if alpha_relax1 > alpha_relax2:
                raise ValueError('STOP25 in sub_solverDirectPartition: alpha_relax1 must be less than or equal to alpha_relax2')
        if iter_relax2 > iter_limit:
            raise ValueError('STOP26 in sub_solverDirectPartition: iter_relax2 must be less than or equal to iter_limit')

    #--- Checking data related to the partition    
    if Igroup.size >=1:           # checking if not empty, Igroup(ngroup)
        if Igroup.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP27 in sub_solverDirectPartition: Igroup must be a 1D array')
    if ngroup <= 0:
        raise ValueError('STOP28 in sub_solverDirectPartition: ngroup must be greater than or equal to 1')
    mmax = MatIgroup.shape[1]  # MatIgroup(ngroup,mmax)
    if mmax <= 0 or mmax > nu:
        raise ValueError('STOP29 in sub_solverDirectPartition: mmax must be in the range of integers [1,nu]')
    n1temp = MatIgroup.shape[0]  # MatIgroup(ngroup,mmax)
    if n1temp != ngroup:
        raise ValueError('STOP30 in sub_solverDirectPartition: the first dimension of MatIgroup must be equal to ngroup')
    nutemp = 0
    for j in range(ngroup):
        mj = Igroup[j]
        nutemp = nutemp + mj
    if nutemp != nu:
        raise ValueError('STOP31 in sub_solverDirectPartition: data in Igroup is not consistent with dimension nu')

    #--- Parameters controlling display, print, plot, and parallel computation
    if ind_display_screen != 0 and ind_display_screen != 1:       
        raise ValueError('STOP32 in sub_solverDirectPartition: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:       
        raise ValueError('STOP33 in sub_solverDirectPartition: ind_print must be equal to 0 or equal to 1')
    if ind_plot != 0 and ind_plot != 1:       
        raise ValueError('STOP34 in sub_solverDirectPartition: ind_plot must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:       
        raise ValueError('STOP35 in sub_solverDirectPartition: ind_parallel must be equal to 0 or equal to 1')
    
    if nbplotHsamples >= 1:  # MatRplotHsamples(nbplotHsamples)
        if np.any(MatRplotHsamples < 1) or np.any(MatRplotHsamples > nu):  # at least one integer is not within the valid range
            raise ValueError('STOP36 in sub_solverDirectPartition: at least one integer in MatRplotHsamples is not in range [1,nu]') 
    if nbplotHClouds >= 1:  # MatRplotHClouds(nbplotHClouds,3)
        if MatRplotHClouds.shape[1] != 3:
            raise ValueError('STOP37 in sub_solverDirectPartition: the second dimension of MatRplotHClouds must be equal to 3') 
        if np.any(MatRplotHClouds < 1) or np.any(MatRplotHClouds > nu):  # At least one integer is not within the valid range
                raise ValueError('STOP38 in sub_solverDirectPartition: at least one integer in MatRplotHClouds is not in range [1,nu]')         
    if nbplotHpdf >= 1:  # MatRplotHpdf(nbplotHpdf)
        if np.any(MatRplotHpdf < 1) or np.any(MatRplotHpdf > nu):  # at least one integer  is not within the valid range
            raise ValueError('STOP39 in sub_solverDirectPartition: at least one integer in MatRplotHpdf is not in [1,nu]')            
    if nbplotHpdf2D >= 1:  # MatRplotHpdf2D(nbplotHpdf2D,2)
        if MatRplotHpdf2D.shape[1] != 2:
            raise ValueError('STOP40 in sub_solverDirectPartition: the second dimension of MatRplotHpdf2D must be equal to 2') 
        if np.any(MatRplotHpdf2D < 1) or np.any(MatRplotHpdf2D > nu):  # at least one integer is not within the valid range
            raise ValueError('STOP41 in sub_solverDirectPartition: at least one integer in MatRplotHpdf2D is not in [1,nu]') 
        
    #--- Computing and loading Ralpha
    Ralpha_relax = sub_solverDirectPartition_parameters_Ralpha(ind_constraints, iter_limit, alpha_relax1, iter_relax2, alpha_relax2)

    #--- Print the data inputs that are used for all the  groups  
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
            fidlisting.write(f' coeffDeltar   = {coeffDeltar:4d}   \n')
            fidlisting.write(f' M0transient   = {M0transient:7d}   \n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(f' epsilonDIFFmin        = {epsilonDIFFmin:14.7e} \n')
            fidlisting.write(f' step_epsilonDIFF      = {step_epsilonDIFF:14.7e} \n')
            fidlisting.write(f' iterlimit_epsilonDIFF = {iterlimit_epsilonDIFF:7d}    \n')
            fidlisting.write(f' comp_ref              = {comp_ref:8.4f}  \n')
            fidlisting.write('\n ')
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
            fidlisting.write(f' ind_display_screen = {ind_display_screen:1d} \n')
            fidlisting.write(f' ind_print          = {ind_print:1d} \n')
            fidlisting.write(f' ind_plot           = {ind_plot:1d} \n')
            fidlisting.write(f' ind_parallel       = {ind_parallel:1d} \n')
            fidlisting.write('\n ')
            fidlisting.write(f' ind_Kullback       = {ind_Kullback:1d} \n')
            fidlisting.write(f' ind_Entropy        = {ind_Entropy:1d} \n')
            fidlisting.write(f' ind_MutualInfo     = {ind_MutualInfo:1d}\n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    #-----------------------------------------------------------------------------------------------------------------------------------------                           
    #                                Generating NnbMC realizations on the independent groups
    #-----------------------------------------------------------------------------------------------------------------------------------------  

    #--- optimal partition in ngroup independent random vectors Y^1,...,Y^ngroup of random vector H of dimension nu
    #    ngroup = number of groups Y^1,...,Y^ngroup
    #    Igroup = vector (ngroup) such that Igroup(j): number mj of the components of  Y^j = (H_jr1,... ,H_jrmj)
    #    MatIgroup = matrix(ngroup,mmax) such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_jrj

    #--- Construction of cellMatReta_d{j} = MatReta_dj(mj,n_d)
    cellMatReta_d = [None] * ngroup                                # cellMatReta_d[j+1] =  MatReta_dj(mj,n_d),  cellMatReta_d{ngroup}
    for j in range(ngroup):
        mj               = Igroup[j]                               # length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
        MatReta_dj       = MatReta_d[MatIgroup[j,:mj] - 1, :]      # MatReta_dj(mj,n_d): realizations of Y^j of length mj = Igroup(j)
        cellMatReta_d[j] = MatReta_dj
    del mj, MatReta_dj

    #--- Generation of random Wienner germs for the Ito equation for each group
    cellGauss               = [None] * ngroup
    cellWiennerM0transientG = [None] * ngroup

    for j in range(ngroup):
        mj                         = Igroup[j]                                  # length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
        cellGauss[j]               = np.random.randn(mj,n_d,nbMC)               # ArrayGaussj(mj,n_d,nbMC)
        cellWiennerM0transientG[j] = np.random.randn(mj,n_d,M0transient,nbMC)   # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)

    #--- Display screen
    if ind_display_screen == 1:
        print('--- Beginning the construction of the parameters for each group')

    #--- Initializing parameters values for the groups   
    Rsg           = np.zeros(ngroup)
    Rshg          = np.zeros(ngroup)
    Rshssg        = np.zeros(ngroup)
    Rf0           = np.zeros(ngroup)
    RDeltarg      = np.zeros(ngroup)
    RM0estimg     = np.zeros(ngroup,dtype=int)
    RepsilonDIFFg = np.zeros(ngroup)
    RmDPg         = np.zeros(ngroup,dtype=int)
    RnbmDMAPg     = np.zeros(ngroup,dtype=int)
    cellMatRg     = [None] * ngroup  # cellMatRg[j+1] =  MatRgj 
    cellMatRa     = [None] * ngroup  # cellMatRa[j+1] =  MatRaj 
    numfig        = 0

    #--- Loop on the groups for computing MatReta_argj(mj,NnbMC)
    for j in range(ngroup):            # DO NOT PARALLELIZE THIS LOOP
        mj = Igroup[j]                 # length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
        MatReta_dj = cellMatReta_d[j]  # MatReta_dj(mj,n_d): realizations of Y^j of length mj = Igroup(j)

        #--- computing the parameters for group j  
        (sj,shj,shssj,f0j,Deltarj,M0estimj) = sub_solverDirectPartition_parameters_groupj(mj,n_d,icorrectif,f0_ref,ind_f0,coeffDeltar)

        #--- Computing the DMAPS basis for group j using the isotropic kernel
        epsilonDIFFj,mDPj,nbmDMAPj,MatRgj,MatRaj = sub_solverDirectPartition_isotropic_kernel_groupj(ind_generator,j+1,mj,n_d,MatReta_dj,
                                                   epsilonDIFFmin,step_epsilonDIFF,iterlimit_epsilonDIFF,comp_ref,
                                                   ind_display_screen,ind_plot,numfig)

        #--- Loading the calculated parameters for group j  
        Rsg[j]           = sj
        Rshg[j]          = shj
        Rshssg[j]        = shssj
        Rf0[j]           = f0j
        RDeltarg[j]      = Deltarj
        RM0estimg[j]     = int(M0estimj)
        RepsilonDIFFg[j] = epsilonDIFFj
        RmDPg[j]         = int(mDPj)
        RnbmDMAPg[j]     = int(nbmDMAPj)
        cellMatRg[j]     = MatRgj         # MatRgj(n_d,nbmDMAPj)
        cellMatRa[j]     = MatRaj         # MatRaj(n_d,nbmDMAPj)

        #--- Print the data inputs that are used for all the  groups  
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(f' --- Parameters for group {j+1:7d} \n ')
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(f' sj           = {Rsg[j]:14.7e} \n')
                fidlisting.write(f' shj          = {Rshg[j]:14.7e} \n')
                fidlisting.write(f' shssj        = {Rshssg[j]:14.7e} \n')
                fidlisting.write('                                 \n ')
                fidlisting.write(f' f0j          = {Rf0[j]:14.7e} \n')
                fidlisting.write(f' Deltarj      = {RDeltarg[j]:14.7e} \n')
                fidlisting.write(f' M0estimj     = {RM0estimg[j]:7d}    \n')
                fidlisting.write('                                 \n ')
                fidlisting.write('                                 \n ')
                fidlisting.write(f' mj           = {mj:7d}    \n')
                fidlisting.write(f' mDPj         = {RmDPg[j]:7d}    \n')
                fidlisting.write(f' nbmDMAPj     = {RnbmDMAPg[j]:7d}    \n')
                fidlisting.write(f' epsilonDIFFj = {RepsilonDIFFg[j]:14.7e} \n')
                fidlisting.write('                                 \n ')
                fidlisting.write('                                 \n ')

    #--- Display screen
    if ind_display_screen == 1:
        print('--- End of the construction of the parameters for each group')
        print(' ')
        print('--- Beginning the learning for each group')

    #--- Initialization cellMatReta_ar
    cellMatReta_ar = [None] * ngroup  # cellMatReta_ar[j+1] = MatReta_arj, with  MatReta_arj(mj,n_ar)

    #--- Loop on the groups 
    for j in range(ngroup):             # DO NOT PARALLELIZE THIS LOOP
        mj         = Igroup[j]
        MatReta_dj = cellMatReta_d[j]
        MatRgj     = cellMatRg[j]       # MatRgj(n_d,nbmDMAPj)
        MatRaj     = cellMatRa[j]       # MatRaj(n_d,nbmDMAPj)

        nbmDMAPj = RnbmDMAPg[j]
        shssj    = Rshssg[j]
        shj      = Rshg[j]
        Deltarj  = RDeltarg[j]
        f0j      = Rf0[j]

        ArrayGaussj              = cellGauss[j]                # ArrayGaussj(mj,n_d,nbMC)
        ArrayWiennerM0transientj = cellWiennerM0transientG[j]  # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)

        #--- Display screen
        if ind_display_screen == 1:
            print(' ')
            print(f'--- Learning group number {j+1} with mj = {mj}')
            print(' ')

        #--- Print
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(f' --- Learning group number j = {j+1:5d} with mj = {mj:5d} \n')
                fidlisting.write('                                 \n ')

        #--- No constraints on H
        if ind_constraints == 0:  # MatReta_arj(mj,n_ar)
            MatReta_arj = sub_solverDirectPartition_constraint0(mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj,MatReta_dj, 
                                                                MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_parallel,ind_print)

        #--- constraints are applied on H
        if ind_constraints >= 1:  # MatReta_arj(mj,n_ar)
            if mj == 1:
                MatReta_arj = sub_solverDirectPartition_constraint0mj1(mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj,MatReta_dj, 
                                                                       MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_parallel,ind_print)
            if mj >= 2:
                MatReta_arj = sub_solverDirectPartition_constraint123(j+1,mj,n_d,nbMC,n_ar,nbmDMAPj,M0transient,Deltarj,f0j,shssj,shj,MatReta_dj,
                                                                      MatRgj,MatRaj,ArrayWiennerM0transientj,ArrayGaussj,ind_constraints,
                                                                      ind_coupling,epsc,iter_limit,Ralpha_relax,minVarH,maxVarH,ind_display_screen,
                                                                      ind_print,ind_parallel,numfig)
        cellMatReta_ar[j] = MatReta_arj

    del MatReta_arj, MatReta_dj, ArrayGaussj, ArrayWiennerM0transientj, cellGauss, cellWiennerM0transientG, cellMatReta_d

    #--- Concatenation of MatReta_arj(mj,n_ar) into MatReta_ar(nu,n_ar) 
    MatReta_ar = np.zeros((nu,n_ar))
    for j in range(ngroup):
        MatReta_arj = cellMatReta_ar[j]                # MatReta_arj(mj,n_ar)
        mj          = Igroup[j]                        # length mj of vector   Y^j = (H_jr1,... ,H_jrmj) of group j
        MatReta_ar[MatIgroup[j,:mj]-1,:] = MatReta_arj   # MatReta_arj(mj,n_ar),MatIgroup(ngroup,nu)

    del MatReta_arj, cellMatReta_ar

    #-----------------------------------------------------------------------------------------------------------------------------------                         
    #                    Estimation of the measure concentration of H_ar with respect to H_d, 
    #                    whose realizations are MatReta_ar(nu,n_ar) and MatReta_ar(nu,n_d)
    #-----------------------------------------------------------------------------------------------------------------------------------

    #--- Estimation of the measure concentration d2mopt_ar = E{|| [H_ar] - [eta_d] ||^2} / || [eta_d] ||^2
    d2mopt_ar = 0
    ArrayH_ar = MatReta_ar.reshape(nu,n_d,nbMC)                                           # ArrayH_ar(nu,n_d,nbMC),MatReta_ar(nu,n_ar)
    for ell in range(nbMC):
        d2mopt_ar = d2mopt_ar + np.linalg.norm(ArrayH_ar[:,:,ell] - MatReta_d, 'fro')**2  # MatReta_d(nu,n_d)

    del ArrayH_ar
    deno2     = np.linalg.norm(MatReta_d,'fro')**2
    d2mopt_ar = d2mopt_ar / (nbMC*deno2)

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

    entropy_Hd = 0
    entropy_Har = 0
    if ind_Entropy == 1:
        entropy_Hd  = sub_solverDirect_Entropy(MatReta_d, ind_parallel)
        entropy_Har = sub_solverDirect_Entropy(MatReta_ar, ind_parallel)

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
    if ind_MutualInfo == 1:  # Unnormalized mutual informations
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
    sub_solverDirectPartition_plot_Hd_Har(n_d,n_ar,MatReta_d,MatReta_ar,nbplotHsamples,nbplotHClouds,nbplotHpdf,nbplotHpdf2D,
                                          MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,MatRplotHpdf2D,numfig)
    
    SAVERANDendDirectPartition   = np.random.get_state()
    ElapsedSolverDirectPartition = time.time() - TimeStartSolverDirectPartition

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('-------   Elapsed time for Task7_SolverDirectPartition \n ')
            fidlisting.write('\n ')
            fidlisting.write(f'Elapsed Time   =  {ElapsedSolverDirectPartition:10.2f}\n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    if ind_display_screen == 1:
        print('--- end Task7_SolverDirectPartition')

    return (n_ar,MatReta_ar,SAVERANDendDirectPartition,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har)
