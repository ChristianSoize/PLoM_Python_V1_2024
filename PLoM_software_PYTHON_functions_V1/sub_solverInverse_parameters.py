import numpy as np

def sub_solverInverse_parameters(nu, n_d, icorrectif, f0_ref, ind_f0, 
                                 coeffDeltar, iter_limit, alpha_relax1, iter_relax2, alpha_relax2):
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (Plom) 
    #  Function name: sub_solverInverse_parameters
    #  Subject      : calculating and loading the parameters used by solverInverse
    #
    #--- INPUT  
    #        nu              : dimension or H = (H_1,...,H_nu)
    #        n_d             : number of realizations of H in the training dataset
    #        icorrectif      = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
    #                        = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified  
    #        f0_ref          : reference value (recommended value f0_ref = 4)
    #        ind_f0          : indicator for generating f0 (recommended value ind_f0 = 0): 
    #                          if ind_f0 = 0, then f0 = f0_ref, and if ind_f0 = 1, then f0 = f0_ref/sh    
    #        coeffDeltar     : coefficient > 0 (usual value is 20) for calculating Deltar
    #        iter_limit      : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
    #                          A relaxation function  iter --> alpha_relax(iter)  controls the convergence of the iterative algorithm 
    #                          that is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
    #        alpha_relax1    : value of alpha_relax for iter = 1  (for instance 0.001)
    #        iter_relax2     : value of iter (for instance, 20) such that  alpha_relax2 = alpha_relax(iter_relax2) 
    #                          if iter_relax2 = 1, then alpha_relax (iter) = alpha_relax2 for all iter >=1   
    #        alpha_relax2    : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = apha_relax2
    #                          NOTE 1: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
    #                          NOTE 2: If iter_relax2 >= 2, then  
    #                                  for iter >= 1 and for iter < iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
    #                                  for iter >= iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax2
    #                          NOTE 3: for decreasing the error err(iter), increase the value of iter_relax2
    #                          NOTE 4: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2
    #
    #--- OUTPUT
    #          s       : usual Silver bandwidth for the GKDE estimate (with the n_d points of the training dataset) 
    #                    of the pdf p_H of H, having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu] 
    #          sh      : modified Silver bandwidth for wich the normalization conditions are satisfied for any value of nu >= 1 
    #          shss    : = 1 if icorrectif  = 0, and = sh/s if icorrectif = 1
    #          f_0     : damping parameter in the ISDE, which controls the speed to reach the stationary response of the ISDE
    #          Deltar  : Stormer-Verlet integration-step of the ISDE        
    #          M0estim : estimate of M0transient provided as a reference to the user
    #          Ralpha_relax(iter_limit): relaxation function for the iteration algorithm that computes the LaGrange multipliers
    
    #--- Generation of parameters for the GKDE estimate, with the training dataset containing n_d points, of the pdf p_H of H,
    #    with H = (H_1,...,H_nu) having to satisfy the normalization condition E{H} = 0_nu and E{H H'} = [I_nu] (for the training)
    #    icorrectif = 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied
    #               = 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified 
    
    s  = (4/((nu+2)*n_d))**(1/(nu+4))                             # usual Silver bandwidth  
    s2 = s*s   
    if icorrectif == 0:                     
        shss = 1
    elif icorrectif == 1:
        shss = 1/np.sqrt(s2 + (n_d-1)/n_d)
    
    sh = s*shss    

    #--- Generation of the damping parameter f_0 in the ISDE, which controls the speed to reach the stationary response
    #    f0_ref:   reference value (recommended value f0_ref = 4)
    #    ind_f0:   indicator for generating f0 (recommended value ind_f0 = 0): 
    #              if ind_f0 = 0, then f0=f0_ref, and if ind_f0 = 1, then f0=f0_ref/sh  

    if ind_f0 == 0:         
        f0 = f0_ref        
    elif ind_f0 == 1:
        f0 = f0_ref/sh  
    
    #--- Generation of the Stormer-Verlet integration-step Deltar of the reduced-order ISDE.
    #    The positive coefficient, coeffDeltar, allows for calculating Deltar 
    #    An usual value for coeff_Deltar is 20

    Deltar = 2*np.pi*sh/coeffDeltar  

    #--- Generation of user information related to the value of M0transient
    #    M0transient : the end-integration value, M0transient, at which the stationary response of the ISDE is reached
    #                  is given by the user. The corresponding final time at which the realization is extrated from 
    #                  solverDirect_Verlet is M0transient*Deltar
    #    M0estim     : estimate of M0transient provided as a reference to the user

    M0estim = 2*np.log(100)*coeffDeltar/(np.pi*f0*sh)    # estimated value of M0transient   

    #--- Generation of Ralpha_relax(iter_limit,1) 
    Ralpha_relax = np.zeros(iter_limit)
    if iter_relax2 == 1:
        Ralpha_relax[:iter_limit] = alpha_relax2
    if iter_relax2 >= 2:
        for iter in range(1, iter_limit + 1):
            if iter >= 1 and iter < iter_relax2:
                Ralpha_relax[iter - 1] = alpha_relax1 + (alpha_relax2 - alpha_relax1) * (iter - 1) / (iter_relax2 - 1)
            if iter >= iter_relax2:
                Ralpha_relax[iter - 1] = alpha_relax2
    
    return s, sh, shss, f0, Deltar, M0estim, Ralpha_relax
