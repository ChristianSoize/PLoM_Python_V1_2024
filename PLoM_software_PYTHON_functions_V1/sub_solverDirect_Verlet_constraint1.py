import numpy as np
from sub_solverDirect_Lrond_constraint1 import  sub_solverDirect_Lrond_constraint1

def sub_solverDirect_Verlet_constraint1(nu, n_d, M0transient, Deltar, f0, MatReta_d, MatRa, MatRg, shss, sh, 
                                        ArrayWiennerM0transient_ell, MatRGauss_ell, Rlambda_iter):

    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM)
    #  Function name: sub_solverDirect_Verlet_constraint1
    #  Subject      : ind_constraints = 1 : constraints E{H_{ar,j}^2} = 1 for j = 1,...,nu  
    #                 Solving the ISDE  with the initial condition MatRz(n,nbmDMAP)
    #                 Time integration scheme is the Verlet algorithm
    #
    #--- INPUTS
    #      nu                  : dimension of H_ar   
    #      n_d                 : number of realizations in the database (initial data)
    #      M0transient         : number of steps for reaching the stationary solution
    #      Deltar              : ISDE integration step by Verlet scheme
    #      f0                  : dissipation coefficient in the ISDE
    #      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
    #      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
    #      shss, sh            : parameters of the GKDE of pdf of H_d (training)
    #      ArrayWiennerM0transient_ell(nu,n_d,M0transient) : realizations of the matrix-valued normalized Wienner process
    #      MatRGauss_ell(nu,n_d)                           : realizations of the Gaussian matrix-valued random variable
    #      Rlambda_iter(mhc)   = (lambda_1,...,lambda_mhc) , mhc = nu
    #
    #--- OUTPUTS
    #      MatRZ_ar_ell(nu,nbmDMAP)
    #
    #--- INTERNAL PARAMETERS
    #      nbmDMAP : number of ISDE-projection basis 
    #      mhc     = nu 

    b = f0 * Deltar / 4
    a0 = Deltar / 2
    a1 = (1 - b) / (1 + b)
    a2 = Deltar / (1 + b)
    a3 = (np.sqrt(f0 * Deltar)) / (1 + b)
    
    #--- realizations ell = 1, ... , nbMC
    MatRZ0 = MatReta_d @ MatRa         # MatRZ0(nu,nbmDMAP) , MatRz(nu,m) : set of realizations following the invariant measure
    MatRV0 = MatRGauss_ell @ MatRa     # MatRV0(nu,nbmDMAP) : [independent normalized Gaussian rv]*MatRa, MatRGauss_ell(nu,n_d)
    MatRZ  = MatRZ0                    # MatRZ(nu,nbmDMAP) 
    MatRV  = MatRV0                    # MatRV(nu,nbmDMAP)  
    
    for k in range(M0transient):              
        MatRZ1s2     = MatRZ + a0 * MatRV                                # MatRLrond1s2(nu,nbmDMAP)
        MatRLrond1s2 = sub_solverDirect_Lrond_constraint1(nu, n_d, MatReta_d, MatRa, MatRg, MatRZ1s2, shss, sh, Rlambda_iter)
        MatRWrond    = ArrayWiennerM0transient_ell[:, :, k] @ MatRa      # MatRWrond(nu,nbmDMAP), MatRa(n_d,nbmDMAP)
        MatRVp1      = a1 * MatRV + a2 * MatRLrond1s2 + a3 * MatRWrond   # ArrayWiennerM0transient_ell(nu,n_d,M0transient)
        MatRZp1      = MatRZ1s2 + a0 * MatRVp1
        MatRZ        = MatRZp1                                           # MatRZp1(nu,nbmDMAP)
        MatRV        = MatRVp1
    
    MatRZ_ar_ell = MatRZ                                                 # MatRZ_ar_ell(nu,nbmDMAP)
    return MatRZ_ar_ell
