import numpy as np
from sub_solverDirect_Lrond_constraint3 import sub_solverDirect_Lrond_constraint3

def sub_solverDirect_Verlet_constraint3(nu, n_d, M0transient, Deltar, f0, MatReta_d, MatRa, MatRg, shss, sh,
                                        ArrayWiennerM0transient_ell, MatRGauss_ell, Rlambda_iter, MatRlambda_iter):
    # ------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM) 
    #  Function name: sub_solverDirect_Verlet_constraint3
    #  Subject      : ind_constraints = 3 :  E{H_ar} = 0 and E{H_ar H_ar'} = [I_nu]     
    #                 Solving the ISDE with the initial condition MatRz(n,nbmDMAP)
    #                 Time integration scheme is the Verlet algorithm
    #
    # --- INPUTS
    #        nu                  : dimension of H_d and H_ar   
    #        n_d                 : number of realizations in the training dataset
    #        M0transient         : number of steps for reaching the stationary solution
    #        Deltar              : ISDE integration step by Verlet scheme
    #        f0                  : dissipation coefficient in the ISDE
    #        MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
    #        MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #        MatRa(n_d,nbmDMAP)  : related to MatRg(n_d, nbmDMAP) 
    #        shss, sh            : parameters of the GKDE of pdf of H_d (training)
    #        ArrayWiennerM0transient_ell(nu, n_d, M0transient) : realizations of the matrix-valued normalized Wiener process
    #        MatRGauss_ell(nu,n_d)                             : realizations of the Gaussian matrix-valued random variable   
    #        Rlambda_iter(mhc)
    #        MatRlambda_iter(nu,nu)
    #
    # --- OUTPUTS
    #        MatRZ_ar_ell(nu, nbmDMAP)
    #
    # --- INTERNAL PARAMETERS
    #        s       = ((4/((nu+2)*n_d))^(1/(nu+4)))   # Silver bandwidth 
    #        s2      = s * s
    #        shss    = 1 if icorrectif = 0 and = 1/sqrt(s2 + (n_d - 1)/n_d) if icorrectif = 1
    #        sh      = s * shss     
    #        nbmDMAP : number of ISDE-projection basis 
    #        mhc     = 2 * nu + nu * (nu - 1) / 2 (note that for nu = 1, we have mhc = 2 * nu)
    #
    # --- COMMENTS using Matlab instructions
    #        if nu = 1: Rlambda_iter(mhc,1)=(lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu})
    #        if nu > 1: Rlambda_iter(mhc,1)=(lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu} , {lambda_{(j,i) + 2 nu}, 1<=j<i<=nu})
    #                   MatRlambda_iter(nu,nu) = Symmetric with zero diagonal and upper part is Rlambda_iter row rise such that:
    #                        MatRlambda_iter = zeros(nu,nu);
    #                       ind = 0;
    #                        for j = 1:nu-1
    #                            for i = j+1:nu
    #                                ind = ind + 1;
    #                                MatRlambda_iter(j,i) = Rlambda_iter(2*nu+ind);
    #                                MatRlambda_iter(i,j) = MatRlambda_iter(j,i);
    #                            end
    #                        end
    
    b = f0 * Deltar / 4
    a0 = Deltar / 2
    a1 = (1 - b) / (1 + b)
    a2 = Deltar / (1 + b)
    a3 = np.sqrt(f0 * Deltar) / (1 + b)
    
    # --- Realizations ell = 1, ..., nbMC
    MatRZ0 = MatReta_d @ MatRa       # MatRZ0(nu,nbmDMAP), MatRz(nu,nbmDMAP) : set of realizations following the invariant measure
    MatRV0 = MatRGauss_ell @ MatRa   # MatRV0(nu,nbmDMAP) : [independent normalized Gaussian rv] * MatRa, MatRGauss_ell(nu, n_d)
    MatRZ  = MatRZ0                  # MatRZ(nu,nbmDMAP)
    MatRV  = MatRV0                  # MatRV(nu,nbmDMAP)
    
    for k in range(M0transient):
        MatRZ1s2     = MatRZ + a0 * MatRV                                    # MatRLrond1s2(nu,nbmDMAP)
        MatRLrond1s2 = sub_solverDirect_Lrond_constraint3(nu,n_d,MatReta_d,MatRa,MatRg,MatRZ1s2,shss,sh,Rlambda_iter,MatRlambda_iter)
        MatRWrond    = ArrayWiennerM0transient_ell[:, :, k] @ MatRa          # MatRWrond(nu,nbmDMAP), MatRa(n_d,nbmDMAP)
        MatRVp1      = a1 * MatRV + a2 * MatRLrond1s2 + a3 * MatRWrond       # ArrayWiennerM0transient_ell(nu,n_d,M0transient)
        MatRZp1      = MatRZ1s2 + a0 * MatRVp1
        MatRZ        = MatRZp1                                               # MatRZp1(nu,m)
        MatRV        = MatRVp1
    
    MatRZ_ar_ell = MatRZ                                                     # MatRZ_ar_ell(nu,nbmDMAP)
    return MatRZ_ar_ell
