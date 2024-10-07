import numpy as np

def sub_solverDirect_Lrond_constraint3(nu,n_d,MatReta_d,MatRa,MatRg,MatRZ,shss,sh,Rlambda_iter,MatRlambda_iter):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_Lrond_constraint3
    #  Subject      : ind_constraints = 3 :  E{H_ar] = 0 and E{H_ar H_ar'} = [I_nu] 
    #
    #--- INPUT
    #        nu                     : dimension of H_d and H_ar   
    #        n_d                    : number of realizations in the training dataset
    #        MatReta_d(nu,n_d)      : n_d realizations of H_d (training)  
    #        MatRa(n_d,nbmDMAP)     : related to MatRg(n_d,nbmDMAP) 
    #        MatRg(n_d,nbmDMAP)     : matrix of the ISDE-projection basis
    #        MatRZ(nu,nbmDMAP)      : projection of MatRH_ar on the ISDE-projection basis
    #        shss,sh                : parameters of the GKDE of pdf of H_d (training)
    #        Rlambda_iter(mhc)      : see the data structure of Rlambda(mhc,1) in the comments below
    #        MatRlambda_iter(nu,nu) : see the data structure of Rlambda(mhc,1) in the comments below
    #
    #--- OUTPUTS
    #        MatRLrond(nu,nbmDMAP)
    #
    #--- INTERNAL PARAMETERS
    #        s       = ((4/((nu+2)*n_d))^(1/(nu+4)))   # Silver bandwidth 
    #        s2      = s*s
    #        shss    = 1 if icorrectif = 0 and = 1/sqrt(s2+(n_d-1)/n_d) if icorrectif = 1
    #        sh      = s*shss     
    #        nbmDMAP : number of ISDE-projection basis 
    #        mhc     = 2*nu + nu*(nu-1)/2 (note that for nu =1, we have mhc = 2*nu)
    #
    #--- COMMENTS using Matlab instructions
    #        if nu = 1: Rlambda_iter(mhc,1)=(lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu})
    #        if nu > 1: Rlambda_iter(mhc,1)=(lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu} , {lambda_{(j,i) + 2 nu}, 1<=j<i<=nu})
    #                   MatRlambda_iter(nu,nu) = Symmetric with zero diagonal and upper part is Rlambda_iter row rise such that:
    #                        MatRlambda_iter = zeros(nu,nu);
    #                        ind = 0;
    #                        for j = 1:nu-1
    #                            for i = j+1:nu
    #                                ind = ind + 1;
    #                                MatRlambda_iter(j,i) = Rlambda_iter(2*nu+ind);
    #                                MatRlambda_iter(i,j) = MatRlambda_iter(j,i);
    #                            end
    #                        end

    sh2   = sh*sh
    sh2m1 = 1/sh2
    co1   = 1/(2*sh2)
    co2   = shss/(sh2*n_d)

    MatRU       = MatRZ @ MatRg.T             # MatRU(nu,n_d), MatRg(n_d,nbmDMAP), MatRZ(nu,nbmDMAP)
    MatRetashss = shss*MatReta_d              # MatReta_d(nu,n_d)
    MatRetaco2  = co2*MatReta_d               # MatRetaco2(nu,n_d)

    #---Vectorial sequence without  parallelization
    MatRL  = np.zeros((nu,n_d))               # MatRL(nu,n_d)
    MatRLc = np.zeros((nu,n_d))               # MatRLc(nu,n_d)
    
    for ell in range(n_d):
        RU       = MatRU[:,ell]                                      # RU(nu),MatRU(nu,n_d)
        MatRexpo = MatRetashss - RU[:,np.newaxis]                    # MatRexpo(nu,n_d),RU(nu)
        Rexpo    = co1*np.sum(MatRexpo**2,axis=0)                    # Rexpo(n_d) 
        expo_min = np.min(Rexpo)                            
        RS       = np.exp(-(Rexpo-expo_min))                         # RS(n_d),Rexpo(n_d)
        q        = np.sum(RS)/n_d     
        Rgraduqp = MatRetaco2 @ RS                                   # Rgraduqp(nu),MatRetaco2(nu,n_d),RS(n_d)
        RL       = -sh2m1*RU + Rgraduqp/q                            # RL(nu),Rgraduqp(nu)
        MatRL[:,ell] = RL                                            # MatRL(nu,n_d)
        
        Rgrad_ell = Rlambda_iter[:nu] + 2*Rlambda_iter[nu:nu+nu]*RU  # MatRLc(nu,n_d), Rlambda_iter(2*nu + nu*(nu-1)/2), RU(nu)
        if nu == 1:                                                  # for nu = 1, the third term does not exist
            MatRLc[:,ell] = - Rgrad_ell                              # MatRLc(nu,n_d)
        
        if nu >= 2:                                                  # for nu >= 2 , the third term exits
            MatRLc[:,ell] = - Rgrad_ell - MatRlambda_iter @ RU       # MatRlambda_iter(nu,nu), RU(nu)
        
    MatRLrond = (MatRL+MatRLc) @ MatRa                               # MatRLrond(nu,nbmDMAP),MatRL(nu,n_d),MatRLc(nu,n_d),MatRa(n_d,nbmDMAP)
    return MatRLrond
