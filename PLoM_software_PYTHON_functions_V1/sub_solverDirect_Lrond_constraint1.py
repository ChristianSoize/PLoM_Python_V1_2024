import numpy as np

def sub_solverDirect_Lrond_constraint1(nu,n_d,MatReta_d,MatRa,MatRg,MatRZ,shss,sh,Rlambda_iter):

    #-------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_Lrond_constraint1
    #  Subject      : ind_constraints = 1 : constraints E{H_{ar,j}^2} = 1 for j = 1,...,nu  
    #
    #--- INPUTS
    #      nu                  : dimension of H_d and H_ar   
    #      n_d                 : number of realizations in the training dataset
    #      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)  
    #      MatRa(n_d,nbmDMAP)  : realted to MatRg(n_d,nbmDMAP) 
    #      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #      MatRZ(nu,nbmDMAP)   : projection of MatRH_ar on the ISDE-projection basis
    #      shss,sh             : parameters of the GKDE of pdf of H_d (training)
    #      Rlambda_iter(mhc)   = ( lambda_1,...,lambda_nu)
    #
    #--- OUTPUTS
    #      MatRLrond(nu,nbmDMAP)
    #
    #--- INTERNAL PARAMETERS
    #      s       = ((4/((nu+2)*n_d))^(1/(nu+4)))   # Silver bandwidth 
    #      s2      = s*s
    #      shss    = 1 if icorrectif = 0 and = 1/sqrt(s2+(n_d-1)/n_d) if icorrectif = 1
    #      sh      = s*shss     
    #      nbmDMAP : number of ISDE-projection basis 
    #      mhc = nu 
    
    sh2   = sh*sh
    sh2m1 = 1/sh2
    co1   = 1/(2*sh2)
    co2   = shss/(sh2*n_d)

    MatRU       = MatRZ @ MatRg.T           # MatRU(nu,n_d), MatRg(n_d,nbmDMAP), MatRZ(nu,nbmDMAP)
    MatRetashss = shss*MatReta_d            # MatReta_d(nu,n_d)
    MatRetaco2  = co2*MatReta_d             # MatRetaco2(nu,n_d)

    #---Vectorial sequence without  parallelization
    MatRL  = np.zeros((nu,n_d))             # MatRL(nu,n_d)
    MatRLc = np.zeros((nu,n_d))             # MatRLc(nu,n_d)
    for ell in range(n_d):
        RU       = MatRU[:,ell]                             # RU(nu),MatRU(nu,n_d)
        MatRexpo = MatRetashss - RU[:,np.newaxis]           # MatRexpo(nu,n_d),RU(nu)
        Rexpo    = co1*np.sum(MatRexpo**2,axis=0)           # Rexpo(n_d) 
        expo_min = np.min(Rexpo)                            
        RS       = np.exp(-(Rexpo-expo_min))                # RS(n_d),Rexpo(n_d)
        q        = np.sum(RS)/n_d     
        Rgraduqp = MatRetaco2 @ RS                          # Rgraduqp(nu),MatRetaco2(nu,n_d),RS(n_d)
        RL       = -sh2m1*RU + Rgraduqp/q                   # RL(nu),Rgraduqp(nu)
        MatRL[:,ell] = RL                                   # MatRL(nu,n_d)
        
        MatRLc[:,ell] = -2*Rlambda_iter*RU                  # MatRLc(nu,n_d), Rlambda_iter(nu), RU(nu)
    
    MatRLrond = (MatRL + MatRLc) @ MatRa    # MatRLrond(nu,nbmDMAP),MatRL(nu,n_d),MatRLc(nu,n_d),MatRa(n_d,nbmDMAP)
    return MatRLrond