import numpy as np

def sub_solverInverse_Lrond_target1(nu, n_d, MatReta_d, MatRa, MatRg, MatRZ, shss, sh, Rlambda_iter, coNr, coNr2, MatReta_targ):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 08 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverInverse_Lrond_target1
    #  Subject      : ind_type_targ = 1 : {h^c_r(H^c)} = Rb_targ1(r,1) for r =1,...,N_r 
    #                 Solving the ISDE  with the initial condition MatRz(n,nbmDMAP)
    #                 Time integration scheme is the Verlet algorithm
    #                               
    #--- INPUT
    #      nu                   : dimension of H_d and H_ar   
    #      n_d                  : number of realizations in the training dataset
    #      MatReta_d(nu,n_d)    : n_d realizations of H_d (training)  
    #      MatRa(n_d,nbmDMAP)   : related to MatRg(n_d,nbmDMAP) 
    #      MatRg(n_d,nbmDMAP)   : matrix of the ISDE-projection basis
    #      MatRZ(nu,nbmDMAP)    : projection of MatRH_ar on the ISDE-projection basis
    #      shss, sh             : parameters of the GKDE of pdf of H_d (training)
    #      Rlambda_iter(mhc)    = (lambda_1,...,lambda_mhc),  mhc = N_r
    #      N_r                  : number of realizations of the targets 
    #      coNr                 : parameter used for evaluating  E{h^c_targ(H^c)}               
    #      coNr2                : parameter used for evaluating  E{h^c_targ(H^c)} 
    #      MatReta_targ(nu,N_r) : N_r realizations of the projection of XX_targ on the model
    #           
    #--- OUTPUT 
    #       MatRLrond(nu,nbmDMAP) 
    
    # s = ((4/((nu+2)*n_d))^(1/(nu+4)));   # Silver bandwidth 
    # s2 = s*s;
    # shss = 1 if icorrectif = 0 and = sqrt(s2+(n_d-1)/n_d) if icorrectif = 1
    # sh = s*shss; 
    # nbmDMAP : number of ISDE-projection basis 
    # mhc = N_r 
    
    sh2   = sh*sh
    sh2m1 = 1/sh2
    co1   = 1/(2*sh2)
    co2   = shss/(sh2*n_d)

    MatRU       = MatRZ @ MatRg.T   # MatRU(nu,n_d), MatRg(n_d,nbmDMAP), MatRZ(nu,nbmDMAP)
    MatRetashss = shss*MatReta_d    # MatReta_d(nu,n_d)
    MatRetaco2  = co2*MatReta_d     # MatRetaco2(nu,n_d)
    
    #---Vectorial sequence without parallelization
    MatRL       = np.zeros((nu,n_d))  # MatRL(nu,n_d)
    MatRL_c     = np.zeros((nu,n_d))  # MatRL_c(nu,n_d)
        
    for j in range(n_d):
        RU_j       = MatRU[:,j]                            # RU_j(nu), MatRU(nu,n_d)
        MatRexpo   = MatRetashss - RU_j[:,np.newaxis]      # MatRexpo(nu,n_d), RU_j(nu)
        Rexpo      = co1*np.sum(MatRexpo**2,axis=0)        # Rexpo(n_d)
        expo_min   = np.min(Rexpo)
        RS         = np.exp(-(Rexpo - expo_min))           # RS(n_d), Rexpo(n_d)
        q          = np.sum(RS)/n_d
        Rgraduqp   = MatRetaco2 @ RS                       # Rgraduqp(nu),MatRetaco2(nu,n_d),RS(n_d)
        RL         = -sh2m1*RU_j + Rgraduqp/q              # RL(nu),Rgraduqp(nu)
        MatRL[:,j] = RL                                    # MatRL(nu,n_d)
        
        MatRexpo_c    = MatReta_targ - RU_j[:,np.newaxis]                         # MatRexpo_c(nu,N_r), MatReta_targ(nu,N_r), RU_j(nu)
        Rexpo_c       = np.exp(-coNr*np.sum(MatRexpo_c**2,axis=0))                # Rexpo_c(N_r)
        MatRL_c[:,j]  = - np.sum(coNr2*MatRexpo_c*(Rexpo_c*Rlambda_iter),axis=1)  # MatRL_c(nu,n_d), MatRexpo_c(nu,N_r), Rexpo_c(N_r), Rlambda_iter(N_r)
    
    MatRLrond = (MatRL + MatRL_c) @ MatRa  # MatRLrond(nu,nbmDMAP), MatRL(nu,n_d), MatRL_c(nu,n_d), MatRa(n_d,nbmDMAP)
    return MatRLrond
