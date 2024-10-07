import numpy as np
import time

def sub_projection_target(n_x, n_d, MatRx_d, ind_exec_solver, ind_scaling,
                          ind_type_targ, Indx_targ_real, Indx_targ_pos, nx_targ, Indx_targ, N_r,
                          MatRxx_targ_real, MatRxx_targ_pos, Rmeanxx_targ,
                          MatRcovxx_targ, nu, RmuPCA, MatRVectPCA, ind_display_screen, ind_print, ind_parallel,
                          Rbeta_scale_real, Ralpham1_scale_real, Rbeta_scale_log, Ralpham1_scale_log):
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_projection_target
    #  Subject      : for ind_exec_solver = 2 (Inverse analysis imposing targets for the leaning), computing and loading information
    #                 that defined the constraints as a function of ind_type_targ that is 1, 2, 3, or 4 
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
    #    n_x                               : dimension of random vector XX_d (unscaled) and X_d (scaled)
    #    n_d                               : number of points in the training set for XX_d and X_d
    #    MatRx_d(n_x,n_d)                  : n_d realizations of X_d (scaled)
    #
    #    ind_exec_solver                   : = 1 Direct analysis : giving a training dataset, generating a learned dataset
    #                                      : = 2 Inverse analysis: giving a training dataset and a target dataset, generating a learned dataset
    #    ind_scaling                       : = 0 no scaling
    #                                      : = 1 scaling
    #    ind_type_targ                     : = 1, targets defined by giving N_r realizations
    #                                      : = 2, targets defined by giving target mean-values 
    #                                      : = 3, targets defined by giving target mean-values and target covariance matrix
    #    Indx_targ_real(nbreal_targ)       : contains the nbreal_targ component numbers of XX, which are real  
    #                                        with 0 <= nbreal_targ <= n_x and for which a "standard scaling" will be used
    #    Indx_targ_pos(nbpos_targ)         : contains the nbpos_targ component numbers of XX, which are strictly positive (specific scaling)
    #                                        with 0 <= nbpos_targ <= n_x  and for which the scaling is {log + "standard scaling"}   
    #    nx_targ                           : number of components of XX with targets such that nx_targ = nbreal_targ + nbpos_targ <= n_x
    #    Indx_targ(nx_targ)                : nx_targ component numbers of XX with targets 
    #
    #                                      --- ind_type_targ = 1: targets defined by giving N_r realizations
    #    N_r                               : number of realizations of the targets               
    #    MatRxx_targ_real(nbreal_targ,N_r) : N_r realizations (unscaled) of the nbreal_targ targets of XX that are real
    #    MatRxx_targ_pos(nbpos_targ,N_r)   : N_r realizations (unscaled) of the nbpos_targ targets of XX that are positive
    #
    #                                      --- ind_type_targ = 2 or 3: targets defined by giving the mean value of unscaled XX_targ 
    #    Rmeanxx_targ(nx_targ)             : nx_targ components of mean value E{XX_targ} of vector-valued random target XX_targ
    #
    #                                      --- ind_type_targ = 3: targets defined by giving the covariance matrix of unscaled XX_targ 
    #    MatRcovxx_targ(nx_targ,nx_targ)   : covariance matrix of XX_targ 
    # 
    #    nu                                : dimension of H
    #    RmuPCA(nu)                        : vector of eigenvalues in descending order
    #    MatRVectPCA(n_x,nu)               : matrix of the eigenvectors associated to the eigenvalues loaded in RmuPCA
    #   
    #    ind_display_screen                : = 0 no display,                = 1 display
    #    ind_print                         : = 0 no print,                  = 1 print
    #    ind_parallel                      : = 0 no parallel computation,   = 1 parallele computation
    #
    #                                      --- ind_exec_solver = 2 and with ind_scaling = 1
    #    Rbeta_scale_real(nbreal)          : loaded if nbreal >= 1 or = [] if nbreal  = 0  
    #    Ralpham1_scale_real(nbreal)       : loaded if nbreal >= 1 or = [] if nbreal  = 0   
    #    Rbeta_scale_log(nbpos)            : loaded if nbpos >= 1  or = [] if nbpos   = 0   
    #    Ralpham1_scale_log(nbpos)         : loaded if nbpos >= 1  or = [] if nbpos   = 0  
    #
    #--- OUPUTS
    #                                      --- ind_type_targ = 1: targets defined by giving N_r realizations of XX_targ 
    #    Rb_targ1(N_r)                    : E{h_targ1(H^c)} = b_targ1  with h_targ1 = (h_{targ1,1}, ... , h_{targ1,N_r})
    #    coNr                             : parameter used for evaluating  E{h^c_targ(H^c)}               
    #    coNr2                            : parameter used for evaluating  E{h^c_targ(H^c)} 
    #    MatReta_targ(nu,N_r)             : N_r realizations of the projection of XX_targ on the model
    #
    #                                      --- ind_type_targ = 2 or 3: targets defined by giving mean value of XX_targ
    #    Rb_targ2(nu)                                               yielding the constraint E{H^c} = b_targ2 
    #
    #                                      --- ind_type_targ = 3: targets defined by giving target covariance matrix of XX_targ
    #    Rb_targ3(nu)                                          yielding the constraint diag(E{H_c H_c'}) = b_targ3   
    #
    #--- COMMENTS
    #             (1) for ind_type_targ = 2 or 3, we have nbpos_targ = 0 and Indx_targ_pos(nbpos_targ,1) = []
    #             (2) Note that the constraints on H^c is not E{H_c H_c'}) = [b_targ3]  but  diag(E{H_c H_c'}) = b_targ3 

    if ind_display_screen != 0 and ind_display_screen != 1:
        raise ValueError('STOP1 in sub_projection_target: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:
        raise ValueError('STOP2 in sub_projection_target: ind_print must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:
        raise ValueError('STOP3 in sub_projection_target: ind_parallel must be equal to 0 or equal to 1')

    if ind_display_screen == 1:
        print('--- beginning Task8_ProjectionTarget')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ------ Task8_ProjectionTarget \n ')
            fidlisting.write('      \n ')

    TimeStart = time.time()

    #--- Loading and checking input data and parameters
    if n_d < 1 or n_x < n_d:
        raise ValueError('STOP4 in sub_projection_target: n_d < 1 or n_x < n_d')
    
    n1temp, n2temp = MatRx_d.shape  # MatRx_d(n_x, n_d) (scaled)
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP5 in sub_projection_target: the dimensions of MatRx_d(n_x,n_d) are not coherent')
    if ind_exec_solver != 2:
        raise ValueError('STOP6 in sub_projection_target: ind_exec_solver should be equal to 2')
    if ind_scaling != 0 and ind_scaling != 1:
        raise ValueError('STOP7 in sub_projection_target: ind_scaling must be equal to 0 or 1')
    if ind_type_targ != 1 and ind_type_targ != 2 and ind_type_targ != 3:
        raise ValueError('STOP8 in sub_projection_target: ind_type_targ must be equal to 1, 2 or 3')

    # Checking parameters and data related to Indx_targ_real(nbreal_targ) and Indx_targ_pos(nbreal_pos)
    if Indx_targ_real.size >= 1:             # checking if not empty
            if Indx_targ_real.ndim != 1:     # ckecking if it is a 1D array 
                raise ValueError('STOP9 in sub_projection_target: Indx_targ_real must be a 1D array')  
    nbreal_targ = len(Indx_targ_real)
    
    if Indx_targ_pos.size >= 1:             # checking if not empty
            if Indx_targ_pos.ndim != 1:     # ckecking if it is a 1D array 
                raise ValueError('STOP10 in sub_projection_target: Indx_targ_pos must be a 1D array') 
    nbpos_targ = len(Indx_targ_pos)

    if nbpos_targ >= 1:
        if ind_type_targ == 2 or ind_type_targ == 3:
            raise ValueError('STOP11 in sub_projection_target: for an inverse problem, when ind_type_targ = 2 or 3, one must have nbpos_targ = 0')
    
    # Checking parameters that control target parameters and data
    if nx_targ - (nbreal_targ + nbpos_targ) != 0:
        raise ValueError('STOP12 in sub_projection_target: nx_targ is not equal to (nbreal_targ + nbpos_targ)')
    if nx_targ < 1 or nx_targ > n_x:
        raise ValueError('STOP13 in sub_projection_target: nx_targ is less than 1 or greater than n_x')
    
    if Indx_targ.size >= 1:             # checking if not empty
            if Indx_targ.ndim != 1:     # ckecking if it is a 1D array 
                raise ValueError('STOP14 in sub_projection_target: Indx_targ must be a 1D array') 
     
    # For ind_type_targ = 1, checking parameters and data related to MatRxx_targ_real and MatRxx_targ_pos
    if ind_type_targ == 1:
        if N_r <= 0:
            raise ValueError('STOP15 in sub_projection_target: for ind_type_targ = 1, one must have N_r greater than or equal to 1')
        
        if nbreal_targ >= 1:
            n1temp, n2temp = MatRxx_targ_real.shape  # MatRxx_targ_real(nbreal_targ,N_r)
            if n1temp != nbreal_targ or n2temp != N_r:
                raise ValueError('STOP16 in sub_projection_target: for ind_type_targ = 1, the dimensions of MatRxx_targ_real(nbreal_targ,N_r) are not correct')

        if nbpos_targ >= 1:
            n1temp, n2temp = MatRxx_targ_pos.shape  # MatRxx_targ_pos(nbpos_targ,N_r)
            if n1temp != nbpos_targ or n2temp != N_r:
                raise ValueError('STOP17 in sub_projection_target: for ind_type_targ = 1, the dimensions of MatRxx_targ_pos(nbpos_targ,N_r) are not correct')

    # For ind_type_targ = 2 or 3, checking parameters and data related to Rmeanxx_targ(nx_targ,1)
    if ind_type_targ == 2 or ind_type_targ == 3:
        if Rmeanxx_targ.size >= 1:              # checking if not empty
            if Rmeanxx_targ.ndim != 1:          # ckecking if it is a 1D array 
                raise ValueError('STOP18 in sub_projection_target: Rmeanxx_targ must be a 1D array')  
        n1temp = len(Rmeanxx_targ)              # Rmeanxx_targ(nx_targ)
        if n1temp != nx_targ: 
            raise ValueError('STOP19 in sub_projection_target: for ind_type_targ >= 2, the dimension of Rmeanxx_targ(nx_targ) is not correct')

    # For ind_type_targ = 3, checking that matrix MatRcovxx_targ is symmetric and positive definite
    if ind_type_targ == 3:
        n1temp, n2temp = MatRcovxx_targ.shape  # MatRcovxx_targ(nx_targ,nx_targ)
        if n1temp != nx_targ or n2temp != nx_targ:
            raise ValueError('STOP20 in sub_projection_target: for ind_type_targ = 3, the dimensions of MatRcovxx_targ(nx_targ,nx_targ) are not correct')
        if not np.allclose(MatRcovxx_targ, MatRcovxx_targ.T):
            raise ValueError('STOP21 in sub_projection_target: matrix MatRcovxx_targ is not symmetric')
        
        (Reigen, _ ) = np.linalg.eigh(MatRcovxx_targ)
        if not np.all(Reigen > 0):
            raise ValueError('STOP22 in sub_projection_target: matrix MatRcovxx_targ is not positive definite')

    # Checking parameters and data related to PCA
    if nu < 1 or nu > n_d:
        raise ValueError('STOP23 in sub_projection_target: nu < 1 or nu > n_d')
    
    if RmuPCA.size >= 1:                   # checking if not empty
            if RmuPCA.ndim != 1:           # ckecking if it is a 1D array 
                raise ValueError('STOP24 in sub_projection_target: RmuPCA must be a 1D array')  
    n1temp = len(RmuPCA)                   # RmuPCA(nu)
    if n1temp != nu:
        raise ValueError('STOP25 in sub_projection_target:  RmuPCA(nu) must be a 1D array')
    
    if nu > n_x or n_x < 1:
        raise ValueError('STOP26 in sub_projection_target: nu > n_x or n_x < 1')
    
    n1temp, n2temp = MatRVectPCA.shape  # MatRVectPCA(n_x,nu)
    if n1temp != n_x or n2temp != nu:
        raise ValueError('STOP27 in sub_projection_target: dimensions of MatRVectPCA(n_x,nu) are not correct')

    # If ind_scaling = 1, 
    # (1) checking the data coherence of Rbeta_scale_real(nbreal), Ralpham1_scale_real(nbreal), Rbeta_scale_log(nbpos), 
    #     and Ralpham1_scale_log(nbpos)      
    # (2) loading Rbeta_scale(n_x) and Ralpham1_scale(n_x)
    # (3) Extracting Rbeta_targ(nx_targ) and Ralpham1_targ(nx_targ) from Rbeta_scale(n_x) and Ralpham1_scale(n_x)
    if ind_scaling == 1:
        if ind_type_targ == 1:
            if Rbeta_scale_real.size >= 1:                   # checking if not empty
                if Rbeta_scale_real.ndim != 1:               # ckecking if it is a 1D array 
                    raise ValueError('STOP28 in sub_projection_target: Rbeta_scale_real must be a 1D array')  
            nbreal = len(Rbeta_scale_real)                   # Rbeta_scale_real(nbreal)
            
            if Rbeta_scale_log.size >= 1:                    # checking if not empty
                if Rbeta_scale_log.ndim != 1:                # ckecking if it is a 1D array 
                    raise ValueError('STOP29 in sub_projection_target: Rbeta_scale_log must be a 1D array')  
            nbpos = len(Rbeta_scale_log)                     # Rbeta_scale_log(nbpos)

            if Ralpham1_scale_real.size >= 1:                   # checking if not empty
                if Ralpham1_scale_real.ndim != 1:               # ckecking if it is a 1D array 
                    raise ValueError('STOP30 in sub_projection_target: Ralpham1_scale_real must be a 1D array')  
            n1temp = len(Ralpham1_scale_real)                   # Ralpham1_scale_real(nbreal)
            if n1temp != nbreal:
                raise ValueError('STOP31 in sub_projection_target: dimension of Ralpham1_scale_real is not correct')  

            if Ralpham1_scale_log.size >= 1:                   # checking if not empty
                if Ralpham1_scale_log.ndim != 1:               # ckecking if it is a 1D array 
                    raise ValueError('STOP32 in sub_projection_target: Ralpham1_scale_real must be a 1D array')  
            n1temp = len(Ralpham1_scale_log)                   # Ralpham1_scale_real(nbreal)
            if n1temp != nbpos:
                raise ValueError('STOP33 in sub_projection_target: dimension of Ralpham1_scale_log is not correct')  

            # Loading Rbeta_scale(n_x) and Ralpham1_scale(n_x)
            Rbeta_scale    = np.concatenate((Rbeta_scale_real, Rbeta_scale_log))        # Rbeta_scale_real(nbreal), Rbeta_scale_log(nbpos)
            Ralpham1_scale = np.concatenate((Ralpham1_scale_real, Ralpham1_scale_log))  # Ralpham1_scale_real(nbreal), Ralpham1_scale_log(nbpos)

        if ind_type_targ == 2 or ind_type_targ == 3:
            if Rbeta_scale_real.size >= 1:                   # checking if not empty
                if Rbeta_scale_real.ndim != 1:               # ckecking if it is a 1D array 
                    raise ValueError('STOP34 in sub_projection_target: Rbeta_scale_real must be a 1D array') 
            nbreal = len(Rbeta_scale_real)                   # Rbeta_scale_real(nbreal)
            if nbreal != n_x:
                raise ValueError('STOP35 in sub_projection_target: for ind_type_targ = 2 or 3, nbreal must be equal to n_x')
            
            if Ralpham1_scale_real.size >= 1:                   # checking if not empty
                if Ralpham1_scale_real.ndim != 1:               # ckecking if it is a 1D array 
                    raise ValueError('STOP36 in sub_projection_target: Ralpham1_scale_real must be a 1D array')  
            n1temp = len(Ralpham1_scale_real)                   # Ralpham1_scale_real(nbreal)
            if n1temp != nbreal:
                raise ValueError('STOP37 in sub_projection_target: dimension of Ralpham1_scale_real is not correct') 

            # Loading Rbeta_scale(n_x,1) and Ralpham1_scale(n_x,1)
            Rbeta_scale    = Rbeta_scale_real                  # Rbeta_scale_real(nbreal)
            Ralpham1_scale = Ralpham1_scale_real               # Ralpham1_scale_real(nbreal)

        # Extraction Rbeta_targ(nx_targ) and Ralpham1_targ(nx_targ) from Rbeta_scale(n_x,1) and Ralpham1_scale(n_x)
        Rbeta_targ       = Rbeta_scale[Indx_targ-1]            # Rbeta_targ(nx_targ), Rbeta_scale(n_x), Indx_targ(nx_targ)
        Ralpham1_targ    = Ralpham1_scale[Indx_targ-1]         # Ralpham1_targ(nx_targ), Ralpham1_scale(n_x), Indx_targ(nx_targ)
        MatRalpham1_targ = np.diag(Ralpham1_targ)              # MatRalpham1_targ(nx_targ,nx_targ)

    #--- Computing Rmeanx_d_targ(nx_targ) and MatRVectPCA_targ(nx_targ,nu) from Rmeanx_d(nx) and MatRVectPCA(nx,nu)
    Rmeanx_d         = np.mean(MatRx_d,axis=1)                 # Rmeanx_d(n_x), MatRx_d(n_x,n_d) (scaled)
    Rmeanx_d_targ    = Rmeanx_d[Indx_targ-1]                   # Rmeanx_d_targ(nx_targ), Indx_targ(nx_targ)
    MatRVectPCA_targ = MatRVectPCA[Indx_targ-1,:]              # MatRVectPCA_targ(nx_targ,nu),MatRVectPCA(n_x,nu)
    Rcoef            = np.sqrt(RmuPCA)                         # RmuPCA(nu)
    MatRdiagRmu1s2   = np.diag(Rcoef)                          # MatRdiagRmu1s2(nu,nu)
    MatRV            = MatRVectPCA_targ @ MatRdiagRmu1s2       # MatRV(nx_targ,nu),MatRVectPCA_targ(nx_targ,nu),MatRdiagRmu1s2(nu,nu)
    MatRVt           = np.linalg.pinv(MatRV)                   # MatRVt(nu,nx_targ),MatRV(nx_targ,nu)

    #--- initialization 
    if ind_type_targ == 1:
        Rb_targ2 = np.array([])
        Rb_targ3 = np.array([])

    if ind_type_targ == 2:
        coNr         = 0
        coNr2        = 0
        Rb_targ1     = np.array([])
        MatReta_targ = np.array([])
        Rb_targ3     = np.array([])

    if ind_type_targ == 3:
        coNr         = 0
        coNr2        = 0
        MatReta_targ = np.array([])
        Rb_targ1     = np.array([])

    #--------------------------------------------------------------------------------------------------------------------------
    #   Case ind_type_targ = 1: targets defined by giving N_r realizations
    #                           computing coNr,coNr2,Rb_targ1(N_r,1)  
    #--------------------------------------------------------------------------------------------------------------------------

    if ind_type_targ == 1:

        MatRxx_targ_log = np.array([])
        if nbpos_targ >= 1:
            MatRxx_targ_log = np.log(MatRxx_targ_pos)  # MatRxx_targ_log(nbpos_targ,N_r), MatRxx_targ_pos(nbpos_targ,N_r)

        #--- Computing MatRx_targ(nx_targ,N_r)  (scaled) from  MatRxx_targ(nx_targ,N_r)  (unscaled)
        MatRxx_targ = np.vstack((MatRxx_targ_real, MatRxx_targ_log))  # MatRxx_targ(nx_targ,N_r)

        # No scaling 
        if ind_scaling == 0:
            MatRx_targ = MatRxx_targ                                                # MatRx_targ(nx_targ,N_r),MatRxx_targ(nx_targ,N_r)

        # Scaling
        if ind_scaling == 1:
            MatRx_targ = Ralpham1_targ[:,np.newaxis] * (MatRxx_targ - Rbeta_targ[:, np.newaxis])  # MatRx_targ(nx_targ,N_r),Ralpham1_targ(nx_targ,1)

        #--- Computing MatReta_targ(nu,N_r)      
        MatRx_targ_tilde = MatRx_targ - Rmeanx_d_targ[:, np.newaxis]     # MatRx_targ_tilde(nx_targ,N_r),MatRx_targ(nx_targ,N_r)
        MatReta_targ     = MatRVt @ MatRx_targ_tilde                     #  MatReta_targ(nu,N_r),MatRVt(nu,nx_targ),MatRx_targ_tilde(nx_targ,N_r)

        #--- Computing error_target as the projection error E{||X_targ - (meanx_targ + V*eta_targ)||^2} / E{||X_targ||^2}
        RtempNum = np.zeros(N_r)
        RtempDen = np.zeros(N_r)
        for r in range(N_r):
            RtempNum[r] = np.linalg.norm(MatRx_targ_tilde[:, r] - MatRV @ MatReta_targ[:, r]) ** 2  # MatRV(nx_targ,nu),MatReta_targ(nu,N_r)
            RtempDen[r] = np.linalg.norm(MatRx_targ[:, r]) ** 2
        error_target = np.sum(RtempNum) / np.sum(RtempDen)

        #---display screen
        if ind_display_screen == 1:
            print(f' Relative projection error of the target dataset onto the model = {error_target}')

        #--- print 
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write('--- Relative projection error of the target dataset onto the model \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f'    error_projection_target_dataset      = {error_target:14.7e} \n ')
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')

        #--- Computing  Rbc(N_r) of the experimental constraints with the PCA representation 
        sNr   = ((4 / ((nu + 2) * N_r)) ** (1 / (nu + 4)))  # Silver bandwidth 
        s2Nr  = sNr * sNr
        coNr  = 1 / (nu * s2Nr)
        coNr2 = 2 * coNr

                    # --- scalar sequence for computing Rb_targ1(N_r) (given for readability of the used algebraic formula)
                    # Rb_targ1 = np.zeros((N_r, 1))
                    # for r in range(N_r):
                    #     b_r = 0
                    #     for rp in range(N_r):
                    #         Reta_rp_r = MatReta_targ[:, rp] - MatReta_targ[:, r]
                    #         expo = np.exp(-coNr * np.sum(Reta_rp_r**2))
                    #         b_r = b_r + expo
                    #     Rb_targ1[r] = b_r/N_r

        #--- vectorized sequence for computing Rb_targ1(N_r)
        Rsumexpo = np.zeros(N_r)
        for rp in range(N_r):
            Reta_targ_rp = MatReta_targ[:, rp]                               # Reta_targ_rp(nu),MatReta_targ(nu,N_r)
            MatRtarg_rp  = Reta_targ_rp[:, np.newaxis] - MatReta_targ        # MatRtarg_rp(nu,N_r),Reta_targ_rp(nu),MatReta_targ(nu,N_r)
            Rtarg_rp     = np.exp(-coNr * np.sum(MatRtarg_rp ** 2, axis=0))  # Rtarg_rp(N_r),MatRtarg_rp(nu,N_r)
            Rsumexpo     = Rsumexpo + Rtarg_rp                               # Rsumexpo(N_r)
        Rb_targ1 = Rsumexpo / N_r                                            # Rb_targ1(N_r)
        normRb_targ1 = np.linalg.norm(Rb_targ1)
        del Rsumexpo, Reta_targ_rp, MatRtarg_rp, Rtarg_rp

        #---display screen
        if ind_display_screen == 1:
            print(f' Sylverman bandwidth sNr  = {sNr}')
            print(f' nu                       = {nu}')
            print(f' coNr  = 1/(nu*sNr*sNr)   = {coNr}')
            print(f' coNr2 = 2*coNr           = {coNr2}')
            print(f' norm of Rb_targ1         = {normRb_targ1}')

        #--- print 
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write(f' Sylverman bandwidth sNr  = {sNr:14.7e} \n ')
                fidlisting.write(f' nu                       = {nu} \n ')
                fidlisting.write(f' coNr  = 1/(nu*sNr*sNr)   = {coNr:14.7e} \n ')
                fidlisting.write(f' coNr2 = 2*coNr           = {coNr2:14.7e} \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f' norm of Rb_targ1         = {normRb_targ1:14.7e} \n ')
                fidlisting.write('      \n ')

    #--------------------------------------------------------------------------------------------------------------------------------------
    #    Case ind_type_targ = 2 or 3: computing Rb_targ2(nu) giving the mean value Rmeanxx_targ(nx_targ) of XX_targ (unscaled)
    #    ind_type_targ : = 2, targets defined by giving the mean value Rmeanxx_targ(nx_targ) of XX_targ 
    #                  : = 3, targets defined by giving the mean value and the covariance matrix MatRcovxx_targ(nx_targ,nx_targ) of XX_targ
    #--------------------------------------------------------------------------------------------------------------------------------------

    if ind_type_targ == 2 or ind_type_targ == 3:
        #--- Computing Rmeanxx_targ(nx_targ)  (scaled) from  Rmeanxx_targ(nx_targ)  (unscaled)
        # No scaling 
        if ind_scaling == 0:
            Rmeanx_targ = Rmeanxx_targ                                    # Rmeanx_targ(nx_targ),Rmeanxx_targ(nx_targ)

        # Scaling
        if ind_scaling == 1:                                              # Rmeanxx_targ(nx_targ),Rbeta_targ(nx_targ)
            Rmeanx_targ = MatRalpham1_targ @ (Rmeanxx_targ - Rbeta_targ)  # Rmeanx_targ(nx_targ),MatRalpham1_targ(nx_targ,nx_targ)

        Rmeanx_targ_tilde = Rmeanx_targ - Rmeanx_d_targ                   # Rmeanx_targ_tilde(nx_targ),Rmeanx_targ(nx_targ),Rmeanx_d_targ(nx_targ)
        Rb_targ2 = MatRVt @ Rmeanx_targ_tilde                             # Rb_targ2(nu),MatRVt(nu,nx_targ),Rmeanx_targ_tilde(nx_targ)

    #--------------------------------------------------------------------------------------------------------------------------------------
    #    Case ind_type_targ = 3: computing Rb_targ3(nu) giving the mean value Rmeanxx_targ(nx_targ) and the covariance 
    #                            matrix MatRcovxx_targ(nx_targ,nx_targ) of XX_targ  (unscaled)
    #--------------------------------------------------------------------------------------------------------------------------------------

    if ind_type_targ == 3:
        MatRcovx_targ = MatRalpham1_targ @ MatRcovxx_targ @ MatRalpham1_targ       # MatRcovxx_targ(nx_targ,nx_targ),MatRalpham1_targ(nx_targ,nx_targ)
        MatRtemp      = MatRcovx_targ + np.outer(Rmeanx_targ_tilde,Rmeanx_targ_tilde)  # MatRtemp(nx_targ,nx_targ),Rmeanx_targ_tilde(nx_targ)
        Rb_targ3      = np.diag(MatRVt @ MatRtemp @ MatRVt.T)                          # Rb_targ3(nu),MatRVt(nu,nx_targ),MatRtemp(nx_targ,nx_targ),
                                                                                       # MatRVt(nu,nx_targ)
    ElapsedTime = time.time() - TimeStart

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(' ----- Elapsed time for Task8_ProjectionTarget \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' Elapsed Time   =  {ElapsedTime:10.2f}\n')
            fidlisting.write('      \n ')

    if ind_display_screen == 1:
        print('--- end Task8_ProjectionTarget')

    return Rb_targ1, coNr, coNr2, MatReta_targ, Rb_targ2, Rb_targ3
