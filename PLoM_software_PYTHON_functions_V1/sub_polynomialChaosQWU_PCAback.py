import numpy as np

def sub_polynomialChaosQWU_PCAback(n_x, n_d, nu, n_ar, nx_obs, MatRx_d, MatReta_ar, Indx_obs, RmuPCA, MatRVectPCA, ind_display_screen, ind_print):
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM)
    #  Function name: sub_PCAback
    #  Subject      : back PCA; computing the n_ar realizations MatRx_obs(nx_obs,n_ar) of the scaled random observation  X_obs 
    #                 from the n_ar realizations MatReta_ar(nu,n_ar) of H_ar 
    #
    #  Publications: 
    #               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #
    # --- INPUTS
    #          n_x                   : dimension of random vector X_ar (scaled)
    #          n_d                   : number of points in the training set for XX_d and X_d  
    #          nu                    : order of the PCA reduction, which is the dimension of H_ar
    #          n_ar                  : number of realizations of H_ar and X_ar
    #          nx_obs                : number of observations extracted from X_ar  
    #          MatRx_d(n_x,n_d)      : n_d realizations of X_d (scaled)
    #          MatReta_ar(nu,n_ar)   : n_ar realizations of H_ar
    #          Indx_obs(nx_obs)      : nx_obs component numbers of X_ar that are observed with nx_obs <= n_x
    #          RmuPCA(nu)            : vector of PCA eigenvalues in descending order
    #          MatRVectPCA(n_x,nu)   : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA
    #          ind_display_screen    : = 0 no display,            = 1 display
    #          ind_print             : = 0 no print,              = 1 print
    #
    # --- OUTPUTS   
    #          MatRx_obs(nx_obs,n_ar)
    
    # --- Checking input data and parameters 
    if nx_obs > n_x:
        raise ValueError('STOP1 in sub_polynomialChaosQWU_PCAback: nx_obs > n_x')
    if MatRx_d.shape != (n_x, n_d):
        raise ValueError('STOP2 in sub_polynomialChaosQWU_PCAback: dimensions in MatRx_d(n_x,n_d) are not correct')
    if MatReta_ar.shape != (nu, n_ar):
        raise ValueError('STOP3 in sub_polynomialChaosQWU_PCAback: dimensions in MatReta_ar(nu,n_ar) are not correct')
    if Indx_obs.size >=1:           # checking if not empty
        if Indx_obs.ndim != 1:      # ckecking if it is a 1D array
            raise ValueError('STOP4 in sub_polynomialChaosQWU_PCAback: Indx_obs must be a 1D array')
    if len(Indx_obs) != nx_obs:
        raise ValueError('STOP5 in sub_polynomialChaosQWU_PCAback: the dimension of Indx_obs must be nx_obs')
    if len(Indx_obs) != len(np.unique(Indx_obs)):
        raise ValueError('STOP6 in sub_polynomialChaosQWU_PCAback: there are repetitions in Indx_obs')  # There are repetitions in Indx_obs
    if np.any(Indx_obs < 1) or np.any(Indx_obs > n_x):                     # At least one integer in Indx_obs is not within the valid range.
        raise ValueError('STOP7 in sub_polynomialChaosQWU_PCAback: at least one  integer in Indx_obs is not in range [1,n_x] ')     

    # --- Print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write(f'n_x    = {n_x:9d}\n')
            fidlisting.write(f'n_d    = {n_d:9d}\n')
            fidlisting.write(f'nu     = {nu:9d}\n')
            fidlisting.write(f'n_ar   = {n_ar:9d}\n')
            fidlisting.write(f'nx_obs = {nx_obs:9d}\n')
            fidlisting.write('\n')

    # --- Computing MatRx_obs(nx_obs, n_ar)
    RXmean       = np.mean(MatRx_d, axis=1)                                # RXmean(n_x),MatRx_d(n_x,n_d)
    MatRtemp     = MatRVectPCA @ np.diag(np.sqrt(RmuPCA))                  # MatRtemp(n_x,nu),MatRVectPCA(n_x,nu),RmuPCA(nu)
    RXmeanx_obs  = RXmean[Indx_obs - 1]                                    # RXmeanx_obs(nx_obs),Indx_obs(nx_obs)
    MatRtemp_obs = MatRtemp[Indx_obs - 1, :]                               # MatRtemp_obs(nx_obs,nu),MatRtemp(n_x,nu),Indx_obs(nx_obs)
    MatRx_obs    = RXmeanx_obs[:, np.newaxis] + MatRtemp_obs @ MatReta_ar  # MatRx_obs(nx_obs,n_ar),MatRtemp_obs(nx_obs,nu),MatReta_ar(nu,n_ar)

    return MatRx_obs
