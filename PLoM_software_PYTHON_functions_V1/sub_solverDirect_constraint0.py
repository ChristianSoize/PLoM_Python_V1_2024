import numpy as np
import gc
from joblib import Parallel, delayed
from sub_solverDirect_Verlet_constraint0 import sub_solverDirect_Verlet_constraint0

def sub_solverDirect_constraint0(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh,
                                 MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,ind_parallel,ind_print):

    #-------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM)
    #  Function name: sub_solverDirect_constraint0
    #  Subject      : ind_constraints = 0 : No constraints on H
    #
    #--- INPUT  
    #      nu                  : dimension of random vector H_d = (H_1, ... H_nu) and H_ar
    #      n_d                 : number of points in the training dataset for H_d
    #      nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar] 
    #      n_ar                : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #      nbmDMAP             : dimension of the ISDE-projection basis
    #      M0transient         : number of steps for reaching the stationary solution
    #      Deltar              : ISDE integration step by Verlet scheme
    #      f0                  : dissipation coefficient in the ISDE   
    #      shss,sh             : parameters of the GKDE of pdf of H_d (training) 
    #      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
    #      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
    #      ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
    #      ArrayGauss(nu,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
    #      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #      ind_print           : = 0 no print,                = 1 print
    #
    #--- OUTPUT
    #      MatReta_ar(nu,n_ar)
    #      ArrayZ_ar(nu,nbmDMAP,nbMC);    # ArrayZ_ar(nu,nbmDMAP,nbMC), this array is used as output for possible use in a postprocessing 
    #                                     # of Z in order to construct its polynomial chaos expansion (PCE)

    #--- Initialization
    ArrayZ_ar = np.zeros((nu,nbmDMAP,nbMC))

    #--- Vectorized computation of ArrayZ_ar(nu,nbmDMAP,nbMC)
    if ind_parallel == 0:
        for ell in range(nbMC):
            MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(nu,n_d,nbMC)
            ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
            
            MatRZ_ar_ell = sub_solverDirect_Verlet_constraint0(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRg,MatRa,shss,sh,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell)
            ArrayZ_ar[:, :, ell] = MatRZ_ar_ell                                   # ArrayZ_ar(nu,nbmDMAP,nbMC), MatRZ_ar_ell(nu,nbmDMAP)
        del MatRZ_ar_ell, MatRGauss_ell, ArrayWiennerM0transient_ell

    #--- Parallel computation of ArrayZ_ar(nu,nbmDMAP,nbMC)
    if ind_parallel == 1:

        def compute_for_ell(ell):
            MatRGauss_ell               = ArrayGauss[:, :, ell]                    # ArrayGauss(nu,n_d,nbMC)
            ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]    # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
            MatRZ_ar_ell = sub_solverDirect_Verlet_constraint0(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRg,MatRa,shss,sh,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell)
            return MatRZ_ar_ell

        results = Parallel(n_jobs=-1)(delayed(compute_for_ell)(ell) for ell in range(nbMC))
        for ell, MatRZ_ar_ell in enumerate(results):
            ArrayZ_ar[:, :, ell] = MatRZ_ar_ell                                   # ArrayZ_ar(nu,nbmDMAP,nbMC),MatRZ_ar_ell(nu,nbmDMAP)
        del results
        gc.collect()    

    #--- Computing MatReta_ar(nu,n_ar) with a Vectorized sequence.
    #    Parallel computation with a parfor loop is not implemented due to possible RAM limitations
    #    and considering that the CPU time is not significant with vectorized computation.
    ArrayH_ar = np.zeros((nu,n_d,nbMC))
    for ell in range(nbMC):
        ArrayH_ar[:, :, ell] = ArrayZ_ar[:, :, ell] @ MatRg.T  # ArrayH_ar(nu,n_d,nbMC),ArrayZ_ar(nu,nbmDMAP,nbMC),MatRg(n_d,nbmDMAP)
    MatReta_ar = np.reshape(ArrayH_ar, (nu,n_ar))              # MatReta_ar(nu,n_ar)

    #--- Print The relative norm of the extradiagonal term that as to be close to 0
    #    and print Hmean_ar and diag(MatRHcov_ar)
    if ind_print == 1:
        RHmean_ar      = np.mean(MatReta_ar,axis=1)
        MatRHcov_ar    = np.cov(MatReta_ar)
        RdiagHcov_ar   = np.diag(MatRHcov_ar)
        normExtra_diag = np.linalg.norm(MatRHcov_ar - np.diag(RdiagHcov_ar)) / np.linalg.norm(np.diag(RdiagHcov_ar))
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('----- RHmean_ar =          \n ')
            fidlisting.write(f'                 {RHmean_ar}\n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('----- diag(MatRHcov_ar) =          \n ')
            fidlisting.write(f'                 {RdiagHcov_ar}\n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(f'----- Relative Frobenius norm of the extra-diagonal terms of MatRHcov_ar = {normExtra_diag:14.7e}\n')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    return MatReta_ar, ArrayZ_ar


