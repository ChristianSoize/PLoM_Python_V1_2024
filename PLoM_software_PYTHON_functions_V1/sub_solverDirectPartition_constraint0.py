import numpy as np
import gc
from joblib import Parallel, delayed
from sub_solverDirect_Verlet_constraint0 import sub_solverDirect_Verlet_constraint0

def sub_solverDirectPartition_constraint0(mj, n_d, nbMC, n_ar, nbmDMAPj, M0transient, Deltarj, f0j, shssj, shj,
                                          MatReta_dj, MatRgj, MatRaj, ArrayWiennerM0transient, ArrayGauss, 
                                          ind_parallel, ind_print):
    
    #---------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirectPartition_constraint0
    #  Subject      : ind_constraints = 0: solver PLoM for direct predictions with the partition and without constraints 
    #                 of normalization, for Y^j for each group j. The notations of the partition are
    #                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
    #                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
    #                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
    #  Comment      : this function is derived from sub_solverDirect_constraint0.m
    #
    #--- INPUT  
    #      mj                  : dimension of random vector Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj) 
    #      n_d                 : number of points in the training dataset for H_d
    #      nbMC                : number of realizations of (mj,n_d)-valued random matrix [Y^j] 
    #      n_ar                : number of realizations of Y^j such that n_ar  = nbMC x n_d
    #      nbmDMAPj            : dimension of the ISDE-projection basis
    #      M0transient         : mjmber of steps for reaching the stationary solution
    #      Deltarj             : ISDE integration step by Verlet scheme
    #      f0j                 : dissipation coefficient in the ISDE   
    #      shssj, shj          : parameters of the GKDE of pdf of Y^j_d (training) 
    #      MatReta_dj(mj,n_d)  : n_d realizations of Y^j_d (training)    
    #      MatRgj(n_d,nbmDMAPj): matrix of the ISDE-projection basis
    #      MatRaj(n_d,nbmDMAPj): related to MatRgj(n_d,nbmDMAPj) 
    #      ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
    #      ArrayGauss(mj,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
    #      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #      ind_print           : = 0 no print,                = 1 print
    #
    #--- OUTPUT
    #      MatReta_arj(mj,n_ar)
    
    #--- Initialization
    ArrayZ_ar = np.zeros((mj,nbmDMAPj,nbMC))

    #--- Vectorized computation of ArrayZ_ar(mj,nbmDMAPj,nbMC)
    if ind_parallel == 0:        
        for ell in range(nbMC):
            MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(mj,n_d,nbMC)
            ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)

            MatRZ_ar_ell = sub_solverDirect_Verlet_constraint0(mj,n_d,M0transient,Deltarj,f0j,MatReta_dj,MatRgj,MatRaj,shssj,shj,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell)
            ArrayZ_ar[:, :, ell] = MatRZ_ar_ell  # ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRZ_ar_ell(mj,nbmDMAPj)
        del MatRZ_ar_ell, MatRGauss_ell, ArrayWiennerM0transient_ell

    #--- Parallel computation of ArrayZ_ar(mj,nbmDMAPj,nbMC)
    if ind_parallel == 1:        
        
        def compute_ell(ell):
            MatRGauss_ell               = ArrayGauss[:,:,ell]                 # ArrayGauss(mj,n_d,nbMC)
            ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]  # ArrayWiennerM0transient(mj,n_d,M0transient,nbMC)

            MatRZ_ar_ell = sub_solverDirect_Verlet_constraint0(mj,n_d,M0transient,Deltarj,f0j,MatReta_dj,MatRgj,MatRaj,shssj,shj,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell)
            return MatRZ_ar_ell

        results = Parallel(n_jobs=-1)(delayed(compute_ell)(ell) for ell in range(nbMC))
        for ell, MatRZ_ar_ell in enumerate(results):
            ArrayZ_ar[:,:,ell] = MatRZ_ar_ell                                 # ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRZ_ar_ell(mj,nbmDMAPj)
        del results
        gc.collect()

    #--- Computing MatReta_arj(mj,n_ar) with a Vectorized sequence.
    #    Parallel computation with workers is not implemented due to possible RAM limitations
    #    and considering that the CPU time is not significant with vectorized computation.
    ArrayH_ar = np.zeros((mj,n_d,nbMC))
    for ell in range(nbMC):
        ArrayH_ar[:,:,ell] = ArrayZ_ar[:,:,ell] @ MatRgj.T   # ArrayH_ar(mj,n_d,nbMC), ArrayZ_ar(mj,nbmDMAPj,nbMC), MatRgj(n_d,nbmDMAPj)
    MatReta_arj = ArrayH_ar.reshape(mj,n_ar)                 # MatReta_arj(mj,n_ar)

    #--- Print The relative norm of the extradiagonal term that has to be close to 0
    #    and print Hmean_ar and diag(MatRHcov_ar)
    if ind_print == 1:       
        RHmean_ar = np.mean(MatReta_arj, axis=1)
        if mj == 1:           
            MatRHcov_ar    = np.var(MatReta_arj, axis=1)  # Variance as a scalar
            RdiagHcov_ar   = MatRHcov_ar                  # Diagonal is just the variance
            normExtra_diag = 0.0                          
        if mj > 1:
            MatRHcov_ar = np.cov(MatReta_arj)    # Covariance matrix
            RdiagHcov_ar = np.diag(MatRHcov_ar)  # Diagonal (variances)
            normExtra_diag = np.linalg.norm(MatRHcov_ar - np.diag(RdiagHcov_ar),'fro')/np.linalg.norm(np.diag(RdiagHcov_ar),'fro')
            
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n\n')
            fidlisting.write('----- RHmean_ar =          \n')
            fidlisting.write('                 ' + ' '.join([f'{x:.2e}' for x in RHmean_ar]) + '\n')
            fidlisting.write('\n\n')
            fidlisting.write('----- diag(MatRHcov_ar) =          \n')
            fidlisting.write('                 ' + ' '.join([f'{x:.2e}' for x in RdiagHcov_ar]) + '\n')
            fidlisting.write('\n\n')
            fidlisting.write(f'----- Relative Frobenius norm of the extra-diagonal terms of MatRHcov_ar = {normExtra_diag:.7e} \n')
            fidlisting.write('\n\n')

    return MatReta_arj
