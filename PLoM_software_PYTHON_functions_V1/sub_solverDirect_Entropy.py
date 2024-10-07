import numpy as np
import gc
from joblib import Parallel, delayed

def sub_solverDirect_Entropy(MatRx,ind_parallel):

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_Entropy
    #  Subject      : Computing the entropy =  - E_X{log(p_X(X))}  in which p_X is the pdf of X
    #                               
    #--- INPUT    
    #         MatRx(n,N)      : N realizations of random vector X of dimension n
    #         ind_parallel    : 0 no parallel computation
    #                           1    parallel computation
    #
    #--- OUTPUT 
    #         entropy         : entropy of probability density function p_X of X

    n = MatRx.shape[0]        # n : dimension of random vector X 
    N = MatRx.shape[1]        # N : number of realizations of random vector X   

    #--- Check data
    if n <= 0 or N <= 0:
        raise ValueError('STOP in sub_solverDirect_Entropy:  n <= 0 or N <= 0')

    #--- Silver bandwidth       
    sx     = (4/((n+2)*N))**(1/(n + 4))
    modifx = 1                             # in the usual theory, modifx = 1;          
    sx     = modifx*sx                     # Silver bandwidth modified 
    cox    = 1/(2*sx*sx)

    #--- std of X 
    Rstd_x = np.std(MatRx,axis=1,ddof=1)    # Rstd_x(n), MatRx(n,N)  
       
    #--- Computation of J0
    Rtemp = np.log(Rstd_x)
    J0    = np.sum(Rtemp,axis=0) + n*np.log(np.sqrt(2*np.pi)*sx)
    del Rtemp

    #--- Computation of 1/std(X) 
    Rstdm1_x = 1./Rstd_x                     # Rstd_x(n)
  
    #--- Computing entropy   
    Rtempx = np.zeros(N)                     # Rtempx(N) 

    #--- Vectorized sequence
    if ind_parallel == 0:
        for j in range(N):
            Rx_j      = MatRx[:, j]                                                    # Rx_j(n), MatRx(n,N)
            MatRxx_j  = (MatRx - Rx_j[:,np.newaxis])*Rstdm1_x[:,np.newaxis]            # MatRxx_j(n,N), MatRx(n,N), Rx_j(n), Rstdm1_x(n)
            Rtempx[j] = np.mean(np.exp(-cox * np.sum(MatRxx_j**2,axis=0)), axis=0)     # Rtempx(N), MatRxx_j(n,N) 

    #--- Parallel sequence
    if ind_parallel == 1:

        def compute_parallel(j,MatRx,Rstdm1_x,cox):
            Rx_j = MatRx[:, j]                                                         # Rx_j(n), MatRx(n,N)
            MatRxx_j = (MatRx - Rx_j[:, np.newaxis]) * Rstdm1_x[:, np.newaxis]         # MatRxx_j(n,N), MatRx(n,N), Rx_j(n), Rstdm1_x(n)
            result = np.mean(np.exp(-cox * np.sum(MatRxx_j**2,axis=0)), axis=0)        # Rtempx(N), MatRxx_j(n,N) 
            return result
        
        Rtempx_list = Parallel(n_jobs=-1)(delayed(compute_parallel)(j,MatRx,Rstdm1_x,cox) for j in range(N))
        Rtempx = np.array(Rtempx_list)
        del Rtempx_list
        gc.collect()
        
    Rlog    = np.log(Rtempx)                             # MatRlog(N)
    entropy = J0 - np.mean(Rlog)
    return entropy
