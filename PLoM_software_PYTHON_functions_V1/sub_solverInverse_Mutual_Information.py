import numpy as np
import gc
from joblib import Parallel, delayed

def sub_solverInverse_Mutual_Information(MatRx, ind_parallel):

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM)
    #  Function name: sub_solverInverse_Mutual_Information
    #  Subject      : Computing the Mutual Information iX = E_X{log(p_X(X)/(p_X1(X1) x ... x pXn(Xn))}
    #                 in which X = (X1,...,Xn), p_X is the pdf of X and p_Xj is the pdf of Xj
    #                 The algorithm used is those of Kullback
    #                               
    #--- INPUT    
    #         MatRx(n,N)      : N realizations of random vector X of dimension n
    #         ind_parallel    : 0 no parallel computation
    #                           1    parallel computation
    #--- OUTPUT 
    #         iX

    n, N = MatRx.shape                   # n : dimension of random vector X, N : number of realizations of random vector X   

    #--- Check data
    if n <= 0 or N <= 0:
        raise ValueError('STOP in sub_solverDirect_Mutual_Information: n <= 0 or N <= 0')

    #--- Silver bandwidth       
    sx     = (4/((n+2)*N))**(1/(n + 4))
    modifx = 1                                          # in the usual theory, modifx=1
    sx     = modifx*sx                                  # Silver bandwidth modified 
    cox    = 1/(2*sx*sx)
   
    #--- std of X 
    Rstd_x = np.std(MatRx,axis=1,ddof=1)                # Rstd_x(n),MatRx(n,N)  

    #--- Computation of 1/std(X) 
    Rstdm1_x = 1.0/Rstd_x                               # Rstdm1_x(n)
   
    #--- Computing iX (the mutual information)
    Rtempx = np.zeros(N)                                # Rtempx(N) 
    MatRy  = np.zeros((n, N))                           # MatRy(n,N)
    
    #--- Vectorial sequence
    if ind_parallel == 0:
        for j in range(N):
            Rx_j         = MatRx[:,j]                                                     # Rx_j(n),MatRx(n,N)
            MatRxx_j     = (MatRx - Rx_j[:,np.newaxis])*Rstdm1_x[:,np.newaxis]            # MatRxx_j(n,N),MatRx(n,N),Rx_j(n),Rstdm1_x(n)
            Rtempx[j]    = np.mean(np.exp(-cox*np.sum(MatRxx_j**2,axis=0)),axis=0)        # Rtempx(N),MatRxx_j(n,N) 
            MatRy[:,j]   = np.mean(np.exp(-cox*(MatRxx_j**2)),axis=1)                     # MatRy(n,N),MatRxx_j(n,N)              
          
    #--- Parallel sequence (placeholder for actual parallel implementation)
    if ind_parallel == 1:

        def compute_parallel(j,MatRx,Rstdm1_x,cox):
            Rx_j     = np.copy(MatRx[:,j])                                                         # Rx_j(n),MatRx(n,N)
            MatRxx_j = (MatRx - Rx_j[:,np.newaxis])*Rstdm1_x[:,np.newaxis]                # MatRxx_j(n,N),MatRx(n,N),Rx_j(n),Rstdm1_x(n)
            tempx    = np.mean(np.exp(-cox*np.sum(MatRxx_j**2,axis=0)),axis=0)            # tempx,MatRxx_j(n,N) 
            Ry       = np.mean(np.exp(-cox*(MatRxx_j**2)),axis=1)                         # Ry(n),MatRxx_j(n,N)
            return tempx,Ry

        results = Parallel(n_jobs=-1)(delayed(compute_parallel)(j,MatRx,Rstdm1_x,cox) for j in range(N))
        for j in range(N):
            Rtempx[j], MatRy[:,j] = results[j]                                            #  MatRy(n,N)
        del results
        gc.collect()

    Rtempy  = np.prod(MatRy,axis=0)                                                       # Rtempy(N)
    Rlog    = np.log(Rtempx/Rtempy)                                                       # Rlog(N)
    iX      = np.mean(Rlog) 
    return iX
