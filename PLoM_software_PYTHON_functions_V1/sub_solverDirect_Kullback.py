import numpy as np
import gc
import sys
from joblib import Parallel, delayed

def sub_solverDirect_Kullback(MatRx, MatRy, ind_parallel):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 07 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_Kullback
    #  Subject      : Computing the Kullback-Leibler divergence: divKL = E_X{log(p_X(X)/p_Y(X))}  in which
    #                 X and Y are random vectors of dimension n with pdf p_X and p_Y    
    #                               
    #--- INPUT    
    #         MatRx(n,Nx)     : Nx realizations of random vector X
    #         MatRy(n,Ny)     : Ny realizations of random vector Y
    #         ind_parallel    : 0 no parallel computation
    #                           1    parallel computation
    #
    #--- OUTPUT 
    #         divKL

    n      = MatRx.shape[0]       # n  : dimension of random vectors X and Y
    nyTemp = MatRy.shape[0]
    Nx     = MatRx.shape[1]       # Nx : number of realizations of random vector X
    Ny     = MatRy.shape[1]       # Ny : number of realizations of random vector Y  

    if n != nyTemp:
        raise ValueError('STOP1 in sub_solverDirect_Kullback: dimension of X and Y must be the same')
    if n <= 0 or Nx <= 0 or Ny <= 0:
        raise ValueError('STOP2 in sub_solverDirect_Kullback: n <= 0 or Nx <= 0 or Ny <= 0')

    #--- Silver bandwidth       
    sx     = (4/((n+2)*Nx))**(1/(n+4))                                
    modifx = 1                                 # in the usual theory, modifx = 1;          
    sx     = modifx*sx                         # Silver bandwidth modified 
    cox    = 1/(2*sx*sx)

    sy     = (4/((n+2)*Ny))**(1/(n+4))                                
    modify = 1                                 # in the usual theory, modify = 1;          
    sy     = modify*sy                         # Silver bandwidth modified 
    coy    = 1/(2*sy*sy)

    #--- std of X and Y
    Rstd_x = np.std(MatRx,axis=1,ddof=1)        # Rstd_x(n), MatRx(n,Nx)  
    Rstd_y = np.std(MatRy,axis=1,ddof=1)        # Rstd_y(n), MatRy(n,Ny)                                                  
    
    #--- Computation of J0
    Rtemp = np.log(Rstd_y/Rstd_x)
    J0    = np.sum(Rtemp) + np.log(Ny/Nx) + n*np.log(sy/sx)

    #--- Computation of 1/std(X) and 1/std(Y)
    Rstdm1_x = 1./Rstd_x                     # Rstd_x(n)
    Rstdm1_y = 1./Rstd_y                     # Rstd_y(n)

    #--- computation of the Kullback
    Rtempx = np.zeros(Nx)                   # Rtempx(Nx) 
    Rtempy = np.zeros(Nx)                   # Rtempy(Nx): it is Nx AND NOT Ny

    # Vectorized sequence
    if ind_parallel == 0:
        for j in range(Nx):
            Rx_j         = MatRx[:, j]                                              # Rx_j(n), MatRx(n,Nx)
            MatRxx_j     = (MatRx - Rx_j[:, np.newaxis]) * Rstdm1_x[:, np.newaxis]  # MatRxx_j(n,Nx),MatRx(n,Nx),Rx_j(n),Rstdm1_x(n)
            Rtempx[j]    = np.sum(np.exp(-cox * np.sum(MatRxx_j**2, axis=0)))       # Rtempx(Nx),MatRxx_j(n,Nx) 
            MatRyy_j     = (MatRy - Rx_j[:, np.newaxis]) * Rstdm1_y[:, np.newaxis]  # MatRyy_j(n,Ny),MatRy(n,Ny),Rx_j(n),Rstdm1_y(n)
            Rtempy[j]    = np.sum(np.exp(-coy * np.sum(MatRyy_j**2, axis=0)))       # Rtempy(Nx), MatRyy_j(n,Nx) 

    # Parallel sequence 
    if ind_parallel == 1:

        def compute_kl(j,MatRx,MatRy,Rstdm1_x,Rstdm1_y,cox,coy):
            Rx_j         = MatRx[:, j]                                              # Rx_j(n), MatRx(n,Nx)
            MatRxx_j     = (MatRx - Rx_j[:, np.newaxis]) * Rstdm1_x[:, np.newaxis]  # MatRxx_j(n,Nx),MatRx(n,Nx),Rx_j(n),Rstdm1_x(n)
            Rtempx_j     = np.sum(np.exp(-cox * np.sum(MatRxx_j**2, axis=0)))       # Rtempx_j(Nx),MatRxx_j(n,Nx) 
            MatRyy_j     = (MatRy - Rx_j[:, np.newaxis]) * Rstdm1_y[:, np.newaxis]  # MatRyy_j(n,Ny),MatRy(n,Ny),Rx_j(n),Rstdm1_y(n)
            Rtempy_j    = np.sum(np.exp(-coy * np.sum(MatRyy_j**2, axis=0)))        # Rtempy_j(Nx), MatRyy_j(n,Nx) 
            return Rtempx_j, Rtempy_j

        results = Parallel(n_jobs=-1)(delayed(compute_kl)(j,MatRx,MatRy,Rstdm1_x,Rstdm1_y,cox,coy) for j in range(Nx))
        Rtempx, Rtempy = zip(*results)
        del results
        gc.collect()

    Rtempx = np.array(Rtempx)            # Rtempx(Nx)
    Rtempy = np.array(Rtempy)            # Rtempy(Nx): it is Nx AND NOT Ny
    Rlog   = np.log(Rtempx/Rtempy)       # MatRlog(Nx)
    divKL  = J0 + np.mean(Rlog) 
    
    return divKL
