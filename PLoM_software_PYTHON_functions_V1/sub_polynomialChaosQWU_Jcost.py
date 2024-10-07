import numpy as np
from scipy.linalg import cholesky
from sub_polynomialChaosQWU_OVL import sub_polynomialChaosQWU_OVL
import sys

def sub_polynomialChaosQWU_Jcost(MatRsample,n_y,Jmax,NnbMC0,n_q,nbqqC,nbpoint,MatRgamma_bar,MatRchol,MatRb,Ralpha_scale_yy, 
                                 RQQmean,MatRVectEig1s2,Ind_qqC,MatRqq_ar0,MatRxipointOVL,RbwOVL,MatRones):
    # WARNING: since a minimization algorithm is used and since we want to maximize the overlap;
    #          the signe "-" (minus) is introduded
    #---- INPUT
    #         MatRsample(n_y,Jmax)
    #         MatRgamma_bar(n_y,Jmax)
    #         MatRchol(n_y,n_y)
    #         MatRb(Jmax,NnbMC0)  
    #         Ralpha_scale_yy(n_y)
    #         RQQmean(n_q)
    #         MatRVectEig1s2(n_q,n_y) 
    #         Ind_qqC(nbqqC)
    #         MatRqq_ar0_Log(n_q,NnbMC0) 
    #         MatRxipointOVL(n_q,nbpoint) 
    #         RbwOVL(n_q)
    #         MatRones(n_y,Jmax) = ones(n_y,Jmax)

    #--- OUTPUT
    #         J = cost function

    MatRtilde = MatRgamma_bar * (MatRones + MatRsample)        # MatRtilde(n_y,Jmax),MatRgamma_bar(n_y,Jmax),MatRsample(n_y,Jmax),MatRones(n_y,Jmax) 
    MatRFtemp = cholesky(MatRtilde @ MatRtilde.T)              # MatRFtemp(n_y,n_y),MatRtilde(n_y,Jmax)
    MatRAtemp = np.linalg.inv(MatRFtemp).T                     # MatRAtemp(n_y,n_y),MatRFtemp(n_y,n_y)
    MatRhat   = MatRAtemp @ MatRtilde                          # MatRhat(n_y,Jmax),MatRAtemp(n_y,n_y),MatRtilde(n_y,Jmax)
    MatRgamma = MatRchol.T @ MatRhat                           # MatRgamma(n_y,Jmax),MatRchol(n_y,n_y),MatRhat(n_y,Jmax)
                                                                      
    MatRy  = MatRgamma @ MatRb                                 # MatRy(n_y,NnbMC0),MatRgamma(n_y,Jmax),MatRb(Jmax,NnbMC0)          
    MatRyy = Ralpha_scale_yy[:, np.newaxis] * MatRy            # MatRyy(n_y,NnbMC0),Ralpha_scale_yy(n_y),MatRy(n_y,NnbMC0)
    MatRqq = RQQmean[:,np.newaxis] + MatRVectEig1s2 @ MatRyy   # MatRqq(n_q,NnbMC0),RQQmean(n_q),MatRVectEig1s2(n_q,n_y),MatRyy(n_y,NnbMC0)
    
    RerrorOVL = sub_polynomialChaosQWU_OVL(nbqqC,NnbMC0,MatRqq_ar0[Ind_qqC-1,:],NnbMC0,MatRqq[Ind_qqC-1,:], 
                                           nbpoint,MatRxipointOVL[Ind_qqC-1,:],RbwOVL[Ind_qqC-1])
    J = - np.sum(RerrorOVL) / nbqqC  
          
    return J                                     