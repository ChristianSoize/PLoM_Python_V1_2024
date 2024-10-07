import numpy as np
from scipy.special import erf
import scipy
import sys

def sub_conditional_confidence_interval(mw, mq, N, MatRqq, MatRww, Rww0, pc):
    #------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_conditional_confidence_interval
    #  Subject      : Let QQ  = (QQ_1,...,QQ_mq)   be the vector-valued random quantity of interest
    #                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
    #                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
    #                 We consider the family of conditional real-valued random variables: QQ_k | WW = ww0,  for k = 1, ..., mq
    #                 For a given value pc of the probability level, and for all k = 1,...,mq, this function computes 
    #                 the lower bound  qqLower_ww0_k  and the upper bound  qqUpper_ww0_k  of the conditional confidence interval of 
    #                 QQ_k | WW = ww0,  for k = 1, ..., mq.
    #
    #  Publications: 
    #               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                         American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020).   
    #               [3] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
    #                         for nonlinear dynamical systems, Computer Methods in Applied Mechanics and Engineering, 
    #                         doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024). 
    #               [ ] For the conditional statistics formula, see the Appendix of paper [3]
    #
    #
    #--- INPUTS
    #          mw                     : dimension of MatRww(mw,N) and Rww_0(mw,1) 
    #          mq                     : dimension of MatRqq(mq,N)
    #          N                      : number of realizations
    #          MatRqq(mq,N)           : N realizations of the vector-valued random quantity of interest QQ = (QQ_1,...,QQ_mq)
    #          MatRww(mw,N)           : N realizations of the vector-valued control random variable WW = (WW_1,...,WW_mw)
    #          Rww0(mw)               : a given value ww0 = (ww0_1,...,ww0_mw) of the vector-valued control parameter
    #          pc                     : the probability level (for instance 0.98)
    #
    #--- OUPUTS
    #          RqqLower_ww0(mq)       : For k=1,...,mq, RqqLower_ww0(k,1) is the lower bound  qqLower_ww0_k  of the conditional 
    #                                   confidence interval of QQ_k | WW = ww0.   
    #          RqqUpper_ww0(mq)       : For k=1,...,mq, RqqUpper_ww0(k,1) is the upper bound  qqUpper_ww0_k  of the conditional 
    #                                   confidence interval of QQ_k | WW = ww0.   
    
    #--- Pre-computation   
    nx = mq + mw
    sx = (4 / (N * (2 + nx))) ** (1 / (nx + 4))  # Silverman bandwidth for XX = (QQ,WW)
    cox = 1 / (2 * sx * sx)
    co2x = 1 / (np.sqrt(2) * sx)

    Rmean_qq = np.mean(MatRqq, axis=1)           # Rmean_qq(mq),MatRqq(mq,N)
    Rstd_qq  = np.std(MatRqq, axis=1, ddof=1)    # Rstd_qq(mq),MatRqq(mq,N) 
    Rmean_ww = np.mean(MatRww, axis=1)           # Rmean_ww(mw),MatRww(mw,N) 
    Rstd_ww  = np.std(MatRww, axis=1, ddof=1)    # Rstd_ww(mw),MatRww(mw,N) 
    
    Rstd_qq[Rstd_qq == 0] = 1
    Rstd_ww[Rstd_ww == 0] = 1

    Rmaxq = np.max(MatRqq, axis=1)  # MatRqq(mq,N), Rmaxq(mq)
    Rminq = np.min(MatRqq, axis=1)  # MatRqq(mq,N), Rminq(mq)   

    # Realizations of the normalized random variable QQtilde(mq), WWtilde(mw), and corresponding Rwwtilde_0(mw)
    MatRqqtilde = (MatRqq - Rmean_qq[:, np.newaxis]) / Rstd_qq[:, np.newaxis]  # MatRqqtilde(mq,N),MatRqq(mq,N),Rmean_qq(mq),Rstd_qq(mq)
    del MatRqq
    MatRwwtilde = (MatRww - Rmean_ww[:, np.newaxis]) / Rstd_ww[:, np.newaxis]  # MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw),Rstd_ww(mw)
    Rwwtilde0 = (Rww0 - Rmean_ww) / Rstd_ww

    #--- Conditional cdf of QQ_k | RWW = Rww0  for k = 1,...,mq  
    nbpoint   = 200
    coeff     = 0.2
    MatRqstar = np.zeros((mq, nbpoint))  # MatRqstar(mq,nbpoint): nbpoint qstar points in abscissa for computation of the cond cdf
    
    for iq in range(mq):
        maxq = Rmaxq[iq]
        minq = Rminq[iq]
        delta = maxq - minq
        maxq = maxq + coeff * delta
        minq = minq - coeff * delta
        MatRqstar[iq, :] = np.linspace(minq, maxq, nbpoint)                          # MatRqstar(mq,nbpoint)
                                                                                     # MatRqstartilde(mq,nbpoint),MatRqstar(mq,nbpoint)
    MatRqstartilde = (MatRqstar - Rmean_qq[:, np.newaxis]) / Rstd_qq[:, np.newaxis]  # Rmean_qq(mq),Rstd_qq(mq)
    MatRexpo       = MatRwwtilde - Rwwtilde0[:, np.newaxis]                          # MatRexpo(mw,N),MatRwwtilde(mw,N),Rwwtilde0(mw)
    MatRS          = np.exp(-cox * np.sum(MatRexpo ** 2, axis=0))                    # MatRS(N)
    den            = np.sum(MatRS)

    ArrayB = np.tile(MatRqqtilde[:, :, np.newaxis], (1, 1, nbpoint))  # ArrayB(mq,N,nbpoint),MatRqqtilde(mq,N)
    del MatRqqtilde
    ArrayB = np.transpose(ArrayB, (0, 2, 1))                          # ArrayB(mq,nbpoint,N)
    ArrayB = ArrayB - np.tile(MatRqstartilde[:, :, np.newaxis], (1, 1, N))
    del MatRqstartilde
    ArrayB = (-co2x) * ArrayB
    ArrayB = erf(ArrayB)
    ArrayB = 0.5 * ArrayB + 0.5
    MatRB = ArrayB.reshape(mq * nbpoint, N)                   # MatRB(mq*nbpoint,N)
    del ArrayB
    MatRA = MatRB @ MatRS                                     # MatRA(mq*nbpoint),MatRB(mq*nbpoint,N),MatRS(N)
    del MatRS
    MatRcdfcondTemp = MatRA.reshape(mq, nbpoint)              # MatRcdfcondTemp(mq,nbpoint)
    MatRcdfcond = MatRcdfcondTemp.T / den                     # MatRcdfcond(nbpoint,mq)

    #--- Computing the confidence interval of QQ_k | WW = ww_0 for k = 1,...,mq and given p_c
    RqqLower_ww0 = np.zeros(mq)
    RqqUpper_ww0 = np.zeros(mq)
    
    for iq in range(mq):
        Rcdfcond_iq = MatRcdfcond[:, iq]                       # MatRcdfcond(nbpoint,mq) 
        ib_upper = np.argmin(np.abs(Rcdfcond_iq - pc))
        ib_lower = np.argmin(np.abs(Rcdfcond_iq - (1 - pc)))
        RqqLower_ww0[iq] = MatRqstar[iq, ib_lower]              # MatRqstar(mq,nbpoint)  
        RqqUpper_ww0[iq] = MatRqstar[iq, ib_upper]
    
    return RqqLower_ww0, RqqUpper_ww0
