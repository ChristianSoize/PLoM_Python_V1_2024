import numpy as np

def sub_conditional_second_order_moment(mw, mq, N, MatRqq, MatRww, Rww0):
    #------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_conditional_second_order_moment
    #  Subject      : Let QQ  = (QQ_1,...,QQ_mq)   be the vector-valued random quantity of interest
    #                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
    #                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
    #                 We consider the family of conditional real-valued random variables: QQ_k | WW = ww0,  for k = 1, ..., mq
    #                 For all k = 1,...,mq, this function computes, for k = 1, ..., mq,
    #                       - the conditional mean value           Eqq_ww0_k  =  E{QQ_k   | WW = ww0}
    #                       - the conditional second-order moment Eqq2_ww0_k =  E{QQ_k^2 | WW = ww0}
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
    #          mw                     : dimension of MatRww(mw,N) and Rww_0(mw) 
    #          mq                     : dimension of MatRqq(mq,N)
    #          N                      : number of realizations
    #          MatRqq(mq,N)           : N realizations of the vector-valued random quantity of interest QQ = (QQ_1,...,QQ_mq)
    #          MatRww(mw,N)           : N realizations of the vector-valued control random variable WW = (WW_1,...,WW_mw)
    #          Rww0(mw)               : a given value ww0 = (ww0_1,...,ww0_mw) of the vector-valued control parameter
    #
    #--- OUTPUTS
    #          REqq_ww0(mq)           : For k=1,...,mq, REqq_ww0(k) is the conditional mean value E{QQ_k | WW = ww0}
    #          REqq2_ww0(mq)          : For k=1,...,mq, REqq2_ww0(k) is the conditional second-order moment E{QQ_k^2 | WW = ww0}
    
    #---  Pre-computation 
    nx = mq + mw
    sx = (4 / (N * (2 + nx))) ** (1 / (nx + 4))  # Silverman bandwidth for XX = (QQ,WW)
    cox = 1 / (2 * sx * sx)

    Rmean_qq = np.mean(MatRqq, axis=1)         # Rmean_qq(mq),MatRqq(mq,N)
    Rstd_qq  = np.std(MatRqq, axis=1, ddof=1)  # Rstd_qq(mq),MatRqq(mq,N)
    Rmean_ww = np.mean(MatRww, axis=1)         # Rmean_ww(mw),MatRww(mw,N)
    Rstd_ww  = np.std(MatRww, axis=1, ddof=1)  # Rstd_ww(mw),MatRww(mw,N)

    Rstd_qq[Rstd_qq == 0] = 1
    Rstd_ww[Rstd_ww == 0] = 1

    # Realizations of the normalized random variable QQtilde(mq), WWtilde(mw), and corresponding Rwwtilde_0(mw)
    MatRqqtilde = (MatRqq - Rmean_qq[:, np.newaxis]) / Rstd_qq[:, np.newaxis]  # MatRqqtilde(mq,N),MatRqq(mq,N),Rmean_qq(mq),Rstd_qq(mq)
    MatRwwtilde = (MatRww - Rmean_ww[:, np.newaxis]) / Rstd_ww[:, np.newaxis]  # MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw),Rstd_ww(mw)
    Rwwtilde0 = (Rww0 - Rmean_ww) / Rstd_ww

    #--- Conditional mean Eqq_ww0_k = E{QQ_k | WW = ww0} and second-order moment Eqq2_ww0_k = E{QQ_k^2 | WW = ww0}, for k = 1,...,mq  
    MatRexpo  = MatRwwtilde - Rwwtilde0[:, np.newaxis]        # MatRexpo(mw,N),MatRwwtilde(mw,N),Rwwtilde0(mw)
    RS        = np.exp(-cox * np.sum(MatRexpo ** 2, axis=0))  # RS(N) is a 1D array
    den       = np.sum(RS)                                    # summation for a 1D array
    Rnum      = MatRqqtilde @ RS                              # Rnum(mq),MatRqqtilde(mq,N),RS(N)
    Rnum2     = (MatRqqtilde ** 2) @ RS                       # MatRqqtilde(mq,N),RS(N)
    Rtemp     = Rstd_qq * Rnum / den                          # Rstd_qq(mq),Rnum(mq)
    REqq_ww0  = Rmean_qq + Rtemp                              # REqq_ww0(mq)
    REqq2_ww0 = Rmean_qq ** 2 + 2 * Rmean_qq * Rtemp + (Rstd_qq ** 2) * (sx ** 2 + Rnum2 / den)  # REqq2_ww0(mq)
    
    return REqq_ww0, REqq2_ww0
