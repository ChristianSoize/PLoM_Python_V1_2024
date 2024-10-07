import numpy as np
import sys

def sub_conditional_pdf(mw, N, Rqq, MatRww, Rww0, nbpoint0):
    #------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 31 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_conditional_pdf
    #  Subject      : Let QQ be the real-valued random quantity of interest
    #                 Let WW  = (WW_1,...,WW_mw)   be the vector-valued control random variable
    #                 Let ww0 = (ww0_1,...,ww0_mw) be a value of the vector-valued control parameter
    #                 We consider the family of conditional real-valued random variables: QQ | WW = ww0
    #                 This function computes the conditional pdf  
    #                 pdfqq_ww0 =  pdf of {QQ | WW = ww0} estimated in nbpoint
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
    #          N                      : number of realizations
    #          Rqq(N)                 : N realizations of the real-valued random quantity of interest QQ
    #          MatRww(mw,N)           : N realizations of the vector-valued control random variable WW = (WW_1,...,WW_mw)
    #          Rww0(mw)               : a given value ww0 = (ww0_1,...,ww0_mw) of the vector-valued control parameter
    #          nbpoint0               : initial number of points from which the number of points will be used for estimating the pdf
    #                                   (for instance nbpoint0 = 200)
    #
    #--- OUTPUTS
    #          nbpoint                : number of points in which the conditional pdf is computed 
    #          Rq(nbpoint)            : discrete graph of the pdf: (Rq,Rpdfqq_ww0)
    #          Rpdfqq_ww0(nbpoint)    : Rpdfqq_ww0(j) is the conditional pdf of {QQ | WW = ww0} at point j = 1,...,nbpoint
    
    #--- Checking dimension for MatRqq(1,N) 
    n1temp = Rqq.ndim
    if n1temp != 1:
        raise ValueError('STOP in sub_conditional_pdf: the dimension of MatRqq must be one')
    
    #---  Pre-computation 
    nx     = 1 + mw
    sx     = (4 / (N * (2 + nx))) ** (1 / (nx + 4))  # Silverman bandwidth for XX = (QQ,WW)
    cox    = 1 / (2 * sx * sx)
    coefsx = 1 / (sx * np.sqrt(2 * np.pi))

    mean_qq  = np.mean(Rqq)                    # Rqq(N)
    std_qq   = np.std(Rqq,ddof=1)              # Rqq(N) 
    Rmean_ww = np.mean(MatRww, axis=1)         # Rmean_ww(mw),MatRww(mw,N) 
    Rstd_ww  = np.std(MatRww, axis=1, ddof=1)  # Rstd_ww(mw),MatRww(mw,N) 
    
    if std_qq == 0:
        std_qq = 1
    Rstd_ww[Rstd_ww == 0] = 1    

    #--- Realizations of the normalized random variable QQtilde, WWtilde(mw), and corresponding Rwwtilde_0(mw,1)
    Rqqtilde    = (Rqq - mean_qq) / std_qq                                     # MatRqqtilde(N),Rqq(N),mean_qq,std_qq
    MatRwwtilde = (MatRww - Rmean_ww[:, np.newaxis]) / Rstd_ww[:, np.newaxis]  # MatRwwtilde(mw,N),MatRww(mw,N),Rmean_ww(mw),Rstd_ww(mw)
    Rww0tilde   = (Rww0 - Rmean_ww) / Rstd_ww                                  # Rww0tilde(mw)  
    
    #--- Conditional pdf of QQ|WW=Rww_0 in nbpoint loaded in Rpdfqq_ww0(nbpoint)
    coeff     = 5
    maxq      = mean_qq + coeff * std_qq
    minq      = mean_qq - coeff * std_qq
    Rq        = np.linspace(minq, maxq, nbpoint0)          
    nbpoint   = len(Rq)                                 # Rq(nbpoint)    
    MatRexpo  = MatRwwtilde - Rww0tilde[:, np.newaxis]  # MatRexpo(mw,N),MatRwwtilde(mw,N),Rww0tilde(mw) 
    Rtempw    = cox * np.sum(MatRexpo ** 2, axis=0)     # Rtempw(N) 
    RS        = np.exp(-Rtempw)                         # RS(N) 
    den       = np.sum(RS) 
    Rpdfqq_ww0  = np.zeros(nbpoint)                     # Rpdfqq_ww0(nbpoint)
    Rqtilde  = (Rq - mean_qq) / std_qq                  # Rqtilde(nbpoint)
    for ib in range(nbpoint):
        qtilde_ib      = Rqtilde[ib]                                    # Rqtilde(nbpoint);
        Rexpo_ib       = Rqqtilde - qtilde_ib                           # Rexpo_ib(N),Rqqtilde(N)
        RS_ib          = np.exp(-Rtempw - cox * (Rexpo_ib ** 2))        # RS_ib(N),Rtempw(N),Rexpo_ib(N)
        num_ib         = np.sum(RS_ib)                                  # RS_ib(N)
        Rpdfqq_ww0[ib] = (coefsx / std_qq) * num_ib / den               # Rpdfqq_ww0(nbpoint)
    
    return nbpoint, Rq, Rpdfqq_ww0
