from joblib import Parallel, delayed
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

def sub_polynomialChaosQWU_OVL(n_q,N_Ref,MatRqq_Ref,N,MatRqq,nbpoint,MatRxipointOVL,RbwOVL):

    #----------------------------------------------------------------------------------------------------------------------------------------------
    #          Copyright C. Soize, 18 September 2024
    #          OVERLAPPING COEFFICIENT
    #----------------------------------------------------------------------------------------------------------------------------------------------
    #
    #          MatRqq(n_q,N)
    #          MatRqq_Ref(n_q,N_Ref)
    #          MatRxipointOVL(n_q,nbpoint)
    #          RbwOVL(n_q)
    
    # Define compute_ovl_for_iq as a nested function
    def compute_ovl_for_iq(iq,MatRxipointOVL,MatRqq,MatRqq_Ref,RbwOVL,nbpoint):
        Rxipoint    = MatRxipointOVL[iq,:]                          # Rxipoint(nbpoint), MatRxipointOVL(n_q,nbpoint)
        RqqRef      = MatRqq_Ref[iq,:]                              # RqqRef(N_Ref),MatRqq_Ref(n_q,N_Ref)
        IndRef      = np.where(np.abs(RqqRef) > 1e-10)[0]           # Removing the 0 values for using ksdensity
        RqqRefb     = RqqRef[IndRef]
        sigmaRefb   = np.std(RqqRefb,ddof=1)                        # used for computing the bandwidth of RqqRef_iq (learning) and of Rqq_iq (chaos)
        kde_RqqRefb = gaussian_kde(RqqRefb, bw_method=RbwOVL[iq]/sigmaRefb)   # RpdfRef(nbpoint),RbwOVL(n_q) 
        RpdfRef     = kde_RqqRefb(Rxipoint) 
        RqqRefVal   = Rxipoint                                       # RqqRefVal(nbpoint),Rxipoint(nbpoint)

        Rqq      = MatRqq[iq,:]                                      # Rqq(N),MatRqq(n_q,N)
        Ind      = np.where(np.abs(Rqq) > 1e-10)[0]                  # Removing the 0 values for using ksdensity
        Rqqb     = Rqq[Ind]                                          # bandwidth of RqqRef_iq (learning) and of Rqq_iq (chaos) is with sigmaRefb
        kde_Rqqb = gaussian_kde(Rqqb,bw_method=RbwOVL[iq]/sigmaRefb) # RbwOVL(n_q) 
        Rpdf     = kde_Rqqb(Rxipoint)                                # Rpdf(nbpoint)
        RqqVal   = Rxipoint                                          # RqqVal(nbpoint),Rxipoint(nbpoint)
        
        MAX = max(RqqVal[0],RqqRefVal[0])
        if MAX <= 0:
            qqmin = 0.999 * MAX
        else:
            qqmin = 1.001 * MAX
        
        MIN = min(RqqVal[nbpoint-1],RqqRefVal[nbpoint-1])
        if MIN <= 0:
            qqmax = 1.001 * MIN
        else:
            qqmax = 0.999 * MIN
        
        nbint = nbpoint + 1
        pasqq = 0.99999 * (qqmax - qqmin) / (nbint - 1)
        Rqqi  = qqmin - pasqq + np.arange(1, nbint + 1) * pasqq                         # Rqqi(nbint)
        
        interp_Ryi = interp1d(RqqVal, Rpdf, bounds_error=False, fill_value=0)
        Ryi = interp_Ryi(Rqqi)                                                          # Ryi(nbint)

        interp_RyiRef = interp1d(RqqRefVal, RpdfRef, bounds_error=False, fill_value=0)
        RyiRef        = interp_RyiRef(Rqqi)                                             # RyiRef(nbint)
        
        deni    = pasqq * np.sum(np.abs(Ryi))
        RpdfQQi = np.abs(Ryi) / deni
        
        deniRef    = pasqq * np.sum(np.abs(RyiRef))
        RpdfQQiRef = np.abs(RyiRef) / deniRef
        
        RerrorOVL_iq = 1 - 0.5 * pasqq * np.sum(np.abs(RpdfQQi - RpdfQQiRef))          # RerrorOVL(n_q)
        return RerrorOVL_iq
    
    # Parallel computation disabled during gradient calculation becausee the gradient calculation is in parallelized
    RerrorOVL = Parallel(n_jobs=1)(delayed(compute_ovl_for_iq)(iq,MatRxipointOVL,MatRqq,MatRqq_Ref,RbwOVL,nbpoint) for iq in range(n_q))

    return np.array(RerrorOVL)


    
