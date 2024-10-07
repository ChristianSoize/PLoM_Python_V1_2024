import numpy as np

def sub_polynomialChaosQWU_surrogate(N,n_w,n_y,MatRww,MatRU,Ralpham1_scale_chaos,Rbeta_scale_chaos,
                                     Ng,K0,MatPower0,MatRa0,ng,KU,MatPowerU,MatRaU,Jmax,MatRgamma_opt,Indm,Indk):
    # Copyright C. Soize 20 April 2023
    #
    # ---INPUT variables
    #         N > or = 1 is the number of evaluation performed: MatRqq = PolChaos(MatRww), MatRqq(n_q,N), Rww(n_w,N) with N > or = to 1
    #         n_w, n_y        with n_w = Ng
    #         MatRww(n_w, N)
    #         MatRU(ng, N)
    #         Ralpham1_scale_chaos(Ng)
    #         Rbeta_scale_chaos(Ng)
    #         Ng, K0
    #         MatPower0(K0, Ng)
    #         MatRa(K0, K0)
    #         ng, KU
    #         MatPowerU(KU, ng)
    #         MatRaU(KU, KU)
    #         Jmax
    #         MatRgamma_opt(n_y, Jmax)
    #         Indm(Jmax)
    #         Indk(Jmax)
    #
    # ---OUTPUT variable
    #         MatRy (n_y, N)

    if Ng != n_w:
        raise ValueError('STOP in sub_polynomialChaosQWU_surrogate: Ng must be equal to n_w')

    # --- Normalization of MatRww(n_w, N) into MatRxiNg(n_w, N) by scaling MatRww between cmin and cmax
    # Ensure MatRww is always 2D by reshaping it if N = 1
    if N == 1:
        MatRww_reshaped = MatRww.reshape(-1,1)
    if N > 1:
        MatRww_reshaped = MatRww
    
    MatRxiNg = Ralpham1_scale_chaos[:,np.newaxis] * (MatRww_reshaped - Rbeta_scale_chaos[:, np.newaxis])  # MatRxiNg(Ng, N), MatRww(n_w, N) with n_w = Ng 
                                                                                                           # Ralpham1_scale_chaos(Ng), Rbeta_scale_chaos(Ng)
    # --- Construction of monomials MatRMM0(K0, N) including multi-index (0,...,0)
    MatRMM0      = np.zeros((K0,N))                               # MatRMM0(K0,N)
    MatRMM0[0,:] = 1
    for k in range(1,K0):
        Rtemp = np.ones(N)                                         # Rtemp(N)
        for j in range(Ng):            
            Rtemp = Rtemp*(MatRxiNg[j,:]**MatPower0[k-1,j])        # MatRxiNg(Ng,N), MatPower0(K0,Ng)
        MatRMM0[k,:] = Rtemp                                       # MatRMM0(K0,N)

    # --- Computing MatRPsi0(K0, N) = MatRa0(K0, K0) * MatRMM0(K0, N)
    MatRPsi0 = MatRa0 @ MatRMM0                                    # MatRPsi0(K0,N)
    del MatRMM0, Rtemp
    
    # --- Construction of monomials MatRMMU(KU, NnbMC0) NOT including multi-index (0,...,0)
    MatRMMU = np.zeros((KU,N))                                     # MatRMM(KU,N)
    MatRMMU[0,:] = 1
    # Ensure MatRU is 2D, even when N = 1
    if N == 1:
        MatRU_reshaped = MatRU.reshape(-1,1)
    if N > 1:
        MatRU_reshaped = MatRU    
    for k in range(1,KU):
        Rtemp = np.ones(N)                                         # Rtemp(N)
        for j in range(ng):
            Rtemp = Rtemp*(MatRU_reshaped[j,:]**MatPowerU[k-1,j])           # MatRU(ng,N), MatPowerU(KU,ng)
        MatRMMU[k,:] = Rtemp                                       # MatRMM(KU,N)

    # --- Computing MatRphiU(KU, N) = MatRaU(KU, KU) * MatRMMU(KU, N)
    MatRphiU = MatRaU @ MatRMMU  # MatRphiU(KU, N)
    del MatRMMU, Rtemp

    # --- Construction of MatRb(Jmax, N) such that MatRb(j, ell) = MatRphiU(m, ell) * MatRPsi0(k, ell)
    MatRb = np.zeros((Jmax,N))
    for j in range(Jmax):
        m = Indm[j]
        k = Indk[j]
        MatRb[j,:] = MatRphiU[m-1,:] * MatRPsi0[k-1,:]      # MatRb(Jmax, N)

    # --- Computing MatRy(n_y,N)
    MatRy = MatRgamma_opt @ MatRb  # MatRy(n_y,N), MatRgamma_opt(n_y,Jmax), MatRb(Jmax,N)
    
    return MatRy
