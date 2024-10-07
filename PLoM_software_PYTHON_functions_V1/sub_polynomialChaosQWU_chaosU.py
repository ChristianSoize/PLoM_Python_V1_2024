import numpy as np
from scipy.linalg import cholesky

def sub_polynomialChaosQWU_chaosU(KU, ndeg, ng, NnbMC0, MatRU):
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 23 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_polynomialChaosQWU_chaosU
    #  Subject      : Computing matrices: MatRphiU(KU,NnbMC0),MatPowerU(KU,ng),MatRaU(KU,KU) of the polynomial chaos 
    #                     phi_{a^(m)}(U) with a^(m) = (a_1^(m),...,a_ng^(m)) in R^ng with m=1,...,KU
    #                     U            = (U_1, ... , U_ng) random vector for normalized Gaussian germ
    #                     ng           = dimension of the germ U = (U_1, ... , U_ng) 
    #                     m            = 1,...,KU indices of multi index (a_1^(m),...,a_ng^(m)) of length ng   
    #                     ndeg         = max degree of the polynomial chaos                                      
    #                     KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
    #                                    of terms including a^(0) = (0, ... ,0) for which phi_a^(0)(U) = 1
    #                 The algorithm used is the one detailed in the following reference:
    #                 [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
    #                     Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
    #
    #---INPUT variables
    #         KU           = factorial(ng+ndeg)/(factorial(ng)*factorial(ndeg)) = number
    #                         of terms including a^(0) = (0, ... ,0) for which phi_a^(0)(U) = 1
    #         ng           = dimension of the germ U = (U_1, ... , U_ng) 
    #         ndeg         = max degree of the polynomial chaos  
    #         NnbMC0       = number of realizations
    #         MatRU(ng,NnbMC0) = NnbMC0 independent realizations of normalized Gaussian random vector U = (U_1, ... , U_ng) 
    #
    #---OUTPUT variable
    #         MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)
    #         MatRaU(KU,KU) such that  MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)
    #         MatPowerU(KU,ng)
    #         NOTE: one has 1/(NnbMC0-1)*MatRphiU(KU,nbMC0)*MatRphiU(KU,nbMC0)' = [I_KU] with
    #               MatRphiU(KU,NnbMC0) = MatRaU(KU,KU)*MatRMM(KU,NnbMC0)

    #--- construction of MatPowerU(KU,ng): MatPowerU(k,:) = (alpha_1^(k), ... , alpha_ng^(k))
    #    which includes the multi-index (0,...,0) located at index  k = KU
    if ndeg == 0:    
        MatRphiU  = np.ones((KU, NnbMC0))    # MatRphiU(KU,NnbMC0): it is (1,...,1) NnbMC0 times
        MatPowerU = np.eye(KU, ng)           # MatPowerU(KU,ng) 
        MatRaU    = np.eye(KU, KU)           # MatRaU(KU,KU)

    if ndeg >= 1:
        MatPowerU = np.eye(KU, ng)
        Ind1 = 1
        Ind2 = ng
        I = ng
        for p in range(2, ndeg + 1):
            for L in range(ng):
                Rtest = np.zeros(ng)
                Rtest[L] = p - 1
                iL = Ind1
                while not np.array_equal(MatPowerU[iL - 1, :], Rtest):
                    iL = iL + 1
                for k in range(iL, Ind2 + 1):
                    I = I + 1
                    MatPowerU[I-1, :] = MatPowerU[L, :] + MatPowerU[k - 1, :]  # MatPowerU(KU,ng)
            Ind1 = Ind2 + 1
            Ind2 = I

        #--- Construction of monomials MatRMM(KU,NnbMC0) including multi-index (0,...,0)
        MatRMM = np.zeros((KU, NnbMC0))                             # MatRMM(KU,NnbMC0)
        MatRMM[0, :] = 1      
        for k in range(1, KU):
            Rtemp = np.ones(NnbMC0)                                 # Rtemp(1,NnbMC0)
            for j in range(ng):
                Rtemp = Rtemp * (MatRU[j, :] ** MatPowerU[k-1, j])  # MatRU(ng,NnbMC0), MatPowerU(KU,ng)
            MatRMM[k, :] = Rtemp                                    # MatRMM(KU,NnbMC0)

        #--- Construction of MatRphiU(KU,NnbMC0)
        MatRFF = (MatRMM @ MatRMM.T) / (NnbMC0 - 1)                 # MatRFF(KU,KU), MatRMM(KU,NnbMC0)
        MatRFF = 0.5 * (MatRFF + MatRFF.T)                          # MatRFF(KU,KU)
        # Removing the null space of MatRFF induced by numerical noise
        MaxD       = np.max(np.diag(MatRFF))
        tolrel     = 1e-12
        MatRFFtemp = MatRFF + tolrel * MaxD * np.eye(KU)            # MatRFFtemp(KU,KU)
        # Cholesky decomposition
        try:
            MatRLLtemp = cholesky(MatRFFtemp, lower=True)  # MatRLLtemp(KU,KU) is a lower triangular matrix    
            MatRLL = MatRLLtemp
        except np.linalg.LinAlgError:
            raise ValueError('STOP in sub_polynomialChaosQWU_chaosU: The matrix MatRFF must be positive definite')
        
        MatRaU   = np.linalg.inv(MatRLL)                           # MatRa(KU,KU): sparse matrix
        MatRphiU = MatRaU @ MatRMM                                 # MatRphiU(KU,NnbMC0), MatRaU(KU,KU), MatRMM(KU,NnbMC0)

    return MatRphiU, MatPowerU, MatRaU
