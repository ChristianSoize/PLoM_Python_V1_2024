import numpy as np
from scipy.linalg import cholesky

def sub_polynomialChaosQWU_chaos0(K0,Ndeg,Ng,NnbMC0,MatRxiNg0):
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 23 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM) 
    #  Function name: sub_polynomialChaosQWU_chaos0
    #  Subject      : Computing matrices: MatRPsi0(K0,NnbMC0),MatPower0(K0,Ng),MatRa0(K0,K0) of the polynomial chaos 
    #                     Psi_{alpha^(k)}(Xi) with alpha^(k) = (alpha_1^(k),...,alpha_Ng^(k)) in R^Ng with k=1,...,K0
    #                     Xi           = (Xi_1, ... , Xi_Ng) random vector for the germ
    #                     Ng           = dimension of the germ Xi = (Xi_1, ... , Xi_Ng)  
    #                     k            = 1,...,K0 indices of multi index (alpha_1^(k),...,alpha_Ng^(k)) of length Ng   
    #                     Ndeg         = max degree of the polynomial chaos   
    #                     K0           = factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg)) number of terms
    #                                    including alpha^0 = (0, ... ,0) for which psi_alpha^0(Xi) = 1
    #                 The algorithm used is the one detailed in the following reference:
    #                 [3] C. Soize, Uncertainty Quantification. An Accelerated Course with Advanced Applications in Computational Engineering,
    #                     Interdisciplinary Applied Mathematics, doi: 10.1007/978-3-319-54339-0, Springer, New York,2017.
    #
    #---INPUT variables
    #         K0 = number of polynomial chaos including (0,...,0): K0 =  fix(1e-12 + factorial(Ng+Ndeg)/(factorial(Ng)*factorial(Ndeg))); 
    #         Ndeg       = maximum degree of polynomial chaos
    #         Ng         = dimension of the germ Xi = (Xi_1, ... , Xi_Ng)   
    #         NnbMC0     = number of samples
    #         MatRxiNg0(Ng,NnbMC0) = NnbMC0 independent realizations of normalized random vector Xi = (Xi_1, ... , Xi_Ng) 
    #
    #---OUTPUT variable
    #         MatRPsi0(K0,NbMC0) such that  MatRPsi0(K0,NnbMC0) = MatRa0(K0,K0)*MatRMM(K0,NnbMC0)
    #         MatRa0(K0,K0)         such that  MatRPsi0(K0,NnbMC0) = MatRa0(K0,K0)*MatRMM(K0,NnbMC0)
    #         MatPower0(K0,Ng)
    #         NOTE: one has (1/(NnbMC0-1))*MatRPsi0(K0,NnbMC0)*MatRPsi0(K0,NnbMC0)' = [I_K0] 

    #--- Construction of MatPower0(K0,Ng): MatPower0(k,:) = (alpha_1^(k), ... , alpha_Ng^(k)),
    #    which includes the multi-index (0,...,0) located at index  k = K0
    if Ndeg == 0:    
        MatRPsi0 = np.ones((K0, NnbMC0))  # MatRPsi0(K0,NnbMC0): it is (1,...,1) NnbMC0 times

    if Ndeg >= 1:
        MatPower0 = np.eye(K0, Ng)
        Ind1 = 1
        Ind2 = Ng
        I = Ng
        for p in range(2, Ndeg + 1):
            for L in range(Ng):
                Rtest = np.zeros(Ng)
                Rtest[L] = p - 1
                iL = Ind1
                while not np.array_equal(MatPower0[iL-1, :], Rtest):
                    iL = iL +  1
                for k in range(iL, Ind2 + 1):
                    I = I + 1
                    MatPower0[I-1,:] = MatPower0[L, :] + MatPower0[k-1, :]   # MatPower0(K0,Ng)
            Ind1 = Ind2 + 1
            Ind2 = I

        #--- Construction of monomials MatRMM(K0,NnbMC0) including multi-index (0,...,0)
        MatRMM = np.zeros((K0, NnbMC0))                                # MatRMM(K0,NnbMC0)
        MatRMM[0, :] = 1      
        for k in range(1, K0):
            Rtemp = np.ones(NnbMC0)                                    # Rtemp(1,NnbMC0);
            for j in range(Ng):
                Rtemp = Rtemp * (MatRxiNg0[j, :] ** MatPower0[k-1, j]) # MatRxiNg0(Ng,NnbMC0), MatPower0(K0,Ng)
            MatRMM[k, :] = Rtemp                                       # MatRMM(K0,NnbMC0)

        #--- Construction of MatRPsi0(K0,NnbMC0)
        MatRFF = (MatRMM @ MatRMM.T) / (NnbMC0 - 1)                    # MatRFF(K0,K0), MatRMM(K0,NnbMC0)
        MatRFF = 0.5 * (MatRFF + MatRFF.T)                             # MatRFF(K0,K0)        
        # Removing the null space of MatRFF induced by numerical noise
        MaxD       = np.max(np.diag(MatRFF))
        tolrel     = 1e-12
        MatRFFtemp = MatRFF + tolrel * MaxD * np.eye(K0)               # MatRFFtemp(K0,K0)
        # Cholesky decomposition
        try:
            MatRLLtemp = cholesky(MatRFFtemp, lower=True)              # MatRLLtemp(K0,K0) is a lower triangular matrix    
            MatRLL = MatRLLtemp
        except np.linalg.LinAlgError:
            raise ValueError('STOP in sub_polynomialChaosQWU_chaos0: The matrix MatRFF must be positive definite')
        
        MatRa0   = np.linalg.inv(MatRLL)                               # MatRa(K0,K0) : sparse matrix
        MatRPsi0 = np.dot(MatRa0, MatRMM)                              # MatRPsi0(K0,NnbMC0),MatRa0(K0,K0),MatRMM(K0,NnbMC0)
    
    return MatRPsi0, MatPower0, MatRa0
