import numpy as np

def sub_solverInverse_pseudo_inverse(MatR, eps_inv):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 09 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PloM) 
    #  Function name: sub_solverInverse_pseudo_inverse
    #  Subject      : compute the pseudo inverse MatRinv(n,n) of a positive definite matrix MatR(n,n): MatRinv = inv(MatR); 
    #
    # INPUT
    #      n                   
    #      MatR(n,n)
    #      eps_inv : tolerance for inverting the eigenvalue (for instance 1e-4)
    #
    # OUTPUT
    #      MatRinv(n,n)
    #
    # INTERNAL PARAMETER
    #      n
    
    MatRS     = 0.5*(MatR + MatR.T)
    (RxiTemp, MatRPsiTemp) = np.linalg.eigh(MatRS)     # MatRPsiTemp(n,n), RxiTemp(n), MatRS(n,n)
    Rxi      = np.sort(RxiTemp)[::-1]                  # Rxi(n,1) in descending order
    Index    = np.argsort(RxiTemp)[::-1]
    MatRPsi  = MatRPsiTemp[:,Index]                    # MatRPsi(n,n)
    xiMin    = Rxi[0]*eps_inv
    Ind      = np.where(Rxi >= xiMin)[0]
    m        = Ind.size                                # m is the number of eigenvalues greater than xiMin
    MatRPsim = MatRPsi[:,:m]                           # MatRPsim(n,m)
    Rxim     = Rxi[:m]                                # Rxim(m)
    MatRinv  = MatRPsim @ np.diag(1./Rxim) @ MatRPsim.T
    
    return MatRinv
