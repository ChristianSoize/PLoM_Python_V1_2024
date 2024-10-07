import numpy as np
from sub_partition9_log_ksdensity_mult import sub_partition9_log_ksdensity_mult
   
def sub_partition13_testINDEPr1r2(NKL, nr, MatRHexp, r1, r2):

    # Copyright C. Soize 24 May 2024 

    #--- SUBJECT
    #          Testing the independence of the non-Gaussian normalized random variables H_r1 et H_r2 by using the MUTUAL INFORMATION criterion
    #          that is written as INDEPr1r2 = Sr1 + Sr2 - Sr1r2 with Sr1 = entropy of H_r1, Sr2 = entropy of H_r2, Sr1r2 =  entropy of (Hr1,Hr2) 
    #          The entropy that is a mathematical expectation is estimated using the same realizations that the one used for estimating the 
    #          pdf by the Gaussian kernel method.
    #
    #---INPUTS 
    #          NKL              : dimension of random vector H
    #          nr               : number of independent realizations of random vector H
    #          MatRHexp(NKL,nr) : nr realizations of H = (H_1,...,H_NKL)
    #          r1 and r2        : indices for testing the statistical independence of H_r1 with H_r2
    #
    #---OUTPUTS  
    #          INDEPr1r2: if INDEPr1r2 ~ 0, H_r1 and H_r2 are independent 
    #                     if INDEPr1r2 > 0, H_r1 and H_r2 are dependent
    
    # Computing log pdf of H_r1 and H_r2
    Rlogpdfr1 = sub_partition9_log_ksdensity_mult(NKL, nr, 1, nr, MatRHexp[r1-1, :], MatRHexp[r1-1, :])  # Rlogpdfr1(nr,1) 
    Rlogpdfr2 = sub_partition9_log_ksdensity_mult(NKL, nr, 1, nr, MatRHexp[r2-1, :], MatRHexp[r2-1, :])  # Rlogpdfr2(nr,1) 
  
    # Computing entropy
    MatRRHData = np.vstack((MatRHexp[r1-1, :], MatRHexp[r2-1, :]))                           # MatRRHData(2,nr) : 2 components H_r1 and H_r2
    Rlogpdfr1r2 = sub_partition9_log_ksdensity_mult(NKL, nr, 2, nr, MatRRHData, MatRRHData)  # Rlogpdfr1r2(nr)
    
    Sr1   = -np.mean(Rlogpdfr1)  # entropy of H_r1
    Sr2   = -np.mean(Rlogpdfr2)  # entropy of H_r2
    Sr1r2 = -np.mean(Rlogpdfr1r2)  # entropy of (H_r1,Hr2)
    
    INDEPr1r2 = Sr1 + Sr2 - Sr1r2  
    
    return INDEPr1r2
