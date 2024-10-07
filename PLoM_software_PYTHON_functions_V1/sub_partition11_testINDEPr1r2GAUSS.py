import numpy as np
from sub_partition9_log_ksdensity_mult import sub_partition9_log_ksdensity_mult

def sub_partition11_testINDEPr1r2GAUSS(NKL,nr,MatRHexpGauss,r1,r2):
    #
    # Copyright C. Soize 24 May 2024 
    #
    # Numerical criterion for testing the independence of two normalized Gaussian random variables HGauss_r1 and HGauss by using 
    # the MUTUAL INFORMATION criterion: INDEPr1r2Gauss = S(HGauss_r1) + S(HGauss_r2) - S(HGauss_r1,HGauss_r2), that is positive or 
    # equal to zero, in which S(HGauss_r1), S(HGauss_r2), and S(HGauss_r1,HGauss_r2) are the entropies. Since HGauss_r1 and HGauss_r2 
    # are normalized random variables, we should have INDEPr1r2Gauss = 0. However as the entropy are estimated with a finite number 
    # of realizations, we have INDEPr1r2Gauss > 0 (and not = 0). This positive value is thus used as a numerical reference for testing 
    # the independence of the non-Gaussian normalized random variables H_r1 and H_r2
    #
    #---INPUT 
    #         nr                    : number of independent realizations of random vector H
    #         MatRHexpGauss(NKL,nr) : nr independent realizations of the random vector HGauss = (HGauss_1,...,HGauss_NKL)
    #         r1 and r2             : indices to estimate the statistical independence criterion of HGauss_r1 with HGauss_r2
    #
    #---OUTPUT 
    #         INDEPr1r2Gauss (theoretically INDEPr1r2Gauss = 0 for all r1 and r2)
    #
    #--- COMMENTS
    #            (1) The independent Gaussian solution is constructed to have the numerical reference of the
    #                independence criterion with the same number of realizations: nr
    #            (2) For each pair icouple = (r1, r2), INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss
    #                where Sr1Gauss is the entropy of HGauss_r1, Sr2Gauss is the entropy of HGauss_r2, and
    #                Sr1r2Gauss is the joint entropy of HGauss_r1 and HGauss_r2.           

    Rlogpdfr1Gauss = sub_partition9_log_ksdensity_mult(NKL,nr,1,nr,MatRHexpGauss[r1-1,:],MatRHexpGauss[r1-1,:]) # Rlogpdfr1Gauss(nr,1)
    Rlogpdfr2Gauss = sub_partition9_log_ksdensity_mult(NKL,nr,1,nr,MatRHexpGauss[r2-1,:],MatRHexpGauss[r2-1,:]) # Rlogpdfr2Gauss(nr,1)
    
    MatRRHDataGauss = np.vstack((MatRHexpGauss[r1-1, :], MatRHexpGauss[r2-1, :]))                               #  MatRRHDataGauss(2,nr)
    
    Rlogpdfr1r2Gauss = sub_partition9_log_ksdensity_mult(NKL,nr,2,nr,MatRRHDataGauss,MatRRHDataGauss)           # Rlogpdfr1r2Gauss(nr,1)
     
    Sr1Gauss   = -np.mean(Rlogpdfr1Gauss)
    Sr2Gauss   = -np.mean(Rlogpdfr2Gauss)
    Sr1r2Gauss = -np.mean(Rlogpdfr1r2Gauss)
    
    INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss  # should be equal to 0 and corresponds to the independence
    
    return INDEPr1r2Gauss

