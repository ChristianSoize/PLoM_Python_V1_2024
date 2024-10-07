import numpy as np
from sub_partition9_log_ksdensity_mult import sub_partition9_log_ksdensity_mult

def sub_partition6_checking_independence_constructed_groups2(NKL, nr, ngroup, Igroup, MatIgroup, MatRHexp, MatRHexpGauss):
    #
    # Copyright C. Soize 24 May 2024 
    #
    # Checking that the constructed groups are independent 

    #--- Constructing the reference entropy criterion with Gauss distribution
    RlogpdfGAUSSch = sub_partition9_log_ksdensity_mult(NKL, nr, NKL, nr, MatRHexpGauss, MatRHexpGauss)  # RlogpdfGAUSSch(nr)     
    SGAUSSch       = - np.mean(RlogpdfGAUSSch)                                                          # entropy of HGAUSSgrgamma
    SGAUSSchgr     = 0
    for j in range(ngroup):
        mj     = Igroup[j]                   # Igroup(ngroup)
        Indgrj = MatIgroup[j,:mj]            # MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup  
        Indgrj = Indgrj.astype(int)  
        # Handle the case when mj == 1 (convert to 2D array)
        if mj == 1:           
            MatRHexpGaussgrj = MatRHexpGauss[Indgrj-1,:].flatten()                                             # Convert to 1D array if mj == 1
        else:
            MatRHexpGaussgrj = MatRHexpGauss[Indgrj-1,:]                                                       # 2D array for mj > 1    
        RlogpdfGAUSSchgrj = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpGaussgrj,MatRHexpGaussgrj)  # RlogpdfGAUSSchgrj(nr)        
        SGAUSSchgrj       = -np.mean(RlogpdfGAUSSchgrj)                                                        # entropy of HGAUSS
        SGAUSSchgr        = SGAUSSchgr + SGAUSSchgrj        
        del Indgrj,MatRHexpGaussgrj,RlogpdfGAUSSchgrj,SGAUSSchgrj

    INDEPGaussRefch = SGAUSSchgr - SGAUSSch  # INDEPGaussRefch should be equal to 0 because the groups are independent
                                             # as nr is small and is finite, INDEPGaussRefch is strictly positive and is chosen as the reference
    del MatRHexpGauss, RlogpdfGAUSSch, SGAUSSch, SGAUSSchgr

    #--- Constructing the criterion INDEPch for checking the independence of the groups

    Rlogpdfch = sub_partition9_log_ksdensity_mult(NKL, nr, NKL, nr, MatRHexp, MatRHexp)    # Rlogpdfch(nr)     
    Sch = - np.mean(Rlogpdfch)                                                             # entropy of H
    Schgr = 0
    for j in range(ngroup):
        mj = Igroup[j]                                                                     # Igroup(ngroup)                                        
        Indgrj = MatIgroup[j, :mj]   
        Indgrj = Indgrj.astype(int)   
        if mj == 1:           
            MatRHexpgrj = MatRHexp[Indgrj-1, :].flatten()                                  # Convert to 1D array if mj == 1
        else:                                                                              # MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup    
            MatRHexpgrj = MatRHexp[Indgrj-1, :]         
        Rlogpdfchgrj = sub_partition9_log_ksdensity_mult(NKL, nr, mj, nr, MatRHexpgrj, MatRHexpgrj)  # Rlogpdfchgrj(nr,1)    
        Schgrj       = - np.mean(Rlogpdfchgrj)                                                       # entropy of Y^j
        Schgr        = Schgr + Schgrj;
        del Indgrj,MatRHexpgrj,Rlogpdfchgrj,Schgrj

    INDEPch = Schgr - Sch
    del Rlogpdfch, Sch, Schgr
    tau = 0
    if INDEPch > 1e-10 and INDEPGaussRefch > 1e-10:
        tau = 1 - INDEPch / INDEPGaussRefch

    return INDEPGaussRefch, INDEPch, tau
