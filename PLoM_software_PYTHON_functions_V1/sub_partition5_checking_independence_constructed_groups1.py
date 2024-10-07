import numpy as np
import sys
from sub_partition9_log_ksdensity_mult import sub_partition9_log_ksdensity_mult

def sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss):

    # Copyright C. Soize 21 January 2016

    # SUJECT
    #           Checking that the constructed groups are independent
    #
    # INPUTS
    #           NKL                   : dimension of random vector H
    #           nr                    : number of independent realizations of random vector H
    #           ngroup                : number of constructed independent groups  
    #           Igroup(ngroup)        : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)  
    #           MatIgroup(ngroup,mmax): MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj with 
    #                                   mmax = max_j mj for j = 1, ... , ngroup
    #           MatRHexp(NKL,nr)      : nr realizations of H = (H_1,...,H_NKL)
    #           MatRHexpGauss(NKL,nr) : nr independent realizations of HGauss = (HGauss_1,...,HGauss_NKL)
    #
    #--- OUPUT
    #           tau : rate

    #--- Constructing the reference entropy criterion with Gauss distribution
    RlogpdfGAUSSch = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexpGauss,MatRHexpGauss)  # RlogpdfGAUSSch(nr,1)
    SGAUSSch       = -np.mean(RlogpdfGAUSSch)                                                      # entropy of HGAUSSgrgamma
    SGAUSSchgr     = 0
    for j in range(ngroup):
        mj = Igroup[j]                                                                             # Igroup(ngroup)
        Indgrj = MatIgroup[j, :mj]                                                                 # MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup 
        Indgrj = Indgrj.astype(int)         
        if mj == 1:
            MatRHexpGaussgrj = MatRHexpGauss[Indgrj-1, :].flatten()                                            # Convert to 1D array if mj == 1
        else:
            MatRHexpGaussgrj = MatRHexpGauss[Indgrj-1, :]                                                      # Keep as 2D array if mj > 1
        RlogpdfGAUSSchgrj = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpGaussgrj,MatRHexpGaussgrj)  # RlogpdfGAUSSchgrj(nr,1)
        SGAUSSchgrj       = -np.mean(RlogpdfGAUSSchgrj)                                                        # entropy of HGAUSS
        SGAUSSchgr        = SGAUSSchgr + SGAUSSchgrj        
        del Indgrj, MatRHexpGaussgrj, RlogpdfGAUSSchgrj, SGAUSSchgrj
    
    INDEPGaussRefch = SGAUSSchgr - SGAUSSch  # INDEPGaussRefch should be equal to 0 because the groups are independent
                                             # as nr is small and is finite, INDEPGaussRefch is strictly positive and is chosen as the reference
    del RlogpdfGAUSSch,SGAUSSch,SGAUSSchgr,j,mj

    #--- Constructing the criterion INDEPch for checking the independence of the groups
    Rlogpdfch = sub_partition9_log_ksdensity_mult(NKL,nr,NKL,nr,MatRHexp,MatRHexp)             # Rlogpdfch(nr)
    Sch = -np.mean(Rlogpdfch)                                                                  # entropy of H
    Schgr = 0
    for j in range(ngroup):
        mj = Igroup[j]                                                                          # Igroup(ngroup)
        Indgrj = MatIgroup[j, :mj]                                                              # MatIgroup(j,1:Igroup(j)), j = 1,...,ngroup
        Indgrj = Indgrj.astype(int)  
        if mj == 1:
            MatRHexpgrj = MatRHexp[Indgrj-1, :].flatten()                                       # Convert to 1D array if mj == 1
        else:
            MatRHexpgrj = MatRHexp[Indgrj-1, :]                                                 # Keep as 2D array if mj > 1
        Rlogpdfchgrj = sub_partition9_log_ksdensity_mult(NKL,nr,mj,nr,MatRHexpgrj,MatRHexpgrj)  # Rlogpdfchgrj(nr,1)
        Schgrj = -np.mean(Rlogpdfchgrj)                                                         # entropy of Y^j
        Schgr = Schgr + Schgrj
        del Indgrj, MatRHexpgrj, Rlogpdfchgrj, Schgrj

    INDEPch = Schgr - Sch
    del Rlogpdfch,Sch,Schgr,j,mj

    #--- Constructing the rate
    tau = 0
    if INDEPch > 1e-10 and INDEPGaussRefch > 1e-10:
        tau = 1 - INDEPch / INDEPGaussRefch
    return tau


