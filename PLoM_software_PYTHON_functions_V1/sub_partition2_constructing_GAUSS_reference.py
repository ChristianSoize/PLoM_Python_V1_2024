import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import gc
import sys
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from sub_partition11_testINDEPr1r2GAUSS import sub_partition11_testINDEPr1r2GAUSS

def sub_partition2_constructing_GAUSS_reference(NKL,nr,MatRHexpGauss,ind_plot,ind_parallel,numfig):
    #
    # Copyright C. Soize 24 May 2024 
    #
    # --- INPUT 
    #         NKL                   : dimension of random vector H
    #         nr                    : number of independent realizations of random vector H
    #         MatRHexpGauss(NKL,nr) : nr independent realizations of the random vector HGauss = (HGauss_1,...,HGauss_NKL)
    #         ind_plot              : = 0 no plot, = 1 plot
    #         ind_parallel          : = 0 no parallel computing, = 1 parallel computing
    #         numfig                : number of generated figures before executing this function
    #
    # --- OUTPUT  
    #         INDEPGaussmax      : maximum observer on all the realizations of INDEPGauss 
    #         meanINDEPGauss     : mean value of INDEPGauss
    #         stdINDEPGauss      : std of INDEPGauss
    #         gust               : gust factor
    #         meanMaxINDEPGauss  : mean value of MaxINDEPGauss 
    #         stdMaxINDEPGauss   : std of MaxINDEPGauss 
    #         numfig             : number of generated figures after executing this function
    #
    # --- INTERNAL PARAMETERS
    #         r1 and r2          : indices to estimate the statistical independence criterion of HGauss_r1 with HGauss_r2
    #         INDEPGauss         : random variable whose realizations are INDEPr1r2Gauss with respect to 1 <= r1 < r2 <= NKL 
    #         MaxINDEPGauss      : random variable corresponding to the extreme values of INDEPGauss
    #
    # --- COMMENTS
    #            (1) The independent Gaussian solution is constructed to have the numerical reference of the
    #                independence criterion with the same number of realizations: nr
    #            (2) For each pair icouple = (r1, r2), INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss
    #                where Sr1Gauss is the entropy of HGauss_r1, Sr2Gauss is the entropy of HGauss_r2, and
    #                Sr1r2Gauss is the joint entropy of HGauss_r1 and HGauss_r2.
    #
    # --- METHOD 
    #
    #    For testing the independence of two normalized Gaussian random variables HGauss_r1 and HGauss_r2, the numerical criterion is
    #    based on the MUTUAL INFORMATION criterion: 
    #
    #    INDEPr1r2Gauss = Sr1Gauss + Sr2Gauss - Sr1r2Gauss >= 0
    #    
    #    in which Sr1Gauss, Sr2Gauss, and Sr1r2Gauss are the entropy of  HGauss_r1, HGauss_r2, and (HGauss_r1,HGauss_r2).
    #    As the Gaussian random vector (HGauss_r1,HGauss_r2) is normalized, we should have INDEPr1r2Gauss  = 0. However as the entropy are 
    #    estimated with a finite number of realizations, we have INDEPr1r2Gauss > 0 (and not = 0).
    #    This positive value could thus be used as a numerical reference for testing the independence of the non-Gaussian normalized random 
    #    variables H_r1 and H_r2. However as the entropy are estimated with a finite number of realizations, we have INDEPr1r2Gauss > 0 (and not = 0). 
    #    This positive value is thus used as a numerical reference for testing the independence of the non-Gaussian normalized random 
    #    variables H_r1 and H_r2
    #  
    #    Let Z be the positive-valued random variable for which the np = NKL*(NKL-1)/2 realizations are z_p = RINDEPGauss(p) = INDEPr1r2Gauss 
    #    with 1 <= r1 < r2 <= NKL.  

    np_val      = NKL* (NKL - 1) // 2         # although that NKL * (NKL - 1) is always even, by security // is used instead of /
    RINDEPGauss = np.zeros(np_val)            # RINDEPGauss(np_val)
    Indr1       = np.zeros(np_val,dtype=int)
    Indr2       = np.zeros(np_val,dtype=int)
   
    p = 0
    for r1 in range(1,NKL):
        for r2 in range(r1 + 1, NKL+1):
            p = p + 1                       
            Indr1[p - 1] = r1
            Indr2[p - 1] = r2
    
    # --- Sequential computation
    if ind_parallel == 0:
        for p in range(np_val):
            r1 = Indr1[p]
            r2 = Indr2[p]
            INDEPr1r2Gauss = sub_partition11_testINDEPr1r2GAUSS(NKL,nr,MatRHexpGauss,r1,r2)
            RINDEPGauss[p] = INDEPr1r2Gauss
   
    # --- Parallel computation
    if ind_parallel == 1:        
        def compute_INDEPr1r2(p,NKL,nr,MatRHexpGauss,Indr1,Indr2):
            r1 = Indr1[p]
            r2 = Indr2[p]
            result = sub_partition11_testINDEPr1r2GAUSS(NKL,nr,MatRHexpGauss,r1,r2)
            return result
        
        RINDEPGauss = Parallel(n_jobs=-1)(delayed(compute_INDEPr1r2)(p,NKL,nr,MatRHexpGauss,Indr1,Indr2) for p in range(np_val))
        RINDEPGauss = np.array(RINDEPGauss)        # 1D array RINDEPGauss(np_val)
    
    meanINDEPGauss = np.mean(RINDEPGauss)          # empirical mean value of Z
    stdINDEPGauss  = np.std(RINDEPGauss,ddof=1)    # empirical standard deviation Z
    INDEPGaussmax  = np.max(RINDEPGauss)           # maximum on the realizations z_p for p=1,...,np
    RINDEPGauss0   = RINDEPGauss - meanINDEPGauss  # centering, 1D array RINDEPGauss0(np_val)
    nup = 0
    for p in range(2, np_val+1):                     # number of upcrossings by 0
        if RINDEPGauss0[p-1] - RINDEPGauss0[p - 2] > 0 and RINDEPGauss0[p-1] * RINDEPGauss0[p - 2] < 0:  # there is one upcrossing by zero
            nup = nup + 1
   
    cons = np.sqrt(2 * np.log(nup))
    gust = cons + 0.577 / cons
    meanMaxINDEPGauss = meanINDEPGauss + gust * stdINDEPGauss
    stdMaxINDEPGauss  = stdINDEPGauss * (np.pi / np.sqrt(6)) / cons
   
    if ind_plot == 1:
        plt.figure()  # --- plot the trajectory, RINDEPGauss(np_val)
        plt.plot(RINDEPGauss, '-b')
        plt.title(r'Graph of $i^\nu(G_{r_1},G_{r_2})$ as a function of the pair $(r_1,r_2)$', fontweight='normal', fontsize=16)
        plt.xlabel(r'number of the pair $(r_1,r_2)$', fontsize=16)
        plt.ylabel(r'$i^\nu(G_{r_1},G_{r_2})$', fontsize=16)
        numfig = numfig + 1 
        plt.savefig(f'figure_PARTITION_{numfig}_i-Gauss.png')
        plt.close()
   
        # --- plot the pdf 
        npoint = 1000
        # For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
        sigma     = np.std(RINDEPGauss,ddof=1) 
        bw        = 1.0592*sigma*np_val**(-1/5)          # bandwidth formula used by Matlab
        kde       = gaussian_kde(RINDEPGauss,bw_method=bw/sigma)
        Rp        = np.linspace(0, RINDEPGauss.max(), npoint)
        Rpdf      = kde.evaluate(Rp)
        plt.figure()  
        plt.plot(Rp, Rpdf, '-b')
        plt.title(r'Graph of the pdf of $Z^\nu$ whose realizations are $z^\nu_p = i^\nu(G_{r_1},G_{r_2})$ with $p = (r_1,r_2)$ (blue solid)', fontweight='normal', fontsize=16)
        plt.xlabel(r'$z$', fontsize=16)
        plt.ylabel(r'$p_{Z^\nu}(z)$', fontsize=16)
        numfig = numfig + 1
        plt.savefig(f'figure_PARTITION_{numfig}_pdf_i-Gauss.png')
        plt.close()
   
    return INDEPGaussmax, meanINDEPGauss, stdINDEPGauss, gust, meanMaxINDEPGauss, stdMaxINDEPGauss, numfig

