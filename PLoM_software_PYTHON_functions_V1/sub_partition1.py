import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import time
import sys
from scipy.stats import norm, gaussian_kde
from sub_ksdensity2D import sub_ksdensity2D
from joblib import Parallel, delayed
from sub_partition2_constructing_GAUSS_reference import sub_partition2_constructing_GAUSS_reference
from sub_partition3_find_groups1 import sub_partition3_find_groups1
from sub_partition4_find_groups2 import sub_partition4_find_groups2
from sub_partition5_checking_independence_constructed_groups1 import sub_partition5_checking_independence_constructed_groups1
from sub_partition6_checking_independence_constructed_groups2 import sub_partition6_checking_independence_constructed_groups2
from sub_partition7_print_plot_groups1 import sub_partition7_print_plot_groups1
from sub_partition8_print_plot_groups2 import sub_partition8_print_plot_groups2
from sub_partition10_ksdensity_mult import sub_partition10_ksdensity_mult

def sub_partition1(nu, n_d, nref, MatReta_d, RINDEPref, SAVERANDstartPARTITION, ind_display_screen, ind_print, ind_plot, MatRHplot, 
                     MatRcoupleRHplot, ind_parallel):
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 24 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_partition1
    #  Subject      : optimal partition in independent groups of random vector H from its n_d independent realizations in MatReta_d(nu,n_d)
    #
    #  Publications: 
    #               [1] C. Soize, Optimal partition in terms of independent random vectors of any non-Gaussian vector defined by 
    #                      a set of realizations,SIAM-ASA Journal on Uncertainty Quantification, 
    #                      doi: 10.1137/16M1062223, 5(1), 176-211 (2017).                 
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                      Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #
    #  Function definition: Decomposition of
    #                       H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
    #                       H    = (H^1,...,H^j,...,H^ngroup) 
    #                       H^j  = (H_rj1,...,Hrjmj)         with j = 1,...,ngroup
    #                                                        with n1 + ... + nngroup = nu
    #
    #--- INPUTS
    #          nu                    : dimension of random vector H = (H_1, ... H_nu)
    #          n_d                   : number of points in the training set for H
    #          nref                  : first dimension of matrix RINDEPref(nref,1)
    #          MatReta_d(nu,n_d)     : n_d realizations of H
    #          RINDEPref(nref,1)     : contains the values of the mutual information for exploring the dependence of two components of H
    #                                  example: RINDEPref =(0.001:0.001:0.016)';  
    #          SAVERANDstartPARTITION: state of the random generator at the end of the PCA step
    #          ind_display_screen    : = 0 no display,            = 1 display
    #          ind_print             : = 0 no print,              = 1 print
    #          ind_plot              : = 0 no plot,               = 1 plot
    #          ind_parallel          : = 0 no parallel computing, = 1 parallel computing
    #          MatRHplot             : 1D array of components H_j of H for which the pdf are estimated and plotted 
    #                                  example 1: MatRHplot = np.array([1 2 5]); plot the 3 pdfs of components 1,2, and 5
    #                                  example 2: MatRHplot = np.array([]);      no plot
    #                                 
    #          MatRcoupleRHplot      : 2D array of pairs H_j - H_j' of components of H for which the joint pdf are estimated and plotted
    #                                  example 1: MatRcoupleRHplot = np.array([[1 2], [1 4] , [8 9]]);  plot the 3 joint pdfs 
    #                                                                                                   of pairs (1,2), (1,4), and (8,9) 
    #                                  example 2: MatRcoupleRHplot = np.array([])  no plot
    #
    #--- OUTPUTS
    #          ngroup                   : number of constructed independent groups  
    #          Igroup(ngroup)           : vector Igroup(ngroup), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)  
    #          MatIgroup(ngroup,mmax)   : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj 
    #                                     with mmax = max_j mj for j = 1, ... , ngroup
    #          SAVERANDendPARTITION     : state of the random generator at the end of the function
    #
    #--- COMMENTS about the internal variables
    #
    #         nedge: number of edges in the graph
    #         MatPrint(nedge,5) such that MatPrint(edge,:) = [edge r1 r2 INDEPr1r2 INDEPGaussRef]  
    #         INDEPGaussRef: Numerical criterion for testing the independence of two normalized Gaussian random variables HGauss_r1 and 
    #                        HGauss_r2 by using the MUTUAL INFORMATION criterion. The random generator for randn is reinitialized 
    #                        for avoiding some variations on the statistical fluctuations in the computation of INDEPGaussRef with respect
    #                        to the different cases analyzed with iexec_TEST. Note that the independent realizations used for constructing 
    #                        INDEPGaussRef can be dependent of all the other random quantities constructed before this one.

    if ind_display_screen == 1:
        print('--- beginning Task4_Partition')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ------ Task4_Partition \n ')
            fidlisting.write('      \n ')

    TimeStartPartition = time.time()

    #--- initializing the random generator at the value of the end of the PCA step 
    np.random.set_state(SAVERANDstartPARTITION)   

    #--- changing the name of nu,n_d, and MatReta for the partition functions                                           
    NKL      = nu
    nr       = n_d  
    MatRHexp = MatReta_d    # MatRHexp(NKL,nr), MatReta_d(nu,n_d)

    #--- initialization of the number of generated figures (plot)
    numfig = 0
    
    #--- checking the input parameters
    (nu_temp, n_d_temp) = MatReta_d.shape                  # MatReta_d(nu,n_d)  
    if nu_temp != nu or n_d_temp != n_d:
        raise ValueError('STOP1 in sub_partition1_main: the dimensions of 2D array MatReta_d(nu,n_d) are not corect')
    
    if RINDEPref.size >=1:           # checking if not empty
        if RINDEPref.ndim >= 2:      # ckecking if it is a 1D array
            raise ValueError('STOP2 in in sub_partition1_main: RINDEPref must be a 1D array')
    else:
        raise ValueError('STOP3 in in sub_partition1_main: RINDEPref cannot be empty')

    if MatRHplot.size >=1:           # checking if not empty
        if MatRHplot.ndim >= 2:      # ckecking if it is a 1D array MatRHplot  
            raise ValueError('STOP4 in in sub_partition1_main: MatRHplot must be a 1D array') 
        
    if np.max(MatRHplot) > NKL:
        raise ValueError('STOP5 in sub_partition1_main: a component in MatRHplot is greater than NKL')
    
    (ncoupleRHplot_temp, dim_temp) = MatRcoupleRHplot.shape  # MatRcoupleRHplot(:,2)
    if dim_temp != 2:
        raise ValueError('STOP6 in sub_partition1_main: the second dimension of the 2D array MatRcoupleRHplot must be 2')
    
    if np.max(MatRcoupleRHplot) > NKL:
        raise ValueError('STOP7 in sub_partition1_main: a component in MatRHplot is greater than NKL')
    
    if ncoupleRHplot_temp > NKL**2:
        raise ValueError('STOP8 in sub_partition1_main: the number of pairs in MatRHplot must be less than or equal to NKL**2')
    
    #==================================================================================================================================  
    #            construction of the Gauss reference for independent group
    #==================================================================================================================================  

    if ind_display_screen == 1:
        print('--- beginning the construction of the Gauss reference for independent group')

    MatRHexpGauss = np.random.randn(NKL,nr)  # (NKL,nr) matrix of the nr independent realizations of HGauss = (HGauss_1,...,HGauss_NKL)
                             
    #--- Constructing the optimal value INDEPopt such that the rate tau(INDEPref) is maximum
    (INDEPGaussmax,meanINDEPGauss,stdINDEPGauss,gust,meanMaxINDEPGauss,stdMaxINDEPGauss,numfig) = \
        sub_partition2_constructing_GAUSS_reference(NKL,nr,MatRHexpGauss,ind_plot,ind_parallel,numfig)
    
    if ind_display_screen == 1:
        print('--- end of the construction of the Gauss reference for independent group')

    #==================================================================================================================================  
    #           optimization loop for independent group
    #==================================================================================================================================  

    if ind_display_screen == 1:
        print('--- beginning the optimization loop for independent group')
    
    #--- sequential computing
    Rtau = np.zeros(nref)
    if ind_parallel == 0:
        for iref in range(nref):       
            INDEPref_value = RINDEPref[iref] 
            # Construction of groups corresponding to the value INDEPref
            (ngroup, Igroup, _, MatIgroup) = sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref_value,ind_parallel)      
            # Rate of independence of the constructed partition               
            tau = sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss)
            Rtau[iref] = tau

    #--- parallel computing
    if ind_parallel == 1:  

        def parallel_computation(iref,RINDEPref,NKL,nr,MatRHexp,MatRHexpGauss,ind_parallel):
            INDEPref_value = RINDEPref[iref]
            (ngroup, Igroup, _, MatIgroup) = sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref_value,ind_parallel)
            tau = sub_partition5_checking_independence_constructed_groups1(NKL,nr,ngroup,Igroup,MatIgroup,MatRHexp,MatRHexpGauss)
            return tau

        Rtau = Parallel(n_jobs=-1)(delayed(parallel_computation)(iref,RINDEPref,NKL,nr,MatRHexp,MatRHexpGauss,ind_parallel) for iref in range(nref))

    if ind_display_screen == 1:
        print('--- end of the optimization loop for independent group')

    if ind_plot == 1:
        plt.figure()
        plt.plot(RINDEPref, Rtau, '-ob')
        plt.title('Graph of the rate $\\tau (i_{\\rm{ref}})$ of mutual information\nfor the partition obtained with the level $i_{\\rm{ref}}$', fontsize=16)
        plt.xlabel('$i_{\\rm{ref}}$', fontsize=16)
        plt.ylabel('$\\tau (i_{\\rm{ref}})$', fontsize=16)
        numfig += 1
        plt.savefig(f'figure_PARTITION_{numfig}_tau.png')
        plt.close()

    #==================================================================================================================================  
    #           construction of the independent groups
    #==================================================================================================================================  

    if ind_display_screen == 1:
        print('--- beginning the construction of the independent groups')

    #--- Calculation of irefopt, tauopt, and INDEPopt 
    tauoptMin  = np.min(Rtau)
    irefoptMin = np.argmin(Rtau) + 1
    tauoptMax  = np.max(Rtau)
    irefoptMax = np.argmax(Rtau) + 1

    #--- General case: all the values of tau are positive or equal to zero and the optimal value corresponds to the maximum value of tau
    if tauoptMin >= 0 or ((tauoptMin < 0 and tauoptMax > 0) and (irefoptMin < irefoptMax)): 
        irefopt  = irefoptMax                                                            
        tauopt   = tauoptMax
        INDEPopt = RINDEPref[irefopt-1]

    #--- Particular case: there exits tau < 0; in this case all the components of H are independent and are Gaussian 
    if tauoptMax < 0 or ((tauoptMin < 0 and tauoptMax > 0) and (irefoptMin > irefoptMax)): 
        irefopt  = irefoptMin                                                                            
        tauopt   = tauoptMin
        INDEPopt = RINDEPref[irefopt-1]
    
    #--- Constructing the groups corresponding to the optimal value INDEPopt of INDEPref
    (ngroup, Igroup, _, MatIgroup, nedge, nindep, npair, MatPrintEdge, MatPrintIndep, MatPrintPair, RplotPair) = \
        sub_partition4_find_groups2(NKL, nr, MatRHexp, INDEPopt, ind_parallel)
    
    if ind_display_screen == 1:
        print('--- end of the construction of the independent groups')

    #==================================================================================================================================  
    #           print, plot, and checking
    #==================================================================================================================================                                     

    Indic = 0  # if Indic = 0 at the end of the loop, then Rtau(iref) < or = tauopt for all iref
    for iref in range(nref):
        if Rtau[iref] > tauopt:
            Indic = 1

    #--- Print and plot the groups
    if Indic == 0 and tauopt < 0:  # then H_1,...,H_nu are mutually independent and Gaussian
        ngroup = NKL 
        Igroup = np.ones(ngroup)
        MatIgroup = np.arange(1, ngroup+1) # 1D array MatIgroup(ngroup)
        numfig = sub_partition7_print_plot_groups1(INDEPopt, ngroup, Igroup, MatIgroup, npair, RplotPair, ind_print, ind_plot, numfig)
    else:  # then H_1,...,H_nu are mutually dependent and not Gaussian
        numfig = sub_partition8_print_plot_groups2(INDEPopt, ngroup, Igroup, MatIgroup, nedge, MatPrintEdge, nindep, MatPrintIndep, npair, 
                                                   MatPrintPair, RplotPair, ind_print, ind_plot, numfig)

    #--- Checking the independence of the constructed groups
    (INDEPGausscheck,INDEPcheck,tauopt) = sub_partition6_checking_independence_constructed_groups2(NKL,nr,ngroup,Igroup, 
                                                                                                  MatIgroup,MatRHexp,MatRHexpGauss)

    #--- Print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ') 
            fidlisting.write('      \n ') 
            fidlisting.write('--------------- SUMMARIZING THE OPTIMAL PARTITION CONSTRUCTED IN INDEPENDENT GROUPS\n')
            fidlisting.write('      \n ') 
            fidlisting.write('      \n ') 
            fidlisting.write(f'Optimal value INDEPopt of INDEPref used for constructing the optimal partition = {INDEPopt:.5f} \n')
            fidlisting.write('      \n ') 
            fidlisting.write(f'Optimal value tauopt of tau corresponding to the construction of the optimal partition = {tauopt:.5f} \n')
            fidlisting.write('      \n ') 
            fidlisting.write(f'INDEPGaussmax     = {INDEPGaussmax:.6f} \n')
            fidlisting.write(f'meanINDEPGauss    = {meanINDEPGauss:.6f} \n')
            fidlisting.write(f'stdINDEPGauss     = {stdINDEPGauss:.6f} \n')
            fidlisting.write(f'gust              = {gust:.6f} \n')
            fidlisting.write(f'meanMaxINDEPGauss = meanINDEPGauss + gust*stdINDEPGauss = {meanMaxINDEPGauss:.6f} \n')
            fidlisting.write(f'stdMaxINDEPGauss  = {stdMaxINDEPGauss:.6f} \n')
            fidlisting.write(f'number of groups  = {ngroup:7d} \n')
            fidlisting.write('      \n ') 
            fidlisting.write(f'maximum number of nodes in the largest group = {np.max(Igroup):7d} \n')  
            fidlisting.write(f'mutual information for the identified decomposition INDEPcheck  = {INDEPcheck:.6f} \n')  
            fidlisting.write(f'mutual information for a normalized Gaussian vector having the identified decomposition INDEPGausscheck = {INDEPGausscheck:.6f} \n')
            fidlisting.write('the groups are independent if INDEPcheck <= INDEPGausscheck')    
            fidlisting.write('      \n ')  

    #--- Plot pdf of H_j and joint pdf of H_j -  H_j' for the training dataset
    if ind_plot == 1:

        # Estimation and plot of the pdf of component H_j
        nRHplot = len(MatRHplot)  # MatRHplot(nRHplot)
        if nRHplot >= 1:
            npoint = 100
            for iplot in range(nRHplot):
                ih = MatRHplot[iplot]
                if ih <= NKL:           # if ih > NKL, component does not exist and plot is skipped
                    RHexp_ih  = MatRHexp[ih-1,:].T                     # MatRHexp(NKL,nr)
                    RHexp_ih  = np.squeeze(RHexp_ih)   
                    MIN       = np.min(RHexp_ih) - 2
                    MAX       = np.max(RHexp_ih) + 2
                    #--- For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
                    sigma_ih  = np.std(RHexp_ih,ddof=1)  
                    bw_ih     = 1.0592*sigma_ih*n_d**(-1/5)       # # Silverman's rule in 1D data
                    kde       = gaussian_kde(RHexp_ih,bw_method=bw_ih/sigma_ih)
                    Rh_exp    = np.linspace(MIN,MAX,npoint)  
                    Rpdf_exp  = kde.evaluate(Rh_exp)
                    Rgauss_ih = norm.pdf(Rh_exp, 0, 1)
                    plt.figure()
                    plt.plot(Rh_exp, Rpdf_exp, '-k', Rh_exp, Rgauss_ih, '--k')
                    plt.title(f'Graph of the training pdf (solid black) and its\nGaussian approximation (dashed black) for $H_{{{ih}}}$', fontsize=16)
                    plt.xlabel(f'$\\eta_{{{ih}}}$', fontsize=16)
                    plt.ylabel(f'$p_{{H_{{{ih}}}}}(\\eta_{{{ih}}})$', fontsize=16)
                    numfig = numfig + 1
                    plt.savefig(f'figure_PARTITION_{numfig}_pdf_H{ih}.png')
                    plt.close()

        # Estimation and plot of the joint pdf of components H_j et H_j'
        ncoupleRHplot = MatRcoupleRHplot.shape[0]
        if ncoupleRHplot >= 1:
            npoint = 100
            for iplot in range(ncoupleRHplot):
                ih = MatRcoupleRHplot[iplot,0]
                jh = MatRcoupleRHplot[iplot,1]
                if ih <= NKL and jh <= NKL:                                  # if ih > NKL or jh > NKL, component does not exist and plot is skipped
                    MINih, MAXih = np.min(MatRHexp[ih-1, :]), np.max(MatRHexp[ih-1, :])  # MatRHexp(NKL,nr)
                    MINjh, MAXjh = np.min(MatRHexp[jh-1, :]), np.max(MatRHexp[jh-1, :])
                    coeff = 0.2
                    deltaih = MAXih - MINih
                    deltajh = MAXjh - MINjh
                    MINih = MINih - coeff*deltaih
                    MAXih = MAXih + coeff*deltaih
                    MINjh = MINjh - coeff*deltajh
                    MAXjh = MAXjh + coeff*deltajh

                    # Compute the joint probability density function
                    # For the joint pdf (dimension 2), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
                    MatRx, MatRy = np.meshgrid(np.linspace(MINih, MAXih, npoint), np.linspace(MINjh, MAXjh, npoint))
                    R_ih         = MatRHexp[ih-1, :].T                                    # R_ih(nr,1)
                    R_jh         = MatRHexp[jh-1, :].T                                    # R_jh(nr,1)
                    MatRHexpT    = np.vstack([R_ih,R_jh]).T                               # MatRHexpT(nr,2)
                    MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
                    N            = n_d                                                    # Number of realizations
                    sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
                    sigma_jh     = np.std(R_jh,ddof=1)  
                    bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman's rule in 2D data for component ih
                    bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman's rule in 2D data for component jh
                    # Kernel density estimation using gaussian_kde in Python
                    # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
                    Rpdf    = sub_ksdensity2D(nr,MatRHexpT,npoint,MatRpts,bw_ih,bw_jh)    # Rpdf(npoint*npoint)
                    MatRpdf = Rpdf.reshape(npoint,npoint)                                 # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)

                    # Plot the contours of the joint PDF
                    plt.figure()
                    plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
                    plt.xlim([MINih, MAXih])
                    plt.ylim([MINjh, MAXjh])
                    plt.colorbar()
                    plt.xlabel(f'$\\eta_{{{ih}}}$', fontsize=16)
                    plt.ylabel(f'$\\eta_{{{jh}}}$', fontsize=16)
                    plt.title(f'Training joint pdf of $H_{{{ih}}}$ with $H_{{{jh}}}$', fontsize=16)
                    numfig = numfig + 1
                    plt.savefig(f'figure_PARTITION_{numfig}_H{ih}_H{jh}.png')
                    plt.close()

    SAVERANDendPARTITION  = np.random.get_state()          
    ElapsedTimePartition  = time.time() - TimeStartPartition

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')                                                                
            fidlisting.write('      \n ') 
            fidlisting.write('-------   Elapsed time for Task4_Partition \n ')
            fidlisting.write('      \n ') 
            fidlisting.write(f'Elapsed Time   =  {ElapsedTimePartition:.2f}\n')
            fidlisting.write('      \n ')  

    if ind_display_screen == 1:
        print('--- end Task4_Partition')

    Igroup    = Igroup.astype(int)
    MatIgroup = MatIgroup.astype(int)

    return ngroup, Igroup, MatIgroup, SAVERANDendPARTITION
