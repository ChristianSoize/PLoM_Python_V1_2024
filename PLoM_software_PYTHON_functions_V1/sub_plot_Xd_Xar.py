import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import time
import sys
from scipy.stats import gaussian_kde
from sub_PCAback import sub_PCAback
from sub_scalingBack import sub_scalingBack
from sub_ksdensity2D import sub_ksdensity2D

def sub_plot_Xd_Xar(n_x, n_q, n_w, n_d, n_ar, nu, MatRxx_d, MatRx_d, MatReta_ar, RmuPCA, MatRVectPCA, Indx_real, Indx_pos, nx_obs,
                    Indx_obs, ind_scaling, Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log, Ralpha_scale_log,
                    MatRplotSamples, MatRplotClouds, MatRplotPDF, MatRplotPDF2D, ind_display_screen, ind_print):

    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel,  une 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_plot_Xd_Xar
    #  Subject      : Plot a subset of MatRqq_obs(nq_obs,:) and MatRww_obs(nw_obs,:) for 
    #                 computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar
    #
    #--- INPUTS 
    #
    #     n_x                         : dimension of random vectors XX_ar  (unscaled) and X_ar (scaled)  
    #     n_q                         : dimension of random vector QQ (unscaled quantity of interest)  1 <= n_q    
    #     n_w                         : dimension of random vector WW (unscaled control variable) with 1 <= n_w  
    #     n_d                         : number of points in the training set for XX_d and X_d  
    #     n_ar                        : number of points in the learning set for H_ar, X_obs, and XX_obs
    #     nu                          : order of the PCA reduction, which is the dimension of H_ar   
    #     MatRxx_d(n_x,n_d)           : n_d realizations of XX_d (unscaled)
    #     MatRx_d(n_x,n_d)            : n_d realizations of X_d (scaled)
    #     MatReta_ar(nu,n_ar)         : n_ar realizations of H_ar 
    #     RmuPCA(nu)                  : vector of PCA eigenvalues in descending order
    #     MatRVectPCA(n_x,nu)         : matrix of the PCA eigenvectors associated to the eigenvalues loaded in RmuPCA
    #     Indx_real(nbreal)           : nbreal component numbers of XX_ar that are real (positive, negative, or zero) 
    #     Indx_pos(nbpos)             : nbpos component numbers of XX_ar that are strictly positive 
    #     nx_obs                      : dimension of random vectors XX_obs (unscaled) and X_obs (scaled) (extracted from X_ar)  
    #     Indx_obs(nx_obs)            : nx_obs component numbers of X_ar and XX_ar that are observed with nx_obs <= n_x
    #     ind_scaling                 : = 0 no scaling
    #                                 : = 1 scaling
    #     Rbeta_scale_real(nbreal)    : loaded if nbreal >= 1 or = [] if nbreal  = 0               
    #     Ralpha_scale_real(nbreal)   : loaded if nbreal >= 1 or = [] if nbreal  = 0    
    #     Rbeta_scale_log(nbpos)      : loaded if nbpos >= 1  or = [] if nbpos = 0                 
    #     Ralpha_scale_log(nbpos)     : loaded if nbpos >= 1  or = [] if nbpos = 0   

    #---- EXAMPLE OF DATA FOR EXPLAINING THE PLOT-DATA STRUCTURE USING MATLAB NOTATION
    #     The components of XX given for the plots must be a subset of the observed components XX_obs, that is to say 
    #     must belong to the set of components declared in mainWorkflow_Data_generation1.m in the array
    #     In this example, using Matlab notation, one has:
    #     Indx_obs = [Indq_obs
    #                 n_q + Indw_obs]
    #     It is assumed that n_q = 15, n_w = 4 
    #     Indq_obs = [2 4 9 13 14]', and Indw_obs = [2 3 4]'
    #     then, Indx_obs = [2 4 9 13 14  17 18 19]'

    #--- Plot data (illustration in examples using the above data)
    #
    #    MatRplotSamples = np.array([4, 13, 18])  1D array of components of XX for which the realizations are plotted 
    #                                             Example 1: plot samples of XX4,XX13,XX18 that is QQ4,QQ13,WW3 and nbplotSamples = 3. 
    #                                                Example 2: MatRplotSamples = np.array([]); no plot, nbplotSamples = 0
    #    MatRplotClouds = np.array([              2D array of components of XX for which the clouds are plotted 
    #                              [9, 17, 19],      Example 1: plot cloud of components XX9,XX17,XX19 that is QQ9,WW2,WW4.
    #                              [13, 14, 17]                 plot cloud of components XX13,XX14,XX17 that is QQ13,WW14,WW2.
    #                              ])                           nbplotClouds = 2. 
    #                                               Example 2: MatRplotHClouds = np.array([]); no plot, nbplotClouds = 0
    #    MatRplotPDF = np.array([4, 13, 18])      1D array of components of XX for which the pdfs are plotted 
    #                                                Example 1: plot the pdf of XX4,XX13,XX18, that is QQ4,QQ13,WW3 with nbplotPDF = 3. 
    #                                                Example 2: MatRplotPDF = np.array([]); no plot, nbplotPDF = 0
    #    MatRplotPDF2D = np.array([               2D array of components of XX for which the joint pdfs are plotted 
    #                             [4, 13],           Example 1: plot the joint pdf of XX4-XX13 that is QQ4-QQ13.
    #                             [13, 18]                      plot the joint pdf of XX13-XX18 that is QQ13-WW3
    #                             ])                            nbplotHpdf2D = 2
    #                                                Example 2: MatRplotPDF2D = np.array([]); no plot, nbplotHpdf2D = 0    
    #
    #     ind_display_screen : = 0 no display, = 1 display
    #     ind_print          : = 0 no print,   = 1 print
    #
    #--- INTERNAL PARAMETERS
    #          nu            : dimension of random vector H = (H_1, ... H_nu)
    #          nbMC          : number of realizations of (nu,n_d)-valued random matrix [H_ar]  

    if ind_display_screen == 1:
        print('--- beginning Task12_PlotXdXar')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write(' ------ Task12_PlotXdXar \n ')
            fidlisting.write('\n ')

    TimeStartPlotXdXar = time.time()
    numfig = 0

    #--- Checking parameters and data
    if n_x <= 0:
        raise ValueError('STOP1 in sub_plot_Xd_Xar: n_x <= 0')
    if n_q <= 0 or n_w < 0:
        raise ValueError('STOP2 in sub_plot_Xd_Xar: n_q <= 0 or n_w < 0')
    nxtemp = n_q + n_w  # dimension of random vector XX = (QQ,WW)
    if nxtemp != n_x:
        raise ValueError('STOP3 in sub_plot_Xd_Xar: n_x not equal to n_q + n_w')
    if n_d <= 0:
        raise ValueError('STOP4 in sub_plot_Xd_Xar: n_d <= 0')
    if n_ar <= 0:
        raise ValueError('STOP5 in sub_plot_Xd_Xar: n_ar <= 0')
    if nu <= 0 or nu >= n_d:
        raise ValueError('STOP6 in sub_plot_Xd_Xar: nu <= 0 or nu >= n_d')
    
    n1temp, n2temp = MatRxx_d.shape
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP7 in sub_plot_Xd_Xar: dimension error in matrix MatRxx_d(n_x,n_d)')
    
    n1temp, n2temp = MatRx_d.shape
    if n1temp != n_x or n2temp != n_d:
        raise ValueError('STOP8 in sub_plot_Xd_Xar: dimension error in matrix MatRx_d(n_x,n_d)')
    
    n1temp, n2temp = MatReta_ar.shape
    if n1temp != nu or n2temp != n_ar:
        raise ValueError('STOP9 in sub_plot_Xd_Xar: dimension error in matrix MatReta_ar(nu,n_ar)')
    
    n1temp = len(RmuPCA)
    if n1temp != nu: 
        raise ValueError('STOP10 in sub_plot_Xd_Xar: dimension error in matrix RmuPCA(nu)')
    
    n1temp, n2temp = MatRVectPCA.shape
    if n1temp != n_x or n2temp != nu:
        raise ValueError('STOP11 in sub_plot_Xd_Xar: dimension error in matrix MatRVectPCA(n_x,nu)')

    if Indx_real.size >=1:           # checking if not empty
        if Indx_real.ndim >= 2:      # ckecking if it is a 1D array
            raise ValueError('STOP12 in sub_plot_Xd_Xar: Indx_real must be a 1D array')
    nbreal = len(Indx_real)    

    if Indx_pos.size >=1:           # checking if not empty
        if Indx_pos.ndim >= 2:      # ckecking if it is a 1D array
            raise ValueError('STOP13 in sub_plot_Xd_Xar: Indx_pos must be a 1D array')
    nbpos = len(Indx_pos)
    
    nxtemp = nbreal + nbpos
    if nxtemp != n_x:
        raise ValueError('STOP14 in sub_plot_Xd_Xar: n_x not equal to nreal + nbpos')

    if nx_obs <= 0:
        raise ValueError('STOP15 in sub_plot_Xd_Xar: nx_obs <= 0')
    if Indx_obs.size >=1:           # checking if not empty
        if Indx_obs.ndim >= 2:      # ckecking if it is a 1D array
            raise ValueError('STOP16 in sub_plot_Xd_Xar: Indx_obs must be a 1D array')
    n1temp = len(Indx_obs)
    if n1temp != nx_obs:
        raise ValueError('STOP17 in sub_plot_Xd_Xar: dimension error in matrix Indx_obs(nx_obs,1)')
    
    if ind_scaling != 0 and ind_scaling != 1:
        raise ValueError('STOP18 in sub_plot_Xd_Xar: ind_scaling must be equal to 0 or to 1')
    
    if nbreal >= 1:
        if Rbeta_scale_real.size >=1:           # checking if not empty
            if Rbeta_scale_real.ndim >= 2:      # ckecking if it is a 1D array
                raise ValueError('STOP19 in sub_plot_Xd_Xar: Rbeta_scale_real must be a 1D array')
        n1temp = len(Rbeta_scale_real)
        if n1temp != nbreal: 
            raise ValueError('STOP20 in sub_plot_Xd_Xar: dimension error in matrix Rbeta_scale_real(nbreal)')
        if Ralpha_scale_real.size >=1:           # checking if not empty
            if Ralpha_scale_real.ndim >= 2:      # ckecking if it is a 1D array
                raise ValueError('STOP21 in sub_plot_Xd_Xar: Ralpha_scale_real must be a 1D array')
        n1temp = len(Ralpha_scale_real)
        if n1temp != nbreal: 
            raise ValueError('STOP22 in sub_plot_Xd_Xar: dimension error in matrix Ralpha_scale_real(nbreal)')
        
    if nbpos >= 1:
        if Rbeta_scale_log.size >=1:           # checking if not empty
            if Rbeta_scale_log.ndim >= 2:      # ckecking if it is a 1D array
                raise ValueError('STOP23 in sub_plot_Xd_Xar: Rbeta_scale_log must be a 1D array')
        n1temp = len(Rbeta_scale_log)
        if n1temp != nbpos: 
            raise ValueError('STOP24 in sub_plot_Xd_Xar: dimension error in matrix Rbeta_scale_log(nbpos)')
        if Ralpha_scale_log.size >=1:           # checking if not empty
            if Ralpha_scale_log.ndim >= 2:      # ckecking if it is a 1D array
                raise ValueError('STOP25 in sub_plot_Xd_Xar: Ralpha_scale_log must be a 1D array')
        n1temp = len(Ralpha_scale_log)
        if n1temp != nbpos: 
            raise ValueError('STOP26 in sub_plot_Xd_Xar: dimension error in matrix Ralpha_scale_log(nbpos)')

    if MatRplotSamples.size >=1:           # checking if not empty
        if MatRplotSamples.ndim >= 2:      # ckecking if it is a 1D array
            raise ValueError('STOP27 in sub_plot_Xd_Xar: MatRplotSamples must be a 1D array')
    if MatRplotClouds.size >=1:           # checking if not empty
        if MatRplotClouds.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP28 in sub_plot_Xd_Xar: MatRplotClouds must be a 2D array')    
    if MatRplotPDF.size >=1:              # checking if not empty
        if MatRplotPDF.ndim >= 2:         # ckecking if it is a 1D array
            raise ValueError('STOP29 in sub_plot_Xd_Xar: MatRplotPDF must be a 1D array')
    if MatRplotPDF2D.size >=1:           # checking if not empty
        if MatRplotPDF2D.ndim != 2:      # ckecking if it is a 2D array
            raise ValueError('STOP30 in sub_plot_Xd_Xar: MatRplotPDF2D must be a 2D array')    
    
    nbplotSamples = len(MatRplotSamples)     # MatRplotSamples(nbplotSamples)
    nbplotClouds  = MatRplotClouds.shape[0]  # MatRplotClouds(nbplotHClouds,3)
    nbplotPDF     = len(MatRplotPDF)         # MatRplotPDF(nbplotPDF)
    nbplotPDF2D   = MatRplotPDF2D.shape[0]   # MatRplotPDF2D(nbplotPDF2D,2)

    if nbplotSamples >= 1:
        if any(MatRplotSamples < 1) or any(MatRplotSamples > n_x):
            raise ValueError('STOP31 in sub_plot_Xd_Xar: at least one integer in MatRplotSamples is not within range [1,n_x]')
        isContained = all(np.isin(MatRplotSamples, Indx_obs))
        if not isContained:
            raise ValueError('STOP32 in sub_plot_Xd_Xar: one or more integers in MatRplotSamples do not belong to Indx_obs')
        
    if nbplotClouds >= 1:
        n2temp = MatRplotClouds.shape[1]
        if n2temp != 3:
            raise ValueError('STOP33 in sub_plot_Xd_Xar: the second dimension of MatRplotClouds must be equal to 3')
        if any(MatRplotClouds.flatten() < 1) or any(MatRplotClouds.flatten() > n_x):
            raise ValueError('STOP34 in sub_plot_Xd_Xar: at least one integer of MatRplotClouds is not within range [1,n_x]')
        isContained = all(np.isin(MatRplotClouds.flatten(), Indx_obs))
        if not isContained:
            raise ValueError('STOP35 in sub_plot_Xd_Xar: one or more integers in MatRplotClouds do not belong to Indx_obs')
        
    if nbplotPDF >= 1:
        if any(MatRplotPDF < 1) or any(MatRplotPDF > n_x):
            raise ValueError('STOP36 in sub_plot_Xd_Xar: at least one integer in MatRplotPDF is not within range [1,n_x]')
        isContained = all(np.isin(MatRplotPDF, Indx_obs))
        if not isContained:
            raise ValueError('STOP37 in sub_plot_Xd_Xar: one or more integers in MatRplotPDF do not belong to Indx_obs')
        
    if nbplotPDF2D >= 1:
        n2temp = MatRplotPDF2D.shape[1]
        if n2temp != 2:
            raise ValueError('STOP38 in sub_plot_Xd_Xar: the second dimension of MatRplotPDF2D must be equal to 2')
        if any(MatRplotPDF2D.flatten() < 1) or any(MatRplotPDF2D.flatten() > n_x):
            raise ValueError('STOP39 in sub_plot_Xd_Xar: at least one integer in MatRplotPDF2D is not within range [1,n_x]')
        isContained = all(np.isin(MatRplotPDF2D.flatten(), Indx_obs))
        if not isContained:
            raise ValueError('STOP40 in sub_plot_Xd_Xar: one or more integers in MatRplotPDF2D do not belong to Indx_obs')

    #--- PCA back: MatRx_obs(nx_obs,n_ar)
    MatRx_obs = sub_PCAback(n_x, n_d, nu, n_ar, nx_obs, MatRx_d, MatReta_ar, Indx_obs, RmuPCA, MatRVectPCA, 
                            ind_display_screen, ind_print)
    
    #--- Scaling back: MatRxx_obs(nx_obs,n_ar)
    MatRxx_obs = sub_scalingBack(nx_obs, n_x, n_ar, MatRx_obs, Indx_real, Indx_pos, Indx_obs, Rbeta_scale_real, Ralpha_scale_real,
                                 Rbeta_scale_log, Ralpha_scale_log, ind_display_screen, ind_print, ind_scaling)

    #--- Plot the 2D graphs ell --> XX_{ar,ih}^ell    
    if nbplotSamples >= 1:
        for iplot in range(nbplotSamples):
            ih = MatRplotSamples[iplot]
            MatRih_ar = MatRxx_obs[Indx_obs - 1 == (ih - 1), :]
            ihq = 0
            ihw = 0
            if ih <= n_q:
                ihq = ih
            if ih > n_q:
                ihw = ih - n_q
            h = plt.figure()
            plt.plot(range(n_ar), MatRih_ar[0, :], 'b-')
            if ihq >= 1:
                plt.title(f'$Q_{{\\rm ar,{ihq}}}$ with $n_{{\\rm ar}} = {n_ar}$', fontsize=16)
                plt.xlabel(r'$\ell$', fontsize=16)
                plt.ylabel(f'$q^\\ell_{{\\rm ar,{ihq}}}$', fontsize=16)
                numfig = numfig + 1
                h.savefig(f'figure_PlotX{numfig}_qar{ihq}.png')
                plt.close(h)
            if ihw >= 1:
                plt.title(f'$W_{{\\rm ar,{ihw}}}$ with $n_{{\\rm ar}} = {n_ar}$', fontsize=16)
                plt.xlabel(r'$\ell$', fontsize=16)
                plt.ylabel(f'$w^\\ell_{{\\rm ar,{ihw}}}$', fontsize=16)
                numfig = numfig + 1
                h.savefig(f'figure_PlotX{numfig}_war{ihw}.png')
                plt.close(h)

    #--- Plot the 3D clouds ell --> XX_{d,ih}^ell,XX_{d,jh}^ell,XX_{d,kh}^ell)  and  ell --> (XX_{ar,ih}^ell,XX_{ar,jh}^ell,XX_{ar,kh}^ell)
    if nbplotClouds >= 1:
        for iplot in range(nbplotClouds):
            ih, jh, kh = MatRplotClouds[iplot]
            MatRih_ar = MatRxx_obs[Indx_obs - 1 == (ih - 1), :]
            MatRjh_ar = MatRxx_obs[Indx_obs - 1 == (jh - 1), :]
            MatRkh_ar = MatRxx_obs[Indx_obs - 1 == (kh - 1), :]
            MINih = np.min(MatRih_ar[0, :])
            MAXih = np.max(MatRih_ar[0, :])
            MINjh = np.min(MatRjh_ar[0, :])
            MAXjh = np.max(MatRjh_ar[0, :])
            MINkh = np.min(MatRkh_ar[0, :])
            MAXkh = np.max(MatRkh_ar[0, :])
            if MINih >= 0: MINih = 0.6 * MINih
            if MINih < 0: MINih = 1.4 * MINih
            if MAXih >= 0: MAXih = 1.4 * MAXih
            if MAXih < 0: MAXih = 0.6 * MAXih
            if MINjh >= 0: MINjh = 0.6 * MINjh
            if MINjh < 0: MINjh = 1.4 * MINjh
            if MAXjh >= 0: MAXjh = 1.4 * MAXjh
            if MAXjh < 0: MAXjh = 0.6 * MAXjh
            if MINkh >= 0: MINkh = 0.6 * MINkh
            if MINkh < 0: MINkh = 1.4 * MINkh
            if MAXkh >= 0: MAXkh = 1.4 * MAXkh
            if MAXkh < 0: MAXkh = 0.6 * MAXkh
            MatRih_d = MatRxx_d[ih - 1, :]
            MatRjh_d = MatRxx_d[jh - 1, :]
            MatRkh_d = MatRxx_d[kh - 1, :]
            ihq, jhq, khq, ihw, jhw, khw = 0, 0, 0, 0, 0, 0
            if ih <= n_q:
                ihq = ih
            if ih > n_q:
                ihw = ih - n_q
            if jh <= n_q:
                jhq = jh
            if jh > n_q:
                jhw = jh - n_q
            if kh <= n_q:
                khq = kh
            if kh > n_q:
                khw = kh - n_q
            h = plt.figure()
            ax = h.add_subplot(111, projection='3d')
            ax.set_xlim([MINih, MAXih])
            ax.set_ylim([MINjh, MAXjh])
            ax.set_zlim([MINkh, MAXkh])
            ax.scatter(MatRih_d, MatRjh_d, MatRkh_d, s=2, c='b', marker='x')
            ax.scatter(MatRih_ar, MatRjh_ar, MatRkh_ar, s=2, c='r', marker='o')
            plt.title(f'clouds $X_d$ with $n_d = {n_d}$ and $X_{{\\rm{{ar}}}}$ with $n_{{\\rm{{ar}}}} = {n_ar}$', fontsize=16)
            if ihq and jhq and khq:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                ax.set_zlabel(f'$q_{{{khq}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_Q{ihq}_Q{jhq}_Q{khq}.png')
                plt.close(h)
            if ihq and jhq and khw:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                ax.set_zlabel(f'$w_{{{khw}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_Q{ihq}_Q{jhq}_W{khw}.png')
                plt.close(h)
            if ihq and jhw and khq:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                ax.set_zlabel(f'$q_{{{khq}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_Q{ihq}_W{jhw}_Q{khq}.png')
                plt.close(h)
            if ihw and jhq and khq:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                ax.set_zlabel(f'$q_{{{khq}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_W{ihw}_Q{jhq}_Q{khq}.png')
                plt.close(h)
            if ihq and jhw and khw:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                ax.set_zlabel(f'$w_{{{khw}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_Q{ihq}_W{jhw}_W{khw}.png')
                plt.close(h)
            if ihw and jhq and khw:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                ax.set_zlabel(f'$w_{{{khw}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_W{ihw}_Q{jhq}_W{khw}.png')
                plt.close(h)
            if ihw and jhw and khq:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                ax.set_zlabel(f'$q_{{{khq}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_W{ihw}_W{jhw}_Q{khq}.png')
                plt.close(h)
            if ihw and jhw and khw:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                ax.set_zlabel(f'$w_{{{khw}}}$', fontsize=16)
                numfig += 1
                h.savefig(f'figure_PlotX{numfig}_clouds_W{ihw}_W{jhw}_W{khw}.png')
                plt.close(h)

    #--- Generation and plot the pdf of XX_{d,ih} and XX_{ar,ih}
    if nbplotPDF >= 1:
        npoint = 200  # number of points for the pdf plot using ksdensity

        for iplot in range(nbplotPDF):
            ih     = MatRplotPDF[iplot]                         # MatRplotPDF(nbplotPDF) 
            Rd_ih  = MatRxx_d[ih-1,:].T;                        #  Rd_ih_d(n_d,1),MatRxx_d(n_x,n_d)
            Rd_ih  = np.squeeze(Rd_ih) 
            Rar_ih = MatRxx_obs[Indx_obs - 1 == (ih - 1), :].T  # Rar_ih(n_ar,1), Indx_obs(nx_obs,1), MatRxx_obs(nx_obs,n_ar)
            Rar_ih = np.squeeze(Rar_ih)         
            MIN    = min(np.min(Rd_ih), np.min(Rar_ih)) - 0.5
            MAX    = max(np.max(Rd_ih), np.max(Rar_ih)) + 0.5

            #--- For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
           
            sigma_ih  = np.std(Rd_ih,ddof=1)              
            bw_ih     = 1.0592*sigma_ih*n_d**(-1/5)                  # Sylverman bandwidth in 1D
            kde       = gaussian_kde(Rd_ih,bw_method=bw_ih/sigma_ih)
            Rh_d      = np.linspace(MIN,MAX,npoint)                         
            Rpdf_d    = kde.evaluate(Rh_d)
            
            sigma_ih  = np.std(Rar_ih,ddof=1)  
            bw_ih     = 1.0592*sigma_ih*n_ar**(-1/5)                # Sylverman bandwidth in 1D
            kde       = gaussian_kde(Rar_ih,bw_method=bw_ih/sigma_ih)
            Rh_ar     = np.linspace(MIN,MAX,npoint)
            Rpdf_ar   = kde.evaluate(Rh_ar)
            
            plt.figure()
            plt.plot(Rh_d, Rpdf_d, 'k-', label=f'$p_{{H_{{d,{ih}}}}}$',linewidth=0.5)
            plt.plot(Rh_ar, Rpdf_ar, 'b-', label=f'$p_{{H_{{ar,{ih}}}}}$',linewidth=1)
            
            ihq = 0
            ihw = 0
            if ih <= n_q:
                ihq = ih
            if ih > n_q:
                ihw = ih - n_q

            if ihq >= 1:
                plt.title(f'$p_{{Q_{{\\rm d,{ihq}}}}}$ (black thin) with $n_d = {n_d}$\n'
                          f'$p_{{Q_{{\\rm ar,{ihq}}}}}$ (blue thick) with $n_{{\\rm ar}} = {n_ar}$', fontsize=16, fontweight='normal')
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$p_{{Q_{{{ihq}}}}}(q_{{{ihq}}})$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_pdf_Qd_Qar{ihq}.png')
                plt.close(h)

            if ihw >= 1:
                plt.title(f'$p_{{W_{{\\rm d,{ihw}}}}}$ (black thin) with $n_d = {n_d}$\n'
                          f'$p_{{W_{{\\rm ar,{ihw}}}}}$ (blue thick) with $n_{{\\rm ar}} = {n_ar}$', fontsize=16, fontweight='normal')
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$p_{{W_{{{ihw}}}}}(w_{{{ihw}}})$', fontsize=16)
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_pdf_Wd_War{ihw}.png')
                plt.close(h)

    #--- Generation and plot the joint pdf of (XX_{d,ih},XX_{d,jh})
    if nbplotPDF2D >= 1:
        npoint = 100
        for iplot in range(nbplotPDF2D):
            ih = MatRplotPDF2D[iplot, 0]  # MatRplotPDF2D(nbplotPDF2D,2)
            jh = MatRplotPDF2D[iplot, 1]
            MatRih_d = MatRxx_d[ih - 1, :]  # MatRih_d(1,n_d), MatRxx_d(n_x,n_d)
            MatRjh_d = MatRxx_d[jh - 1, :]  # MatRjh_d(1,n_d), MatRxx_d(n_x,n_d)
            MINih = MatRih_d.min()
            MAXih = MatRih_d.max()
            MINjh = MatRjh_d.min()
            MAXjh = MatRjh_d.max()
            coeff = 0.2
            deltaih = MAXih - MINih
            deltajh = MAXjh - MINjh
            MINih = MINih - coeff*deltaih;
            MAXih = MAXih + coeff*deltaih;
            MINjh = MINjh - coeff*deltajh;
            MAXjh = MAXjh + coeff*deltajh;

            # Compute the joint probability density function
            # For the joint pdf (dimension 2), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
            MatRx, MatRy = np.meshgrid(np.linspace(MINih, MAXih, npoint), np.linspace(MINjh, MAXjh, npoint))
            R_ih         = MatRxx_d[ih - 1, :].T                                  # R_ih(n_d,1)           
            R_jh         = MatRxx_d[jh - 1, :].T                                  # R_jh(n_d,1)
            MatRxxdT     = np.vstack([R_ih,R_jh]).T                               # MatRxxdT(n_d,2)
            MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
            N            = n_d                                                    # Number of realizations
            sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
            sigma_jh     = np.std(R_jh,ddof=1)                                    # Standard deviation of component jh
            bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman bandwidth in 2D
            bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman bandwidth in 2D
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(n_d,MatRxxdT,npoint,MatRpts,bw_ih,bw_jh)   # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint,npoint)   

            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()

            ihq = 0
            jhq = 0
            ihw = 0
            jhw = 0
            if ih <= n_q:
                ihq = ih
            if ih > n_q:
                ihw = ih - n_q
            if jh <= n_q:
                jhq = jh
            if jh > n_q:
                jhw = jh - n_q
            
            if ihq >= 1 and jhq >= 1:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                plt.title(f'Joint pdf of $Q_{{\\rm{{d}},{ihq}}}$ with $Q_{{\\rm{{d}},{jhq}}}$ for $n_d = {n_d}$', fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Qd{ihq}_Qd{jhq}.png')
                plt.close(h)
            if ihq >= 1 and jhw >= 1:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                plt.title(f'Joint pdf of $Q_{{\\rm{{d}},{ihq}}}$ with $W_{{\\rm{{d}},{jhw}}}$ for $n_d = {n_d}$', fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Qd{ihq}_Wd{jhw}.png')
                plt.close(h)
            if ihw >= 1 and jhq >= 1:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                plt.title(f'Joint pdf of $W_{{\\rm{{d}},{ihw}}}$ with $Q_{{\\rm{{d}},{jhq}}}$ for $n_d = {n_d}$', fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Wd{ihw}_Qd{jhq}.png')
                plt.close(h)
            if ihw >= 1 and jhw >= 1:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                plt.title(f'Joint pdf of $W_{{\\rm{{d}},{ihw}}}$ with $W_{{\\rm{{d}},{jhw}}}$ for $n_d = {n_d}$', fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Wd{ihw}_Wd{jhw}.png')
                plt.close(h)    
    
    #--- Generation and plot the joint pdf of (XX_{ar,ih},XX_{ar,ih})
    if nbplotPDF2D >= 1:
        npoint = 100
        for iplot in range(nbplotPDF2D):
            ih = MatRplotPDF2D[iplot, 0]  # MatRplotPDF2D(nbplotPDF2D,2)
            jh = MatRplotPDF2D[iplot, 1]
            MatRih_ar = MatRxx_obs[Indx_obs - 1 == (ih - 1), :]  # MatRih_ar(1,n_ar), Indx_obs(nx_obs,1), MatRxx_obs(nx_obs,n_ar)
            MatRjh_ar = MatRxx_obs[Indx_obs - 1 == (jh - 1), :]  # MatRjh_ar(1,n_ar), MatRxx_obs(nx_obs,n_ar)
            MINih = MatRih_ar.min()
            MAXih = MatRih_ar.max()
            MINjh = MatRjh_ar.min()
            MAXjh = MatRjh_ar.max()
            coeff = 0.2
            deltaih = MAXih - MINih
            deltajh = MAXjh - MINjh
            MINih = MINih - coeff*deltaih;
            MAXih = MAXih + coeff*deltaih;
            MINjh = MINjh - coeff*deltajh;
            MAXjh = MAXjh + coeff*deltajh;

            # Compute the joint probability density function
            # For the joint pdf (dimension 2), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
            MatRx, MatRy = np.meshgrid(np.linspace(MINih, MAXih, npoint), np.linspace(MINjh, MAXjh, npoint))
            R_ih         = MatRxx_obs[Indx_obs - 1 == (ih - 1), :].T  # R_ih(n_ar,1), Indx_obs(nx_obs,1), MatRxx_obs(nx_obs,n_ar)
            R_jh         = MatRxx_obs[Indx_obs - 1 == (jh - 1), :].T  # R_jh(n_ar,1), Indx_obs(nx_obs,1), MatRxx_obs(nx_obs,n_ar)
            R_ih         = np.squeeze(R_ih) 
            R_jh         = np.squeeze(R_jh)
            MatRxxarT    = np.vstack([R_ih,R_jh]).T                               # MatRxxarT(n_ar,2)
            MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)          
            N            = n_ar                                                   # Number of realizations
            sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
            sigma_jh     = np.std(R_jh,ddof=1)                                    # Standard deviation of component jh
            bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman bandwidth in 2D
            bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman bandwidth in 2D
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(n_ar,MatRxxarT,npoint,MatRpts,bw_ih,bw_jh)  # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint,npoint)                                 # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)

            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()

            ihq = 0
            jhq = 0
            ihw = 0
            jhw = 0
            if ih <= n_q:
                ihq = ih
            if ih > n_q:
                ihw = ih - n_q
            if jh <= n_q:
                jhq = jh
            if jh > n_q:
                jhw = jh - n_q

            if ihq >= 1 and jhq >= 1:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                plt.title(f'Joint pdf of $Q_{{\\rm{{ar}},{ihq}}}$ with $Q_{{\\rm{{ar}},{jhq}}}$ for $n_{{\\rm{{ar}}}} = {n_ar}$',fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Qar{ihq}_Qar{jhq}.png')
                plt.close(h)
            if ihq >= 1 and jhw >= 1:
                plt.xlabel(f'$q_{{{ihq}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                plt.title(f'Joint pdf of $Q_{{\\rm{{ar}},{ihq}}}$ with $W_{{\\rm{{ar}},{jhw}}}$ for $n_{{\\rm{{ar}}}} = {n_ar}$',fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_Qar{ihq}_War{jhw}.png')
                plt.close(h)
            if ihw >= 1 and jhq >= 1:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$q_{{{jhq}}}$', fontsize=16)
                plt.title(f'Joint pdf of $W_{{\\rm{{ar}},{ihw}}}$ with $Q_{{\\rm{{ar}},{jhq}}}$ for $n_{{\\rm{{ar}}}} = {n_ar}$',fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_War{ihw}_Qar{jhq}.png')
                plt.close(h)   
            if ihw >= 1 and jhw >= 1:
                plt.xlabel(f'$w_{{{ihw}}}$', fontsize=16)
                plt.ylabel(f'$w_{{{jhw}}}$', fontsize=16)
                plt.title(f'Joint pdf of $W_{{\\rm{{ar}},{ihw}}}$ with $W_{{\\rm{{ar}},{jhw}}}$ for $n_{{\\rm{{ar}}}} = {n_ar}$',fontsize=16, fontweight='normal')
                numfig += 1
                plt.savefig(f'figure_PlotX{numfig}_joint_pdf_War{ihw}_War{jhw}.png')
                plt.close(h)   

    # Elapsed time for plotting
    ElapsedTimePlotXdXar = time.time() - TimeStartPlotXdXar

    # Print or display elapsed time if required
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n\n ----- Elapsed time for Task12_PlotXdXar \n\n')
            fidlisting.write(f' Elapsed Time   =  {ElapsedTimePlotXdXar:10.2f}\n\n')

    if ind_display_screen == 1:
        print('--- end Task12_PlotXdXar')

# End of script




