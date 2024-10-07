import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import gaussian_kde
from sub_ksdensity2D import sub_ksdensity2D
import sys

def sub_polynomial_chaosZWiener_plot_Har_HPCE(n_ar, nar_PCE, MatReta_ar, MatReta_PCE, nbplotHClouds, nbplotHpdf, \
                                              nbplotHpdf2D, MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, numfig):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 12 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM)
    #  Function name: sub_polynomial_chaosZWiener_plot_Har_HPCE
    #  Subject      : solver PlOM for direct predictions with or without the constraints of normalization fo H_ar
    #                 computation of n_ar learned realizations MatReta_ar(nu, n_ar) of H_ar
    #
    #--- INPUTS    
    #          n_ar                              : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #          nar_PCE                           : number of realizations of H_PCE such that nar_PCE  = nbMC_PCE x n_d
    #          MatReta_ar(nu, n_ar)              : n_ar realizations of H_ar   
    #          MatReta_PCE(nu, nar_PCE)          : nar_PCE realizations of H_PCE 
    #          nbplotHClouds                     : number of the 3 components numbers of H_ar for which the plot of the clouds are made   
    #          nbplotHpdf                        : number of the components numbers of H_ar for which the plot of the pdfs are made  
    #          nbplotHpdf2D                      : number of the 2 components numbers of H_ar for which the plot of joint pdfs are made
    #          MatRplotHClouds(nbplotHClouds,3)  : contains the 3 components numbers of H_ar for which the plot of the clouds are made
    #          MatRplotHpdf(nbplotHpdf)          : contains the components numbers of H_ar for which the plot of the pdfs are made 
    #          MatRplotHpdf2D(nbplotHpdf2D,2)    : contains the 2 components numbers of H_ar for which the plot of joint pdfs are made 
    #
    #--- INTERNAL PARAMETERS
    #          nu                                : dimension of random vector H = (H_1, ... H_nu)
    #          nbMC                              : number of realizations of (nu, n_d)-valued random matrix [H_ar]  
    
    #--- Plot the 3D clouds ell --> (H_{ar,ih}^ell, H_{ar,jh}^ell, H_{ar,kh}^ell)  and ell --> (H_{PCE,ih}^ell, H_{PCE,jh}^ell, H_{PCE,kh}^ell)
    if nbplotHClouds >= 1:  
        for iplot in range(nbplotHClouds):
            ih, jh, kh = MatRplotHClouds[iplot]
            MINih = np.min(MatReta_ar[ih-1, :])
            MAXih = np.max(MatReta_ar[ih-1, :])
            MINjh = np.min(MatReta_ar[jh-1, :])
            MAXjh = np.max(MatReta_ar[jh-1, :])
            MINkh = np.min(MatReta_ar[kh-1, :])
            MAXkh = np.max(MatReta_ar[kh-1, :])

            # plot clouds of H_ar
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim([MINih - 0.5, MAXih + 0.5])
            ax.set_ylim([MINjh - 0.5, MAXjh + 0.5])
            ax.set_zlim([MINkh - 0.5, MAXkh + 0.5])
            plt.title(f'clouds $H_{{ar}}$ with $n_{{ar}} = {n_ar}$', fontsize=16)
            ax.set_xlabel(f'$H_{{{ih}}}$', fontsize=16)
            ax.set_ylabel(f'$H_{{{jh}}}$', fontsize=16)
            ax.set_zlabel(f'$H_{{{kh}}}$', fontsize=16)
            ax.scatter(MatReta_ar[ih-1, :], MatReta_ar[jh-1, :], MatReta_ar[kh-1, :], s=2, marker='h', color='b')
            numfig = numfig + 1
            plt.savefig(f'figure_PolynomialChaosZWiener_{numfig}_clouds_Har_{ih}_{jh}_{kh}.png')
            plt.close()

            # plot clouds of H_PCE
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim([MINih - 0.5, MAXih + 0.5])
            ax.set_ylim([MINjh - 0.5, MAXjh + 0.5])
            ax.set_zlim([MINkh - 0.5, MAXkh + 0.5])
            plt.title(f'clouds $H_{{PCE}}$ with $n_{{ar,PCE}} = {nar_PCE}$', fontsize=16)
            ax.set_xlabel(f'$H_{{{ih}}}$', fontsize=16)
            ax.set_ylabel(f'$H_{{{jh}}}$', fontsize=16)
            ax.set_zlabel(f'$H_{{{kh}}}$', fontsize=16)
            ax.scatter(MatReta_PCE[ih-1, :], MatReta_PCE[jh-1, :], MatReta_PCE[kh-1, :], s=2, marker='h', color='r')
            numfig += 1
            plt.savefig(f'figure_PolynomialChaosZWiener_{numfig}_clouds_HPCE_{ih}_{jh}_{kh}.png')
            plt.close()

    #--- Generation and plot the pdf of H_{ar,ih} and H_{PCE,ih}
    if nbplotHpdf >= 1:
        npoint = 200  # number of points for the pdf plot using gaussian_kde
        for iplot in range(nbplotHpdf):
            ih      = MatRplotHpdf[iplot]          # MatRplotHpdf(nbplotHpdf)
            Rar_ih  = MatReta_ar[ih-1,:].T  
            Rar_ih  = np.squeeze(Rar_ih) 
            RPCE_ih = MatReta_PCE[ih-1,:].T  
            RPCE_ih = np.squeeze(RPCE_ih)         
            MIN     = min(np.min(Rar_ih), np.min(RPCE_ih))
            MAX     = max(np.max(Rar_ih), np.max(RPCE_ih)) 

            #--- For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
            sigma_ih  = np.std(Rar_ih,ddof=1)              
            bw_ih     = 1.0592*sigma_ih*n_ar**(-1/5)                       # Sylverman bandwidth in 1D
            kde       = gaussian_kde(Rar_ih,bw_method=bw_ih/sigma_ih)
            Rh_ar      = np.linspace(MIN,MAX,npoint)  
            Rpdf_ar    = kde.evaluate(Rh_ar)
            
            sigma_ih  = np.std(RPCE_ih,ddof=1)                           
            bw_ih     = 1.0592*sigma_ih*nar_PCE**(-1/5)                    # Sylverman bandwidth in 1D
            kde       = gaussian_kde(RPCE_ih,bw_method=bw_ih/sigma_ih)
            Rh_PCE    = np.linspace(MIN,MAX,npoint)
            Rpdf_PCE  = kde.evaluate(Rh_PCE)

            plt.figure()
            plt.plot(Rh_ar, Rpdf_ar, 'k-', label=f'$p_{{H_{{ar,{ih}}}}}$',linewidth=0.5)
            plt.plot(Rh_PCE, Rpdf_PCE, 'b-', label=f'$p_{{H_{{PCE,{ih}}}}}$',linewidth=1)
            plt.xlim([MIN - 1, MAX + 1])
            plt.title(f'$p_{{H_{{ar,{ih}}}}}$ (black thin) with $n_ar = {n_ar}$\n$p_{{H_{{PCE,{ih}}}}}$ (blue thick) with $nar_{{PCE}} = {nar_PCE}$', fontsize=16)
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$p_{{H_{{{ih}}}}}(\eta_{{{ih}}})$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PolynomialChaosZWiener_{numfig}_pdf_Har_HPCE_{ih}.png')
            plt.close()

    #--- Generation and plot the joint pdf of (H_{ar,ih}, H_{ar,jh})
    if nbplotHpdf2D >= 1:
        npoint = 100
        for iplot in range(nbplotHpdf2D):
            ih, jh = MatRplotHpdf2D[iplot]
            MINih = np.min(MatReta_ar[ih-1, :])
            MAXih = np.max(MatReta_ar[ih-1, :])
            MINjh = np.min(MatReta_ar[jh-1, :])
            MAXjh = np.max(MatReta_ar[jh-1, :])
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
            R_ih         = MatReta_ar[ih-1, :].T                                  # R_ih(n_ar,1)
            R_jh         = MatReta_ar[jh-1, :].T                                  # R_jh(n_ar,1)
            MatRetaT     = np.vstack([R_ih,R_jh]).T                               # MatRetaT(n_ar,2)
            MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
            N            = n_ar                                                   # Number of realizations
            sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
            sigma_jh     = np.std(R_jh,ddof=1)                                    # Standard deviation of component jh
            bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman bandwidth in 2D 
            bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman bandwidth in 2D 
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(n_ar,MatRetaT,npoint,MatRpts,bw_ih,bw_jh)   # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint,npoint)                                 # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)

            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$\eta_{{{jh}}}$', fontsize=16)
            plt.title(f'Joint pdf of $H_{{ar,{ih}}}$ with $H_{{ar,{jh}}}$ for $n_ar = {n_ar}$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PolynomialChaosZWiener_{numfig}_joint_pdf_Har{ih}_Har{jh}.png')
            plt.close()

    #--- Generation and plot the joint pdf of (H_{PCE,ih}, H_{PCE,jh})
    if nbplotHpdf2D >= 1:
        npoint = 100
        for iplot in range(nbplotHpdf2D):
            ih, jh = MatRplotHpdf2D[iplot]
            MINih = np.min(MatReta_PCE[ih-1, :])
            MAXih = np.max(MatReta_PCE[ih-1, :])
            MINjh = np.min(MatReta_PCE[jh-1, :])
            MAXjh = np.max(MatReta_PCE[jh-1, :])
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
            R_ih         = MatReta_PCE[ih-1, :].T                                 # R_ih(nar_PCE,1)
            R_jh         = MatReta_PCE[jh-1, :].T                                 # R_jh(nar_PCE,1)
            MatRetaT     = np.vstack([R_ih,R_jh]).T                               # MatRetaT(nar_PCE,2)
            MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
            N            = nar_PCE                                                # Number of realizations
            sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
            sigma_jh     = np.std(R_jh,ddof=1)                                    # Standard deviation of component jh
            bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman bandwidth in 2D 
            bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman bandwidth in 2D 
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(nar_PCE,MatRetaT,npoint,MatRpts,bw_ih,bw_jh)  # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint,npoint)                                   # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)

            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$\eta_{{{jh}}}$', fontsize=16)
            plt.title(f'Joint pdf of $H_{{PCE,{ih}}}$ with $H_{{PCE,{jh}}}$ for $nar_PCE = {nar_PCE}$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PolynomialChaosZWiener_{numfig}_joint_pdf_HPCE{ih}_HPCE{jh}.png')
            plt.close()
