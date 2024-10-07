import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import gaussian_kde
from sub_ksdensity2D import sub_ksdensity2D

def sub_solverInverse_plot_Hd_Har(n_d, n_ar, MatReta_d, MatReta_ar, nbplotHsamples, nbplotHClouds, nbplotHpdf, nbplotHpdf2D, 
                                 MatRplotHsamples, MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, numfig):

    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 26 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverInverse_plot_Hd_Har
    #  Subject      : solver PLoM for direct predictions with or without the constraints of normalization for H_ar
    #                 computation of n_ar learned realizations MatReta_ar(nu,n_ar) of H_ar
    #
    #--- INPUTS    
    #          n_d                               : number of points in the training set for H   
    #          n_ar                              : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #          MatReta_d(nu,n_d)                 : n_d realizations of H   
    #          MatReta_ar(nu,n_ar)               : n_ar realizations of H_ar 
    #          nbplotHsamples                    : number of the components numbers of H_ar for which the plot of the realizations are made  
    #          nbplotHClouds                     : number of the 3 components numbers of H_ar for which the plot of the clouds are made   
    #          nbplotHpdf                        : number of the components numbers of H_d and H_ar for which the plot of the pdfs are made  
    #          nbplotHpdf2D                      : number of the 2 components numbers of H_d and H_ar for which the plot of joint pdfs are made
    #          MatRplotHsamples(nbplotHsamples)  : contains the components numbers of H_ar for which the plot of the realizations are made
    #          MatRplotHClouds(nbplotHClouds,3)  : contains the 3 components numbers of H_ar for which the plot of the clouds are made
    #          MatRplotHpdf(nbplotHpdf)          : contains the components numbers of H_d and H_ar for which the plot of the pdfs are made 
    #          MatRplotHpdf2D(nbplotHpdf2D,2)    : contains the 2 components numbers of H_d and H_ar for which the plot of joint pdfs are made 
    #
    #--- INTERNAL PARAMETERS
    #          nu                                : dimension of random vector H = (H_1, ... H_nu)
    #          nbMC                              : number of realizations of (nu,n_d)-valued random matrix [H_ar]  

    #--- Plot the 2D graphs ell --> H_{ar,ih}^ell    
    if nbplotHsamples >= 1:
        for iplot in range(nbplotHsamples):
            ih     = MatRplotHsamples[iplot]   # MatRplotHsamples(nbplotHsamples)
            Rih_ar = MatReta_ar[ih-1, :]       # Rih_ar(n_ar)
            plt.figure()
            plt.plot(range(n_ar), Rih_ar, 'b-')
            plt.title(f'$H_{{ar,{ih}}}$ with $n_{{ar}} = {n_ar}$', fontsize=16)
            plt.xlabel(r' $\ell$', fontsize=16)
            plt.ylabel(f'$\\eta^\\ell_{{ar,{ih}}}$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_SolverInverse_{numfig}_eta_ar_{ih}.png')
            plt.close()

    #--- Plot the 3D clouds ell --> (H_{d,ih}^ell,H_{d,jh}^ell,H_{d,kh}^ell) and ell --> (H_{ar,ih}^ell,H_{ar,jh}^ell,H_{ar,kh}^ell) 
    if nbplotHClouds >= 1:
        for iplot in range(nbplotHClouds):
            ih, jh, kh = MatRplotHClouds[iplot]
            MINih, MAXih = np.min(MatReta_ar[ih-1, :]), np.max(MatReta_ar[ih-1, :])
            MINjh, MAXjh = np.min(MatReta_ar[jh-1, :]), np.max(MatReta_ar[jh-1, :])
            MINkh, MAXkh = np.min(MatReta_ar[kh-1, :]), np.max(MatReta_ar[kh-1, :])
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(54, 25)
            ax.set_xlim([MINih - 0.5, MAXih + 0.5])
            ax.set_ylim([MINjh - 0.5, MAXjh + 0.5])
            ax.set_zlim([MINkh - 0.5, MAXkh + 0.5])
            ax.scatter(MatReta_d[ih-1, :], MatReta_d[jh-1, :], MatReta_d[kh-1, :], s=2, c='b', marker='h')
            ax.scatter(MatReta_ar[ih-1, :], MatReta_ar[jh-1, :], MatReta_ar[kh-1, :], s=2, c='r', marker='h')
            plt.title(f'clouds $H_d$ with $n_d = {n_d}$ and $H_{{ar}}$ with $n_{{ar}} = {n_ar}$', fontsize=16)
            ax.set_xlabel(f'$H_{{{ih}}}$', fontsize=16)
            ax.set_ylabel(f'$H_{{{jh}}}$', fontsize=16)
            ax.set_zlabel(f'$H_{{{kh}}}$', fontsize=16)
            numfig += 1
            plt.savefig(f'figure_SolverInverse_{numfig}_eta_clouds_{ih}_{jh}_{kh}.png')
            plt.close()

    #--- Generation and plot the pdf of H_{d,ih} and H_{ar,ih}   
    if nbplotHpdf >= 1:
        npoint = 200                              # number of points for the pdf plot using gaussian_kde
        for iplot in range(nbplotHpdf):
            ih     = MatRplotHpdf[iplot]          # MatRplotHpdf(nbplotHpdf)
            Rd_ih  = MatReta_d[ih-1,:].T  
            Rd_ih  = np.squeeze(Rd_ih) 
            Rar_ih = MatReta_ar[ih-1,:].T  
            Rar_ih = np.squeeze(Rar_ih)         
            MIN    = min(np.min(Rd_ih), np.min(Rar_ih)) - 1
            MAX    = max(np.max(Rd_ih), np.max(Rar_ih)) + 1

            #--- For the pdf (dimension 1), modifying the Python bandwidth to obtain a bandwidth close to the Matlab bandwidth
            sigma_ih  = np.std(Rd_ih,ddof=1)  
            bw_ih     = 1.0592*sigma_ih*n_d**(-1/5)                     # Silverman bandwidth in 1D
            kde       = gaussian_kde(Rd_ih,bw_method=bw_ih/sigma_ih)
            Rh_d      = np.linspace(MIN,MAX,npoint)  
            Rpdf_d    = kde.evaluate(Rh_d)
            
            sigma_ih  = np.std(Rar_ih,ddof=1)  
            bw_ih     = 1.0592*sigma_ih*n_ar**(-1/5)                     # Silverman bandwidth in 1D
            kde       = gaussian_kde(Rar_ih,bw_method=bw_ih/sigma_ih)
            Rh_ar     = np.linspace(MIN,MAX,npoint)
            Rpdf_ar   = kde.evaluate(Rh_ar)
                       
            plt.figure()
            plt.plot(Rh_d, Rpdf_d, 'k-', label=f'$p_{{H_{{d,{ih}}}}}$',linewidth=0.5)
            plt.plot(Rh_ar, Rpdf_ar, 'b-', label=f'$p_{{H_{{ar,{ih}}}}}$',linewidth=1)
            plt.xlim([MIN - 1, MAX + 1])
            plt.title(f'$p_{{H_{{d,{ih}}}}}$ (black thin) with $n_d = {n_d}$\n$p_{{H_{{ar,{ih}}}}}$ (blue thick) with $n_{{ar}} = {n_ar}$', fontsize=16)
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$p_{{H_{{{ih}}}}}(\eta_{{{ih}}})$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_SolverInverse_{numfig}_pdf_H_d_H_ar{ih}.png')
            plt.close()

    #--- Generation and plot the joint pdf of (H_{d,ih},H_{d,jh}) 
    if nbplotHpdf2D >= 1:
        npoint = 100
        for iplot in range(nbplotHpdf2D):
            ih, jh = MatRplotHpdf2D[iplot]
            MINih, MAXih = np.min(MatReta_d[ih-1, :]), np.max(MatReta_d[ih-1, :])
            MINjh, MAXjh = np.min(MatReta_d[jh-1, :]), np.max(MatReta_d[jh-1, :])
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
            R_ih         = MatReta_d[ih-1, :].T                                   # R_ih(n_d,1)
            R_jh         = MatReta_d[jh-1, :].T                                   # R_jh(n_d,1)
            MatRetaT     = np.vstack([R_ih,R_jh]).T                               # MatRetaT(n_d,2)
            MatRpts      = np.column_stack([MatRx.ravel(), MatRy.ravel()])        # MatRpts(npoint*npoint,2),MatRx(npoint,npoint),MatRy(npoint,npoint)
            N            = n_d                                                    # Number of realizations
            sigma_ih     = np.std(R_ih,ddof=1)                                    # Standard deviation of component ih
            sigma_jh     = np.std(R_jh,ddof=1)                                    # Standard deviation of component jh
            bw_ih        = sigma_ih*N**(-1/6)                                     # Silverman bandwidth in 2D
            bw_jh        = sigma_jh*N**(-1/6)                                     # Silverman bandwidth in 2D
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(n_d,MatRetaT,npoint,MatRpts,bw_ih,bw_jh)   # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint, npoint)      # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)

            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$\eta_{{{jh}}}$', fontsize=16)
            plt.title(f'Joint pdf of $H_{{d,{ih}}}$ with $H_{{d,{jh}}}$ for $n_d = {n_d}$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_SolverInverse_{numfig}_joint_pdf_Hd{ih}_Hd{jh}.png')
            plt.close()

    #--- Generation and plot the joint pdf of (H_{ar,ih},H_{ar,jh}) 
    if nbplotHpdf2D > 0:
        npoint = 100
        for iplot in range(nbplotHpdf2D):
            ih, jh = MatRplotHpdf2D[iplot]
            MINih, MAXih = np.min(MatReta_ar[ih-1, :]), np.max(MatReta_ar[ih-1, :])
            MINjh, MAXjh = np.min(MatReta_ar[jh-1, :]), np.max(MatReta_ar[jh-1, :])
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
            Rbw          = np.array([bw_ih, bw_jh])                               # Bandwidth vector
            # Kernel density estimation using gaussian_kde in Python
            # Note: Bandwidth scaling in gaussian_kde is a factor applied to the covariance matrix, so we scale appropriately
            Rpdf    = sub_ksdensity2D(n_ar,MatRetaT,npoint,MatRpts,bw_ih,bw_jh)   # Rpdf(npoint*npoint)
            MatRpdf = Rpdf.reshape(npoint, npoint)                                # MatRpdf(npoint,npoint), Rpdf(npoint*npoint)
            
            # Plot the contours of the joint PDF
            plt.figure()
            plt.pcolormesh(MatRx, MatRy, MatRpdf, shading='gouraud', cmap='jet')  # 'shading=gouraud' 
            plt.xlim([MINih, MAXih])
            plt.ylim([MINjh, MAXjh])
            plt.colorbar()
            plt.xlabel(f'$\eta_{{{ih}}}$', fontsize=16)
            plt.ylabel(f'$\eta_{{{jh}}}$', fontsize=16)            
            plt.title(f'Joint pdf of $H_{{ar,{ih}}}$ with $H_{{ar,{jh}}}$ for $n_{{ar}} = {n_ar}$', fontsize=16)
            numfig += 1
            plt.savefig(f'figure_SolverInverse_{numfig}_joint_pdf_Har{ih}_Har{jh}.png')
            plt.close()

    return
