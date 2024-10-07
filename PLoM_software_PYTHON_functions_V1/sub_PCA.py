import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import time

def sub_PCA(n_x, n_d, MatRx_d, error_PCA, ind_display_screen, ind_print, ind_plot):
    
    #---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (Plom) 
    #  Function name: sub_PCA
    #  Subject      : PCA of the scaled random vector X_d using the n_d scaled realizations MatRx_d(n_x,n_d) of X_d 
    #
    #  Publications: 
    #               [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                         Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).               
    #               [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                          American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #
    #--- INPUTS
    #          n_x                   : dimension of random vector X_d (scaled)
    #          n_d                   : number of realizations of X_d
    #          MatRx_d(n_x,n_d)      : n_d realizations of X_d
    #          error_PCA             : relative error on the mean-square norm (related to the eigenvalues of the covariance matrix of X_d)
    #                                  for the truncation of the PCA representation
    #          ind_display_screen    : = 0 no display,            = 1 display
    #          ind_print             : = 0 no print,              = 1 print
    #          ind_plot              : = 0 no plot,               = 1 plot
    #
    #--- OUTPUTS   
    #          nu                    : order of the PCA reduction
    #          nnull                 : = n_x - nu dimension of the null space  
    #          MatReta_d(nu,n_d)     : n_d realizations of random vector H = (H_1,...,H_nu)  
    #          RmuPCA(nu)            : vector of eigenvalues in descending order
    #          MatRVectPCA(n_x,nu)   : matrix of the eigenvectors associated to the eigenvalues loaded in RmuPCA
    #
    
    if ind_display_screen == 1:   
        print(' ')
        print('--- beginning Task3_PCA')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ') 
            fidlisting.write(' ------ Task3_PCA   \n ')
            fidlisting.write('      \n ')  

    TimeStartPCA = time.time()
    numfig       = 0   # initialization of the number of figures

    #--- Computating the trace of the estimated covariance matrix: traceMatRXcov = trace(MatRXcov);
    RXmean        = np.mean(MatRx_d, axis=1)                                    # MatRx_d(n_x,n_d),RXmean(n_x)
    MatRXmean     = np.tile(RXmean, (n_d, 1)).T                                 # MatRXmean(n_x,n_d)
    MatRtemp      = (MatRx_d - MatRXmean) ** 2                                  # MatRtemp(n_x,n_d)
    Rtemp         = np.sum(MatRtemp, axis=1) / (n_d - 1)                          
    traceMatRXcov = np.sum(Rtemp)                                               # traceMatRXcov = trace(MatRXcov)
    del MatRtemp, Rtemp

    #---------------------------------------------------------------------------------------------------------------------------------------
    #       Case for which n_x <= n_d : construction of the estimated covariance matrix and solving the eigenvalue problem
    #---------------------------------------------------------------------------------------------------------------------------------------

    if n_x <= n_d:  
        #--- Constructing the covariance matrix
        MatRXcov = np.cov(MatRx_d)                                          # MatRXcov(n_x,n_x),MatRx_d(n_x,n_d)
        MatRXcov = 0.5 * (MatRXcov + MatRXcov.T)                            # symmetrization 

        #--- Solving the eigenvalue problem
        (RmuTemp,MatRVectTemp) = np.linalg.eigh(MatRXcov)                   # MatRVectTemp(n_x,n_x),RmuTemp(n_x)
        # Align the sign of each vector by ensuring the first element is positive
        for i in range(MatRVectTemp.shape[1]):
            if MatRVectTemp[0, i] < 0:
               MatRVectTemp[:, i] = -MatRVectTemp[:, i]

        # Ordering the eigenvalues in descending order and replacing the values in RmuPCA that are less than 0 with 0
        RmuPCA = np.sort(RmuTemp)[::-1]                                     # RmuPCA(n_x)
        RmuPCA[RmuPCA < 0] = 0

        # Associate the ordering of eigenvectors with the ordering of the eigenvalues
        MatRVectPCA = MatRVectTemp[:, np.argsort(RmuTemp)[::-1]]            # MatRVectPCA(n_x,n_x)  

        # Find the indices where RerrPCA is less than 0 and replacing the values at those indices with RerrPCA(1) * 1e-14
        RerrPCA = 1 - np.cumsum(RmuPCA) / traceMatRXcov                     # RerrPCA(n_x)
        Rneg_indices = RerrPCA < 0                                          # Rneg_indices(n_x): logical array with 0 if > 0 and 1 if < 0
        RerrPCA[Rneg_indices] = RerrPCA[0] * 1e-14

        #--- Print
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')  
                fidlisting.write('      \n ') 
                fidlisting.write('RmuPCA =          \n ') 
                fidlisting.write(' '.join([f'{x:9.2e}' for x in RmuPCA]) + ' \n ')
                fidlisting.write('      \n ') 
                fidlisting.write('      \n ') 
                fidlisting.write('errPCA =          \n ') 
                fidlisting.write(' '.join([f'{x:14.7e}' for x in RerrPCA]) + ' \n ')
                fidlisting.write('      \n ') 
                fidlisting.write('      \n ')  

        #--- Plot
        if ind_plot == 1:
            plt.figure()
            plt.semilogy(np.arange(1, n_x + 1), RmuPCA, '-')
            plt.title(r'Graph of the PCA eigenvalues in ${\rm{log}}_{10}$ scale', fontsize=16)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\mu(\alpha)$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PCA_{numfig}_eigenvaluePCA.png')
            plt.close()

            plt.figure()
            plt.semilogy(np.arange(1, n_x + 1), RerrPCA, '-')
            plt.title(r'Graph of function $\rm{err}_{\rm{PCA}}$ in ${\rm{log}}_{10}$ scale', fontsize=16)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\rm{err}_{\rm{PCA}}(\alpha)$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PCA_{numfig}_errorPCA.png')
            plt.close()

        #--- Truncation of the PCA representation
        Ind = np.where(RerrPCA <= error_PCA)[0]

        # Adapting the dimension of RmuPCA(nu,1) and MatRVectPCA(n_x,nu)
        if Ind.size > 0:                                                    #    nu < n_x
            nu = n_x - Ind.size + 1
            if nu + 1 <= n_x:
                RmuPCA      = RmuPCA[:nu]                                   #    RmuPCA(nu)
                MatRVectPCA = MatRVectPCA[:, :nu]                           #    MatRVectPCA(n_x,nu)
        else:                                                               #    nu = n_x
            nu = n_x

        #--- Dimension of the null space
        nnull = n_x - nu

    #---------------------------------------------------------------------------------------------------------------------------------------
    #  Case for which n_x > n_d : the estimated covariance matrix is not constructed and eigenvalue problem is solved with a SVD on MatRx_d
    #---------------------------------------------------------------------------------------------------------------------------------------

    if n_x > n_d:   
        #--- solving with a "thin SVD" without assembling the covariance matrix of X_d
        (MatRVectTemp, RSigmaTemp, _) = np.linalg.svd(MatRx_d - MatRXmean, full_matrices=False)  # MatRVectTemp(n_x,n_d), RSigmaTemp(n_d)
        # Align the sign of each vector by ensuring the first element is positive
        for i in range(MatRVectTemp.shape[1]):
            if MatRVectTemp[0, i] < 0:
               MatRVectTemp[:, i] = -MatRVectTemp[:, i]

        # Ordering the singular values in descending order 
        RSigma = np.sort(RSigmaTemp)[::-1]                                   # RSigma(n_d)

        # Computing the eigenvalues in descending order
        RmuPCA = (RSigma ** 2) / (n_d - 1)                                  # RmuPCA(n_d) 

        # Associate the ordering of eigenvectors with the ordering of the eigenvalues
        MatRVectPCA = MatRVectTemp[:, np.argsort(RSigma)[::-1]]          # MatRVectPCA(n_x,n_d)  

        # Find the indices where RerrPCA is less than 0 and replacing the values at those indices with RerrPCA(1) * 1e-14
        RerrPCA               = 1 - np.cumsum(RmuPCA) / traceMatRXcov       # RerrPCA(n_d)
        Rneg_indices          = RerrPCA < 0                                 # Rneg_indices(n_d,1): logical array with 0 if > 0 and 1 if < 0
        RerrPCA[Rneg_indices] = RerrPCA[0] * 1e-14

        #--- Print
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')  
                fidlisting.write('      \n ') 
                fidlisting.write('RmuPCA =          \n ') 
                fidlisting.write(' '.join([f'{x:9.2e}' for x in RmuPCA]) + ' \n ')
                fidlisting.write('      \n ') 
                fidlisting.write('      \n ') 
                fidlisting.write('errPCA =          \n ') 
                fidlisting.write(' '.join([f'{x:14.7e}' for x in RerrPCA]) + ' \n ')
                fidlisting.write('      \n ') 
                fidlisting.write('      \n ')  

        #--- Plot
        if ind_plot == 1:
            plt.figure()
            plt.semilogy(np.arange(1, n_d + 1), RmuPCA, '-')
            plt.title(r'Graph of the PCA eigenvalues in ${\rm{log}}_{10}$ scale', fontsize=16)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\mu(\alpha)$', fontsize=16)
            numfig += 1
            plt.savefig(f'figure_PCA_{numfig}_eigenvaluePCA.png')
            plt.close()

            plt.figure()
            plt.semilogy(np.arange(1, n_d + 1), RerrPCA, '-')
            plt.title(r'Graph of function $\rm{err}_{\rm{PCA}}$ in ${\rm{log}}_{10}$ scale', fontsize=16)
            plt.xlabel(r'$\alpha$', fontsize=16)
            plt.ylabel(r'$\rm{err}_{\rm{PCA}}(\alpha)$', fontsize=16)
            numfig += 1
            plt.savefig(f'figure_PCA_{numfig}_errorPCA.png')
            plt.close()

        #--- Truncation of the PCA representation
        Ind = np.where(RerrPCA <= error_PCA)[0]   
        if Ind.size > 0:                                                    # nu < n_d
            nu = n_d - Ind.size + 1
            if nu + 1 <= n_d:
                RmuPCA      = RmuPCA[:nu]                                   # RmuPCA(nu,1)
                MatRVectPCA = MatRVectPCA[:, :nu]                           # MatRVectPCA(n_x,nu)
        else:
            nu = n_d

        #--- Dimension of the null space
        nnull = n_d - nu

    #---------------------------------------------------------------------------------------------------------------------------------------
    #             Computing  MatReta_d(nu,n_d) : n_d realizations of random vector H = (H_1,...,H_nu)  
    #---------------------------------------------------------------------------------------------------------------------------------------
    #--- Construction of  the samples of RH = (H_1,...,H_nu)
    #    MatReta_d(nu,n_d)
    Rcoef = 1.0 / np.sqrt(RmuPCA)
    MatRcoef = np.diag(Rcoef)
    MatReta_d = MatRcoef @ MatRVectPCA.T @ (MatRx_d - MatRXmean)

    #---------------------------------------------------------------------------------------------------------------------------------------
    #             Print PCA results  
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    # Computing the L2 error and the second-order moments of X_d      
    MatRX_nu = MatRXmean + MatRVectPCA @ (np.diag(np.sqrt(RmuPCA))) @ MatReta_d            # MatRX_nu(n_x,n_d)
    error_nu = np.linalg.norm(MatRx_d - MatRX_nu, 'fro') / np.linalg.norm(MatRx_d, 'fro') 
                                                                            
    with open('listing.txt', 'a+') as fidlisting:  
        fidlisting.write('      \n ') 
        fidlisting.write('      \n ')                     
        fidlisting.write(f'error_PCA                    = {error_PCA:14.7e} \n ') 
        fidlisting.write('      \n ')  
        fidlisting.write(f'Number n_d of samples of X_d = {n_d:4d} \n ') 
        fidlisting.write(f'Dimension n_x of X_d         = {n_x:4d} \n ') 
        fidlisting.write(f'Dimension nu  of H           = {nu:4d} \n ') 
        fidlisting.write(f'Null-space dimension         = {nnull:4d} \n ') 
        fidlisting.write('      \n ')  
        fidlisting.write(f'L2 error error_nu            = {error_nu:14.7e} \n ') 
        fidlisting.write('      \n ') 

    if ind_plot == 1:
        with open('listing.txt', 'a+') as fidlisting:  
            fidlisting.write('      \n ')                     
            fidlisting.write('RmuPCA =          \n ') 
            fidlisting.write(' '.join([f'{x:9.2e}' for x in RmuPCA]) + ' \n ')
            fidlisting.write('      \n ') 
            fidlisting.write('      \n ') 

    ElapsedTimePCA = time.time() - TimeStartPCA

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')                                                                
            fidlisting.write('      \n ') 
            fidlisting.write(' ----- Elapsed time for Task3_PCA \n ')
            fidlisting.write('      \n ') 
            fidlisting.write(f' Elapsed Time   =  {ElapsedTimePCA:10.2f}\n')   
            fidlisting.write('      \n ')  

    if ind_display_screen == 1:   
        print('--- end Task3_PCA')
        print(' ')
    
    return nu, nnull, MatReta_d, RmuPCA, MatRVectPCA
