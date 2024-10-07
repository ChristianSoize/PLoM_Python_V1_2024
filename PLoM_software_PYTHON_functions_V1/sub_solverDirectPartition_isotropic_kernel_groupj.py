import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np

def sub_solverDirectPartition_isotropic_kernel_groupj(ind_generator, j, mj, n_d, MatReta_dj, 
                                                      epsilonDIFFmin, step_epsilonDIFF, iterlimit_epsilonDIFF, 
                                                      comp_ref, ind_display_screen, ind_plot, numfig):
    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirectPartition_isotropic_kernel.m
    #  Subject      : for group j, generating matrix MatRgj(n_d,nbmDMAP) of the nbmDMAPj projection basis vectors
    #                 solving the eigenvalue problem related to the isotropic kernel
    #  Comment      : this function is the adaptation of the function: sub_projection_basis_isotropic_kernel.m
    #
    #  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). .
    #                [3] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                       mjmerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #
    #--- INPUTS 
    #        ind_generator       : = 0 the projection basis is the canonical basis of R^n_d
    #                              = 1 the projection basis is the DMAPS baisis associated with the isotropice kernel
    #        mj                  : dimension of random vector Y^j = (H_jr1,... ,H_jrmj) of group j
    #        n_d                 : number of points in the training set for Y^j
    #        MatReta_dj(mj,n_d)  : n_d realizations of Y^j                            
    #                            --- The following parameters are required for generating the projection basis by solving the 
    #                                eigenvalue problem related to the isotropic kernel: the smooting paramameter epsilonDIFFj 
    #                                is searched with an iteration algorithm              
    #        epsilonDIFFmin        : epsilonDIFFj is searched in interval [epsilonDIFFmin , +infty[                                    
    #        step_epsilonDIFF      : step for searching the optimal value epsilonDIFFj starting from epsilonDIFFmin
    #        iterlimit_epsilonDIFF : maximum number of the iteration algorithm for computing epsilonDIFFj                              
    #        comp_ref              : value in  [ 0.1 , 0.5 [  used for stopping the iteration algorithm.
    #                                if comp =  Rlambda(nbmDMAPj+1)/Rlambda(nbmDMAPj) <= comp_ref, then algorithm is stopped
    #                                The standard value for comp_ref is 0.1 
    #                            --- parameters and variables controling execution
    #        ind_display_screen  : 0, no display, if 1 display
    #        ind_print           : 0, no print, if 1 print
    #        ind_plot            : 0, no plot, if  1 plot
    #        ind_parallel        : 0, no parallel computation, if 1 parallel computation
    #
    #--- OUTPUTS
    #        epsilonDIFFj          : optimal value of the smooting parameter in isotropic kernel
    #        mDPj                  : maximum mjmber of projection-basis vectors with nbmDMAPj <= mDPj <= n_d
    #        nbmDMAPj              : dimension of the projection basis such that nbmDMAPj <= mDPj      
    #        MatRgj(n_d,nbmDMAPj)  : matrix of the projection basis
    #        MatRaj(n_d,nbmDMAPj)  : matrix [a] =[g]([g]'[g])^(-1) 

    #----------------------------------------------------------------------------------------------------------------------------------
    #     ind_generator = 0:  the projection basis is the canonical basis of R^{n_d}                                
    #----------------------------------------------------------------------------------------------------------------------------------       
    if ind_generator == 0:  # the parameter nbmDMAPj entered by the used is modified  
        epsilonDIFFj = 0  # not used
        mDPj         = n_d
        nbmDMAPj     = n_d                   
        MatRgj       = np.eye(n_d)  # MatRg(n_d,nbmDMAPj) : matrix of the nbmDMAPj projection basis vectors
        MatRaj       = np.eye(n_d)  # MatRa(n_d,nbmDMAPj) = MatRg*(MatRg'*MatRg)^{-1} 

    #----------------------------------------------------------------------------------------------------------------------------------
    #     ind_generator = 1  and  mj = 1                         
    #----------------------------------------------------------------------------------------------------------------------------------     
    if ind_generator == 1 and mj == 1:
        epsilonDIFFj = 0  # not used
        mDPj         = 1
        nbmDMAPj     = 1
        MatRgj       = MatReta_dj[0, :].reshape(-1, 1)    # MatRgj(n_d,1), MatReta_dj(1,n_d)
        MatRaj       = MatRgj / np.dot(MatRgj.T, MatRgj)  # MatRaj(n_d,nbmDMAPj), [a] =[g]([g]'[g])^(-1) 

    #----------------------------------------------------------------------------------------------------------------------------------
    #     ind_generator = 1 and mj >= 2:  the projection basis  is the DMAPS related to the isotropic kernel                             
    #---------------------------------------------------------------------------------------------------------------------------------- 
    if ind_generator == 1 and mj >= 2:
        nbmDMAPj = mj + 1
        mDPj     = min(nbmDMAPj + 10, n_d)  # mDPj: maximum mjmber of projection-basis vectors with nbmDMAPj <= mDPj <= n_d   

        #--- Constructing the column matrix containing the iterlimit_epsilonDIFF values of epsilonDIFFj 
        RepsilonDIFF = np.zeros(iterlimit_epsilonDIFF)
        for iter in range(iterlimit_epsilonDIFF):
            RepsilonDIFF[iter] = epsilonDIFFmin + step_epsilonDIFF*iter

        #--- Finding the optimal value epsilonDIFF by solving the iteration algorithm
        Rcomp = np.zeros(iterlimit_epsilonDIFF)                                    # Rcomp(iterlimit_epsilonDIFF,1)
        for iter in range(iterlimit_epsilonDIFF):
            # Display
            if ind_display_screen == 1:
                print(f'  iter = {iter + 1}')
            epsilonDIFF_iter = RepsilonDIFF[iter]

            # Isotropic kernel, computing MatRKernel_iter(n_d,n_d) 
            co_iter = 1/(4*epsilonDIFF_iter)
            MatRtemp1 = np.zeros((n_d,n_d))
            for ell in range(n_d):
                MatRtemp2 = np.zeros((mj,n_d))
                for k in range(mj):
                    MatRtemp2[k,:] = (MatReta_dj[k,:] - MatReta_dj[k,ell])**2     # MatRtemp2(mj,n_d), MatReta_dj(mj,n_d)               
                MatRtemp1[ell,:] = np.sum(MatRtemp2,axis=0)                       # MatRtemp1(n_d,n_d)

            MatRKernel_temp = np.exp(-co_iter*MatRtemp1)                          # MatRKernel_temp(n_d,n_d) 

            # Removing the negative zeros of the eigenvalues of MatRKernel_temp
            scaling         = np.max(MatRKernel_temp)                                       
            MatRKernel_iter = MatRKernel_temp + scaling * 1e-12 * np.eye(n_d)     # MatRKernel_iter(n_d,n_d) 

            # Computing the transition probability matrix MatRPs       
            MatRPs_iter = np.zeros((n_d,n_d))                                     # MatRPs_iter(n_d,n_d)
            Rd          = np.sum(MatRKernel_iter,axis=1)
            Rdm1s2      = 1.0 / np.sqrt(Rd)
            for i in range(n_d):
                MatRPs_iter[i,:] = Rdm1s2[i] * MatRKernel_iter[i,:] * Rdm1s2      # MatRPs_iter(n_d,n_d), Symmetric matrix

            MatRPs_iter = 0.5 * (MatRPs_iter + MatRPs_iter.T)                     # MatRPs_iter(n_d,n_d), imposing a perfect symmetry

            # Computing the eigenvalues of  MatRPs_iter * MatRphi_iter = MatRphi_iter * diag(Rlambda_iter) 
            (Rlambda1, MatRphi1) = np.linalg.eigh(MatRPs_iter)                    # Rlambda1(n_d),MatRPs_iter(n_d,n_d)
            Index                = np.argsort(Rlambda1)[::-1]                     # descending order
            Rlambda_iter         = Rlambda1[Index]                                # Rlambda_iter(n_d) 
            MatRphi_iter         = MatRphi1[:,Index]                              # MatRphi_iter(n_d,n_d)

            #--- checking the jump in the eigenvalues spectrum with comp = Rlambda_iter(nbmDMAPj+1)/Rlambda_iter(nbmDMAPj)<= comp_ref 
            #    given by the user in [ 0.1 , 0.5 [
            comp_iter = Rlambda_iter[nbmDMAPj] / Rlambda_iter[nbmDMAPj - 1]
            Rcomp[iter] = comp_iter                                              # Rcomp(iterlimit_epsilonDIFF)                                  
            if comp_iter <= comp_ref:                                            # solution is epsilonDIFFj
                epsilonDIFFj = epsilonDIFF_iter
                iter_conv = iter
                break  

            #--- no convergence
            if iter == iterlimit_epsilonDIFF - 1:  
                # Plot
                if ind_plot == 1:
                    plt.figure()
                    plt.plot(np.arange(1, iterlimit_epsilonDIFF + 1), Rcomp, 'b-o')
                    plt.title(f'Graph iter $\\mapsto$ Rcomp(iter) for group {j}', fontsize=16, weight='normal')
                    plt.xlabel('iter', fontsize=16) 
                    plt.ylabel('Rcomp(iter)', fontsize=16) 
                    numfig = numfig + 1
                    plt.savefig(f'figure_isotropic_kernel_basis_{numfig}_Rcomp_group{j}.png')
                    plt.close()

                # Display
                print(f'STOP in sub_solverDirectPartition_isotropic_kernel: for group {j} iterlimit_epsilonDIFF reached without finding a solution')
                # Print
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n ')
                    fidlisting.write(f'STOP in sub_solverDirectPartition_isotropic_kernel: for group {j}, ...\n')
                    fidlisting.write('iterlimit_epsilonDIFF reached without finding a solution\n')

                raise RuntimeError(f'STOP in sub_solverDirectPartition_isotropic_kernel: for group {j} iterlimit_epsilonDIFF reached without finding a solution')

        #--- Convergence is reached
        Rlambda     = Rlambda_iter[:mDPj]     # Rlambda(mDPj),Rlambda_iter(n_d)  
        MatRphi     = MatRphi_iter[:, :mDPj]  # MatRphi(n_d,mDPj),MatRphi_iter(n_d,n_d)
        MatRgj_mDPj = np.zeros((n_d, mDPj))   # MatRgj_mDPj(n_d,mDPj)                         
        for beta in range(mDPj):
            MatRgj_mDPj[:,beta] = MatRphi[:,beta] * Rdm1s2

        #--- Loading the nbmDMAP projection basis vectors (diffusion maps)
        MatRgj = MatRgj_mDPj[:,:nbmDMAPj]                                  # MatRgj(n_d,nbmDMAPj),MatRgj_mDPj(n_d,mDPj) 

        #--- Computing MatRaj(n_d,nbmDMAPj): [aj] =[gj]([gj]'[gj])^(-1) 
        MatRaj = np.linalg.solve(MatRgj.T @ MatRgj, MatRgj.T).T            # MatRaj(n_d,nbmDMAPj)

        #--- Plot
        if ind_plot == 1:
            if iterlimit_epsilonDIFF - iter_conv >= 1:
                iter_plot = iter_conv + 1
            plt.figure()
            plt.plot(np.arange(1, iter_plot + 1), Rcomp[:iter_plot], 'b-o')
            plt.title(f'Graph iter $\\mapsto$ Rcomp(iter) for group {j}', fontsize=16, weight='normal')
            plt.xlabel('iter', fontsize=16) 
            plt.ylabel('Rcomp(iter)', fontsize=16) 
            numfig = numfig + 1
            plt.savefig(f'figure_SolverDirectPartition_isotropic_kernel_basis_{numfig}_Rcomp_group{j}.png')
            plt.close()

            plt.figure()
            plt.semilogy(np.arange(1, mDPj + 1), Rlambda[:mDPj], '-o')
            plt.title(f'Eigenvalues $\\alpha\\mapsto\\lambda_\\alpha$ of the transition matrix for group {j}', fontsize=16, weight='normal')                                        
            plt.xlabel('$\\alpha$', fontsize=16)                                                                
            plt.ylabel('$\\lambda_\\alpha$', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_SolverDirectPartition_isotropic_kernel_basis_{numfig}_eigenvalueDMAPS_group{j}.png')
            plt.close()

    return epsilonDIFFj, mDPj, nbmDMAPj, MatRgj, MatRaj
