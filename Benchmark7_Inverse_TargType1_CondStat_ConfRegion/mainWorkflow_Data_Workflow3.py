import os
import numpy as np
import scipy.io as sio
from mainWorkflow_Data_generation1 import mainWorkflow_Data_generation1
from mainWorkflow_Data_generation2 import mainWorkflow_Data_generation2

def mainWorkflow_Data_Workflow3():
    #----------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 26 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow_Data_WorkFlow3
    #  Subject      : this function must be used by the user for defining the data for the Inverse Solver without partition 
    #                 ind_workflow = 3 for Benchmark7_Inverse_TargType1_CondStat_ConfRegion
    #  WARNING      : The structure of the function, the names of the parameters and variables must not be changed. 
    #                 Only the numerical values should be modified according to the application being processed.

    #---------- INDICATE THE STEPS TO EXECUTE. FOR STEPS 3 AND 4, SPECIFY IF SAVING THE FILE IS REQUIRED AT THE END OF STEP 3 AND/OR STEP 4
    ind_step1 = 1  # 0: no step1 (pre-analysis), 1: step1
    ind_step2 = 1  # 0: no step2 (analysis), 1: step2
    ind_step3 = 1  # 0: no step3 (post analysis: print and plot), 1: step3
    ind_step4 = 1  # 0: no step4 (conditional statistics and polynomial chaos expansions), 1: step4

    # The following parameters are used when ind_step4 = 1 and are used as an initialization when ind_step4 = 0
    ind_step4_task13  = 1   # 0: task13 of step4 is not executed (conditional statistics: second_order moments, pdf's, confidence regions), 
                            # 1: task13 of step4 is executed
    ind_step4_task14  = 0   # 0: task14 of step4 is not executed (polynomial Chaos of Z with Wiener as germs), 
                            # 1: task14 of step4 is executed
    ind_step4_task15  = 0   # 0: task15 of step4 is not executed (polynomial Chaos of Q with W and U as germs), 
                            # 1: task15 of step4 is executed
    ind_SavefileStep3 = 0   # 0: SavefileStep3.mat not save at the end of Step3, 
                            # 1: SavefileStep3.mat save at the end of Step3
    ind_SavefileStep4 = 0   # 0: SavefileStep3.mat not save at the end of Step4, 
                            # 1: SavefileStep3.mat save at the end of Step4

    #------------------------- DATA BLOCK STEP1 ENTERED BY THE USER ----------------------------------------------------
    if ind_step1 == 1:
        #--- parameters and variables controlling execution of step1  
        ind_display_screen = 1  # 0 no display, 1 display
        ind_print          = 1  # 0 no print, 1 print
        ind_plot           = 1  # 0 no plot, 1 plot
        ind_parallel       = 1  # 0 no parallel computation, 1 parallel computation

        #===== Data block for 'Task1_DataStructureCheck','Task2_Scaling', and 'Task3_PCA'
        ind_scaling = 1      # 0 no scaling, 1 scaling
        error_PCA   = 1e-3   # relative error on the mean-square norm (related to the eigenvalues of the covariance matrix of X_d) 
                             # for the truncation of the PCA representation

        (n_q,Indq_real,Indq_pos,n_w,Indw_real,Indw_pos,Indq_obs,Indw_obs,n_d,MatRxx_d) = mainWorkflow_Data_generation1()

        (ind_type_targ,Indq_targ_real,Indq_targ_pos,Indw_targ_real,Indw_targ_pos,N_r,MatRxx_targ_real,MatRxx_targ_pos, 
            Rmeanxx_targ,MatRcovxx_targ) = mainWorkflow_Data_generation2()

        #==== WARNING: save instruction must not be modified by the user 
        sio.savemat('FileDataWorkFlow3Step1.mat', {
            'ind_display_screen': ind_display_screen,
            'ind_print': ind_print,
            'ind_plot': ind_plot,
            'ind_parallel': ind_parallel,
            'ind_scaling': ind_scaling,
            'error_PCA': error_PCA,
            'n_q': n_q,
            'Indq_real': Indq_real,
            'Indq_pos': Indq_pos,
            'n_w': n_w,
            'Indw_real': Indw_real,
            'Indw_pos': Indw_pos,
            'Indq_obs': Indq_obs,
            'Indw_obs': Indw_obs,
            'n_d': n_d,
            'MatRxx_d': MatRxx_d,
            'ind_type_targ': ind_type_targ,
            'Indq_targ_real': Indq_targ_real,
            'Indq_targ_pos': Indq_targ_pos,
            'Indw_targ_real': Indw_targ_real,
            'Indw_targ_pos': Indw_targ_pos,
            'N_r': N_r,
            'MatRxx_targ_real': MatRxx_targ_real,
            'MatRxx_targ_pos': MatRxx_targ_pos,
            'Rmeanxx_targ': Rmeanxx_targ,
            'MatRcovxx_targ': MatRcovxx_targ
        }, do_compression=True)

    #------------------------- DATA BLOCK STEP2 ENTERED BY THE USER ----------------------------------------------------
    if ind_step2 == 1:
        #--- parameters and variables controlling execution of step2  
        ind_display_screen = 1  # 0 no display, 1 display
        ind_print          = 1  # 0 no print, 1 print
        ind_plot           = 1  # 0 no plot, 1 plot
        ind_parallel       = 1  # 0 no parallel computation, 1 parallel computation

        #===== Data block required for executing 'Task5_ProjectionBasisNoPartition' 
        mDP = 100           # maximum number of projection-basis vectors that are read on a file or that are generated 
                            # by the isotropic kernel, and which must be such that  nbmDMAP <= mDP <= n_d. 
                            # Parameter mDP is not used if ind_generator = 0.
        ind_generator  = 1  # 0 : generator without using projection basis = dissipative-Hamiltonian-based MCMC, 
                            # 1 : generator using the projection basis 
        ind_basis_type = 2  # 1 : read nd,mDP,MatRg_mDP(nd,mDP) on a Binary File or a Text File  where nd should be equal to n_d, 
                            # 2 : generation of the projection basis solving the eigenvalue problem related to the isotropic kernel
        ind_file_type  = 0  # used if ind_basis_type = 1, 
                            # 1 : Binary File type and ind_basis_type = 1, with filename = 'ISDEprojectionBasisBIN.bin', 
                            # 2 : Text File type and ind_basis_type = 1, with filename = 'ISDEprojectionBasisTXT.txt'

                                      # --- if ind_basis_type = 2, then the following parameters are required for generating 
                                      # the projection basis by solving the eigenvalue problem related to the isotropic kernel:
                                      # the smoothing paramameter epsilonDIFF is searched with an iteration algorithm 
        epsilonDIFFmin        = 10    # epsilonDIFF is searched in interval [epsilonDIFFmin , +infty[
        step_epsilonDIFF      = 1.0   # step for searching the optimal value epsilonDIFF starting from epsilonDIFFmin
        iterlimit_epsilonDIFF = 220   # maximum number of the iteration algorithm for computing epsilonDIFF                              
        comp_ref              = 0.1   # value in  [ 0.1 , 0.5 [  used for stopping the iteration algorithm. 
                                      # if comp =  Rlambda(nbmDMAP+1)/Rlambda(nbmDMAP) <= comp_ref, then algorithm is stopped. 
                                      # The standard value for comp_ref is 0.2                       

        #===== Data block required for executing 'Task6_SolverDirect'   
        nbMC = 100  # number of realizations of (nu,n_d)-valued random matrix [H_ar]. 
                    # The number of learned realizations n_ar is n_ar  = n_d x nbMC.

        #--- parameters controlling the time-integration scheme of the ISDE       
        icorrectif = 1     # 0: usual Silveman-bandwidth formulation for which the normalization conditions are not exactly satisfied, 
                           # 1: modified Silverman bandwidth implying that, for any value of nu, the normalization conditions are verified  
        f0_ref = 4         # reference value (recommended value f0_ref = 4)
        ind_f0 = 0         # indicator for generating f0 (recommended value ind_f0 = 0): 
                           # if ind_f0 = 0, then f0 = f0_ref, and if ind_f0 = 1, then f0 = f0_ref/sh    
        coeffDeltar = 20   # coefficient > 0 (usual value is 20) for calculating Deltar
        M0transient = 30   # the end-integration value, M0transient (for instance, 30), at which the stationary response of the ISDE is 
                           # reached, is given by the user. The corresponding final time at which the realization is extracted from 
                           # solverDirect_Verlet is M0transient * Deltar 

        # for the inverse problem, if ind_type_targ = 1 (targets defined by giving N_r realizations)
        eps_inv = 0.001  # Tolerance for computing the pseudo-inverse of matrix MatRGammaS_iter with 
                         # sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv) for ind_type_targ = 1, 
                         # in sub_solverInverse_constrainedByTargets. An adapted value is 0.001. If problems occurs 
                         # increase the value to 0.01.

        # parameters controlling the convergence of the iteration algorithm related to the Lagrange multipliers for imposing the targets
        ind_coupling = 0    # 0: for ind_constraints = 2 or 3, no coupling in matrix MatRGammaS_iter (HIGHLY RECOMMENDED), 
                            # 1: for ind_constraints = 2 or 3, coupling all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
        iter_limit = 4000   # maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
        epsc = 0.01         # relative tolerance (for instance 1e-3) for the iteration-algorithm convergence 
                            # NOTE: on the convergence criteria for the iteration algorithm computing the Lagrange multipliers:
                            # Criterion 1: if iter > 10 and Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained 
                            #              for iter - 1. Convergence is then assumed to be reached and then, exit from the loop on iter
                            # Criterion 2: if Rerr(iter) <= epsc, the relative error of the constraint satisfaction is less than or equal to 
                            #              the tolerance. Convergence is reached, and exit from the loop on iter.
                            # Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc, the error is stationary and thus 
                            #              convergence is assumed to be reached and exit from the loop on iter

        #--- relaxation function iter --> alpha_relax(iter) controlling convergence of the iterative algorithm 
        #    is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
        alpha_relax1 = 0.001 # value of alpha_relax for iter = 1  (for instance 0.001)
        iter_relax2  = 20    # value of iter (for instance, 20) such that alpha_relax2 = alpha_relax(iter_relax2). 
                             # If iter_relax2 = 1, then alpha_relax(iter) = alpha_relax2 for all iter >= 1   
        alpha_relax2 = 0.009  # value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = alpha_relax2. 
                             # NOTE 1: If iter_relax2 = 1, then Ralpha_relax(iter) = alpha_relax2 for all iter >= 1
                             # NOTE 2: If iter_relax2 >= 2, then  
                             #         For iter >= 1 and for iter < iter_relax2, we have:
                             #             alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
                             #         For iter >= iter_relax2, we have:
                             #             alpha_relax(iter) = alpha_relax2
                             # NOTE 3: To decrease the error err(iter), increase the value of iter_relax2
                             # NOTE 4: If the iteration algorithm does not converge, decrease alpha_relax2 and/or increase iter_relax2  

        #--- data for the plots of H_d and H_ar
        #    in the example below, nu >=  9
        MatRplotHsamples = np.array([])           # 1D array of the components numbers of H_ar for which the realizations are plotted 
                                                  #         no plot, nbplotHsamples = 0
        MatRplotHClouds = np.array([              # 2D array containing the 3 components numbers of H_ar for which the clouds are plotted 
                                   [1,2,3],       #    plot of components 1,2,3 
                                   [65,66,67]     #    plot of components 65,66,67 
                                   ])             #    nbplotHClouds  = 2                                                               
                                                  #   
        MatRplotHpdf = np.array([1,2,3,65,66,67]) # 1D array containing the components numbers of H_ar for which the pdfs are plotted 
                                                  #   plot of components 1,2,3,65,66,67
                                                  #   nbplotHpdf = 6
        MatRplotHpdf2D = np.array([               # 2D array containing the 2 components numbers of H_ar for which the joint pdfs are plotted 
                                   [1,2],         #    plot for the components 1 and 2
                                   [66,67]        #    plot for the components 66 and 67.
                                   ])             #    nbplotHpdf2D = 2 

        #--- data for post-processing of Information Theory                                  
        ind_Kullback   = 1    # 0: no computation of the Kullback-Leibler divergence of H_ar with respect to H_d, 
                              # 1: computation of the Kullback-Leibler divergence of H_ar with respect to H_d
        ind_Entropy    = 1    # 0: no computation of the Entropy of Hd and Har, 
                              # 1: computation of the Entropy of Hd and Har 
        ind_MutualInfo = 1    # 0: no computation of the Mutual Information iHd and iHar for Hd and Har, 
                              # 1: computation of the Mutual Information iHd and iHar for Hd and Har

        #==== WARNING: save instruction must not be modified by the user 
        sio.savemat('FileDataWorkFlow3Step2.mat', {
            'ind_display_screen': ind_display_screen,
            'ind_print': ind_print,
            'ind_plot': ind_plot,
            'ind_parallel': ind_parallel,
            'mDP': mDP,
            'ind_generator': ind_generator,
            'ind_basis_type': ind_basis_type,
            'ind_file_type': ind_file_type,
            'epsilonDIFFmin': epsilonDIFFmin,
            'step_epsilonDIFF': step_epsilonDIFF,
            'iterlimit_epsilonDIFF': iterlimit_epsilonDIFF,
            'comp_ref': comp_ref,
            'nbMC': nbMC,
            'icorrectif': icorrectif,
            'f0_ref': f0_ref,
            'ind_f0': ind_f0,
            'coeffDeltar': coeffDeltar,
            'M0transient': M0transient,
            'eps_inv': eps_inv,
            'ind_coupling': ind_coupling,
            'iter_limit': iter_limit,
            'epsc': epsc,
            'alpha_relax1': alpha_relax1,
            'iter_relax2': iter_relax2,
            'alpha_relax2': alpha_relax2,
            'MatRplotHsamples': MatRplotHsamples,
            'MatRplotHClouds': MatRplotHClouds,
            'MatRplotHpdf': MatRplotHpdf,
            'MatRplotHpdf2D': MatRplotHpdf2D,
            'ind_Kullback': ind_Kullback,
            'ind_Entropy': ind_Entropy,
            'ind_MutualInfo': ind_MutualInfo
        }, do_compression=True)

    #------------------------- DATA BLOCK STEP3 ENTERED BY THE USER ----------------------------------------------------
    if ind_step3 == 1:
        #--- parameters and variables controlling execution of step3  
        ind_display_screen = 1  # 0 no display, 1 display
        ind_print          = 1  # 0 no print, 1 print
        ind_plot           = 1  # 0 no plot, 1 plot
        ind_parallel       = 1  # 0 no parallel computation, 1 parallel computation

        #===== Data block required for executing 'Task12_PlotXdXar' 

        # WARNING: the components of XX given for the plots must be a subset of the observed components XX_obs, that is to say 
        #          must belong to the set of components declared in mainWorkflow_Data_generation1.m in the array, using Matlab notation, 
        #          Indx_obs = [Indq_obs
        #                      n_q + Indw_obs]
        #          In the example below, it is assumed that n_q = 15, n_w = 4, 
        #          Indq_obs = [2 4 9 13 14]', and Indw_obs = [2 3 4]'
        #          then, Indx_obs = [2 4 9 13 14  17 18 19]'

        MatRplotSamples = np.array([])             # 1D array of components of XX for which the realizations are plotted 
                                                   #    no plot, nbplotSamples = 0
        MatRplotClouds = np.array([                # 2D array of components of XX for which the clouds are plotted 
                                  [34,57,256],     #    plot cloud of components XX34,XX57,XX256 that is QQ34,QQ57,QQ256
                                  [401,402,34],    #    plot cloud of components XX401,XX402,XX34 that is WW1,WW2,QQ34
                                  [401,402,256]    #    plot cloud of components XX401,XX402,XX256 that is WW1,WW2,QQ256
                                  ])               #    nbplotClouds = 3
        MatRplotPDF = np.array([34,40,57,245,256]) # 1D array of components of XX for which the pdfs are plotted 
                                                   #    plot the pdf of XX34,XX40,XX57,XX245,XX256 that is QQ34,QQ40,QQ57,QQ245,QQ256 
                                                   # nbplotPDF = 5. 
        MatRplotPDF2D = np.array([                 # 2D array of components of XX for which the joint pdfs are plotted 
                                 [33,57],          #    plot the joint pdf of XX33-XX57 that is QQ33-QQ57.
                                 [245,256],        #    plot the joint pdf of XX245-XX256 that is QQ245-QQ256
                                 [57,401],         #    plot the joint pdf of XX57-XX401 that is QQ57-WW1
                                 [57,402]          #    plot the joint pdf of XX57-XX402 that is QQ57-WW2
                                 ])                #     nbplotHpdf2D = 2   

        #==== WARNING: save instruction must not be modified by the user 
        sio.savemat('FileDataWorkFlowStep3.mat', {
            'ind_display_screen': ind_display_screen,
            'ind_print': ind_print,
            'ind_plot': ind_plot,
            'ind_parallel': ind_parallel,
            'MatRplotSamples': MatRplotSamples,
            'MatRplotClouds': MatRplotClouds,
            'MatRplotPDF': MatRplotPDF,
            'MatRplotPDF2D': MatRplotPDF2D
        }, do_compression=True)

    #------------------------- DATA BLOCK STEP4 ENTERED BY THE USER ----------------------------------------------------
    if ind_step4 == 1:
        #--- parameters and variables controlling execution of step4  
        ind_display_screen = 1  # 0 no display, 1 display
        ind_print          = 1  # 0 no print, 1 print
        ind_plot           = 1  # 0 no plot, 1 plot
        ind_parallel       = 1  # 0 no parallel computation, 1 parallel computation

        #===== Data block required for executing 'Task13_ConditionalStatistics' if ind_step4_task13 = 1
        if ind_step4_task13 == 1:
            ind_mean       = 0  # 0: No estimation of the conditional mean, 
                                # 1:    estimation of the conditional mean
            ind_mean_som   = 0  # 0: No estimation of the conditional mean and second-order moment, 
                                # 1:    estimation of the conditional mean and second-order moment
            ind_pdf        = 1  # 0: No estimation of the conditional pdf of component jcomponent <= nq_obs, 
                                # 1:    estimation of the conditional pdf of component jcomponent <= nq_obs
            ind_confregion = 1  # 0: No estimation of the conditional confidence region, 
                                # 1:    estimation of the conditional confidence region

            nbParam = 100   # WARNING: For the analysis of the conditional statistics of Step4, the organization of the components of the 
                            # QQ vector of the quantity of interest QoI is as follows (this organization must be planned from the 
                            # creation of the data in "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2. m" .
                            #    
                            # If the QoI depends on the sampling in nbParam points of a physical system parameter
                            # (such as time or frequency), if QoI_1, QoI_2, ... are the scalar quantities of interest, and if 
                            # % f_1,...,f_nbParam are the nbParam sampling points, the components of the QQ vector must be organized 
                            # as follows: 
                            # [(QoI_1,f_1) , (QoI_1,f_2), ... ,(QoI_1,f_nbParam), (QoI_2,f_1), (QoI_2,f_2), ... ,(QoI_2,f_nbParam), ... ]'.
                            #
                            # If nbParam > 1, this means that nq_obs is equal to nqobsPhys*nbParam, in which nqobsPhys is the number
                            # of the components of the state variables that are observed. Consequently, nq_obs/nbParam must be 
                            # an integer, if not, there is an error in the given value of nbParam or in the Data generation in 
                            # "mainWorkflow_Data_generation1.m" and "mainWorkflow_Data_generation2.m" 
                            #
                            # WARNING: NOTE THAT if such a parameter does not exist, it must be considered that nbParam = 1, but the 
                            # information structure must be consistent with the case nbParam > 1.   

            nbw0_obs = 2                        # number of vectors Rww0_obs de WW_obs used for conditional statistics Q_obs | WW_obs = Rww0_obs. 
                                                # 2D array MatRww0_obs(nw_obs,nbw0_obs) : MatRww0_obs(:,kw0) = Rww0_obs_kw0(nw_obs,1)
            MatRww0_obs = np.array([            # n_w = 2 and nbw0_obs = 2 
                                    [1,1.05], 
                                    [0.005,0.004]
                                    ])  

                                    #--- For ind_pdf = 1:
            Ind_Qcomp = np.array([34,40,57,245,256]) # 1D array Ind_Qcomp(nbQcomp): pdf of Q_obs_kk | WW_obs = Rww0_obs in which Q_obs_kk = MatRqq_obs(k,:). 
                                          # In which k is such that kk = Indq_obs(k). 
                                          # The components in Ind_Qcomp must be a subset of the observed components QQ_obs, that is to say 
                                          # must belong to the set of components declared in mainWorkflow_Data_generation1.m in the array 
                                          # Indq_obs. 
            nbpoint_pdf   = 200           # Number of points in which the pdf is computed
            pc_confregion = 0.95          # Only used if ind_confregion = 1

        #===== Data block required for executing 'Task14_PolynomialChaosZwiener' if ind_step4_task14 = 1
        if ind_step4_task14 == 1:
            ind_PCE_ident = 0    # 0: no identification of the PCE, 
                                 # 1:    identification of the PCE 
            ind_PCE_compt = 0    # 0: no computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M, 
                                 # 1:    computation of the PCE with plot for given values mu_PCE of mu and M_PCE of M
            nbMC_PCE      = 0    # number of realizations generated for [Z_PCE] and consequently for [H_PCE] with nbMC_PCE <= nbMC 
                                 # (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible)

            #--- parameters for PCE identification (ind_PCE_ident = 1):
            Rmu = np.array([])                    # 1D array Rmu(nbmu): nbmu values of the dimension mu of the germ (Xi_1,...,Xi_mu) of the PCE 
                                                  # with 1 <= mu <= nu, which are explored to find the optimal value muopt
            RM = np.array([])  # 1D array RM(nbM): nbM values of the maximum degree M of the PCE with 0 <= M, which are explored 
                                                  # to find the optimal value Mopt. 
                                                  # M = 0 : multi-index (0,0,...,0). 
                                                  # M = 1 : multi-indices (0,...,0) and (1,0,...,0),(0,1,...,0),...,(0,...,1): Gaussian representation

            #--- parameters for computing the PCE for given values mu = mu_PCE and M = M_PCE (ind_PCE_compt = 1):
            mu_PCE = 0     # value of mu for computing the realizations with the PCE with mu >= 1
            M_PCE  = 0     # value of M for computing the realizations with the PCE with M >= 0 

            #--- For ind_PCE_compt = 1 and ind_plot = 1, parameters and data controlling plots for H_ar and H_PCE
            #    in the example below, nu >=  9
            MatRplotHsamples = np.array([])        # 1D array of the components numbers of H_ar for which the realizations are plotted 
                                                   #    no plot, nbplotHsamples = 0
            MatRplotHClouds = np.array([])         # 2D array containing the 3 components numbers of H_ar for which the clouds are plotted                                    
                                                   #    no plot, nbplotHClouds = 0
            MatRplotHpdf = np.array([])            # 1D array containing the components numbers of H_ar for which the pdfs are plotted 
                                                   #    no plot, nbplotHpdf = 0
            MatRplotHpdf2D = np.array([])          # 2D array containing the 2 components numbers of H_ar for which the joint pdfs are plotted 
                                                   #    no plot, nbplotHpdf2D = 0

        #===== Data block required for executing 'Task15_PolynomialChaosQWU' if ind_step4_task15 = 1
        if ind_step4_task15 == 1:
            nbMC_PCE = 0   # number of learned realizations used for PCE is nar_PCE= nbMC_PCE x n_d 
                           # (HIGHLY RECOMMENDED TO TAKE nbMC_PCE = nbMC if possible) 
            Ng      = 0    # dimension of the germ for polynomial chaos Psi_alpha that is such that Ng = nw_obs   
            Ndeg    = 0    # max degree of the polynomial chaos Psi_alpha with Ndeg >= 1  
            ng      = 0    # dimension of the germ for polynomial chaos phi_a that is such that ng >= 1  
            ndeg    = 0    # max degree of the polynomial chaos phi_a with ndeg >= 0 (if ndeg = 0, then KU = 1)  
            MaxIter = 0    # maximum number of iteration used by the quasi-Newton optimization algorithm (example 400)

        #==== WARNING: save instruction must not be modified by the user 
        if ind_step4_task13 == 1:
            sio.savemat('FileDataWorkFlowStep4_task13.mat', {
                'ind_display_screen': ind_display_screen,
                'ind_print': ind_print,
                'ind_plot': ind_plot,
                'ind_parallel': ind_parallel,
                'ind_mean': ind_mean,
                'ind_mean_som': ind_mean_som,
                'ind_pdf': ind_pdf,
                'ind_confregion': ind_confregion,
                'nbParam': nbParam,
                'nbw0_obs': nbw0_obs,
                'MatRww0_obs': MatRww0_obs,
                'Ind_Qcomp': Ind_Qcomp,
                'nbpoint_pdf': nbpoint_pdf,
                'pc_confregion': pc_confregion
            }, do_compression=True)
        if ind_step4_task14 == 1:
            sio.savemat('FileDataWorkFlowStep4_task14.mat', {
                'ind_display_screen': ind_display_screen,
                'ind_print': ind_print,
                'ind_plot': ind_plot,
                'ind_parallel': ind_parallel,
                'ind_PCE_ident': ind_PCE_ident,
                'ind_PCE_compt': ind_PCE_compt,
                'nbMC_PCE': nbMC_PCE,
                'Rmu': Rmu,
                'RM': RM,
                'mu_PCE': mu_PCE,
                'M_PCE': M_PCE,
                'MatRplotHsamples': MatRplotHsamples,
                'MatRplotHClouds': MatRplotHClouds,
                'MatRplotHpdf': MatRplotHpdf,
                'MatRplotHpdf2D': MatRplotHpdf2D
            }, do_compression=True)
        if ind_step4_task15 == 1:
            sio.savemat('FileDataWorkFlowStep4_task15.mat', {
                'ind_display_screen': ind_display_screen,
                'ind_print': ind_print,
                'ind_plot': ind_plot,
                'ind_parallel': ind_parallel,
                'nbMC_PCE': nbMC_PCE,
                'Ng': Ng,
                'Ndeg': Ndeg,
                'ng': ng,
                'ndeg': ndeg,
                'MaxIter': MaxIter
            }, do_compression=True)

    return (ind_step1, ind_step2, ind_step3, ind_step4, ind_step4_task13, ind_step4_task14, ind_step4_task15, ind_SavefileStep3, ind_SavefileStep4)