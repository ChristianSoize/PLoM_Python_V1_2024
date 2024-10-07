import os
import numpy as np
import scipy.io as sio
import sys
import pickle
import types
from joblib import parallel_backend
from typing import Callable
from sub_data_structure_and_check import sub_data_structure_and_check
from sub_scaling import sub_scaling
from sub_PCA import sub_PCA
from sub_partition1 import sub_partition1
from sub_projection_target import sub_projection_target
from sub_projection_basis_NoPartition import sub_projection_basis_NoPartition
from sub_solverDirect import sub_solverDirect
from sub_solverDirectPartition import sub_solverDirectPartition
from sub_solverInverse import sub_solverInverse
from sub_PCAback import sub_PCAback
from sub_scalingBack import sub_scalingBack 
from sub_plot_Xd_Xar import sub_plot_Xd_Xar
from sub_conditional_statistics import sub_conditional_statistics
from sub_polynomial_chaosZWiener import sub_polynomial_chaosZWiener
from sub_polynomialChaosQWU import sub_polynomialChaosQWU

def mainWorkflow(max_workers:int,ind_workflow:int,mainWorkflow_Data_Workflow:Callable):
    #----------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 3 August 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: mainWorkflow
    #  Subject      : managing the work flow and data for Benchmark1_Direct_NoPartition_PCE
    #
    #  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #                [3] C. Soize, R. Ghanem, Physics-constrained non-Gaussian probabilistic learning on manifolds, 
    #                       International Journal for Numerical Methods in Engineering, doi: 10.1002/nme.6202, 121 (1), 110-145 (2020).
    #                [4] C. Soize, R. Ghanem, Probabilistic learning on manifolds constrained by nonlinear partial differential equations 
    #                       from small datasets, Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2021.113777, 
    #                       380, 113777 (2021).
    #                [5] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #                [6] C. Soize, Probabilistic learning inference of boundary value problem with uncertainties based on Kullback-Leibler 
    #                       divergence under implicit constraints, Computer Methods in Applied Mechanics and Engineering,
    #                       doi:10.1016/j.cma.2022.115078, 395, 115078 (2022). 
    #                [7] C. Soize, Probabilistic learning constrained by realizations using a weak formulation of Fourier transform of 
    #                       probability measures, Computational Statistics, doi:10.1007/s00180-022-01300-w, 38(4),1879–1925 (2023).
    #                [8] C. Soize, R. Ghanem, Probabilistic-learning-based stochastic surrogate model from small incomplete datasets 
    #                       for nonlinear dynamical systems,Computer Methods in Applied Mechanics and Engineering, 
    #                       doi:10.1016/j.cma.2023.116498, 418, 116498, pp.1-25 (2024).
    #                [9] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
    #                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
    #               [10] C. Soize, R. Ghanem, Physical systems with random uncertainties: Chaos representation with arbitrary probability 
    #                       measure, SIAM Journal on Scientific Computing, doi:10.1137/S1064827503424505, 26}2), 395-410 (2004).
    #               [11] C. Soize, C. Desceliers, Computational aspects for constructing realizations of polynomial chaos in high 
    #                       dimension}, SIAM Journal On Scientific Computing, doi:10.1137/100787830, 32(5), 2820-2831 (2010).
    #               [13] C. Soize, Q-D. To, Polynomial-chaos-based conditional statistics for probabilistic learning with heterogeneous
    #                       data applied to atomic collisions of Helium on graphite substrate, Journal of Computational Physics,
    #                       doi:10.1016/j.jcp.2023.112582, 496, 112582, pp.1-20 (2024).
    #
    #--------------------------------------------------------------------------------------------------------------------------------------
    #                  WorkFlow1                 |                  WorkFlow2                 |                  WorkFlow3 |              |  
    #        SolverDirect_WithoutPartition       |          SolverDirect_WithPartition        |        SolverInverse_WithoutPartition     |     
    #--------------------------------------------|--------------------------------------------|-------------------------------------------|
    #  Step1_Pre_processing                      |  Step1_Pre_processing                      |   Step1_Pre_processing                    |
    #        Task1_DataStructureCheck            |        Task1_DataStructureCheck            |         Task1_DataStructureCheck          |
    #        Task2_Scaling                       |        Task2_Scaling                       |         Task2_Scaling                     |
    #        Task3_PCA                           |        Task3_PCA                           |         Task3_PCA                         |
    #                                            |        Task4_Partition                     |         Task8_ProjectionTarget            |
    #--------------------------------------------|--------------------------------------------|-------------------------------------------|
    #  Step2_Processing                          |  Step2_Processing                          |   Step2_Processing                        |
    #        Task5_ProjectionBasisNoPartition    |                                            |         Task5_ProjectionBasisNoPartition  |
    #        Task6_SolverDirect                  |        Task7_SolverDirectPartition         |         Task9_SolverInverse               |
    #--------------------------------------------|--------------------------------------------|-------------------------------------------|
    #  Step3_Post_processing                     |  Step3_Post_processing                     |   Step3_Post_processing                   |
    #        Task10_PCAback                      |        Task10_PCAback                      |         Task10_PCAback                    |
    #        Task11_ScalingBack                  |        Task11_ScalingBack                  |         Task11_ScalingBack                |
    #        Task12_PlotXdXar                    |        Task12_PlotXdXar                    |         Task12_PlotXdXar                  |
    #--------------------------------------------|--------------------------------------------|-------------------------------------------|
    #  Step4_Conditional_statistics_processing   |  Step4_Conditional_statistics_processing   |   Step4_Conditional_statistics_processing |
    #        Task13_ConditionalStatistics        |        Task13_ConditionalStatistics        |         Task13_ConditionalStatistics      |
    #        Task14_PolynomialChaosZwiener       |                                            |         Task14_PolynomialChaosZwiener     |
    #        Task15_PolynomialChaosQWU           |        Task15_PolynomialChaosQWU           |         Task15_PolynomialChaosQWU         |
    #--------------------------------------------|--------------------------------------------|-------------------------------------------|
    #
    #--- IMPORTANT
    #    For the supervised analysis the information structure is 
    #    XX = (QQ,WW) with dimension n_x = n_q + n_w, 
    #
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                           BEGIN CODE SEQUENCE - DO NOT MODIFY    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Initialization
    ind_scaling = 0
    error_PCA = 0.0
    n_q = 0
    Indq_real = np.array([],dtype=int)
    Indq_pos = np.array([],dtype=int)
    n_w = 0
    Indw_real = np.array([],dtype=int)
    Indw_pos = np.array([],dtype=int)
    Indq_obs = np.array([],dtype=int)
    Indw_obs = np.array([],dtype=int)
    n_d = 0
    MatRxx_d = np.array([])
    mDP = 0
    nbmDMAP = 0
    ind_generator = 0
    ind_basis_type = 0
    ind_file_type = 0
    epsilonDIFFmin = 0.0
    step_epsilonDIFF = 0.0
    iterlimit_epsilonDIFF = 0
    comp_ref = 0.0
    nbMC = 0
    icorrectif = 0
    f0_ref = 0.0
    ind_f0 = 0
    coeffDeltar = 0.0
    M0transient = 0
    ind_constraints = 0
    ind_coupling = 0
    iter_limit = 0
    epsc = 0.0
    minVarH = 0.0
    maxVarH = 0.0
    alpha_relax1 = 0.0
    iter_relax2 = 0
    alpha_relax2 = 0.0
    MatRplotHsamples = np.array([],dtype=int)
    MatRplotHClouds = np.array([],dtype=int)
    MatRplotHpdf = np.array([],dtype=int)
    MatRplotHpdf2D = np.array([],dtype=int)
    ind_Kullback = 0
    ind_Entropy = 0
    ind_MutualInfo = 0
    MatRplotSamples = np.array([],dtype=int)
    MatRplotClouds = np.array([],dtype=int)
    MatRplotPDF = np.array([],dtype=int)
    MatRplotPDF2D = np.array([],dtype=int)
    ind_mean = 0
    ind_mean_som = 0
    ind_pdf = 0
    ind_confregion = 0
    nbParam = 0
    nbw0_obs = 0
    MatRww0_obs = np.array([])
    Ind_Qcomp = np.array([],dtype=int)
    nbpoint_pdf = 0
    pc_confregion = 0.0
    ind_PCE_ident = 0
    ind_PCE_compt = 0
    nbMC_PCE = 0
    Rmu = np.array([])
    RM = np.array([],dtype=int)
    mu_PCE = 0
    M_PCE = 0
    Ng = 0
    Ndeg = 0
    ng = 0
    ndeg = 0
    MaxIter = 0
    RINDEPref = np.array([])
    MatRHplot = np.array([],dtype=int)
    MatRcoupleRHplot = np.array([],dtype=int)
    ind_type_targ = 0
    Indq_targ_real = np.array([],dtype=int)
    Indq_targ_pos = np.array([],dtype=int)
    Indw_targ_real = np.array([],dtype=int)
    Indw_targ_pos = np.array([],dtype=int)
    N_r = 0
    MatRxx_targ_real = np.array([])
    MatRxx_targ_pos = np.array([])
    Rmeanxx_targ = np.array([])
    MatRcovxx_targ = np.array([])
    eps_inv = 0.0
    ind_SavefileStep3 = 0
    ind_SavefileStep4 = 0
    ngroup    = 0
    Igroup    = np.array([],dtype=int)
    MatIgroup = np.array([],dtype=int)
    Rb_targ1     = np.array([])
    coNr         = 0.0
    coNr2        = 0.0
    MatReta_targ = np.array([])
    Rb_targ2     = np.array([])
    Rb_targ3     = np.array([])
    ArrayWienner = np.array([])
    ArrayZ_ar    = np.array([]) 
    MatRg        = np.array([]) 
    MatRa        = np.array([]) 

    # Defining the workflow types, which must not be modified by the user
    nbworkflow = 3
    cellRworkflow = [
        'WorkFlow1_SolverDirect_WithoutPartition',
        'WorkFlow2_SolverDirect_WithPartition',
        'WorkFlow3_SolverInverse_WithoutPartition'
    ]

    # Defining all the existing tasks for all the workflows, which must not be modified by the user
    nbtask = 15
    cellRtask = [
        'Task1_DataStructureCheck',
        'Task2_Scaling',
        'Task3_PCA',
        'Task4_Partition',
        'Task5_ProjectionBasisNoPartition',
        'Task6_SolverDirect',
        'Task7_SolverDirectPartition',
        'Task8_ProjectionTarget',
        'Task9_SolverInverse',
        'Task10_PCAback',
        'Task11_ScalingBack',
        'Task12_PlotXdXar',
        'Task13_ConditionalStatistics',
        'Task14_PolynomialChaosZwiener',
        'Task15_PolynomialChaosQWU'
    ]

    # Initialization of the execution of the tasks, which must not be modified by the user
    exec_task1 = 0
    exec_task2 = 0
    exec_task3 = 0
    exec_task4 = 0
    exec_task5 = 0
    exec_task6 = 0
    exec_task7 = 0
    exec_task8 = 0
    exec_task9 = 0
    exec_task10 = 0
    exec_task11 = 0
    exec_task12 = 0
    exec_task13 = 0
    exec_task14 = 0
    exec_task15 = 0

    # Defining the steps, which must not be modified by the user
    nbstep = 4
    cellRstep = [
        'Step1_Pre_processing',
        'Step2_Processing',
        'Step3_Post_processing',
        'Step4_Conditional_statistics_processing'
    ]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                           BEGIN CODE SEQUENCE - DO NOT MODIFY    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                     BEGIN USER DATA DEFINITION SEQUENCE
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # the user must define the maximum number of workers used for parallel computation
    # max_workers = 96      # in this example the max number is 96
    
    # use of parallel_backend to configure gobally the max number of workers
    with parallel_backend('loky', n_jobs = max_workers):
    
        # ind_workflow must be given by the user
        # ind_workflow   # = 1: WorkFlow1_SolverDirect_WithoutPartition
                         # = 2: WorkFlow2_SolverDirect_WithPartition
                         # = 3: WorkFlow3_SolverInverse_WithoutPartition

        # DATA FOR "WorkFlow1_SolverDirect_WithoutPartition" (ind_workflow = 1)
        # The user must describe the data inside function mainWorkflow_Data_Workflow1
        if ind_workflow == 1:
            (ind_step1, ind_step2, ind_step3, ind_step4, ind_step4_task13, ind_step4_task14,
            ind_step4_task15, ind_SavefileStep3, ind_SavefileStep4) = mainWorkflow_Data_Workflow()

        # DATA FOR "WorkFlow2_SolverDirect_WithPartition" (ind_workflow = 2)
        # The user must describe the data inside function mainWorkflow_Data_Workflow2
        if ind_workflow == 2:
            (ind_step1, ind_step2, ind_step3, ind_step4, ind_step4_task13, ind_step4_task14,
            ind_step4_task15, ind_SavefileStep3, ind_SavefileStep4) = mainWorkflow_Data_Workflow()

        # DATA FOR "WorkFlow3_SolverInverse_WithoutPartition" (ind_workflow = 3)
        # The user must describe the data inside function mainWorkflow_Data_Workflow3
        if ind_workflow == 3:
            (ind_step1, ind_step2, ind_step3, ind_step4, ind_step4_task13, ind_step4_task14,
            ind_step4_task15, ind_SavefileStep3, ind_SavefileStep4) = mainWorkflow_Data_Workflow()

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                                           END USER DATA DEFINITION SEQUENCE
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                                           BEGIN CODE SEQUENCE - DO NOT MODIFY    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        #===============================================================================================================
        #                                                  CHECKING DATA
        #===============================================================================================================

        # Save the current values of the control parameters on a temporary file of the job
        
        sio.savemat('FileTemporary.mat', {
            'ind_step1': ind_step1,
            'ind_step2': ind_step2,
            'ind_step3': ind_step3,
            'ind_step4': ind_step4,
            'ind_step4_task13': ind_step4_task13,
            'ind_step4_task14': ind_step4_task14,
            'ind_step4_task15': ind_step4_task15,
            'ind_SavefileStep3': ind_SavefileStep3,
            'ind_SavefileStep4': ind_SavefileStep4
        })        
        
        # The two following "if sequence" must not be modified by the user
        if ind_workflow == 1 or ind_workflow == 2:
            ind_exec_solver = 1
            Indq_targ_real = np.array([],dtype=int)
            Indq_targ_pos  = np.array([],dtype=int)
            Indw_targ_real = np.array([],dtype=int)
            Indw_targ_pos  = np.array([],dtype=int)
            ind_type_targ  = 0
            N_r            = 0
            MatRxx_targ_real = np.array([])
            MatRxx_targ_pos  = np.array([])
            Rmeanxx_targ     = np.array([])
            MatRcovxx_targ   = np.array([])

        if ind_workflow == 3:
            ind_exec_solver = 2

        # Checking the consistency of parameters
        if ind_workflow != 1 and ind_workflow != 2 and ind_workflow != 3:
            raise ValueError('STOP1 in mainWorkflow: ind_workflow must be equal to 1, 2 or 3')
        if ind_step1 != 0 and ind_step1 != 1:
            raise ValueError('STOP2 in mainWorkflow: ind_step1 must be equal to 0 or 1')
        if ind_step2 != 0 and ind_step2 != 1:
            raise ValueError('STOP3 in mainWorkflow: ind_step2 must be equal to 0 or 1')
        if ind_step3 != 0 and ind_step3 != 1:
            raise ValueError('STOP4 in mainWorkflow: ind_step3 must be equal to 0 or 1')
        if ind_step4 != 0 and ind_step4 != 1:
            raise ValueError('STOP5 in mainWorkflow: ind_step4 must be equal to 0 or 1')
        if ind_step4_task13 != 0 and ind_step4_task13 != 1:
            raise ValueError('STOP6 in mainWorkflow: ind_step4_task13 must be equal to 0 or 1')
        if ind_step4_task14 != 0 and ind_step4_task14 != 1:
            raise ValueError('STOP7 in mainWorkflow: ind_step4_task14 must be equal to 0 or 1')
        if ind_step4_task15 != 0 and ind_step4_task15 != 1:
            raise ValueError('STOP8 in mainWorkflow: ind_step4_task15 must be equal to 0 or 1')
        if ind_SavefileStep3 != 0 and ind_SavefileStep3 != 1:
            raise ValueError('STOP9 in mainWorkflow: ind_SavefileStep3 must be equal to 0 or 1')
        if ind_SavefileStep4 != 0 and ind_SavefileStep4 != 1:
            raise ValueError('STOP10 in mainWorkflow: ind_SavefileStep4 must be equal to 0 or 1')

        # Print information
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(f' ---------------- {cellRworkflow[ind_workflow - 1]} ---------------- \n')
            fidlisting.write('      \n ')
            fidlisting.write(f'  {cellRstep[0]}                    = {ind_step1:1d} \n')
            fidlisting.write(f'   {cellRstep[1]}                        = {ind_step2:1d} \n')
            fidlisting.write(f'   {cellRstep[2]}                   = {ind_step3:1d} \n')
            fidlisting.write(f'   {cellRstep[3]} = {ind_step4:1d} \n')
        
        #---------------------------------------------------------------------------------------------------------------
        #                                                 Step1_Pre_processing
        #---------------------------------------------------------------------------------------------------------------
        if ind_step1 == 1:
            np.random.seed(0) # Initialize random generator

            # Execution of the tasks of Step2
            if ind_workflow == 1 or ind_workflow == 2 or ind_workflow == 3:

                if ind_workflow == 1:
                    # Load FileDataWorkFlow1Step1.mat
                    fileName = 'FileDataWorkFlow1Step1.mat'
                    if os.path.isfile(fileName):
                        data = sio.loadmat(fileName)
                        ind_display_screen = int(data['ind_display_screen'][0, 0])
                        ind_print = int(data['ind_print'][0, 0])
                        ind_plot = int(data['ind_plot'][0, 0])
                        ind_parallel = int(data['ind_parallel'][0, 0])
                        ind_scaling = int(data['ind_scaling'][0, 0])
                        error_PCA = data['error_PCA'][0, 0]
                        n_q = int(data['n_q'][0, 0])
                        Indq_real = data['Indq_real'].flatten().astype(int)  # Conversion in 1D
                        Indq_pos = data['Indq_pos'].flatten().astype(int)    # Conversion in 1D
                        n_w = int(data['n_w'][0, 0])
                        Indw_real = data['Indw_real'].flatten().astype(int)  # Conversion in 1D
                        Indw_pos = data['Indw_pos'].flatten().astype(int)    # Conversion in 1D
                        Indq_obs = data['Indq_obs'].flatten().astype(int)    # Conversion in 1D
                        Indw_obs = data['Indw_obs'].flatten().astype(int)    # Conversion in 1D
                        n_d = int(data['n_d'][0, 0])
                        MatRxx_d = data['MatRxx_d']  # 2D array, pas de conversion nécessaire
                    else:
                        raise ValueError('STOP11 in mainWorkflow: File FileDataWorkFlow1Step1.mat does not exist')

                if ind_workflow == 2:
                    # Load FileDataWorkFlow2Step1.mat
                    fileName = 'FileDataWorkFlow2Step1.mat'
                    if os.path.isfile(fileName):
                        data = sio.loadmat(fileName)
                        ind_display_screen = int(data['ind_display_screen'][0][0])
                        ind_print    = int(data['ind_print'][0][0])
                        ind_plot     = int(data['ind_plot'][0][0])
                        ind_parallel = int(data['ind_parallel'][0][0])
                        ind_scaling  = int(data['ind_scaling'][0][0])
                        error_PCA = data['error_PCA'][0][0]
                        n_q       = int(data['n_q'][0][0])
                        Indq_real = data['Indq_real'].flatten().astype(int)
                        Indq_pos  = data['Indq_pos'].flatten().astype(int)
                        n_w       = int(data['n_w'][0][0])
                        Indw_real = data['Indw_real'].flatten().astype(int)
                        Indw_pos  = data['Indw_pos'].flatten().astype(int)
                        Indq_obs  = data['Indq_obs'].flatten().astype(int)
                        Indw_obs  = data['Indw_obs'].flatten().astype(int)
                        n_d       = int(data['n_d'][0][0])
                        MatRxx_d  = data['MatRxx_d']
                        RINDEPref = data['RINDEPref'].flatten() 
                        MatRHplot = data['MatRHplot'].flatten().astype(int) 
                        MatRcoupleRHplot = data['MatRcoupleRHplot'].astype(int)
                    else:
                        raise ValueError('STOP12 in mainWorkflow: File FileDataWorkFlow2Step1.mat does not exist')

                if ind_workflow == 3:
                    # Load FileDataWorkFlow3Step1.mat
                    fileName = 'FileDataWorkFlow3Step1.mat'
                    if os.path.isfile(fileName):
                        data = sio.loadmat(fileName)
                        ind_display_screen = int(data['ind_display_screen'][0][0])
                        ind_print    = int(data['ind_print'][0][0])
                        ind_plot     = int(data['ind_plot'][0][0])
                        ind_parallel = int(data['ind_parallel'][0][0])
                        ind_scaling  = int(data['ind_scaling'][0][0])
                        error_PCA    = data['error_PCA'][0][0]
                        n_q       = int(data['n_q'][0][0])
                        Indq_real = data['Indq_real'].flatten().astype(int)
                        Indq_pos  = data['Indq_pos'].flatten().astype(int)
                        n_w       = int(data['n_w'][0][0])
                        Indw_real = data['Indw_real'].flatten().astype(int)
                        Indw_pos  = data['Indw_pos'].flatten().astype(int)
                        Indq_obs  = data['Indq_obs'].flatten().astype(int)
                        Indw_obs  = data['Indw_obs'].flatten().astype(int)
                        n_d       = int(data['n_d'][0][0])
                        MatRxx_d  = data['MatRxx_d']
                        ind_type_targ  = int(data['ind_type_targ'][0][0])
                        Indq_targ_real = data['Indq_targ_real'].flatten().astype(int)
                        Indq_targ_pos  = data['Indq_targ_pos'].flatten().astype(int)
                        Indw_targ_real = data['Indw_targ_real'].flatten().astype(int)
                        Indw_targ_pos  = data['Indw_targ_pos'].flatten().astype(int)
                        N_r            = int(data['N_r'][0][0])
                        MatRxx_targ_real = data['MatRxx_targ_real']
                        MatRxx_targ_pos  = data['MatRxx_targ_pos']
                        Rmeanxx_targ     = data['Rmeanxx_targ'].flatten() 
                        MatRcovxx_targ   = data['MatRcovxx_targ']
                    else:
                        raise ValueError('STOP13 in mainWorkflow: File FileDataWorkFlow3Step1.mat does not exist')

                if ind_display_screen == 1:
                    print(' ================================ Step1 Pre_processing ================================ ')

                if ind_print == 1:
                    with open('listing.txt', 'a+') as fidlisting:
                        fidlisting.write('      \n ')
                        fidlisting.write(' ================================ Step1 Pre_processing ================================ \n ')
                        fidlisting.write('      \n ')

                # Task1_DataStructureCheck
                (n_x, Indx_real, Indx_pos, nx_obs, Indx_obs, Indx_targ_real, Indx_targ_pos, nx_targ, Indx_targ) = \
                    sub_data_structure_and_check(n_q, Indq_real, Indq_pos, n_w, Indw_real, Indw_pos, n_d, MatRxx_d,
                                                Indq_obs, Indw_obs, ind_display_screen, ind_print, ind_exec_solver,
                                                Indq_targ_real, Indq_targ_pos, Indw_targ_real, Indw_targ_pos, ind_type_targ,
                                                N_r, MatRxx_targ_real, MatRxx_targ_pos, Rmeanxx_targ, MatRcovxx_targ)
                exec_task1 = 1

                # Task2_Scaling
                (MatRx_d, Rbeta_scale_real, Ralpha_scale_real, Ralpham1_scale_real, Rbeta_scale_log, Ralpha_scale_log,
                Ralpham1_scale_log) = \
                    sub_scaling(n_x, n_d, MatRxx_d, Indx_real, Indx_pos, ind_display_screen, ind_print, ind_scaling)
                exec_task2 = 1

                # Task3_PCA
                (nu, nnull, MatReta_d, RmuPCA, MatRVectPCA) = sub_PCA(n_x, n_d, MatRx_d, error_PCA, ind_display_screen, ind_print, ind_plot)
                exec_task3 = 1
                SAVERANDendSTEP1 = np.random.get_state()
                
            if ind_workflow == 2:
                # Task4_Partition
                SAVERANDstartPARTITION = np.random.get_state()
                nref = RINDEPref.shape[0]
                (ngroup, Igroup, MatIgroup,SAVERANDendPARTITION) = sub_partition1(nu, n_d, nref, MatReta_d, RINDEPref, SAVERANDstartPARTITION,
                              ind_display_screen,ind_print, ind_plot, MatRHplot, MatRcoupleRHplot, ind_parallel)
                exec_task4 = 1
                SAVERANDendSTEP1 = SAVERANDendPARTITION
                
            if ind_workflow == 3:
                # Task8_ProjectionTarget
                (Rb_targ1, coNr, coNr2, MatReta_targ, Rb_targ2, Rb_targ3) = \
                    sub_projection_target(n_x, n_d, MatRx_d, ind_exec_solver, ind_scaling, ind_type_targ, Indx_targ_real,
                                        Indx_targ_pos, nx_targ, Indx_targ, N_r, MatRxx_targ_real, MatRxx_targ_pos,
                                        Rmeanxx_targ, MatRcovxx_targ, nu, RmuPCA, MatRVectPCA, ind_display_screen, ind_print, ind_parallel,
                                        Rbeta_scale_real, Ralpham1_scale_real, Rbeta_scale_log, Ralpham1_scale_log)
                exec_task8 = 1
                SAVERANDendSTEP1 = np.random.get_state()
            
            # save the random generator state    
            with open("SaveGeneratorSTEP1.pkl", "wb") as file:
                pickle.dump(SAVERANDendSTEP1, file)
                       
            # SavefileStep1.mat   
            filename = 'SavefileStep1.mat'

            # Check if the file already exists
            if os.path.exists(filename):
                print(f"The file '{filename}' already exists and will be overwritten.")
            else:
                print(f"The file '{filename}' does not exist. Creating new file.")

            # save
            sio.savemat('SavefileStep1.mat', {
                'Ind_Qcomp': Ind_Qcomp,
                'Indq_obs': Indq_obs,
                'Indq_pos': Indq_pos,
                'Indq_real': Indq_real,
                'Indq_targ_pos': Indq_targ_pos,
                'Indq_targ_real': Indq_targ_real,
                'Indw_obs': Indw_obs,
                'Indw_pos': Indw_pos,
                'Indw_real': Indw_real,
                'Indw_targ_pos': Indw_targ_pos,
                'Indw_targ_real': Indw_targ_real,
                'Indx_obs': Indx_obs,
                'Indx_pos': Indx_pos,
                'Indx_real': Indx_real,
                'Indx_targ': Indx_targ,
                'Indx_targ_pos': Indx_targ_pos,
                'Indx_targ_real': Indx_targ_real,
                'M0transient': M0transient,
                'M_PCE': M_PCE,
                'MatRHplot': MatRHplot,
                'MatRVectPCA': MatRVectPCA,
                'MatRcoupleRHplot': MatRcoupleRHplot,
                'MatRcovxx_targ': MatRcovxx_targ,
                'MatReta_d': MatReta_d,
                'MatRplotClouds': MatRplotClouds,
                'MatRplotHClouds': MatRplotHClouds,
                'MatRplotHpdf': MatRplotHpdf,
                'MatRplotHpdf2D': MatRplotHpdf2D,
                'MatRplotHsamples': MatRplotHsamples,
                'MatRplotPDF': MatRplotPDF,
                'MatRplotPDF2D': MatRplotPDF2D,
                'MatRplotSamples': MatRplotSamples,
                'MatRww0_obs': MatRww0_obs,
                'MatRx_d': MatRx_d,
                'MatRxx_d': MatRxx_d,
                'MatRxx_targ_pos': MatRxx_targ_pos,
                'MatRxx_targ_real': MatRxx_targ_real,
                'MaxIter': MaxIter,
                'N_r': N_r,
                'Ndeg': Ndeg,
                'Ng': Ng,
                'RINDEPref': RINDEPref,
                'RM': RM,
                'Ralpha_scale_log': Ralpha_scale_log,
                'Ralpha_scale_real': Ralpha_scale_real,
                'Ralpham1_scale_log': Ralpham1_scale_log,
                'Ralpham1_scale_real': Ralpham1_scale_real,
                'Rbeta_scale_log': Rbeta_scale_log,
                'Rbeta_scale_real': Rbeta_scale_real,
                'Rmeanxx_targ': Rmeanxx_targ,
                'Rmu': Rmu,
                'RmuPCA': RmuPCA,
                'alpha_relax1': alpha_relax1,
                'alpha_relax2': alpha_relax2,
                'coeffDeltar': coeffDeltar,
                'comp_ref': comp_ref,
                'eps_inv': eps_inv,
                'epsc': epsc,
                'epsilonDIFFmin': epsilonDIFFmin,
                'error_PCA': error_PCA,
                'exec_task1': exec_task1,
                'exec_task10': exec_task10,
                'exec_task11': exec_task11,
                'exec_task12': exec_task12,
                'exec_task13': exec_task13,
                'exec_task14': exec_task14,
                'exec_task15': exec_task15,
                'exec_task2': exec_task2,
                'exec_task3': exec_task3,
                'exec_task4': exec_task4,
                'exec_task5': exec_task5,
                'exec_task6': exec_task6,
                'exec_task7': exec_task7,
                'exec_task8': exec_task8,
                'exec_task9': exec_task9,
                'f0_ref': f0_ref,
                'icorrectif': icorrectif,
                'ind_Entropy': ind_Entropy,
                'ind_Kullback': ind_Kullback,
                'ind_MutualInfo': ind_MutualInfo,
                'ind_PCE_compt': ind_PCE_compt,
                'ind_PCE_ident': ind_PCE_ident,
                'ind_SavefileStep3': ind_SavefileStep3,
                'ind_SavefileStep4': ind_SavefileStep4,
                'ind_basis_type': ind_basis_type,
                'ind_confregion': ind_confregion,
                'ind_constraints': ind_constraints,
                'ind_coupling': ind_coupling,
                'ind_display_screen': ind_display_screen,
                'ind_exec_solver': ind_exec_solver,
                'ind_f0': ind_f0,
                'ind_file_type': ind_file_type,
                'ind_generator': ind_generator,
                'ind_mean': ind_mean,
                'ind_mean_som': ind_mean_som,
                'ind_parallel': ind_parallel,
                'ind_pdf': ind_pdf,
                'ind_plot': ind_plot,
                'ind_print': ind_print,
                'ind_scaling': ind_scaling,
                'ind_step1': ind_step1,
                'ind_step2': ind_step2,
                'ind_step3': ind_step3,
                'ind_step4': ind_step4,
                'ind_step4_task13': ind_step4_task13,
                'ind_step4_task14': ind_step4_task14,
                'ind_step4_task15': ind_step4_task15,
                'ind_type_targ': ind_type_targ,
                'ind_workflow': ind_workflow,
                'iter_limit': iter_limit,
                'iter_relax2': iter_relax2,
                'iterlimit_epsilonDIFF': iterlimit_epsilonDIFF,
                'mDP': mDP,
                'maxVarH': maxVarH,
                'minVarH': minVarH,
                'mu_PCE': mu_PCE,
                'n_d': n_d,
                'n_q': n_q,
                'n_w': n_w,
                'n_x': n_x,
                'nbMC': nbMC,
                'nbMC_PCE': nbMC_PCE,
                'nbmDMAP': nbmDMAP,
                'nbParam': nbParam,
                'nbpoint_pdf': nbpoint_pdf,
                'nbstep': nbstep,
                'nbtask': nbtask,
                'nbw0_obs': nbw0_obs,
                'nbworkflow': nbworkflow,
                'ndeg': ndeg,
                'ng': ng,
                'nnull': nnull,
                'nu': nu,
                'nx_obs': nx_obs,
                'nx_targ': nx_targ,
                'pc_confregion': pc_confregion,
                'step_epsilonDIFF': step_epsilonDIFF,
                'ngroup': ngroup,
                'Igroup': Igroup,
                'MatIgroup': MatIgroup,
                'Rb_targ1': Rb_targ1,
                'coNr': coNr,
                'coNr2': coNr2,       
                'MatReta_targ': MatReta_targ,
                'Rb_targ2': Rb_targ2,
                'Rb_targ3': Rb_targ3,
                'ArrayWienner': ArrayWienner,
                'ArrayZ_ar': ArrayZ_ar,
                'MatRg': MatRg,
                'MatRa': MatRa
            }, do_compression=True)
          
            
        #-----------------------------------------------------------------------------------------------------------------
        #                                                Step2_Processing
        #-----------------------------------------------------------------------------------------------------------------

        if ind_step2 == 1:

            # Load SavefileStep1.mat
            filename = "SavefileStep1.mat"

            # Check if the file exists before loading
            if os.path.exists(filename):
                data = sio.loadmat(filename) 
                Ind_Qcomp = data['Ind_Qcomp'].flatten().astype(int) 
                Indq_obs  = data['Indq_obs'].flatten().astype(int) 
                Indq_pos  = data['Indq_pos'].flatten().astype(int) 
                Indq_real = data['Indq_real'].flatten().astype(int) 
                Indq_targ_pos  = data['Indq_targ_pos'].flatten().astype(int) 
                Indq_targ_real = data['Indq_targ_real'].flatten().astype(int) 
                Indw_obs  = data['Indw_obs'].flatten().astype(int) 
                Indw_pos  = data['Indw_pos'].flatten().astype(int)
                Indw_real = data['Indw_real'].flatten().astype(int) 
                Indw_targ_pos  = data['Indw_targ_pos'].flatten().astype(int) 
                Indw_targ_real = data['Indw_targ_real'].flatten().astype(int) 
                Indx_obs  = data['Indx_obs'].flatten().astype(int) 
                Indx_pos  = data['Indx_pos'].flatten().astype(int) 
                Indx_real = data['Indx_real'].flatten().astype(int) 
                Indx_targ = data['Indx_targ'].flatten().astype(int) 
                Indx_targ_pos  = data['Indx_targ_pos'].flatten().astype(int) 
                Indx_targ_real = data['Indx_targ_real'].flatten().astype(int)
                M0transient    = int(data['M0transient'][0][0])
                M_PCE = int(data['M_PCE'][0][0])
                MatRHplot = data['MatRHplot'].flatten().astype(int) 
                MatRVectPCA = data['MatRVectPCA']
                MatRcoupleRHplot = data['MatRcoupleRHplot'].astype(int)
                MatRcovxx_targ  = data['MatRcovxx_targ']
                MatReta_d       = data['MatReta_d']
                MatRplotClouds  = data['MatRplotClouds'].astype(int)
                MatRplotHClouds = data['MatRplotHClouds'].astype(int)
                MatRplotHpdf    = data['MatRplotHpdf'].flatten().astype(int) 
                MatRplotHpdf2D  = data['MatRplotHpdf2D'].astype(int)
                MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int) 
                MatRplotPDF = data['MatRplotPDF'].flatten().astype(int) 
                MatRplotPDF2D = data['MatRplotPDF2D'].astype(int)
                MatRplotSamples = data['MatRplotSamples'].flatten().astype(int) 
                MatRww0_obs = data['MatRww0_obs']
                MatRx_d = data['MatRx_d']
                MatRxx_d = data['MatRxx_d']
                MatRxx_targ_pos = data['MatRxx_targ_pos']
                MatRxx_targ_real = data['MatRxx_targ_real']
                MaxIter = data['MaxIter'].astype(int)[0][0]
                N_r  = int(data['N_r'][0][0])
                Ndeg = int(data['Ndeg'][0][0])
                Ng   = int(data['Ng'][0][0])
                RINDEPref           = data['RINDEPref'].flatten()
                RM                  = data['RM'].flatten().astype(int)
                Ralpha_scale_log    = data['Ralpha_scale_log'].flatten()
                Ralpha_scale_real   = data['Ralpha_scale_real'].flatten()
                Ralpham1_scale_log  = data['Ralpham1_scale_log'].flatten()
                Ralpham1_scale_real = data['Ralpham1_scale_real'].flatten()
                Rbeta_scale_log     = data['Rbeta_scale_log'].flatten()
                Rbeta_scale_real    = data['Rbeta_scale_real'].flatten()
                Rmeanxx_targ = data['Rmeanxx_targ'].flatten()
                Rmu          = data['Rmu'].flatten().astype(int)
                RmuPCA       = data['RmuPCA'].flatten()
                alpha_relax1 = data['alpha_relax1'][0][0]
                alpha_relax2 = data['alpha_relax2'][0][0]
                coeffDeltar  = data['coeffDeltar'][0][0]
                comp_ref     = data['comp_ref'][0][0]
                eps_inv      = data['eps_inv'][0][0]
                epsc         = data['epsc'][0][0]
                epsilonDIFFmin = data['epsilonDIFFmin'][0][0]
                error_PCA   = data['error_PCA'][0][0]
                exec_task1  = data['exec_task1'][0][0]
                exec_task10 = data['exec_task10'][0][0]
                exec_task11 = data['exec_task11'][0][0]
                exec_task12 = data['exec_task12'][0][0]
                exec_task13 = data['exec_task13'][0][0]
                exec_task14 = data['exec_task14'][0][0]
                exec_task15 = data['exec_task15'][0][0]
                exec_task2  = data['exec_task2'][0][0]
                exec_task3  = data['exec_task3'][0][0]
                exec_task4  = data['exec_task4'][0][0]
                exec_task5  = data['exec_task5'][0][0]
                exec_task6 = data['exec_task6'][0][0]
                exec_task7 = data['exec_task7'][0][0]
                exec_task8 = data['exec_task8'][0][0]
                exec_task9 = data['exec_task9'][0][0]
                f0_ref     = data['f0_ref'][0][0]
                icorrectif   = int(data['icorrectif'][0][0])
                ind_Entropy  = int(data['ind_Entropy'][0][0])
                ind_Kullback = int(data['ind_Kullback'][0][0])
                ind_MutualInfo = int(data['ind_MutualInfo'][0][0])
                ind_PCE_compt = int(data['ind_PCE_compt'][0][0])
                ind_PCE_ident = int(data['ind_PCE_ident'][0][0])
                ind_SavefileStep3 = int(data['ind_SavefileStep3'][0][0])
                ind_SavefileStep4 = int(data['ind_SavefileStep4'][0][0])
                ind_basis_type = int(data['ind_basis_type'][0][0])
                ind_confregion = int(data['ind_confregion'][0][0])
                ind_constraints = int(data['ind_constraints'][0][0])
                ind_coupling = int(data['ind_coupling'][0][0])
                ind_display_screen = int(data['ind_display_screen'][0][0])
                ind_exec_solver = int(data['ind_exec_solver'][0][0])
                ind_f0          = int(data['ind_f0'][0][0])
                ind_file_type   = int(data['ind_file_type'][0][0])
                ind_generator   = int(data['ind_generator'][0][0])
                ind_mean        = int(data['ind_mean'][0][0])
                ind_mean_som    = int(data['ind_mean_som'][0][0])
                ind_parallel    = int(data['ind_parallel'][0][0])
                ind_pdf         = int(data['ind_pdf'][0][0])
                ind_plot        = int(data['ind_plot'][0][0])
                ind_print       = int(data['ind_print'][0][0])
                ind_scaling     = int(data['ind_scaling'][0][0])
                ind_step1       = int(data['ind_step1'][0][0])
                ind_step2       = int(data['ind_step2'][0][0])
                ind_step3       = int(data['ind_step3'][0][0])
                ind_step4       = int(data['ind_step4'][0][0])
                ind_step4_task13 = int(data['ind_step4_task13'][0][0])
                ind_step4_task14 = int(data['ind_step4_task14'][0][0])
                ind_step4_task15 = int(data['ind_step4_task15'][0][0])
                ind_type_targ    = int(data['ind_type_targ'][0][0])
                ind_workflow     = int(data['ind_workflow'][0][0])
                iter_limit       = int(data['iter_limit'][0][0])
                iter_relax2      = int(data['iter_relax2'][0][0])
                iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'][0][0])
                mDP     = int(data['mDP'][0][0])
                maxVarH = data['maxVarH'][0][0]
                minVarH = data['minVarH'][0][0]
                mu_PCE   = int(data['mu_PCE'][0][0])
                n_d      = int(data['n_d'][0][0])
                n_q      = int(data['n_q'][0][0])
                n_w      = int(data['n_w'][0][0])
                n_x      = int(data['n_x'][0][0])
                nbMC     = int(data['nbMC'][0][0])
                nbMC_PCE = int(data['nbMC_PCE'][0][0])
                nbmDMAP  = int(data['nbmDMAP'][0][0])
                nbParam  = int(data['nbParam'][0][0])
                nbpoint_pdf = int(data['nbpoint_pdf'][0][0])
                nbstep      = int(data['nbstep'][0][0])
                nbtask      = int(data['nbtask'][0][0])
                nbw0_obs    = int(data['nbw0_obs'][0][0])
                nbworkflow  = int(data['nbworkflow'][0][0])
                ndeg        = int(data['ndeg'][0][0])
                ng          = int(data['ng'][0][0])
                nnull       = int(data['nnull'][0][0])
                nu          = int(data['nu'][0][0])
                nx_obs      = int(data['nx_obs'][0][0])
                nx_targ     = int(data['nx_targ'][0][0])
                pc_confregion = data['pc_confregion'][0][0]
                step_epsilonDIFF = data['step_epsilonDIFF'][0][0]
                ngroup    = int(data['ngroup'][0][0])
                Igroup    = data['Igroup'].flatten().astype(int) 
                MatIgroup = data['MatIgroup'].astype(int) 
                Rb_targ1     = data['Rb_targ1'].flatten()
                coNr         = data['coNr'][0][0]
                coNr2        = data['coNr2'][0][0]     
                MatReta_targ = data['MatReta_targ']
                Rb_targ2     = data['Rb_targ2'].flatten()
                Rb_targ3     = data['Rb_targ3'].flatten()
                ArrayWienner = data['ArrayWienner']
                ArrayZ_ar    = data['ArrayZ_ar'],
                MatRg        = data['MatRg'],
                MatRa        = data['MatRa']
            else:
                print(f"The file '{filename}' does not exist.")

            # restore the random generator state
            with open("SaveGeneratorSTEP1.pkl", "rb") as file:
                 SAVERANDendSTEP1_loaded = pickle.load(file)
                 np.random.set_state(SAVERANDendSTEP1_loaded) 
            
            if ind_workflow == 1:
                # Load FileDataWorkFlow1Step2.mat
                fileName = 'FileDataWorkFlow1Step2.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print          = int(data['ind_print'][0][0])
                    ind_plot           = int(data['ind_plot'][0][0])
                    ind_parallel       = int(data['ind_parallel'][0][0])
                    mDP                = int(data['mDP'][0][0])
                    ind_generator      = int(data['ind_generator'][0][0])
                    ind_basis_type     = int(data['ind_basis_type'][0][0])
                    ind_file_type      = int(data['ind_file_type'][0][0])                    
                    epsilonDIFFmin   = data['epsilonDIFFmin'][0][0]
                    step_epsilonDIFF = data['step_epsilonDIFF'][0][0]
                    iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'].astype(int)[0][0])
                    comp_ref    = data['comp_ref'][0][0]
                    nbMC        = int(data['nbMC'][0][0])
                    icorrectif  = int(data['icorrectif'][0][0])
                    f0_ref      = data['f0_ref'][0][0]
                    ind_f0      = int(data['ind_f0'][0][0])
                    coeffDeltar = data['coeffDeltar'][0][0]
                    M0transient = int(data['M0transient'][0][0])
                    ind_constraints = int(data['ind_constraints'][0][0])
                    ind_coupling    = int(data['ind_coupling'][0][0])
                    iter_limit      = int(data['iter_limit'][0][0])
                    epsc    = data['epsc'][0][0]
                    minVarH = data['minVarH'][0][0]
                    maxVarH = data['maxVarH'][0][0]
                    alpha_relax1 = data['alpha_relax1'][0][0]
                    iter_relax2  = int(data['iter_relax2'][0][0])
                    alpha_relax2 = data['alpha_relax2'][0][0]
                    MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                    MatRplotHClouds  = data['MatRplotHClouds'].astype(int) 
                    MatRplotHpdf     = data['MatRplotHpdf'].flatten().astype(int)  
                    MatRplotHpdf2D   = data['MatRplotHpdf2D'].astype(int) 
                    ind_Kullback     = int(data['ind_Kullback'][0][0])
                    ind_Entropy      = int(data['ind_Entropy'][0][0])
                    ind_MutualInfo   = int(data['ind_MutualInfo'][0][0])
                else:
                    raise ValueError('STOP15 in mainWorkflow: File FileDataWorkFlow1Step2.mat does not exist')

            if ind_workflow == 2:
                # Load FileDataWorkFlow2Step2.mat
                fileName = 'FileDataWorkFlow2Step2.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print          = int(data['ind_print'][0][0])
                    ind_plot           = int(data['ind_plot'][0][0])
                    ind_parallel       = int(data['ind_parallel'][0][0])
                    nbMC               = int(data['nbMC'][0][0])
                    ind_generator      = int(data['ind_generator'][0][0])
                    icorrectif         = int(data['icorrectif'][0][0])
                    f0_ref             = data['f0_ref'][0][0]
                    ind_f0             = int(data['ind_f0'][0][0])
                    coeffDeltar        = data['coeffDeltar'][0][0]
                    M0transient        = int(data['M0transient'][0][0])
                    epsilonDIFFmin     = data['epsilonDIFFmin'][0][0]
                    step_epsilonDIFF   = data['step_epsilonDIFF'][0][0]
                    iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'][0][0])
                    comp_ref        = data['comp_ref'][0][0]
                    ind_constraints = int(data['ind_constraints'][0][0])
                    ind_coupling    = int(data['ind_coupling'][0][0])
                    iter_limit      = int(data['iter_limit'][0][0])
                    epsc         = data['epsc'][0][0]
                    minVarH      = data['minVarH'][0][0]
                    maxVarH      = data['maxVarH'][0][0]
                    alpha_relax1 = data['alpha_relax1'][0][0]
                    iter_relax2  = int(data['iter_relax2'][0][0])
                    alpha_relax2 = data['alpha_relax2'][0][0]
                    MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                    MatRplotHClouds  = data['MatRplotHClouds'].astype(int) 
                    MatRplotHpdf     = data['MatRplotHpdf'].flatten().astype(int)  
                    MatRplotHpdf2D   = data['MatRplotHpdf2D'].astype(int) 
                    ind_Kullback     = int(data['ind_Kullback'][0][0])
                    ind_Entropy      = int(data['ind_Entropy'][0][0])
                    ind_MutualInfo   =int( data['ind_MutualInfo'][0][0])
                else:
                    raise ValueError('STOP16 in mainWorkflow: File FileDataWorkFlow2Step2.mat does not exist')

            if ind_workflow == 3:
                # Load FileDataWorkFlow3Step2.mat
                fileName = 'FileDataWorkFlow3Step2.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print          = int(data['ind_print'][0][0])
                    ind_plot           = int(data['ind_plot'][0][0])
                    ind_parallel       = int(data['ind_parallel'][0][0])
                    mDP                = int(data['mDP'][0][0])
                    ind_generator      = int(data['ind_generator'][0][0])
                    ind_basis_type     = int(data['ind_basis_type'][0][0])
                    ind_file_type      = int(data['ind_file_type'][0][0])
                    epsilonDIFFmin     = data['epsilonDIFFmin'][0][0]
                    step_epsilonDIFF  = data['step_epsilonDIFF'][0][0]
                    iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'][0][0])
                    comp_ref   = data['comp_ref'][0][0]
                    nbMC        = int(data['nbMC'][0][0])
                    icorrectif  = int(data['icorrectif'][0][0])
                    f0_ref      = data['f0_ref'][0][0]
                    ind_f0      = int(data['ind_f0'][0][0])
                    coeffDeltar = data['coeffDeltar'][0][0]
                    M0transient = int(data['M0transient'][0][0])
                    eps_inv     = data['eps_inv'][0][0]
                    ind_coupling = int(data['ind_coupling'][0][0])
                    iter_limit   = int(data['iter_limit'][0][0])
                    epsc         = data['epsc'][0][0]
                    alpha_relax1 = data['alpha_relax1'][0][0]
                    iter_relax2  = int(data['iter_relax2'][0][0])
                    alpha_relax2 = data['alpha_relax2'][0][0]
                    MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                    MatRplotHClouds  = data['MatRplotHClouds'].astype(int) 
                    MatRplotHpdf     = data['MatRplotHpdf'].flatten().astype(int)  
                    MatRplotHpdf2D   = data['MatRplotHpdf2D'].astype(int) 
                    ind_Kullback     = int(data['ind_Kullback'][0][0])
                    ind_Entropy      = int(data['ind_Entropy'][0][0])
                    ind_MutualInfo   =int( data['ind_MutualInfo'][0][0])
                else:
                    raise ValueError('STOP17 in mainWorkflow: File FileDataWorkFlow3Step2.mat does not exist')

            if ind_display_screen == 1:
                print(' ================================ Step2 Processing ================================ ')

            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' ================================ Step2 Processing ================================ \n ')
                    fidlisting.write('      \n ')
            # Checking that all the required tasks in Step 1 have correctly been executed before executing the tasks of Step2
            if exec_task1 != 1:
                raise ValueError('STOP18 in mainWorkflow: Step 2 cannot be executed because Task1_DataStructureCheck was not correctly executed')
            if exec_task2 != 1:
                raise ValueError('STOP19 in mainWorkflow: Step 2 cannot be executed because Task2_Scaling was not correctly executed')
            if exec_task3 != 1:
                raise ValueError('STOP20 in mainWorkflow: Step 2 cannot be executed because Task3_PCA was not correctly executed')
            if ind_workflow == 2:
                if exec_task4 != 1:
                    raise ValueError('STOP21 in mainWorkflow: Step 2 cannot be executed because Task4_Partition was not correctly executed')
            if ind_workflow == 3:
                if exec_task8 != 1:
                    raise ValueError('STOP22 in mainWorkflow: Step 2 cannot be executed because Task8_ProjectionTarget was not correctly executed')
            if ind_workflow == 3:
                if exec_task4 != 0:
                    raise ValueError('STOP23 in mainWorkflow: Step 2 cannot be executed because Inverse Analysis cannot be done with partition')
           
            # Execution of the tasks of Step2
            if ind_workflow == 1 or ind_workflow == 3:

                # Task5_ProjectionBasisNoPartition
                if ind_generator == 0:
                    nbmDMAP = n_d
                if ind_generator == 1:
                    nbmDMAP = nu + 1
                (MatRg, MatRa) = sub_projection_basis_NoPartition(nu, n_d, MatReta_d, ind_generator, mDP, nbmDMAP, ind_basis_type,
                                                                ind_file_type, ind_display_screen, ind_print, ind_plot, ind_parallel,
                                                                epsilonDIFFmin, step_epsilonDIFF, iterlimit_epsilonDIFF, comp_ref)
                exec_task5 = 1

            if ind_workflow == 1:
                # Task6_SolverDirect 
                SAVERANDstartDirect = np.random.get_state() 
                (n_ar, MatReta_ar, ArrayZ_ar, ArrayWienner, SAVERANDendDirect, d2mopt_ar, divKL, iHd, iHar, entropy_Hd, entropy_Har) = \
                    sub_solverDirect(nu, n_d, nbMC, MatReta_d, ind_generator, icorrectif, f0_ref, ind_f0, coeffDeltar, M0transient,
                                    nbmDMAP, MatRg, MatRa, ind_constraints, ind_coupling, iter_limit, epsc, minVarH, maxVarH, alpha_relax1,
                                    iter_relax2, alpha_relax2, SAVERANDstartDirect, ind_display_screen, ind_print, ind_plot, ind_parallel,
                                    MatRplotHsamples, MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, ind_Kullback, ind_Entropy, ind_MutualInfo)
                
                exec_task6 = 1
                SAVERANDendSTEP2 = SAVERANDendDirect

            if ind_workflow == 2:
                # Task7_SolverDirectPartition
                SAVERANDstartDirectPartition = np.random.get_state() 
                (n_ar, MatReta_ar, SAVERANDendDirectPartition, d2mopt_ar, divKL, iHd, iHar, entropy_Hd, entropy_Har) = \
                    sub_solverDirectPartition(nu, n_d, nbMC, MatReta_d, ind_generator, icorrectif, f0_ref, ind_f0, coeffDeltar, M0transient,
                                            epsilonDIFFmin, step_epsilonDIFF, iterlimit_epsilonDIFF, comp_ref, ind_constraints, ind_coupling,
                                            iter_limit, epsc, minVarH, maxVarH, alpha_relax1, iter_relax2, alpha_relax2, ngroup, Igroup, MatIgroup,
                                            SAVERANDstartDirectPartition, ind_display_screen, ind_print, ind_plot, ind_parallel, MatRplotHsamples,
                                            MatRplotHClouds, MatRplotHpdf, MatRplotHpdf2D, ind_Kullback, ind_Entropy, ind_MutualInfo)
                exec_task7 = 1
                SAVERANDendSTEP2 = SAVERANDendDirectPartition

            if ind_workflow == 3:
                # Task9_SolverInverse
                SAVERANDstartInverse = np.random.get_state()
                (n_ar,MatReta_ar,ArrayZ_ar,ArrayWienner,SAVERANDendInverse,d2mopt_ar,divKL,iHd,iHar,entropy_Hd,entropy_Har) = \
                    sub_solverInverse(nu,n_d,nbMC,MatReta_d,ind_generator,icorrectif,f0_ref,ind_f0,coeffDeltar,M0transient,
                                    nbmDMAP,MatRg,MatRa,ind_type_targ,N_r,Rb_targ1,coNr,coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3,
                                    ind_coupling,iter_limit,epsc,alpha_relax1,iter_relax2,alpha_relax2,SAVERANDstartInverse, 
                                    ind_display_screen,ind_print,ind_plot,ind_parallel,MatRplotHsamples,MatRplotHClouds,MatRplotHpdf,
                                    MatRplotHpdf2D,ind_Kullback,ind_Entropy,ind_MutualInfo)
                exec_task9 = 1
                SAVERANDendSTEP2 = SAVERANDendInverse  

            # save the random generator state    
            with open("SaveGeneratorSTEP2.pkl", "wb") as file:
                pickle.dump(SAVERANDendSTEP2, file)

            # SavefileStep1.mat   
            filename = 'SavefileStep2.mat'

            # Check if the file already exists
            if os.path.exists(filename):
                print(f"The file '{filename}' already exists and will be overwritten.")
            else:
                print(f"The file '{filename}' does not exist. Creating new file.")

            # save
            sio.savemat('SavefileStep2.mat', {
                'ArrayWienner': ArrayWienner,
                'ArrayZ_ar': ArrayZ_ar,
                'Ind_Qcomp': Ind_Qcomp,
                'Indq_obs': Indq_obs,
                'Indq_pos': Indq_pos,
                'Indq_real': Indq_real,
                'Indq_targ_pos': Indq_targ_pos,
                'Indq_targ_real': Indq_targ_real,
                'Indw_obs': Indw_obs,
                'Indw_pos': Indw_pos,
                'Indw_real': Indw_real,
                'Indw_targ_pos': Indw_targ_pos,
                'Indw_targ_real': Indw_targ_real,
                'Indx_obs': Indx_obs,
                'Indx_pos': Indx_pos,
                'Indx_real': Indx_real,
                'Indx_targ': Indx_targ,
                'Indx_targ_pos': Indx_targ_pos,
                'Indx_targ_real': Indx_targ_real,
                'M0transient': M0transient,
                'M_PCE': M_PCE,
                'MatRHplot': MatRHplot,
                'MatRVectPCA': MatRVectPCA,
                'MatRa': MatRa,
                'MatRcoupleRHplot': MatRcoupleRHplot,
                'MatRcovxx_targ': MatRcovxx_targ,
                'MatReta_ar': MatReta_ar,
                'MatReta_d': MatReta_d,
                'MatRg': MatRg,
                'MatRplotClouds': MatRplotClouds,
                'MatRplotHClouds': MatRplotHClouds,
                'MatRplotHpdf': MatRplotHpdf,
                'MatRplotHpdf2D': MatRplotHpdf2D,
                'MatRplotHsamples': MatRplotHsamples,
                'MatRplotPDF': MatRplotPDF,
                'MatRplotPDF2D': MatRplotPDF2D,
                'MatRplotSamples': MatRplotSamples,
                'MatRww0_obs': MatRww0_obs,
                'MatRx_d': MatRx_d,
                'MatRxx_d': MatRxx_d,
                'MatRxx_targ_pos': MatRxx_targ_pos,
                'MatRxx_targ_real': MatRxx_targ_real,
                'MaxIter': MaxIter,
                'N_r': N_r,
                'Ndeg': Ndeg,
                'Ng': Ng,
                'RINDEPref': RINDEPref,
                'RM': RM,
                'Ralpha_scale_log': Ralpha_scale_log,
                'Ralpha_scale_real': Ralpha_scale_real,
                'Ralpham1_scale_log': Ralpham1_scale_log,
                'Ralpham1_scale_real': Ralpham1_scale_real,
                'Rbeta_scale_log': Rbeta_scale_log,
                'Rbeta_scale_real': Rbeta_scale_real,
                'Rmeanxx_targ': Rmeanxx_targ,
                'Rmu': Rmu,
                'RmuPCA': RmuPCA,
                'alpha_relax1': alpha_relax1,
                'alpha_relax2': alpha_relax2,
                'coeffDeltar': coeffDeltar,
                'comp_ref': comp_ref,
                'd2mopt_ar': d2mopt_ar,
                'divKL': divKL,
                'entropy_Har': entropy_Har,
                'entropy_Hd': entropy_Hd,
                'eps_inv': eps_inv,
                'epsc': epsc,
                'epsilonDIFFmin': epsilonDIFFmin,
                'error_PCA': error_PCA,
                'exec_task1': exec_task1,
                'exec_task10': exec_task10,
                'exec_task11': exec_task11,
                'exec_task12': exec_task12,
                'exec_task13': exec_task13,
                'exec_task14': exec_task14,
                'exec_task15': exec_task15,
                'exec_task2': exec_task2,
                'exec_task3': exec_task3,
                'exec_task4': exec_task4,
                'exec_task5': exec_task5,
                'exec_task6': exec_task6,
                'exec_task7': exec_task7,
                'exec_task8': exec_task8,
                'exec_task9': exec_task9,
                'f0_ref': f0_ref,
                'iHar': iHar,
                'iHd': iHd,
                'icorrectif': icorrectif,
                'ind_Entropy': ind_Entropy,
                'ind_Kullback': ind_Kullback,
                'ind_MutualInfo': ind_MutualInfo,
                'ind_PCE_compt': ind_PCE_compt,
                'ind_PCE_ident': ind_PCE_ident,
                'ind_SavefileStep3': ind_SavefileStep3,
                'ind_SavefileStep4': ind_SavefileStep4,
                'ind_basis_type': ind_basis_type,
                'ind_confregion': ind_confregion,
                'ind_constraints': ind_constraints,
                'ind_coupling': ind_coupling,
                'ind_display_screen': ind_display_screen,
                'ind_exec_solver': ind_exec_solver,
                'ind_f0': ind_f0,
                'ind_file_type': ind_file_type,
                'ind_generator': ind_generator,
                'ind_mean': ind_mean,
                'ind_mean_som': ind_mean_som,
                'ind_parallel': ind_parallel,
                'ind_pdf': ind_pdf,
                'ind_plot': ind_plot,
                'ind_print': ind_print,
                'ind_scaling': ind_scaling,
                'ind_step1': ind_step1,
                'ind_step2': ind_step2,
                'ind_step3': ind_step3,
                'ind_step4': ind_step4,
                'ind_step4_task13': ind_step4_task13,
                'ind_step4_task14': ind_step4_task14,
                'ind_step4_task15': ind_step4_task15,
                'ind_type_targ': ind_type_targ,
                'ind_workflow': ind_workflow,
                'iter_limit': iter_limit,
                'iter_relax2': iter_relax2,
                'iterlimit_epsilonDIFF': iterlimit_epsilonDIFF,
                'mDP': mDP,
                'maxVarH': maxVarH,
                'minVarH': minVarH,
                'mu_PCE': mu_PCE,
                'n_ar': n_ar,
                'n_d': n_d,
                'n_q': n_q,
                'n_w': n_w,
                'n_x': n_x,
                'nbMC': nbMC,
                'nbMC_PCE': nbMC_PCE,
                'nbParam': nbParam,
                'nbmDMAP': nbmDMAP,
                'nbpoint_pdf': nbpoint_pdf,
                'nbstep': nbstep,
                'nbtask': nbtask,
                'nbw0_obs': nbw0_obs,
                'nbworkflow': nbworkflow,
                'ndeg': ndeg,
                'ng': ng,
                'nnull': nnull,
                'nu': nu,
                'nx_obs': nx_obs,
                'nx_targ': nx_targ,
                'pc_confregion': pc_confregion,
                'step_epsilonDIFF': step_epsilonDIFF
            }, do_compression=True)
            
        #-----------------------------------------------------------------------------------------------------------------
        #                                                 Step3_Post_processing
        #-----------------------------------------------------------------------------------------------------------------

        if ind_step3 == 1:

            # Load SavefileStep2.mat
            filename = "SavefileStep2.mat"

            # Check if the file exists before loading
            if os.path.exists(filename):
                data = sio.loadmat(filename) 
                ArrayWienner = data['ArrayWienner']                
                ArrayZ_ar    = data['ArrayZ_ar']
                Ind_Qcomp = data['Ind_Qcomp'].flatten().astype(int)  
                Indq_obs = data['Indq_obs'].flatten().astype(int)  
                Indq_pos = data['Indq_pos'].flatten().astype(int)  
                Indq_real = data['Indq_real'].flatten().astype(int)  
                Indq_targ_pos = data['Indq_targ_pos'].flatten().astype(int)  
                Indq_targ_real = data['Indq_targ_real'].flatten().astype(int)  
                Indw_obs = data['Indw_obs'].flatten().astype(int)  
                Indw_pos = data['Indw_pos'].flatten().astype(int)  
                Indw_real = data['Indw_real'].flatten().astype(int)  
                Indw_targ_pos = data['Indw_targ_pos'].flatten().astype(int)  
                Indw_targ_real = data['Indw_targ_real'].flatten().astype(int)  
                Indx_obs = data['Indx_obs'].flatten().astype(int)  
                Indx_pos = data['Indx_pos'].flatten().astype(int)  
                Indx_real = data['Indx_real'].flatten().astype(int)  
                Indx_targ = data['Indx_targ'].flatten().astype(int)  
                Indx_targ_pos = data['Indx_targ_pos'].flatten().astype(int)  
                Indx_targ_real = data['Indx_targ_real'].flatten().astype(int)  
                M0transient = int(data['M0transient'][0][0])
                M_PCE       = int(data['M_PCE'][0][0])
                MatRHplot   = data['MatRHplot'].flatten().astype(int)  
                MatRVectPCA = data['MatRVectPCA']
                MatRa       = data['MatRa']
                MatRcoupleRHplot = data['MatRcoupleRHplot'].astype(int) 
                MatRcovxx_targ = data['MatRcovxx_targ']
                MatReta_ar = data['MatReta_ar']
                MatReta_d = data['MatReta_d']
                MatRg = data['MatRg']
                MatRplotClouds   = data['MatRplotClouds'].astype(int) 
                MatRplotHClouds  = data['MatRplotHClouds'].astype(int) 
                MatRplotHpdf     = data['MatRplotHpdf'].flatten().astype(int)  
                MatRplotHpdf2D   = data['MatRplotHpdf2D'].astype(int) 
                MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                MatRplotPDF      = data['MatRplotPDF'].flatten().astype(int)  
                MatRplotPDF2D    = data['MatRplotPDF2D'].astype(int) 
                MatRplotSamples  = data['MatRplotSamples'].flatten().astype(int)  
                MatRww0_obs      = data['MatRww0_obs']
                MatRx_d          = data['MatRx_d']
                MatRxx_d         = data['MatRxx_d']
                MatRxx_targ_pos  = data['MatRxx_targ_pos']
                MatRxx_targ_real = data['MatRxx_targ_real']
                MaxIter          = int(data['MaxIter'][0][0])
                N_r              = int(data['N_r'][0][0])
                Ndeg             = int(data['Ndeg'][0][0])
                Ng               = int(data['Ng'][0][0])
                RINDEPref        = data['RINDEPref'].flatten() 
                RM               = data['RM'].flatten().astype(int)  
                Ralpha_scale_log    = data['Ralpha_scale_log'].flatten() 
                Ralpha_scale_real   = data['Ralpha_scale_real'].flatten() 
                Ralpham1_scale_log  = data['Ralpham1_scale_log'].flatten() 
                Ralpham1_scale_real = data['Ralpham1_scale_real'].flatten() 
                Rbeta_scale_log     = data['Rbeta_scale_log'].flatten() 
                Rbeta_scale_real    = data['Rbeta_scale_real'].flatten() 
                Rmeanxx_targ        = data['Rmeanxx_targ'].flatten() 
                Rmu    = data['Rmu'].flatten().astype(int) 
                RmuPCA = data['RmuPCA'].flatten() 
                alpha_relax1 = data['alpha_relax1'][0][0]
                alpha_relax2 = data['alpha_relax2'][0][0]
                coeffDeltar  = data['coeffDeltar'][0][0]
                comp_ref     = data['comp_ref'][0][0]
                d2mopt_ar    = data['d2mopt_ar'][0][0]
                divKL        = data['divKL'][0][0]
                entropy_Har  = data['entropy_Har'][0][0]
                entropy_Hd   = data['entropy_Hd'][0][0]
                eps_inv      = data['eps_inv'][0][0]
                epsc         = data['epsc'][0][0]
                epsilonDIFFmin = data['epsilonDIFFmin'][0][0]
                error_PCA   = data['error_PCA'][0][0]
                exec_task1  = data['exec_task1'][0][0]
                exec_task10 = data['exec_task10'][0][0]
                exec_task11 = data['exec_task11'][0][0]
                exec_task12 = data['exec_task12'][0][0]
                exec_task13 = data['exec_task13'][0][0]
                exec_task14 = data['exec_task14'][0][0]
                exec_task15 = data['exec_task15'][0][0]
                exec_task2 = data['exec_task2'][0][0]
                exec_task3 = data['exec_task3'][0][0]
                exec_task4 = data['exec_task4'][0][0]
                exec_task5 = data['exec_task5'][0][0]
                exec_task6 = data['exec_task6'][0][0]
                exec_task7 = data['exec_task7'][0][0]
                exec_task8 = data['exec_task8'][0][0]
                exec_task9 = data['exec_task9'][0][0]
                f0_ref  = data['f0_ref'][0][0]
                iHar    = data['iHar'][0][0]
                iHd     = data['iHd'][0][0]
                icorrectif     = int(data['icorrectif'][0][0])
                ind_Entropy    = int(data['ind_Entropy'][0][0])
                ind_Kullback   = int(data['ind_Kullback'][0][0])
                ind_MutualInfo = int(data['ind_MutualInfo'][0][0])
                ind_PCE_compt  = int(data['ind_PCE_compt'][0][0])
                ind_PCE_ident  = int(data['ind_PCE_ident'][0][0])
                ind_SavefileStep3 = int(data['ind_SavefileStep3'][0][0])
                ind_SavefileStep4 = int(data['ind_SavefileStep4'][0][0])
                ind_basis_type  = int(data['ind_basis_type'][0][0])
                ind_confregion  = int(data['ind_confregion'][0][0])
                ind_constraints = int(data['ind_constraints'][0][0])
                ind_coupling    = int(data['ind_coupling'][0][0])
                ind_display_screen = int(data['ind_display_screen'][0][0])
                ind_exec_solver    = int(data['ind_exec_solver'][0][0])
                ind_f0        = int(data['ind_f0'][0][0])
                ind_file_type = int(data['ind_file_type'][0][0])
                ind_generator = int(data['ind_generator'][0][0])
                ind_mean      = int(data['ind_mean'][0][0])
                ind_mean_som  = int(data['ind_mean_som'][0][0])
                ind_parallel  = int(data['ind_parallel'][0][0])
                ind_pdf       = int(data['ind_pdf'][0][0])
                ind_plot      = int(data['ind_plot'][0][0])
                ind_print     = int(data['ind_print'][0][0])
                ind_scaling   = int(data['ind_scaling'][0][0])
                ind_step1     = int(data['ind_step1'][0][0])
                ind_step2     = int(data['ind_step2'][0][0])
                ind_step3     = int(data['ind_step3'][0][0])
                ind_step4     = int(data['ind_step4'][0][0])
                ind_step4_task13 = int(data['ind_step4_task13'][0][0])
                ind_step4_task13 = int(data['ind_step4_task13'][0][0])
                ind_step4_task14 = int(data['ind_step4_task14'][0][0])
                ind_step4_task15 = int(data['ind_step4_task15'][0][0])
                ind_type_targ    = int(data['ind_type_targ'][0][0])
                ind_workflow     = int(data['ind_workflow'][0][0])
                iter_limit       = int(data['iter_limit'][0][0])
                iter_relax2      = int(data['iter_relax2'][0][0])
                iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'][0][0])
                mDP     = int(data['mDP'][0][0])
                maxVarH = data['maxVarH'][0][0]
                minVarH = data['minVarH'][0][0]
                mu_PCE  = int(data['mu_PCE'][0][0])
                n_ar    = int(data['n_ar'][0][0])
                n_d     = int(data['n_d'][0][0])
                n_q     = int(data['n_q'][0][0])
                n_w     = int(data['n_w'][0][0])
                n_x     = int(data['n_x'][0][0])
                nbMC    = int(data['nbMC'][0][0])
                nbMC_PCE = int(data['nbMC_PCE'][0][0])
                nbParam  = int(data['nbParam'][0][0])
                nbmDMAP  = int(data['nbmDMAP'][0][0])
                nbpoint_pdf = int(data['nbpoint_pdf'][0][0])
                nbstep      = int(data['nbstep'][0][0])
                nbtask      = int(data['nbtask'][0][0])
                nbw0_obs    = int(data['nbw0_obs'][0][0])
                nbworkflow  = int(data['nbworkflow'][0][0])
                ndeg        = int(data['ndeg'][0][0])
                ng          = int(data['ng'][0][0])
                nnull       = int(data['nnull'][0][0])
                nu          = int(data['nu'][0][0])
                nx_obs      = int(data['nx_obs'][0][0])
                nx_targ     = int(data['nx_targ'][0][0])
                pc_confregion    = data['pc_confregion'][0][0]
                step_epsilonDIFF = data['step_epsilonDIFF'][0][0]                
            else:
                print(f"The file '{filename}' does not exist.")

            # Restore the current values of the control parameters
            temp_data = sio.loadmat('FileTemporary.mat')
            ind_step1 = int(temp_data['ind_step1'][0][0])
            ind_step2 = int(temp_data['ind_step2'][0][0])
            ind_step3 = int(temp_data['ind_step3'][0][0])
            ind_step4 = int(temp_data['ind_step4'][0][0])
            ind_step4_task13  = int(temp_data['ind_step4_task13'][0][0])
            ind_step4_task14  = int(temp_data['ind_step4_task14'][0][0])
            ind_step4_task15  = int(temp_data['ind_step4_task15'][0][0])
            ind_SavefileStep3 = int(temp_data['ind_SavefileStep3'][0][0])
            ind_SavefileStep4 = int(temp_data['ind_SavefileStep4'][0][0])

            # restore the random generator state
            with open("SaveGeneratorSTEP2.pkl", "rb") as file:
                 SAVERANDendSTEP2_loaded = pickle.load(file)
                 np.random.set_state(SAVERANDendSTEP2_loaded) 

            # Load FileDataWorkFlowStep3.mat for ind_workflow = 1, 2, or 3
            fileName = 'FileDataWorkFlowStep3.mat'
            if os.path.isfile(fileName):
                data = sio.loadmat(fileName)
                ind_display_screen = int(data['ind_display_screen'][0][0])
                ind_print          = int(data['ind_print'][0][0])
                ind_plot           = int(data['ind_plot'][0][0])
                ind_parallel       = int(data['ind_parallel'][0][0])
                MatRplotSamples    = data['MatRplotSamples'].flatten().astype(int)  
                MatRplotClouds     = data['MatRplotClouds'].astype(int) 
                MatRplotPDF        = data['MatRplotPDF'].flatten().astype(int)  
                MatRplotPDF2D      = data['MatRplotPDF2D'].astype(int) 
            else:
                raise ValueError('STOP25 in mainWorkflow: File FileDataWorkFlowStep3.mat does not exist')

            if ind_display_screen == 1:
                print(' ================================ Step3 Post_processing ================================ ')

            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' ================================ Step3 Post_processing ================================ \n ')
                    fidlisting.write('      \n ')

            # Checking that all the required tasks in Step 2 have correctly been executed before executing the tasks of Step3
            if ind_workflow == 1 or ind_workflow == 3:
                if exec_task5 != 1:
                    raise ValueError('STOP26 in mainWorkflow: Step 3 cannot be executed because Task5_ProjectionBasisNoPartition was not correctly executed')
            if ind_workflow == 1:
                if exec_task6 != 1:
                    raise ValueError('STOP27 in mainWorkflow: Step 3 cannot be executed because Task6_SolverDirect was not correctly executed')
            if ind_workflow == 2:
                if exec_task7 != 1:
                    raise ValueError('STOP28 in mainWorkflow: Step 3 cannot be executed because Task7_SolverDirectPartition was not correctly executed')
            if ind_workflow == 3:
                if exec_task9 != 1:
                    raise ValueError('STOP29 in mainWorkflow: Step 3 cannot be executed because Task9_SolverInverse was not correctly executed')
                      
            # Execution of the tasks of Step3
            if ind_workflow == 1 or  ind_workflow == 2 or  ind_workflow == 3:
                # Task10_PCAback
                MatRx_obs = sub_PCAback(n_x, n_d, nu, n_ar, nx_obs, MatRx_d, MatReta_ar, Indx_obs, RmuPCA, MatRVectPCA,
                                        ind_display_screen, ind_print)
                exec_task10 = 1

                # Task11_ScalingBack
                MatRxx_obs = sub_scalingBack(nx_obs, n_x, n_ar, MatRx_obs, Indx_real, Indx_pos, Indx_obs, Rbeta_scale_real,
                                            Ralpha_scale_real, Rbeta_scale_log, Ralpha_scale_log, ind_display_screen, ind_print, ind_scaling)
                exec_task11 = 1
                
                # Task12_PlotXdXar
                sub_plot_Xd_Xar(n_x, n_q, n_w, n_d, n_ar, nu, MatRxx_d, MatRx_d, MatReta_ar, RmuPCA, MatRVectPCA, Indx_real,
                                Indx_pos, nx_obs, Indx_obs, ind_scaling, Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log,
                                Ralpha_scale_log, MatRplotSamples, MatRplotClouds, MatRplotPDF, MatRplotPDF2D, ind_display_screen, ind_print)
                exec_task12 = 1
           
            ## save file of STEP3 is not implemented in Version V1
            # SAVERANDendSTEP3 = np.random.get_state() 

            ## save the random generator state    
            # with open("SaveGeneratorSTEP3.pkl", "wb") as file:
            #    pickle.dump(SAVERANDendSTEP3, file)

            # SavefileStep3.mat
            fileName = 'SavefileStep3.mat'
            if ind_SavefileStep3 == 1:               
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(f'  The file "{fileName}" is not saved in version V1\n')
                    fidlisting.write('      \n ')
                print(f'The file "{fileName}" is not saved in version V1')

        #------------------------------------------------------------------------------------------------------------------------------------
        #                                                 Step4_Conditional_statistics_processing
        #------------------------------------------------------------------------------------------------------------------------------------
    
        if ind_step4 == 1:            
            # Load SavefileStep2.mat
            filename = "SavefileStep2.mat"

            # Check if the file exists before loading
            if os.path.exists(filename):
                data = sio.loadmat(filename) 
                ArrayWienner = data['ArrayWienner']
                ArrayZ_ar = data['ArrayZ_ar']
                Ind_Qcomp = data['Ind_Qcomp'].flatten().astype(int)  
                Indq_obs = data['Indq_obs'].flatten().astype(int)  
                Indq_pos = data['Indq_pos'].flatten().astype(int)  
                Indq_real = data['Indq_real'].flatten().astype(int)  
                Indq_targ_pos = data['Indq_targ_pos'].flatten().astype(int)  
                Indq_targ_real = data['Indq_targ_real'].flatten().astype(int)  
                Indw_obs = data['Indw_obs'].flatten().astype(int)  
                Indw_pos = data['Indw_pos'].flatten().astype(int)  
                Indw_real = data['Indw_real'].flatten().astype(int)  
                Indw_targ_pos = data['Indw_targ_pos'].flatten().astype(int)  
                Indw_targ_real = data['Indw_targ_real'].flatten().astype(int)  
                Indx_obs = data['Indx_obs'].flatten().astype(int)  
                Indx_pos = data['Indx_pos'].flatten().astype(int)  
                Indx_real = data['Indx_real'].flatten().astype(int)  
                Indx_targ = data['Indx_targ'].flatten().astype(int)  
                Indx_targ_pos = data['Indx_targ_pos'].flatten().astype(int)  
                Indx_targ_real = data['Indx_targ_real'].flatten().astype(int)  
                M0transient = int(data['M0transient'][0][0])
                M_PCE = int(data['M_PCE'][0][0])
                MatRHplot = data['MatRHplot'].flatten().astype(int)  
                MatRVectPCA = data['MatRVectPCA']
                MatRa = data['MatRa']
                MatRcoupleRHplot = data['MatRcoupleRHplot'].astype(int) 
                MatRcovxx_targ = data['MatRcovxx_targ']
                MatReta_ar = data['MatReta_ar']
                MatReta_d = data['MatReta_d']
                MatRg = data['MatRg']
                MatRplotClouds = data['MatRplotClouds'].astype(int) 
                MatRplotHClouds = data['MatRplotHClouds'].astype(int) 
                MatRplotHpdf = data['MatRplotHpdf'].flatten().astype(int)  
                MatRplotHpdf2D = data['MatRplotHpdf2D'].astype(int) 
                MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                MatRplotPDF = data['MatRplotPDF'].flatten().astype(int)  
                MatRplotPDF2D = data['MatRplotPDF2D'].astype(int) 
                MatRplotSamples = data['MatRplotSamples'].flatten().astype(int)  
                MatRww0_obs = data['MatRww0_obs']
                MatRx_d = data['MatRx_d']
                MatRxx_d = data['MatRxx_d']
                MatRxx_targ_pos = data['MatRxx_targ_pos']
                MatRxx_targ_real = data['MatRxx_targ_real']
                MaxIter = int(data['MaxIter'][0][0])
                N_r = int(data['N_r'][0][0])
                Ndeg = int(data['Ndeg'][0][0])
                Ng = int(data['Ng'][0][0])
                RINDEPref = data['RINDEPref'].flatten() 
                RM = data['RM'].flatten().astype(int)  
                Ralpha_scale_log = data['Ralpha_scale_log'].flatten() 
                Ralpha_scale_real = data['Ralpha_scale_real'].flatten() 
                Ralpham1_scale_log = data['Ralpham1_scale_log'].flatten() 
                Ralpham1_scale_real = data['Ralpham1_scale_real'].flatten() 
                Rbeta_scale_log = data['Rbeta_scale_log'].flatten() 
                Rbeta_scale_real = data['Rbeta_scale_real'].flatten() 
                Rmeanxx_targ = data['Rmeanxx_targ'].flatten() 
                Rmu = data['Rmu'].flatten().astype(int) 
                RmuPCA = data['RmuPCA'].flatten() 
                alpha_relax1 = data['alpha_relax1'][0][0]
                alpha_relax2 = data['alpha_relax2'][0][0]
                coeffDeltar = data['coeffDeltar'][0][0]
                comp_ref = data['comp_ref'][0][0]
                d2mopt_ar = data['d2mopt_ar'][0][0]
                divKL = data['divKL'][0][0]
                entropy_Har = data['entropy_Har'][0][0]
                entropy_Hd = data['entropy_Hd'][0][0]
                eps_inv = data['eps_inv'][0][0]
                epsc = data['epsc'][0][0]
                epsilonDIFFmin = data['epsilonDIFFmin'][0][0]
                error_PCA = data['error_PCA'][0][0]
                exec_task1 = data['exec_task1'][0][0]
                exec_task10 = data['exec_task10'][0][0]
                exec_task11 = data['exec_task11'][0][0]
                exec_task12 = data['exec_task12'][0][0]
                exec_task13 = data['exec_task13'][0][0]
                exec_task14 = data['exec_task14'][0][0]
                exec_task15 = data['exec_task15'][0][0]
                exec_task2 = data['exec_task2'][0][0]
                exec_task3 = data['exec_task3'][0][0]
                exec_task4 = data['exec_task4'][0][0]
                exec_task5 = data['exec_task5'][0][0]
                exec_task6 = data['exec_task6'][0][0]
                exec_task7 = data['exec_task7'][0][0]
                exec_task8 = data['exec_task8'][0][0]
                exec_task9 = data['exec_task9'][0][0]
                f0_ref = data['f0_ref'].astype(int)[0][0]
                iHar = data['iHar'][0][0]
                iHd = data['iHd'][0][0]
                icorrectif = int(data['icorrectif'][0][0])
                ind_Entropy = int(data['ind_Entropy'][0][0])
                ind_Kullback = int(data['ind_Kullback'][0][0])
                ind_MutualInfo = int(data['ind_MutualInfo'][0][0])
                ind_PCE_compt = int(data['ind_PCE_compt'][0][0])
                ind_PCE_ident = int(data['ind_PCE_ident'][0][0])
                ind_SavefileStep3 = int(data['ind_SavefileStep3'][0][0])
                ind_SavefileStep4 = int(data['ind_SavefileStep4'][0][0])
                ind_basis_type = int(data['ind_basis_type'][0][0])
                ind_confregion = int(data['ind_confregion'][0][0])
                ind_constraints = int(data['ind_constraints'][0][0])
                ind_coupling = int(data['ind_coupling'][0][0])
                ind_display_screen = int(data['ind_display_screen'][0][0])
                ind_exec_solver = int(data['ind_exec_solver'][0][0])
                ind_f0 = int(data['ind_f0'][0][0])
                ind_file_type = int(data['ind_file_type'][0][0])
                ind_generator = int(data['ind_generator'][0][0])
                ind_mean = int(data['ind_mean'][0][0])
                ind_mean_som = int(data['ind_mean_som'][0][0])
                ind_parallel = int(data['ind_parallel'][0][0])
                ind_pdf = int(data['ind_pdf'][0][0])
                ind_plot = int(data['ind_plot'][0][0])
                ind_print = int(data['ind_print'][0][0])
                ind_scaling = int(data['ind_scaling'][0][0])
                ind_step1 = int(data['ind_step1'][0][0])
                ind_step2 = int(data['ind_step2'][0][0])
                ind_step3 = int(data['ind_step3'][0][0])
                ind_step4 = int(data['ind_step4'][0][0])
                ind_step4_task13 = int(data['ind_step4_task13'][0][0])
                ind_step4_task14 = int(data['ind_step4_task14'][0][0])
                ind_step4_task15 = int(data['ind_step4_task15'][0][0])
                ind_type_targ = int(data['ind_type_targ'][0][0])
                ind_workflow = int(data['ind_workflow'][0][0])
                iter_limit = int(data['iter_limit'][0][0])
                iter_relax2 = int(data['iter_relax2'][0][0])
                iterlimit_epsilonDIFF = int(data['iterlimit_epsilonDIFF'][0][0])
                mDP = int(data['mDP'][0][0])
                maxVarH = data['maxVarH'][0][0]
                minVarH = data['minVarH'][0][0]
                mu_PCE = int(data['mu_PCE'][0][0])
                n_ar = int(data['n_ar'][0][0])
                n_d = int(data['n_d'][0][0])
                n_q = int(data['n_q'][0][0])
                n_w = int(data['n_w'][0][0])
                n_x = int(data['n_x'][0][0])
                nbMC = int(data['nbMC'][0][0])
                nbMC_PCE = int(data['nbMC_PCE'][0][0])
                nbParam = int(data['nbParam'][0][0])
                nbmDMAP = int(data['nbmDMAP'][0][0])
                nbpoint_pdf = int(data['nbpoint_pdf'][0][0])
                nbstep = int(data['nbstep'][0][0])
                nbtask = int(data['nbtask'][0][0])
                nbw0_obs = int(data['nbw0_obs'][0][0])
                nbworkflow = int(data['nbworkflow'][0][0])
                ndeg = int(data['ndeg'][0][0])
                ng = int(data['ng'][0][0])
                nnull = int(data['nnull'][0][0])
                nu = int(data['nu'][0][0])
                nx_obs = int(data['nx_obs'][0][0])
                nx_targ = int(data['nx_targ'][0][0])
                pc_confregion = data['pc_confregion'][0][0]
                step_epsilonDIFF = data['step_epsilonDIFF'][0][0]                
            else:
                print(f"The file '{filename}' does not exist.")
            
            # Restore the current values of the control parameters
            temp_data = sio.loadmat('FileTemporary.mat')
            ind_step1 = int(temp_data['ind_step1'][0][0])
            ind_step2 = int(temp_data['ind_step2'][0][0])
            ind_step3 = int(temp_data['ind_step3'][0][0])
            ind_step4 = int(temp_data['ind_step4'][0][0])
            ind_step4_task13 = int(temp_data['ind_step4_task13'][0][0])
            ind_step4_task14 = int(temp_data['ind_step4_task14'][0][0])
            ind_step4_task15 = int(temp_data['ind_step4_task15'][0][0])
            ind_SavefileStep3 = int(temp_data['ind_SavefileStep3'][0][0])
            ind_SavefileStep4 = int(temp_data['ind_SavefileStep4'][0][0])
            
            # restore the random generator state
            with open("SaveGeneratorSTEP2.pkl", "rb") as file:
                 SAVERANDendSTEP2_loaded = pickle.load(file)
                 np.random.set_state(SAVERANDendSTEP2_loaded) 

            # Load FileDataWorkFlowStep4.mat_task13, task14, or task15 for ind_workflow = 1, 2, or 3
            if ind_step4_task13 == 1:
                fileName = 'FileDataWorkFlowStep4_task13.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print = int(data['ind_print'][0][0])
                    ind_plot = int(data['ind_plot'][0][0])
                    ind_parallel = int(data['ind_parallel'][0][0])
                    ind_mean = int(data['ind_mean'][0][0])
                    ind_mean_som = int(data['ind_mean_som'][0][0])
                    ind_pdf = int(data['ind_pdf'][0][0])
                    ind_confregion = int(data['ind_confregion'][0][0])
                    nbParam = int(data['nbParam'][0][0])
                    nbw0_obs = int(data['nbw0_obs'][0][0])
                    MatRww0_obs = data['MatRww0_obs']
                    Ind_Qcomp = data['Ind_Qcomp'].flatten().astype(int) 
                    nbpoint_pdf = int(data['nbpoint_pdf'][0][0])
                    pc_confregion = data['pc_confregion'][0][0]
                else:
                    raise ValueError('STOP31 in mainWorkflow: File FileDataWorkFlowStep4_task13.mat does not exist')

            if ind_step4_task14 == 1:
                fileName = 'FileDataWorkFlowStep4_task14.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print = int(data['ind_print'][0][0])
                    ind_plot = int(data['ind_plot'][0][0])
                    ind_parallel = int(data['ind_parallel'][0][0])
                    ind_PCE_ident = int(data['ind_PCE_ident'][0][0])
                    ind_PCE_compt = int(data['ind_PCE_compt'][0][0])
                    nbMC_PCE = int(data['nbMC_PCE'][0][0])
                    Rmu = data['Rmu'].flatten().astype(int) 
                    RM = data['RM'].flatten().astype(int) 
                    mu_PCE = int(data['mu_PCE'][0][0])
                    M_PCE = int(data['M_PCE'][0][0])
                    MatRplotHsamples = data['MatRplotHsamples'].flatten().astype(int)  
                    MatRplotHClouds = data['MatRplotHClouds'].astype(int) 
                    MatRplotHpdf = data['MatRplotHpdf'].flatten().astype(int)  
                    MatRplotHpdf2D = data['MatRplotHpdf2D'].astype(int) 
                else:
                    raise ValueError('STOP32 in mainWorkflow: File FileDataWorkFlowStep4_task14.mat does not exist')

            if ind_step4_task15 == 1:
                fileName = 'FileDataWorkFlowStep4_task15.mat'
                if os.path.isfile(fileName):
                    data = sio.loadmat(fileName)
                    ind_display_screen = int(data['ind_display_screen'][0][0])
                    ind_print = int(data['ind_print'][0][0])
                    ind_plot = int(data['ind_plot'][0][0])
                    ind_parallel = int(data['ind_parallel'][0][0])
                    nbMC_PCE = int(data['nbMC_PCE'][0][0])
                    Ng = int(data['Ng'][0][0])
                    Ndeg = int(data['Ndeg'][0][0])
                    ng = int(data['ng'][0][0])
                    ndeg = int(data['ndeg'][0][0])
                    MaxIter = int(data['MaxIter'][0][0])
                else:
                    raise ValueError('STOP33 in mainWorkflow: File FileDataWorkFlowStep4_task15.mat does not exist')

            if ind_display_screen == 1:
                print(' ============================ Step4 Conditional statistics processing ============================ ')

            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' ============================ Step4 Conditional statistics processing ============================ \n ')
                    fidlisting.write('      \n ')

            # Checking that all the required tasks in Step 2 have correctly been executed before executing the tasks of Step4
            if ind_workflow == 1:
                if exec_task6 != 1:
                    raise ValueError('STOP34 in mainWorkflow: Step 4 cannot be executed because Task6_SolverDirect was not correctly executed')
            if ind_workflow == 2:
                if exec_task7 != 1:
                    raise ValueError('STOP35 in mainWorkflow: Step 4 cannot be executed because Task7_SolverDirectPartition was not correctly executed')
            if ind_workflow == 3:
                if exec_task9 != 1:
                    raise ValueError('STOP36 in mainWorkflow: Step 4 cannot be executed because Task9_SolverInverse was not correctly executed')
            if ind_workflow == 2 and ind_step4_task14 == 1:
                raise ValueError('STOP37 in mainWorkflow: for ind_workflow = 2 we must have ind_step4_task14 = 0')
           
            if ind_workflow == 1 or ind_workflow == 2 or ind_workflow == 3:
                # Task13_ConditionalStatistics
                if ind_step4_task13 == 1:
                    sub_conditional_statistics(ind_mean, ind_mean_som, ind_pdf, ind_confregion,
                                            n_x, n_q, nbParam, n_w, n_d, n_ar, nbMC, nu, MatRx_d, MatRxx_d, MatReta_ar, RmuPCA,
                                            MatRVectPCA, Indx_real, Indx_pos, Indq_obs, Indw_obs, nx_obs, Indx_obs, ind_scaling,
                                            Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log, Ralpha_scale_log, nbw0_obs,
                                            MatRww0_obs, Ind_Qcomp, nbpoint_pdf, pc_confregion, ind_display_screen, ind_print)
                    exec_task13 = 1
                    SAVERANDendSTEP4 = np.random.get_state()

                # Task14_PolynomialChaosZwiener
                if ind_step4_task14 == 1:
                    SAVERANDstartPCE = np.random.get_state()
                    (nar_PCE,MatReta_PCE,SAVERANDendPCE) = sub_polynomial_chaosZWiener(nu, n_d, nbMC, nbmDMAP, 
                                    MatRg, MatRa, n_ar, MatReta_ar, ArrayZ_ar,
                                    ArrayWienner, icorrectif, coeffDeltar, ind_PCE_ident, ind_PCE_compt,
                                    nbMC_PCE, Rmu, RM, mu_PCE, M_PCE, SAVERANDstartPCE, ind_display_screen,
                                    ind_print, ind_plot, ind_parallel, MatRplotHsamples, MatRplotHClouds,
                                    MatRplotHpdf, MatRplotHpdf2D)
                    exec_task14 = 1
                    SAVERANDendSTEP4 = SAVERANDendPCE

                    #--- save:                   
                    #  SAVERANDendPCE : state of the random generator at the end of the function
                    #  nu, nar_PCE, MatReta_PCE(nu,nar_PCE) : nar_PCE = n_d*nbMC_PCE realizations of H_PCE as the reshaping of the nbMC_PCE realizations of [H_PCE]

                    # save the random generator state SAVERANDendPCE on SaveSTEP4task14generator.pkl
                    with open("SaveSTEP4task14generator.pkl", "wb") as file:
                        pickle.dump(SAVERANDendPCE, file)

                    # save nu, n_ar_PCE, MatReta_PCE(nu,nar_PCE) on SaveSTEP4task14etaPCE.mat
                    filename = 'SaveSTEP4task14etaPCE.mat'
                    if os.path.exists(filename):
                        print(f"The file '{filename}' already exists and will be overwritten.")
                    else:
                        print(f"The file '{filename}' does not exist. Creating new file.")
                    sio.savemat('SaveSTEP4task14etaPCE.mat', {
                        'nu': nu,
                        'nar_PCE': nar_PCE,                 
                        'MatReta_PCE': MatReta_PCE,
                    }, do_compression=True)

                # Task15_PolynomialChaosQWU
                if ind_step4_task15 == 1:
                    SAVERANDstartPolynomialChaosQWU = np.random.get_state()
                    sub_polynomialChaosQWU(n_x, n_q, n_w, n_d, n_ar, nbMC, nu, MatRx_d, MatReta_ar, RmuPCA,
                                    MatRVectPCA, Indx_real, Indx_pos, Indq_obs, Indw_obs, nx_obs, Indx_obs,
                                    ind_scaling, Rbeta_scale_real, Ralpha_scale_real, Rbeta_scale_log,
                                    Ralpha_scale_log, nbMC_PCE, Ng, Ndeg, ng, ndeg, MaxIter, 
                                    SAVERANDstartPolynomialChaosQWU, ind_display_screen, ind_print, ind_plot, ind_parallel)
                    exec_task15 = 1
                    SAVERANDendSTEP4 = SAVERANDstartPolynomialChaosQWU

            np.random.set_state(SAVERANDendSTEP4)            
            ## save file of STEP4 is not implemented in Version V1
            # SAVERANDendSTEP4 = np.random.get_state() 

            ## save the random generator state    
            # with open("SaveGeneratorSTEP4.pkl", "wb") as file:
            #    pickle.dump(SAVERANDendSTEP4, file)

            # SavefileStep4.mat
            fileName = 'SavefileStep4.mat'
            if ind_SavefileStep4 == 1:               
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(f'  The file "{fileName}" is not saved in version V1\n')
                    fidlisting.write('      \n ')
                print(f'The file "{fileName}" is not saved in version V1')

        # Delete temporary file: 'FileTemporary.mat'
        os.remove('FileTemporary.mat')

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                           END CODE SEQUENCE - DO NOT MODIFY    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    return
