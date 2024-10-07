import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import gc
from joblib import Parallel, delayed
from sub_solverDirect_Verlet_constraint1 import sub_solverDirect_Verlet_constraint1
from sub_solverDirect_Verlet_constraint2 import sub_solverDirect_Verlet_constraint2
from sub_solverDirect_Verlet_constraint3 import sub_solverDirect_Verlet_constraint3

def sub_solverDirect_constraint123(nu, n_d, nbMC, n_ar, nbmDMAP, M0transient, Deltar, f0, shss, sh, 
                                   MatReta_d, MatRg, MatRa, ArrayWiennerM0transient, ArrayGauss, 
                                   ind_constraints, ind_coupling, epsc, iter_limit, Ralpha_relax, 
                                   minVarH, maxVarH, ind_display_screen, ind_print, ind_parallel, numfig):

    #-------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 27 May 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_constraint123
    #  Subject      : ind_constraints = 1, 2, or 3: Constraints on H_ar
    #
    #--- INPUT  
    #      nu                  : dimension of random vector H_d and H_ar
    #      n_d                 : number of points in the training set for H_d
    #      nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar] 
    #      n_ar                : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #      nbmDMAP             : dimension of the ISDE-projection basis
    #      M0transient         : number of steps for reaching the stationary solution
    #      Deltar              : ISDE integration step by Verlet scheme
    #      f0                  : dissipation coefficient in the ISDE   
    #      shss,sh             : parameters of the GKDE of pdf of H_d (training) 
    #      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
    #      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
    #      ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
    #      ArrayGauss(nu,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
    #      ind_constraints     : type of constraints (= 1,2, or 3)
    #      ind_coupling        : 0 (no coupling), = 1 (coupling for the Lagrange mutipliers computation)
    #      epsc                : relative tolerance for the iteration convergence of the constraints on H_ar 
    #      iter_limit          : maximum number of iteration for computing the Lagrange multipliers
    #      Ralpha_relax        : Ralpha_relax(iter_limit,1) relaxation parameter in ] 0 , 1 ] for the iterations. 
    #      minVarH             : minimum imposed on Var{H^2} with respect to 1 (for instance 0.99) for the convergence test
    #      maxVarH             : maximum imposed on Var{H^2} with respect to 1 (for instance 1.01) for the convergence test
    #      ind_display_screen  : = 0 no display,              = 1 display
    #      ind_print           : = 0 no print,                = 1 print
    #      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #
    #--- OUTPUT
    #      MatReta_ar(nu,n_ar)
    #      ArrayZ_ar(nu,nbmDMAP,nbMC);    # ArrayZ_ar(nu,nbmDMAP,nbMC), this array is used as ouput for possible use in a postprocessing 
    #                                     # of Z in order to construct its polynomial chaos expansion (PCE)
    #--- COMMENTS
    #      ind_constraints     = 1 : constraints E{H_j^2} = 1 for j =1,...,nu   
    #                          = 2 : constraints E{H] = 0 and E{H_j^2} = 1 for j =1,...,nu
    #                          = 3 : constraints E{H] = 0 and E{H H'} = [I_nu]
    #      ind_coupling        = 0 : for ind_constraints = 2 or 3, no coupling between the Lagrange multipliers in matrix MatRGammaS_iter 
    #                          = 1 : for ind_constraints = 2 or 3, coupling, all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
 
    #--- Initialization and preallocation    
    if ind_constraints == 1:                   #--- Constraints E{H_j^2} = 1 for j =1,...,nu are used
        mhc = nu
        Rbc = np.ones(nu)                      # Rbc(mhc):  E{h^c(H)} = b^c corresponds to  E{H_j^2} = 1 for j =1,...,nu
        Rlambda_iter_m1 = np.zeros(mhc)        # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_nu)
        MatRlambda_iter_m1 = np.array([])      # MatRlambda_iter_m1(nu,nu) is not used

    if ind_constraints == 2:                   #--- constraints E{H} = 0 and E{H_j^2} = 1 for j =1,...,nu are used
        mhc = 2*nu
        Rbc = np.zeros(mhc)                    # Rbc(mhc):  E{h^c(H)} = b^c corresponds to E{H} = 0 and E{H_j^2} = 1 for j =1,...,nu
        Rbc[nu:2*nu] = np.ones(nu)
        Rlambda_iter_m1 = np.zeros(mhc)        # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_nu,lambda_{1+nu},...,lambda_{nu+nu})
        MatRlambda_iter_m1 = np.array([])      # MatRlambda_iter_m1(nu,nu) is not used

    if ind_constraints == 3:                   #--- constraints E{H] = 0 and E{H H'} = [I_nu] are used
        mhc = 2*nu + nu*(nu - 1) // 2          # note that for nu = 1, we have mhc = 2*nu
                                               # if nu = 1: Rbc(mhc) = (bc_1,...,bc_nu , bc_{1+nu},...,bc_{nu+nu} )
                                               # if nu > 1: Rbc(mhc) = (bc_1,...,bc_nu , bc_{1+nu},...,bc_{nu+nu} , {bc_{(j,i) + 2 nu} 
                                               #            for 1 <= j < i <= nu} )
        Rbc = np.zeros(mhc)                    # Rbc(mhc):  E{h^c(H)} = b^c corresponds to E{H] = 0  and  E{H H'} = [I_nu] 
        Rbc[nu:2*nu] = np.ones(nu)
        Rlambda_iter_m1 = np.zeros(mhc)        # Rlambda_iter_m1(mhc)      
        if nu == 1:                            # Rlambda_iter(mhc,1) = (lambda_1,...,lambda_nu,lambda_{1+nu},...,lambda_{nu+nu})
            MatRlambda_iter_m1 = np.array([])  # MatRlambda_iter_m1(nu,nu) is not used  
        if nu >= 2:                            # Rlambda_iter(mhc) = ( lambda_1,...,lambda_nu , lambda_{1+nu},...,lambda_{nu+nu} ,
                                               # {lambda_{(j,i) + 2 nu} for 1 <= j < i <= nu} ), and MatRlambda_iter_m1(nu,nu) = symmetric 
            MatRlambda_iter_m1 = np.zeros((nu,nu))  # with zero diagonal and for which upper part is Rlambda_iter row rise
            ind = 0
            for j in range(nu - 1):
                for i in range(j+1,nu):
                    ind = ind + 1
                    MatRlambda_iter_m1[j, i] = Rlambda_iter_m1[2*nu + ind - 1]    # Rlambda_iter_m1(mhc); 
                    MatRlambda_iter_m1[i, j] = MatRlambda_iter_m1[j, i]           # MatRlambda_iter_m1(nu,nu)

    #--- Print parameters
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('--- Constraints parameters in solverDirect with constraints \n ')
            fidlisting.write('\n ')
            fidlisting.write(f'    ind_constraints = {ind_constraints:7d} \n ')
            fidlisting.write(f'    nu              = {nu:7d} \n ')
            fidlisting.write(f'    mhc             = {mhc:7d} \n ')
            fidlisting.write('\n ')

    #--- Preallocation for the Lagrange-multipliers computation by the iteration algorithm and generation  of nbMC realizations                                                        
    Rerr         = np.zeros(iter_limit)   # Rerr(iter_limit)      
    RnormRlambda = np.zeros(iter_limit)   # RnormRlambda(iter_limit)
    RcondGammaS  = np.zeros(iter_limit)   # RcondGammaS(iter_limit)
    Rtol         = np.zeros(iter_limit)   # Rtol(iter_limit) 
    Rtol[0]      = 1
    ind_conv     = 0
   
    #--- Loop of the iteration algorithm (iter: lambda_iter given and compute lambda_iterplus1)
    for iter in range(iter_limit):
        if ind_display_screen == 1:
            print(f'------------- iter number: {iter + 1}')

        # Constraints E{H_{ar,j}^2} = 1 for j =1,...,nu 
        if ind_constraints == 1:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))  # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)  
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, 
                                                                        ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1) 
                    ArrayZ_ar_iter[:, :, ell] = MatRZ_ar_iter                            # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell

            # Parallel sequence
            if ind_parallel == 1:
                def compute_MatRZ_ar_iter(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1):
                    MatRGauss_ell = ArrayGauss[:, :, ell]  # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh, 
                                                                 ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:, :, ell] = results[ell]                             # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del results
                gc.collect()

        # Constraints E{H_{ar,j}} = 0 and E{H_{ar,j}^2} = 1 for j =1,...,nu
        if ind_constraints == 2:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))  # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell = ArrayGauss[:, :, ell]                                # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,  
                                                                        ArrayWiennerM0transient_ell, MatRGauss_ell, Rlambda_iter_m1)                                                  
                    ArrayZ_ar_iter[:, :, ell] = MatRZ_ar_iter                            # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell

            # Parallel sequence
            if ind_parallel == 1:
                def compute_MatRZ_ar_iter(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1):
                    MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,  
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:, :, ell] = results[ell]                             # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del results
                gc.collect()

        # Constraints E{H_ar] = 0 and E{H_ar H_ar'} = [I_nu]  
        if ind_constraints == 3:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))  # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa, 
                                                                        MatRg,shss,sh,ArrayWiennerM0transient_ell,MatRGauss_ell, 
                                                                        Rlambda_iter_m1,MatRlambda_iter_m1) 
                    ArrayZ_ar_iter[:, :, ell] = MatRZ_ar_iter                            # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell

            # Parallel sequence
            if ind_parallel == 1:
                def compute_MatRZ_ar_iter(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1,MatRlambda_iter_m1):
                    MatRGauss_ell               = ArrayGauss[:, :, ell]                  # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]  # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa, 
                                                               MatRg,shss,sh,ArrayWiennerM0transient_ell,MatRGauss_ell, 
                                                               Rlambda_iter_m1,MatRlambda_iter_m1) 
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell,ArrayGauss,ArrayWiennerM0transient,nu,n_d,M0transient,Deltar,f0,
                                          MatReta_d,MatRa,MatRg,shss,sh,Rlambda_iter_m1,MatRlambda_iter_m1) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:, :, ell] = results[ell]                             # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
                del results
                gc.collect()

        # Computing ArrayH_ar_iter(nu,n_d,nbMC)   
        ArrayH_ar_iter = np.zeros((nu,n_d,nbMC))                                    # ArrayH_ar_iter(nu,n_d,nbMC)                                                            
        for ell in range(nbMC):                                                     # ArrayZ_ar_iter(nu,nbmDMAP,nbMC), MatRg(n_d,nbmDMAP) 
            ArrayH_ar_iter[:, :, ell] = ArrayZ_ar_iter[:, :, ell] @ MatRg.T         # ArrayH_ar_iter(nu,n_d,nbMC)

        # Computing h^c(H) loaded in MatRhc_iter(mhc,n_ar)
        MatReta_ar_iter = np.reshape(ArrayH_ar_iter, (nu,n_ar))                     # MatReta_ar_iter(nu,n_ar)
        del ArrayH_ar_iter

        # Test if there NaN or Inf is obtained in the ISDE solver  
        testNaN = np.isnan(np.linalg.norm(MatReta_ar_iter, 'fro'))
        if testNaN:
            print(' ')
            print('----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver ')
            print(' ')
            print(f'iter         {iter+1}')
            print(' ')
            print(' If NaN or Inf is obtained after a small number of iterations, decrease the value of alpha_relax1')
            print(' If NaN or Inf is still obtained, decrease alpha_relax2 and/or increase iter_relax2')
            print(' ')
            print('Rlambda_iter_m1 = ')
            print(Rlambda_iter_m1)

            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write('----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver \n ')
                fidlisting.write('\n ')             
                fidlisting.write(f'      iter             = {iter+1:7d} \n ')
                fidlisting.write('\n ')  
                fidlisting.write('      If indetermination is reached after a small number of iterations, decrease the value of alpha_relax1. \n')
                fidlisting.write('       If indetermination is still reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
                fidlisting.write('\n ')  
                fidlisting.write('      Rlambda_iter_m1  = \n')
                fidlisting.write(f'                         {Rlambda_iter_m1} \n ')
                fidlisting.write('\n ')
                fidlisting.write('\n ')

            raise ValueError('STOP: divergence of ISDE in sub_solverDirect_constraint123')
       
        # Computing and loading MatRhc_iter(mhc,n_ar);
        if ind_constraints == 1:                            # Constraints E{H_j^2} = 1 for j =1,...,nu
            MatRhc_iter         = np.zeros((mhc,n_ar))      # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:nu, :] = MatReta_ar_iter ** 2      # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)

        if ind_constraints == 2:                            # Constraints E{H} = 0 and E{H_j^2} = 1 for j =1,...,nu
            MatRhc_iter             = np.zeros((mhc,n_ar))  # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:nu, :]     = MatReta_ar_iter       # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
            MatRhc_iter[nu:2*nu, :] = MatReta_ar_iter ** 2  # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)

        if ind_constraints == 3:                            # Constraints E{H] = 0 and E{H H'} = [I_nu]
            MatRhc_iter             = np.zeros((mhc,n_ar))  # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:nu, :]     = MatReta_ar_iter       # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
            MatRhc_iter[nu:2*nu, :] = MatReta_ar_iter ** 2  # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
            if nu >= 2:
                ind = 0
                for j in range(nu - 1):
                    for i in range(j + 1, nu):
                        ind = ind + 1
                        MatRhc_iter[2*nu + ind - 1, :] = MatReta_ar_iter[j, :] * MatReta_ar_iter[i, :]

        # Computing the values of the quantities at iteration iter
        RVarH_iter   = np.var(MatReta_ar_iter,axis=1,ddof=1)
        minVarH_iter = np.min(RVarH_iter)
        maxVarH_iter = np.max(RVarH_iter)
        del RVarH_iter
                                               
        Rmeanhc_iter         = np.mean(MatRhc_iter,axis=1)                    # Rmeanhc_iter(mhc),MatRhc_iter(mhc,n_ar)           
        RGammaP_iter         = Rbc - Rmeanhc_iter                             # RGammaP_iter(mhc),Rbc(mhc),Rmeanhc_iter(mhc) 
        MatRGammaS_iter_temp = np.cov(MatRhc_iter)                            # MatRGammaS_iter_temp(mhc,mhc) 
        
        # Updating MatRGammaS_iter(mhc,mhc) if ind_coupling = 0:  For ind_constraints = 2 or 3, if ind_coupling = 0, there is no coupling 
        # between the Lagrange multipliers in the matrix MatRGammaS_iter
        if ind_coupling == 0:                                                  
            if ind_constraints == 1:                                          # Constraints E{H_j^2} = 1 for j =1,...,nu
                MatRGammaS_iter = MatRGammaS_iter_temp                        # MatRGammaS_iter(mhc,mhc)  

            if ind_constraints == 2:  # constraints E{H} = 0 and E{H_j^2} = 1 for j =1,...,nu
                MatRGammaS_iter_temp[:nu, nu:nu + nu] = np.zeros((nu,nu))
                MatRGammaS_iter_temp[nu:nu + nu, :nu] = np.zeros((nu,nu))
                MatRGammaS_iter = MatRGammaS_iter_temp                        # MatRGammaS_iter(mhc,mhc)  

            if ind_constraints == 3:  # Constraints E{H] = 0 and E{H H'} = [I_nu]
                MatRGammaS_iter_temp[:nu, nu:mhc]            = np.zeros((nu,mhc-nu))
                MatRGammaS_iter_temp[nu:mhc, :nu]            = np.zeros((mhc-nu,nu))
                MatRGammaS_iter_temp[nu:nu+nu, 2*nu:mhc]     = np.zeros((nu,mhc-2*nu))
                MatRGammaS_iter_temp[2*nu:mhc, nu:nu+nu]     = np.zeros((mhc-2*nu,nu))
                MatRGammaS_iter = MatRGammaS_iter_temp                        # MatRGammaS_iter(mhc,mhc)  

        if ind_coupling == 1:
            MatRGammaS_iter = MatRGammaS_iter_temp                             # MatRGammaS_iter(mhc,mhc)  

        del MatRGammaS_iter_temp
        
        # Testing the convergence at iteration iter
        Rerr[iter]         = np.linalg.norm(RGammaP_iter) / np.linalg.norm(Rbc)  # Rerr(iter_limit)  
        RnormRlambda[iter] = np.linalg.norm(Rlambda_iter_m1)                     # RnormRlambda(iter_limit) 
        RcondGammaS[iter]  = np.linalg.cond(MatRGammaS_iter)                     # RcondGammaS(iter_limit) 

        if ind_display_screen == 1:
            print(f'err_iter         = {Rerr[iter]}')
            print(f'norm_lambda_iter = {RnormRlambda[iter]}')

        if iter >= 1:
            Rtol[iter] = 2*abs(Rerr[iter] - Rerr[iter - 1]) / abs(Rerr[iter] + Rerr[iter - 1])
            if ind_display_screen == 1:
                print(f'tol_iter         = {Rtol[iter]}')

        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write(f'         --- iter number = {iter + 1:7d} \n ')
                fidlisting.write(f'             err_iter    = {Rerr[iter]:14.7e} \n ')
                fidlisting.write(f'             tol_iter    = {Rtol[iter]:14.7e} \n ')
                fidlisting.write('\n ')
       
        # Criterion 1: if iter > 10 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained for iter - 1.
        # convergence is then assumed to be reached and then, exit from the loop on iter  
        if iter > 9 and (Rerr[iter] - Rerr[iter - 1]) > 0:                     
            ind_conv = 1
            MatReta_ar = MatReta_ar_iter_m1  # MatReta_ar(nu,n_ar), MatReta_ar_iter_m1(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter_m1   # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter_m1(nu,nbmDMAP,nbMC)   
            iter_plot  = iter
            if ind_display_screen == 1:
                print('Convergence with criterion 1: local minimum reached')
                print('If convergence is not sufficiently good, decrease the value of alpha_relax2')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n ')
                    fidlisting.write(' --- Convergence with criterion 1: local minimum reached \n ')
                    fidlisting.write('     If convergence is not sufficiently good, decrease the value of alpha_relax2 \n ')
                    fidlisting.write('\n ')
            del MatReta_ar_iter_m1, MatReta_ar_iter, MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break    # exit from the loop on iter  

        # Criterion 2: if {minVarH_iter >= minVarH and maxVarH_iter <= maxVarH} or Rerr(iter) <= epsc, the variance of each component is greater 
        # than or equal to minVarH and less than or equal to maxVarH, or the relative error of the constraint satisfaction is less 
        # than or equal to the tolerance. The convergence is reached, and then exit from the loop on iter.
        if (minVarH_iter >= minVarH and maxVarH_iter <= maxVarH) or Rerr[iter] <= epsc:  
            ind_conv   = 1
            MatReta_ar = MatReta_ar_iter  # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter   # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
            iter_plot  = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 2: convergence obtained either with variance-values of H-components satisfied')
                print('                              or relative error of the constraint satisfaction is less than the tolerance')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n ')
                    fidlisting.write(' --- Convergence with criterion 2: convergence obtained either with variance-values \n')
                    fidlisting.write('                                    of H-components satisfied or relative error of the \n')
                    fidlisting.write('                                    constraint satisfaction is less than the tolerance \n')
                    fidlisting.write('                \n ')
            del MatReta_ar_iter_m1, MatReta_ar_iter, MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break  # exit from the loop on iter   

        # Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
        # the convergence is assumed to be reached and then, exit from the loop on iter  
        if (iter + 1 > min(20, iter_limit) and Rerr[iter] < epsc) and (Rtol[iter] < epsc):    
            ind_conv   = 1
            MatReta_ar = MatReta_ar_iter  # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter   # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
            iter_plot  = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n ')
                    fidlisting.write(' --- Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary \n ')
                    fidlisting.write('\n ')
            del MatReta_ar_iter_m1, MatReta_ar_iter, MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break  # exit from the loop on iter
      
        # Convergence not reached: the variance of each component is less than minVarH or greater than maxVarH, 
        # the relative error of the constraint satisfaction is greater than the tolerance, there is no local minimum,
        # and there is no stationary point.  
        if ind_conv == 0:
            Rtemp_iter      = np.linalg.solve(MatRGammaS_iter,RGammaP_iter)        # Rtemp_iter = inv(MatRGammaS_iter)*RGammaP_iter
            Rlambda_iter    = Rlambda_iter_m1 - Ralpha_relax[iter] * Rtemp_iter    # Rlambda_iter(mhc), Rlambda_iter_m1(mhc)
            MatRlambda_iter = np.array([])
            if ind_constraints == 3 and nu >= 2:                                   # constraints E{H] = 0 and E{H H'} = [I_nu]
                MatRlambda_iter = np.zeros((nu,nu))                                # MatRlambda_iter(nu,nu) 
                ind = 0
                for j in range(nu - 1):
                    for i in range(j + 1, nu):
                        ind =  ind + 1
                        MatRlambda_iter[j,i] = Rlambda_iter[2*nu + ind -1]
                        MatRlambda_iter[i,j] = MatRlambda_iter[j,i]

            Rlambda_iter_m1    = Rlambda_iter
            MatRlambda_iter_m1 = MatRlambda_iter
            MatReta_ar_iter_m1 = MatReta_ar_iter
            ArrayZ_ar_iter_m1  = ArrayZ_ar_iter
            del MatRhc_iter, Rlambda_iter, MatRlambda_iter, Rmeanhc_iter, RGammaP_iter, MatRGammaS_iter, Rtemp_iter

    #--- if ind_conv = 0, then iter_limit is reached without convergence
    if ind_conv == 0: 
        MatReta_ar = MatReta_ar_iter  # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
        ArrayZ_ar = ArrayZ_ar_iter  # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
        iter_plot = iter_limit    
        if ind_display_screen == 1:
            print('------ No convergence of the iteration algorithm in sub_solverDirect_constraint123')
            print(f'       iter_plot = {iter_plot}')
            print('        If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n')
            print('        If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write(' --- No convergence of the iteration algorithm in sub_solverDirect_constraint123 \n ')
                fidlisting.write('                                                  \n ')
                fidlisting.write(f'     iter             = {iter_plot:7d}    \n ')
                fidlisting.write(f'     err_iter         = {Rerr[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     tol_iter         = {Rtol[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     norm_lambda_iter = {RnormRlambda[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     condGammaS_iter  = {RcondGammaS[iter_plot-1]:14.7e} \n ')
                fidlisting.write('                                                  \n ') 
                fidlisting.write('                                                  \n ') 
                fidlisting.write('     If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n')
                fidlisting.write('      If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
                fidlisting.write('      \n ')

    #--- if ind_conv = 1, then convergence is reached
    if ind_conv == 1:
        if ind_display_screen == 1:
            print('------ Convergence of the iteration algorithm in sub_solverDirect_constraint123')
            print(f'       iter_plot = {iter_plot}')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n ')
                fidlisting.write('\n ')
                fidlisting.write(' --- Convergence of the iteration algorithm in sub_solverDirect_constraint123   \n ') 
                fidlisting.write('                                                    \n ')
                fidlisting.write(f'     iter             = {iter_plot:7d}    \n ')
                fidlisting.write(f'     err_iter         = {Rerr[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     tol_iter         = {Rtol[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     norm_lambda_iter = {RnormRlambda[iter_plot-1]:14.7e} \n ')
                fidlisting.write(f'     condGammaS_iter  = {RcondGammaS[iter_plot-1]:14.7e} \n ')         
                fidlisting.write('      \n ')

    #--- Plot
    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), Rerr[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\rm{err}(\iota)$', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)                                                                
    plt.ylabel(r'$\rm{err}(\iota)$', fontsize=16)  
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirect_constraint123_{numfig}_Rerr.png')
    plt.close(h)

    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), RnormRlambda[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\Vert \lambda_{\iota}\Vert $', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)                                                                
    plt.ylabel(r'$\Vert \lambda_{\iota}\Vert $', fontsize=16)  
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirect_constraint123_{numfig}_RnormRlambda.png')
    plt.close(h)

    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), RcondGammaS[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)  
    plt.ylabel(r'$\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16)
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirect_constraint123_{numfig}_RcondGammaS.png')
    plt.close(h)

    #--- Print The relative norm of the extradiagonal term that as to be close to 0R
    #    and print Hmean_ar and diag(MatRHcov_ar)
    if ind_print == 1:
        RHmean_ar      = np.mean(MatReta_ar,axis=1)
        MatRHcov_ar    = np.cov(MatReta_ar)
        RdiagHcov_ar   = np.diag(MatRHcov_ar)
        normExtra_diag = np.linalg.norm(MatRHcov_ar - np.diag(RdiagHcov_ar), 'fro') / np.linalg.norm(np.diag(RdiagHcov_ar), 'fro')
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('----- RHmean_ar =          \n ')
            fidlisting.write(f'                 {RHmean_ar} \n ')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write('----- diag(MatRHcov_ar) =          \n ')
            fidlisting.write(f'                 {RdiagHcov_ar} \n ')
            fidlisting.write('\n ')
            fidlisting.write('\n ')
            fidlisting.write(f'----- Relative Frobenius norm of the extra-diagonal terms of MatRHcov_ar = {normExtra_diag:14.7e} \n ')
            fidlisting.write('\n ')
            fidlisting.write('\n ')

    return MatReta_ar, ArrayZ_ar
