import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import gc
from joblib import Parallel, delayed
from sub_solverDirect_Verlet_constraint1 import sub_solverDirect_Verlet_constraint1
from sub_solverDirect_Verlet_constraint2 import sub_solverDirect_Verlet_constraint2
from sub_solverDirect_Verlet_constraint3 import sub_solverDirect_Verlet_constraint3

def sub_solverDirectPartition_constraint123(j, mj, n_d, nbMC, n_ar, nbmDMAPj, M0transient, Delatarj, f0j, shssj, shj,
                                            MatReta_dj, MatRgj, MatRaj, ArrayWiennerM0transientj, ArrayGaussj, ind_constraints, ind_coupling,
                                            epsc, iter_limit, Ralpha_relax, minVarH, maxVarH, ind_display_screen, ind_print, ind_parallel, numfig):
    
    # -------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverDirect_constraint123
    #  Subject      : ind_constraints = 1, 2, or 3: solver PLoM for direct predictions with the partition and with constraints 
    #                 of normalization for Y^j for each group j. The notations of the partition are
    #                 H    = (H_1,...,H_r,...,H_nu)    in ngroup subsets (groups) H^1,...,H^j,...,H^ngroup
    #                 H    = (Y^1,...,Y^j,...,Y^ngroup) 
    #                 Y^j  = = (Y^j_1,...,Y^j_mj) = (H_rj1,...,Hrjmj)   with j = 1,...,ngroup and with n1 + ... + nngroup = nu
    #  Comment      : this function is derived from sub_solverDirect_constraint123.m
    #
    # --- INPUT 
    #      For each group j
    #      j                   : group number in {1,...,ngroup}
    #      mj                  : dimension of random vector Y^j
    #      n_d                 : number of points in the training set for Y^j_d (also of H_d)
    #      nbMC                : number of realizations of (mj,n_d)-valued random matrix [Y^j_ar] 
    #      n_ar                : number of realizations of H_ar = (Y^1_ar,...Y^ngroup_ar) such that n_ar  = nbMC x n_d
    #      nbmDMAPj            : dimension of the ISDE-projection basis
    #      Delatarj            : ISDE integration step by Verlet scheme
    #      f0j                 : dissipation coefficient in the ISDE   
    #      shssj,shj           : parameters of the GKDE of pdf of Y^j_d (training) 
    #      MatReta_dj(mj,n_d)  : n_d realizations of Y^j_d (training)    
    #      MatRgj(n_d,nbmDMAPj): matrix of the ISDE-projection basis
    #      MatRaj(n_d,nbmDMAPj): related to MatRgj(n_d,nbmDMAPj) 
    #      ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
    #      ArrayGaussj(mj,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
    #
    #      For all the groups (independent of j)
    #      ind_constraints     : type of constraints (= 1,2, or 3)
    #      ind_coupling        : 0 (no coupling), = 1 (coupling for the Lagrange mutipliers computation)
    #      epsc                : relative tolerance for the iteration convergence of the constraints on Y^j_ar  for any j
    #      iter_limit          : maximum number of iteration for computing the Lagrange multipliers
    #      Ralpha_relax        : Ralpha_relax(iter_limit,1) relaxation parameter in ] 0 , 1 ] for the iterations. 
    #      minVarH             : minimum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 0.999)  (for any j)
    #      maxVarH             : maximum imposed on E{(Y^j_k)^2} with respect to 1 (for instance 1.001)  (for any j)
    #      ind_display_screen  : = 0 no display,              = 1 display
    #      ind_print           : = 0 no print,                = 1 print
    #      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #
    # --- OUTPUT
    #      MatReta_arj(mj,n_ar)
    #
    # --- COMMENTS
    #    ind_constraints     = 0 : no constraints concerning E{Y^j] = 0 and E{Y^j (Y^j)'} = [I_mj]
    #                        = 1 : constraints E{(Y^j_k)^2} = 1 for k =1,...,mj   
    #                        = 2 : constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
    #                        = 3 : constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
    #      ind_coupling      = 0 : for ind_constraints = 2 or 3, no coupling between the Lagrange multipliers in matrix MatRGammaS_iter 
    #                        = 1 : for ind_constraints = 2 or 3, coupling, all the extra-diagonal blocs in matrix MatRGammaS_iter are kept
    
    # --- Initialization and preallocation    
    if ind_constraints == 1:                       # Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj  are used
        mhc                = mj
        Rbc                = np.ones(mj)           # Rbc(mhc):  E{h^c(Y^j)} = b^c corresponds to  E{(Y^j_k)^2} = 1 for k =1,...,mj
        Rlambda_iter_m1    = np.zeros((mhc, 1))    # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_mj)
        MatRlambda_iter_m1 = np.array([])          # MatRlambda_iter_m1(mj,mj) is not used

    if ind_constraints == 2:                       # constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj are used
        mhc                = 2*mj
        Rbc                = np.zeros(mhc)         # Rbc(mhc):  E{h^c(Y^j)} = b^c corresponds to E{H} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
        Rbc[mj:2*mj]       = np.ones(mj)
        Rlambda_iter_m1    = np.zeros(mhc)         # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_mj,lambda_{1+mj},...,lambda_{mj+mj})
        MatRlambda_iter_m1 = np.array([])          # MatRlambda_iter_m1(mj,mj) is not used

    if ind_constraints == 3:                       # constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] are used
        mhc             = 2*mj + mj*(mj - 1) // 2  # note that for mj = 1, we have mhc = 2*mj
        Rbc             = np.zeros(mhc)            # Rbc(mhc):  E{h^c(Y^j)} = b^c corresponds to E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
        Rbc[mj:2*mj]    = np.ones(mj)
        Rlambda_iter_m1 = np.zeros(mhc)             # Rlambda_iter_m1(mhc)
        if mj == 1:
            MatRlambda_iter_m1 = np.array([])  # MatRlambda_iter_m1(mj,mj) is not used
        if mj >= 2:
            MatRlambda_iter_m1 = np.zeros((mj, mj))  # MatRlambda_iter_m1(mj,mj)
            ind = 0
            for k in range(mj - 1):
                for i in range(k + 1, mj):
                    ind = ind + 1
                    MatRlambda_iter_m1[k, i] = Rlambda_iter_m1[2*mj + ind - 1]
                    MatRlambda_iter_m1[i, k] = MatRlambda_iter_m1[k, i]
    
    # --- Print parameters
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n')
            fidlisting.write('--- Constraints parameters in solverDirect with constraints \n')
            fidlisting.write('\n')
            fidlisting.write(f'    ind_constraints = {ind_constraints:7d} \n')
            fidlisting.write(f'    mj              = {mj:7d} \n')
            fidlisting.write(f'    mhc             = {mhc:7d} \n')
            fidlisting.write('\n')
    
    # --- Preallocation for the Lagrange-multipliers computation by the iteration algorithm and generation  of nbMC realizations                                                        
    Rerr         = np.zeros(iter_limit)  # Rerr(iter_limit)      
    RnormRlambda = np.zeros(iter_limit)  # RnormRlambda(iter_limit)
    RcondGammaS  = np.zeros(iter_limit)  # RcondGammaS(iter_limit)
    Rtol         = np.zeros(iter_limit)  # Rtol(iter_limit) 
    Rtol[0]      = 1
    ind_conv     = 0
    
    # --- Loop of the iteration algorithm (iter: lambda_iter given and compute lambda_iterplus1)
    for iter in range(iter_limit):
        if ind_display_screen == 1:
            print(f'------------- iter number: {iter+1}')
        
        # Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj
        if ind_constraints == 1:
            ArrayZ_ar_iter = np.zeros((mj,nbmDMAPj,nbMC))                 # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

            if ind_parallel == 0:                                                         # Vectorized sequence
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGaussj[:, :, ell]                  # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:, :, :, ell]  # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint1(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                                        ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                               # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell

            if ind_parallel == 1:                                                         # Parallel sequence
                
                def compute_MatRZ_ar_iter(ell):
                    MatRGauss_ell               = ArrayGaussj[:, :, ell]                  # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:, :, :, ell]  # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint1(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:,:,ell] = results[ell]                               # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del results
                gc.collect()
                    
        # Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
        if ind_constraints == 2:
            ArrayZ_ar_iter = np.zeros((mj,nbmDMAPj,nbMC))                                 # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

            if ind_parallel == 0:                                                         # Vectorized sequence
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGaussj[:,:,ell]                    # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:,:,:,ell]     # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint2(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                                        ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                               # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del MatRZ_ar_iter,MatRGauss_ell,ArrayWiennerM0transient_ell

            if ind_parallel == 1:                                                         # Parallel sequence

                def compute_MatRZ_ar_iter(ell):
                    MatRGauss_ell = ArrayGaussj[:, :, ell]                                # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:, :, :, ell]  # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint2(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:,:,ell] = results[ell]                               # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del results
                gc.collect()            

        # Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]
        if ind_constraints == 3:
            ArrayZ_ar_iter = np.zeros((mj,nbmDMAPj,nbMC))                                 # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)

            if ind_parallel == 0:                                                         # Vectorized sequence
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGaussj[:,:,ell]                    # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:,:,:,ell]     # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    MatRZ_ar_iter = sub_solverDirect_Verlet_constraint3(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                                        ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,MatRlambda_iter_m1)
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                               # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del MatRZ_ar_iter,MatRGauss_ell,ArrayWiennerM0transient_ell

            if ind_parallel == 1:  # Parallel sequence
                def compute_MatRZ_ar_iter(ell):
                    MatRGauss_ell = ArrayGaussj[:,:,ell]                                  # ArrayGaussj(mj,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transientj[:,:,:,ell]     # ArrayWiennerM0transientj(mj,n_d,M0transient,nbMC)
                    return sub_solverDirect_Verlet_constraint3(mj,n_d,M0transient,Delatarj,f0j,MatReta_dj,MatRaj,MatRgj,shssj,shj,
                                                               ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,MatRlambda_iter_m1)
                results = Parallel(n_jobs=-1)(delayed(compute_MatRZ_ar_iter)(ell) for ell in range(nbMC))
                for ell in range(nbMC):
                     ArrayZ_ar_iter[:,:,ell] = results[ell]                                # ArrayZ_ar_iter(mj,nbmDMAPj,nbMC)
                del results
                gc.collect()            
        
        # Computing ArrayH_ar_iter(mj,n_d,nbMC)
        ArrayH_ar_iter = np.zeros((mj,n_d,nbMC))
        for ell in range(nbMC):
            ArrayH_ar_iter[:,:,ell] = ArrayZ_ar_iter[:,:,ell] @ MatRgj.T

        # Computing h^c(Y^j) loaded in MatRhc_iter(mhc,n_ar)
        MatReta_ar_iter = ArrayH_ar_iter.reshape(mj,n_ar)            # MatReta_ar_iter(mj,n_ar)
        del ArrayH_ar_iter

        # Test if there NaN or Inf is obtained in the ISDE solver  
        if np.isnan(np.linalg.norm(MatReta_ar_iter, 'fro')) >= 1:
            print(' ')
            print('----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver ')
            print(' ')
            print(f'iter         {iter+1}')
            print(' ')
            print(' If NaN or Inf is obtained after a small number of iterations, decrease the value of alpha_relax1')
            print(' If NaN or Inf is still obtained, decrease alpha_relax2 and/or increase iter_relax2')
            print(' ')
            print('Rlambda_iter_m1 = ')
            print(Rlambda_iter_m1.T)

            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n\n')
                    fidlisting.write('----- STOP in sub_solverDirect_constraint123: NaN or Inf is obtained in the ISDE solver \n')
                    fidlisting.write('\n')
                    fidlisting.write(f'      iter             = {iter+1:7d} \n')
                    fidlisting.write('\n')
                    fidlisting.write('      If indetermination is reached after a small number of iterations, decrease the value of alpha_relax1. \n')
                    fidlisting.write('       If indetermination is still reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
                    fidlisting.write('\n')
                    fidlisting.write('      Rlambda_iter_m1  = \n')
                    fidlisting.write(f'                         {Rlambda_iter_m1} \n ') 
                    fidlisting.write('\n ')
                    fidlisting.write('\n ')

            raise ValueError('STOP: divergence of ISDE in sub_solverDirect_constraint123')

        # Computing and loading MatRhc_iter(mhc,n_ar);
        if ind_constraints == 1:                             # Constraints E{(Y^j_k)^2} = 1 for k =1,...,mj
            MatRhc_iter         = np.zeros((mhc,n_ar))       # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:mj, :] = MatReta_ar_iter ** 2       # MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)

        if ind_constraints == 2:                             # Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
            MatRhc_iter             = np.zeros((mhc,n_ar))   # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:mj, :]     = MatReta_ar_iter        # MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
            MatRhc_iter[mj:2*mj, :] = MatReta_ar_iter**2     # MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)

        if ind_constraints == 3:  # Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj]  
            MatRhc_iter             = np.zeros((mhc, n_ar))  # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[:mj, :]     = MatReta_ar_iter        # MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
            MatRhc_iter[mj:2*mj, :] = MatReta_ar_iter**2     # MatReta_ar_iter(mj,n_ar),MatRhc_iter(mhc,n_ar)
            if mj >= 2:
                ind = 0
                for k in range(mj - 1):
                    for i in range(k + 1, mj):
                        ind = ind + 1
                        MatRhc_iter[2*mj + ind - 1, :] = MatReta_ar_iter[k, :] * MatReta_ar_iter[i, :]

        # Computing the values of the quantities at iteration iter
        RVarH_iter   = np.var(MatReta_ar_iter,axis=1,ddof=1)
        minVarH_iter = np.min(RVarH_iter)
        maxVarH_iter = np.max(RVarH_iter)
        del RVarH_iter

        Rmeanhc_iter         = np.mean(MatRhc_iter,axis=1)         # Rmeanhc_iter(mhc),MatRhc_iter(mhc,n_ar)           
        RGammaP_iter         = Rbc - Rmeanhc_iter                  # RGammaP_iter(mhc),Rbc(mhc),Rmeanhc_iter(mhc) 
        MatRGammaS_iter_temp = np.cov(MatRhc_iter)                 # MatRGammaS_iter_temp(mhc,mhc) 

        # Updating MatRGammaS_iter(mhc,mhc) if ind_coupling = 0:  For ind_constraints = 2 or 3, if ind_coupling = 0, there is no coupling 
        # between the Lagrange multipliers in the matrix MatRGammaS_iter
        if ind_coupling == 0:
            if ind_constraints == 1:                                               # Constraints  E{(Y^j_k)^2} = 1 for k =1,...,mj
                MatRGammaS_iter = MatRGammaS_iter_temp                             # MatRGammaS_iter(mhc,mhc)  

            if ind_constraints == 2:                                               # Constraints E{Y^j} = 0 and E{(Y^j_k)^2} = 1 for k =1,...,mj
                MatRGammaS_iter_temp[:mj, mj:2*mj] = np.zeros((mj,mj))
                MatRGammaS_iter_temp[mj:2*mj, :mj] = np.zeros((mj,mj))
                MatRGammaS_iter = MatRGammaS_iter_temp                             # MatRGammaS_iter(mhc,mhc)  

            if ind_constraints == 3:                                               # Constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
                MatRGammaS_iter_temp[:mj, mj:]          = np.zeros((mj,mhc-mj))
                MatRGammaS_iter_temp[mj:, :mj]          = np.zeros((mhc-mj,mj))
                MatRGammaS_iter_temp[mj:2*mj, 2*mj:mhc] = np.zeros((mj,mhc-2*mj))
                MatRGammaS_iter_temp[2*mj:mhc, mj:2*mj] = np.zeros((mhc-2*mj,mj))
                MatRGammaS_iter = MatRGammaS_iter_temp                             # MatRGammaS_iter(mhc,mhc) 

        if ind_coupling == 1:
            MatRGammaS_iter = MatRGammaS_iter_temp                                 # MatRGammaS_iter(mhc,mhc)  

        del MatRGammaS_iter_temp

        # Testing the convergence at iteration iter
        Rerr[iter]         = np.linalg.norm(RGammaP_iter) / np.linalg.norm(Rbc)    # Rerr(iter_limit,1)  
        RnormRlambda[iter] = np.linalg.norm(Rlambda_iter_m1)                       # RnormRlambda(iter_limit,1) 
        RcondGammaS[iter]  = np.linalg.cond(MatRGammaS_iter)                       # RcondGammaS(iter_limit,1) 
        if ind_display_screen == 1:
            print(f'err_iter         = {Rerr[iter]}')
            print(f'norm_lambda_iter = {RnormRlambda[iter]}')
        if iter >= 2:
            Rtol[iter] = 2*abs(Rerr[iter] - Rerr[iter - 1]) / abs(Rerr[iter] + Rerr[iter - 1])
            if ind_display_screen == 1:
                print(f'tol_iter         = {Rtol[iter]}')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n')
                fidlisting.write(f'         --- iter number = {iter+1:7d} \n')
                fidlisting.write(f'             err_iter    = {Rerr[iter]:14.7e} \n')
                fidlisting.write(f'             tol_iter    = {Rtol[iter]:14.7e} \n')
                fidlisting.write('\n')

        # Criterion 1: if iter > 10 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained for iter - 1.
        # convergence is then assumed to be reached and then, exit from the loop on iter  
        if (iter > 10) and (Rerr[iter] - Rerr[iter - 1] > 0):
            ind_conv = 1
            MatReta_arj = MatReta_ar_iter_m1  # MatReta_arj(mj,n_ar), MatReta_ar_iter_m1(mj,n_ar)
            iter_plot = iter
            if ind_display_screen == 1:
                print('Convergence with criterion 1: local minimum reached')
                print('If convergence is not sufficiently good, decrease the value of alpha_relax2')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n')
                    fidlisting.write(' --- Convergence with criterion 1: local minimum reached \n')
                    fidlisting.write('     If convergence is not sufficiently good, decrease the value of alpha_relax2 \n')
                    fidlisting.write('\n')
            del MatReta_ar_iter_m1, MatReta_ar_iter
            del MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break  # exit from the loop on iter

        # Criterion 2: if {minVarH_iter >= minVarH and maxVarH_iter <= maxVarH} or Rerr(iter) <= epsc, the variance of each component is greater 
        # than or equal to minVarH and less than or equal to maxVarH, or the relative error of the constraint satisfaction is less 
        # than or equal to the tolerance. The convergence is reached, and then exit from the loop on iter.
        if ((minVarH_iter >= minVarH) and (maxVarH_iter <= maxVarH)) or Rerr[iter] <= epsc:
            ind_conv = 1
            MatReta_arj = MatReta_ar_iter  # MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
            iter_plot = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 2: convergence obtained either with variance-values of H-components satisfied')
                print('                              or relative error of the constraint satisfaction is less than the tolerance')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n')
                    fidlisting.write(' --- Convergence with criterion 2: convergence obtained either with variance-values \n')
                    fidlisting.write('                                    of H-components satisfied or relative error of the \n')
                    fidlisting.write('                                    constraint satisfaction is less than the tolerance \n')
                    fidlisting.write('                \n')
            del MatReta_ar_iter_m1, MatReta_ar_iter
            del MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break  # exit from the loop on iter

        # Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
        # the convergence is assumed to be reached and then, exit from the loop on iter  
        if (iter > min(20, iter_limit) and Rerr[iter] < epsc) and (Rtol[iter] < epsc):
            ind_conv = 1
            MatReta_arj = MatReta_ar_iter  # MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
            iter_plot = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('\n')
                    fidlisting.write(' --- Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary \n')
                    fidlisting.write('\n')
            del MatReta_ar_iter_m1, MatReta_ar_iter
            del MatRhc_iter, Rmeanhc_iter, RGammaP_iter, Rlambda_iter_m1
            break  # exit from the loop on iter

        # Convergence not reached: the variance of each component is less than minVarH or greater than maxVarH, 
        # the relative error of the constraint satisfaction is greater than the tolerance, there is no local minimum,
        # and there is no stationary point.  
        if ind_conv == 0:
            Rtemp_iter      = np.linalg.solve(MatRGammaS_iter,RGammaP_iter)       # Rtemp_iter = inv(MatRGammaS_iter)*RGammaP_iter
            Rlambda_iter    = Rlambda_iter_m1 - Ralpha_relax[iter] * Rtemp_iter   # Rlambda_iter(mhc), Rlambda_iter_m1(mhc)
            MatRlambda_iter = np.array([])
            if ind_constraints == 3 and mj >= 2:                                  # constraints E{Y^j} = 0 and E{Y^j (Y^j)'} = [I_mj] 
                MatRlambda_iter = np.zeros((mj,mj))                              # MatRlambda_iter(mj,mj)
                ind = 0
                for k in range(mj - 1):
                    for i in range(k + 1, mj):
                        ind = ind + 1
                        MatRlambda_iter[k,i] = Rlambda_iter[2*mj + ind - 1]
                        MatRlambda_iter[i,k] = MatRlambda_iter[k,i]
            Rlambda_iter_m1    = Rlambda_iter
            MatRlambda_iter_m1 = MatRlambda_iter
            MatReta_ar_iter_m1 = MatReta_ar_iter
            del MatRhc_iter, Rlambda_iter, MatRlambda_iter, Rmeanhc_iter, RGammaP_iter, MatRGammaS_iter, Rtemp_iter

    # --- if ind_conv = 0, then iter_limit is reached without convergence
    if ind_conv == 0:
        MatReta_arj = MatReta_ar_iter  # MatReta_arj(mj,n_ar), MatReta_ar_iter(mj,n_ar)
        iter_plot = iter_limit
        if ind_display_screen == 1:
            print('------ No convergence of the iteration algorithm in sub_solverDirect_constraint123')
            print(f'       iter_plot = {iter_plot}')
            print('        If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n')
            print('        If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n')
                fidlisting.write(' --- No convergence of the iteration algorithm in sub_solverDirect_constraint123 \n')
                fidlisting.write('\n')
                fidlisting.write(f'     iter             = {iter_plot:7d}    \n')
                fidlisting.write(f'     err_iter         = {Rerr[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     tol_iter         = {Rtol[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     norm_lambda_iter = {RnormRlambda[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     condGammaS_iter  = {RcondGammaS[iter_plot-1]:14.7e} \n')
                fidlisting.write('\n\n')
                fidlisting.write('     If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n')
                fidlisting.write('      If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
                fidlisting.write('\n')

    # --- if ind_conv = 1, then convergence is reached
    if ind_conv == 1:
        if ind_display_screen == 1:
            print('------ Convergence of the iteration algorithm in sub_solverDirect_constraint123')
            print(f'       iter_plot = {iter_plot}')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n\n')
                fidlisting.write(' --- Convergence of the iteration algorithm in sub_solverDirect_constraint123   \n')
                fidlisting.write('\n')
                fidlisting.write(f'     iter             = {iter_plot:7d}    \n')
                fidlisting.write(f'     err_iter         = {Rerr[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     tol_iter         = {Rtol[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     norm_lambda_iter = {RnormRlambda[iter_plot-1]:14.7e} \n')
                fidlisting.write(f'     condGammaS_iter  = {RcondGammaS[iter_plot-1]:14.7e} \n')
                fidlisting.write('\n')

    # --- Plot
    plt.figure()
    plt.plot(range(1, iter_plot + 1), Rerr[:iter_plot], 'b-')
    plt.title(r'Graph of function $\rm{err}(\iota)$', fontsize=16, weight='normal')    
    plt.xlabel(r'$\iota$', fontsize=16)
    plt.ylabel(r'$\rm{err}(\iota)$', fontsize=16)
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirectPartition_constraint123_{numfig}_Rerr_group{j}.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, iter_plot + 1), RnormRlambda[:iter_plot], 'b-')
    plt.title(r'Graph of function $\Vert \lambda_{\iota}\Vert $', fontsize=16, weight='normal')  
    plt.xlabel(r'$\iota$', fontsize=16)
    plt.ylabel(r'$\Vert \lambda_{\iota}\Vert $', fontsize=16)
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirectPartition_constraint123_{numfig}_RnormRlambda_group{j}.png')
    plt.close()

    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), RcondGammaS[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)  
    plt.ylabel(r'$\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16)
    numfig = numfig + 1
    plt.savefig(f'figure_solverDirectPartition_constraint123_{numfig}_RcondGammaS_group{j}.png')
    plt.close(h)
    
    # --- Print The relative norm of the extradiagonal term that as to be close to 0R
    #    and print Hmean_ar and diag(MatRHcov_ar)
    #                                                     
    if ind_print == 1:
        RHmean_ar      = np.mean(MatReta_arj,axis=1)
        MatRHcov_ar    = np.cov(MatReta_arj)
        RdiagHcov_ar   = np.diag(MatRHcov_ar)
        normExtra_diag = np.linalg.norm(MatRHcov_ar - np.diag(RdiagHcov_ar),'fro') / np.linalg.norm(np.diag(RdiagHcov_ar),'fro')
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
            
    return MatReta_arj
