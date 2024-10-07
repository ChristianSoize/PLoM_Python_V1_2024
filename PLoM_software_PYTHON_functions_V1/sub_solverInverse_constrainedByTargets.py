import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np
import gc
from joblib import Parallel, delayed
from sub_solverInverse_Verlet_target1 import sub_solverInverse_Verlet_target1
from sub_solverInverse_Verlet_target2 import sub_solverInverse_Verlet_target2
from sub_solverInverse_Verlet_target3 import sub_solverInverse_Verlet_target3
from sub_solverInverse_pseudo_inverse import sub_solverInverse_pseudo_inverse

def sub_solverInverse_constrainedByTargets(nu,n_d,nbMC,n_ar,nbmDMAP,M0transient,Deltar,f0,shss,sh,
                                           MatReta_d,MatRg,MatRa,ArrayWiennerM0transient,ArrayGauss,ind_type_targ,N_r,Rb_targ1,coNr,
                                           coNr2,MatReta_targ,eps_inv,Rb_targ2,Rb_targ3,ind_coupling,epsc,iter_limit,Ralpha_relax,
                                           ind_display_screen,ind_print,ind_parallel,numfig):

    #------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 08 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_solverInverse_constrainedByTargets
    #  Subject      : solver for ind_type_targ = 1, 2, or 3: Constraints on H^c
    #
    #--- INPUT  
    #      nu                  : dimension of random vector H = (H_1, ... H_nu)
    #      n_d                 : number of points in the training set for H
    #      nbMC                : number of realizations of (nu,n_d)-valued random matrix [H_ar] 
    #      n_ar                : number of realizations of H_ar such that n_ar  = nbMC x n_d
    #      nbmDMAP             : dimension of the ISDE-projection basis
    #      M0transient         : number of steps for reaching the stationary solution
    #      Deltar              : ISDE integration step by Verlet scheme
    #      f0                  : dissipation coefficient in the ISDE   
    #      shss, sh            : parameters of the GKDE of pdf of H_d (training) 
    #      MatReta_d(nu,n_d)   : n_d realizations of H_d (training)    
    #      MatRg(n_d,nbmDMAP)  : matrix of the ISDE-projection basis
    #      MatRa(n_d,nbmDMAP)  : related to MatRg(n_d,nbmDMAP) 
    #      ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)  : realizations of the matrix-valued normalized Wienner process
    #      ArrayGauss(nu,n_d,nbMC)                           : realizations of the Gaussian matrix-valued random variable
    #
    #      ind_type_targ       : = 1, targets defined by giving N_r realizations
    #                          : = 2, targets defined by giving target mean-values 
    #                          : = 3, targets defined by giving target mean-values and target covariance matrix
    #                          --- ind_type_targ = 1: targets defined by giving N_r realizations of XX_targ 
    #      N_r                     nummber of realizations of the targets     
    #      Rb_targ1(N_r)           E{h_targ1(H^c)} = b_targ1  with h_targ1 = (h_{targ1,1}, ... , h_{targ1,N_r})
    #      coNr                    parameter used for evaluating  E{h^c_targ(H^c)}               
    #      coNr2                   parameter used for evaluating  E{h^c_targ(H^c)} 
    #      MatReta_targ(nu,N_r)    N_r realizations of the projection of XX_targ on the model
    #      eps_inv                 : tolerance for computing the pseudo-inverse of matrix MatRGammaS_iter with 
    #                                sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv) for ind_type_targ = 1,
    #                                in sub_solverInverse_constrainedByTargets. An adapted value is 0.001. If problems occurs
    #                                increase the value to 0.01.
    #                          --- ind_type_targ = 2 or 3: targets defined by giving mean value of XX_targ
    #      Rb_targ2(nu)            yielding the constraint E{H^c} = b_targ2 
    #                          --- ind_type_targ = 3: targets defined by giving target covariance matrix of XX_targ
    #      Rb_targ3(nu)            yielding the constraint diag(E{H_c H_c'}) = b_targ3  
    #
    #      ind_coupling        : 0 (no coupling). = 1, used if ind_type_targ = 3 and consisting by keeping the extradiagobal blocks 
    #                            in matrix MatRGammaS_iter(2*nu,2*nu) related to [H^c ; diag(H_c H_c')] for the computation
    #                            Lagrange mutipliers by using the iteration algorithm
    #      epsc                : relative tolerance for the iteration convergence of the constraints on H 
    #      iter_limit          : maximum number of iteration for computing the Lagrange multipliers
    #      Ralpha_relax        : Ralpha_relax(iter_limit,1) relaxation parameter in ] 0 , 1 ] for the iterations. 
    #      ind_display_screen  : = 0 no display,              = 1 display
    #      ind_print           : = 0 no print,                = 1 print
    #      ind_parallel        : = 0 no parallel computation, = 1 parallel computation
    #
    #--- OUTPUT
    #      MatReta_ar(nu,n_ar)
    #      ArrayZ_ar(nu,nbmDMAP,nbMC);    # ArrayZ_ar(nu,nbmDMAP,nbMC), this array is used as ouput for possible use in a postprocessing 
    #                                     # of Z in order to construct its polynomial chaos expansion (PCE)
 
    #--- Initialization and preallocation  
    if ind_type_targ == 1:                      #--- Constraints E{h^c_r(H^c)} = Rb_targ1(r) for r =1,...,N_r 
        mhc = N_r
        Rbc = Rb_targ1                          # Rbc(mhc) = Rb_targ1(N_r)
        Rlambda_iter_m1 = np.zeros(mhc)         # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_N_r)

    if ind_type_targ == 2:                      #--- Constraints E{H^c_j} = Rb_targ2(j,1) for j =1,...,nu 
        mhc = nu
        Rbc = Rb_targ2                          # Rbc(mhc) = Rb_targ2(nu)
        Rlambda_iter_m1 = np.zeros(mhc)         # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_nu)

    if ind_type_targ == 3:                      #--- constraints E{H^c_j} = Rb_targ2(j,1) and E{H^c_j^2} = Rb_targ3(j,1) for j =1,...,nu 
        mhc = 2*nu                              # Rbc(mhc) = [Rb_targ2 ; Rb_targ3] (en Matlab)
        Rbc = np.zeros(mhc)
        Rbc[0:nu] = Rb_targ2                    # Rb_targ2(nu)
        Rbc[nu:2*nu] = Rb_targ3                 # Rb_targ3(nu)
        Rlambda_iter_m1 = np.zeros(mhc)         # Rlambda_iter_m1(mhc) = (lambda_1,...,lambda_nu,lambda_{1+nu},...,lambda_{nu+nu})
   
    #--- Print parameters
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('--- Parameters for solver Inverse constrained by targets    \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'    ind_type_targ = {ind_type_targ:7d} \n ')
            fidlisting.write(f'    nu            = {nu:7d} \n ')
            fidlisting.write(f'    mhc           = {mhc:7d} \n ')
            fidlisting.write('      \n ')
   
    #--- Preallocation for the Lagrange-multipliers computation by the iteration algorithm and generation  of nbMC realizations                                                        
    Rerr         = np.zeros(iter_limit)   # Rerr(iter_limit)      
    RnormRlambda = np.zeros(iter_limit)   # RnormRlambda(iter_limit)
    RcondGammaS  = np.zeros(iter_limit)   # RcondGammaS(iter_limit)
    Rtol         = np.zeros(iter_limit)   # Rtol(iter_limit) 
    Rtol[0]      = 1
    ind_conv     = 0
   
    #--- Loop of the iteration algorithm (iter: lambda_iter given and compute lambda_iterplus1

    for iter in range(iter_limit):
        if ind_display_screen == 1:
            print(f'------------- iter number: {iter+1}')

        # Constraints E{h^c_r(H^c)} = Rbc(r,1) for r =1,...,N_r 
        if ind_type_targ == 1:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))      # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGauss[:,:,ell]                   # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]    # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,coNr,coNr2, 
                                                                     MatReta_targ)                                                   
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                             # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell 

            # Parallel sequence
            if ind_parallel == 1:

                def process_ell(ell):
                    MatRGauss_ell               = ArrayGauss[:, :, ell]                   # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:, :, :, ell]    # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target1(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1,coNr,coNr2, 
                                                                     MatReta_targ)               
                    return MatRZ_ar_iter

                results = Parallel(n_jobs=-1)(delayed(process_ell)(ell) for ell in range(nbMC))
                for ell, result in enumerate(results):
                    ArrayZ_ar_iter[:, :, ell] = result
                del results
                gc.collect()
            

        # Constraints E{H^c_j} = Rb_targ2(j,1) for j =1,...,nu 
        if ind_type_targ == 2:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))                    # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGauss[:,:,ell]                   # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]    # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell, Rlambda_iter_m1)
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                           # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell

            # Parallel sequence
            if ind_parallel == 1:

                def process_ell(ell):
                    MatRGauss_ell               = ArrayGauss[:,:,ell]                     # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]      # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target2(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell, Rlambda_iter_m1)          
                    return MatRZ_ar_iter

                results = Parallel(n_jobs=-1)(delayed(process_ell)(ell) for ell in range(nbMC))
                for ell, result in enumerate(results):
                    ArrayZ_ar_iter[:,:,ell] = result          # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)  
                del results
                gc.collect()
             
       
        # Constraints E{H^c_j} = Rb_targ2(j,1) and E{H^c_j^2} = Rb_targ3(j,1) for j =1,...,nu
        if ind_type_targ == 3:
            ArrayZ_ar_iter = np.zeros((nu,nbmDMAP,nbMC))                                   # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)

            # Vectorized sequence
            if ind_parallel == 0:
                for ell in range(nbMC):
                    MatRGauss_ell               = ArrayGauss[:,:,ell]                      # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]       # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)                                                
                    ArrayZ_ar_iter[:,:,ell] = MatRZ_ar_iter                             # ArrayZ_ar_iter(nu,nbmDMAP,nbMC)                                
                del MatRZ_ar_iter, MatRGauss_ell, ArrayWiennerM0transient_ell 

            # Parallel sequence
            elif ind_parallel == 1:

                def process_ell(ell):
                    MatRGauss_ell               = ArrayGauss[:,:,ell]                      # ArrayGauss(nu,n_d,nbMC)
                    ArrayWiennerM0transient_ell = ArrayWiennerM0transient[:,:,:,ell]       # ArrayWiennerM0transient(nu,n_d,M0transient,nbMC)   
                    MatRZ_ar_iter = sub_solverInverse_Verlet_target3(nu,n_d,M0transient,Deltar,f0,MatReta_d,MatRa,MatRg,shss,sh,
                                                                     ArrayWiennerM0transient_ell,MatRGauss_ell,Rlambda_iter_m1)                                                   
                    return MatRZ_ar_iter

                results = Parallel(n_jobs=-1)(delayed(process_ell)(ell) for ell in range(nbMC))
                for ell, result in enumerate(results):
                    ArrayZ_ar_iter[:,:,ell] = result
                del results
                gc.collect()            
       
        # Computing ArrayH_ar_iter(nu,n_d,nbMC)   
        ArrayH_ar_iter = np.zeros((nu,n_d,nbMC))                                                            
        for ell in range(nbMC):                                                # ArrayZ_ar_iter(nu,nbmDMAP,nbMC), MatRg(n_d,nbmDMAP) 
            ArrayH_ar_iter[:,:,ell] = ArrayZ_ar_iter[:,:,ell] @ MatRg.T        # ArrayH_ar_iter(nu,n_d,nbMC)
       
        # Computing h^c(H) loaded in MatRhc_iter(mhc,n_ar)
        MatReta_ar_iter = np.reshape(ArrayH_ar_iter,(nu,n_ar))                 # MatReta_ar_iter(nu,n_ar)
        del ArrayH_ar_iter   
       
        # Test if there NaN or Inf is obtained in the ISDE solver  
        testNaN = np.isnan(np.linalg.norm(MatReta_ar_iter,'fro'))
        if testNaN:
            print(' ')
            print('----- STOP in sub_solverInverse_constrainedByTargets: NaN or Inf is obtained in the ISDE solver')
            print(' ')
            print(f'iter         {iter+1}')
            print(' ')
            print(' If NaN or Inf is obtained after a small number of iterations, decrease the value of alpha_relax1')
            print(' If NaN or Inf is still obtained, decrease alpha_relax2 and/or increase iter_relax2')
            print(' ')
            print('Rlambda_iter_m1 = ')
            print(Rlambda_iter_m1)

            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')
                fidlisting.write('----- STOP in sub_solverInverse_constrainedByTargets: NaN or Inf is obtained in the ISDE solver \n ') 
                fidlisting.write('      \n ')             
                fidlisting.write(f'      iter             = {iter+1:7d} \n ')
                fidlisting.write('      \n ')  
                fidlisting.write('      If indetermination is reached after a small number of iterations, decrease the value of alpha_relax1. \n')
                fidlisting.write('       If indetermination is still reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
                fidlisting.write('      \n ')  
                fidlisting.write('      Rlambda_iter_m1  = \n')
                fidlisting.write(f'                         {Rlambda_iter_m1} \n ') 
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')
                
            raise ValueError('STOP:divergence of ISDE in sub_solverInverse_constrainedByTargets')
       
        # Computing and loading MatRhc_iter(mhc,n_ar);
        if ind_type_targ == 1:                                            # Constraints E{h^c_r(H^c)}=Rb_targ1(r,1), r = 1,...,N_r 
            MatRhc_iter = np.zeros((N_r,n_ar))                            # MatRhc_iter(N_r,n_ar), mhc = N_r
            # Vectorized sequence
            if ind_parallel == 0:
                for r in range(N_r):
                    MatRexp_r = MatReta_ar_iter - MatReta_targ[:, r][:, np.newaxis] # MatRexp_r(nu,n_ar),MatReta_ar_iter(nu,n_ar),MatReta_targ(nu,N_r)
                    Rexp_r = np.exp(-coNr*(np.sum(MatRexp_r**2,axis=0)))            # Rexp_r(n_ar) 
                    MatRhc_iter[r,:] = Rexp_r                                        # MatRhc_iter(N_r,n_ar)   
   
            # Parallel sequence
            if ind_parallel == 1:

                def process_r(r,MatReta_ar_iter,MatReta_targ,coNr):
                    MatRexp_r = MatReta_ar_iter - MatReta_targ[:,r][:, np.newaxis] # MatRexp_r(nu,n_ar),MatReta_ar_iter(nu,n_ar),MatReta_targ(nu,N_r)
                    Rexp_r = np.exp(-coNr*(np.sum(MatRexp_r**2,axis=0)))           # Rexp_r(n_ar) 
                    return Rexp_r

                results = Parallel(n_jobs=-1)(delayed(process_r)(r,MatReta_ar_iter,MatReta_targ,coNr) for r in range(N_r))
                for r, result in enumerate(results):
                    MatRhc_iter[r,:] = result
                del results
                gc.collect()
   
        if ind_type_targ == 2:                                 # Constraints E{H_{ar,j}} = Rb_targ2(j,1) for j = 1,...,nu
            MatRhc_iter         = np.zeros((mhc,n_ar))         # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[0:nu,:] = MatReta_ar_iter              # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)

        if ind_type_targ == 3:                                 # Constraints E{H_{ar,j}}=Rb_targ2(j,1) and E{H_{ar,j}^2}=Rb_targ3(j,1), j=1,...,nu
            MatRhc_iter            = np.zeros((mhc,n_ar))      # MatRhc_iter(mhc,n_ar);
            MatRhc_iter[0:nu,:]    = MatReta_ar_iter           # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
            MatRhc_iter[nu:2*nu,:] = MatReta_ar_iter**2        # MatReta_ar_iter(nu,n_ar),MatRhc_iter(mhc,n_ar)
            
        # Computing the values of the quantities at iteration iter
        Rmeanhc_iter         = np.mean(MatRhc_iter,axis=1)     #  Rmeanhc_iter(mhc),MatRhc_iter(mhc,n_ar)           
        RGammaP_iter         = Rbc - Rmeanhc_iter              #  RGammaP_iter(mhc),Rbc(mhc),Rmeanhc_iter(mhc) 
        MatRGammaS_iter_temp = np.cov(MatRhc_iter)             #  MatRGammaS_iter_temp(mhc,mhc) 
       
        # Updating MatRGammaS_iter_temp(mhc,mhc) if ind_coupling = 0: there is no coupling between the extradiagonal blocks in 
        # matrix  MatRGammaS_iter
        if ind_coupling == 0:
            if ind_type_targ == 1 or ind_type_targ == 2:                 # Constraints E{h^c(H_{ar})} = Rbc (no extradiagonal blocks)
                MatRGammaS_iter = MatRGammaS_iter_temp                   # MatRGammaS_iter(mhc,mhc)   
            if ind_type_targ == 3:                                       # constraints E{H_{ar,j}} = 0 and E{H_{ar,j}^2} = 1 for j =1,...,nu
                MatRGammaS_iter_temp[0:nu,nu:nu+nu] = np.zeros((nu,nu))  # there are two extradiagonal blocks: [E{H_{ar}} , E{H_{ar}^.2}}]
                MatRGammaS_iter_temp[nu:nu+nu,0:nu] = np.zeros((nu,nu))  # and its symmetric block:  [E{H_{ar}^.2} , E{H_{ar}} ]
                MatRGammaS_iter = MatRGammaS_iter_temp                   # MatRGammaS_iter(mhc,mhc)  
        if ind_coupling == 1:
            MatRGammaS_iter = MatRGammaS_iter_temp                       # MatRGammaS_iter(mhc,mhc)  
        del MatRGammaS_iter_temp 

        # Testing the convergence at iteration iter
        normbc = np.linalg.norm(Rbc)
        if normbc == 0:
            normbc = 1
        Rerr[iter]         = np.linalg.norm(RGammaP_iter)/normbc         #  Rerr(iter_limit)  
        RnormRlambda[iter] = np.linalg.norm(Rlambda_iter_m1)             #  RnormRlambda(iter_limit) 
        RcondGammaS[iter]  = np.linalg.cond(MatRGammaS_iter)             #  RcondGammaS(iter_limit) 
        if ind_display_screen == 1:
            print(f'err_iter         = {Rerr[iter]}')
            print(f'norm_lambda_iter = {RnormRlambda[iter]}')
        if iter >= 1:
            denom = (abs(Rerr[iter] + Rerr[iter-1])) / 2
            if denom == 0:
                denom = 1
            Rtol[iter] = abs(Rerr[iter] - Rerr[iter-1]) / denom
            if ind_display_screen == 1:
                print(f'tol_iter         = {Rtol[iter]}')

        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write(f'         --- iter number = {iter+1:7d} \n ')
                fidlisting.write(f'             err_iter    = {Rerr[iter]:14.7e} \n ')
                fidlisting.write(f'             tol_iter    = {Rtol[iter]:14.7e} \n ')
                fidlisting.write('      \n ')
       
        # Criterion 1: if iter > 10-1 and if Rerr(iter)-Rerr(iter-1) > 0, there is a local minimum, obtained for iter - 1.
        # convergence is then assumed to be reached and then, exit from the loop on iter  
        if (iter > 9) and (Rerr[iter] - Rerr[iter-1] > 0):
            ind_conv   = 1 
            MatReta_ar = MatReta_ar_iter_m1                         # MatReta_ar(nu,n_ar), MatReta_ar_iter_m1(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter_m1                          # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter_m1(nu,nbmDMAP,nbMC)   
            iter_plot  = iter
            if ind_display_screen == 1:
                print('Convergence with criterion 1: local minimum reached')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' --- Convergence with criterion 1: local minimum reached \n ')
                    fidlisting.write('      \n ')
                
            del MatReta_ar_iter_m1,MatReta_ar_iter,MatRhc_iter,Rmeanhc_iter,RGammaP_iter,Rlambda_iter_m1
            break    # exit from the loop on iter  

        # Criterion 2: if the relative error of the constraint satisfaction is less than or equal to the tolerance. The convergence 
        #              is reached, and then exit from the loop on iter.
        if Rerr[iter] <= epsc:
            ind_conv   = 1
            MatReta_ar = MatReta_ar_iter                                    # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter                                     # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
            iter_plot  = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 2: convergence obtained either with variance-values of H-components satisfied')
                print('                              or relative error of the constraint satisfaction is less than the tolerance')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' --- Convergence with criterion 2: convergence obtained either with variance-values \n')
                    fidlisting.write('                                    of H-components satisfied or relative error of the \n')
                    fidlisting.write('                                    constraint satisfaction is less than the tolerance \n')
                    fidlisting.write('                \n ')
            del MatReta_ar_iter_m1,MatReta_ar_iter,MatRhc_iter,Rmeanhc_iter,RGammaP_iter,Rlambda_iter_m1
            break                          # exit from the loop on iter

        # Criterion 3: if iter > min(20,iter_limit) and Rtol(iter) < epsc,  the error is stationary and thus
        # the convergence is assumed to be reached and then, exit from the loop on iter  
        if (iter > min(20,iter_limit) and Rerr[iter] < epsc) and (Rtol[iter] < epsc):
            ind_conv   = 1
            MatReta_ar = MatReta_ar_iter                                    # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
            ArrayZ_ar  = ArrayZ_ar_iter                                     # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
            iter_plot  = iter + 1
            if ind_display_screen == 1:
                print('Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary')
            if ind_print == 1:
                with open('listing.txt', 'a+') as fidlisting:
                    fidlisting.write('      \n ')
                    fidlisting.write(' --- Convergence with criterion 3: iteration number greater that iter_limit and the error is stationary \n ')
                    fidlisting.write('      \n ')
            del MatReta_ar_iter_m1,MatReta_ar_iter,MatRhc_iter,Rmeanhc_iter,RGammaP_iter,Rlambda_iter_m1
            break                         # exit from the loop on iter
      
        # Convergence not reached
        if ind_conv == 0:
            if ind_type_targ == 1:
                MatRGammaSinv_iter = sub_solverInverse_pseudo_inverse(MatRGammaS_iter,eps_inv)
                Rtemp_iter = MatRGammaSinv_iter @ RGammaP_iter                        # Rtemp_iter = pseudo_inv(MatRGammaS_iter)*RGammaP_iter                 
            if ind_type_targ == 2 or ind_type_targ == 3:
                Rtemp_iter = np.linalg.solve(MatRGammaS_iter,RGammaP_iter)            # Rtemp_iter = inv(MatRGammaS_iter)*RGammaP_iter
            Rlambda_iter       = Rlambda_iter_m1 - Ralpha_relax[iter] * Rtemp_iter;   # Rlambda_iter(mhc), Rlambda_iter_m1(mhc)
            Rlambda_iter_m1    = Rlambda_iter
            MatReta_ar_iter_m1 = MatReta_ar_iter
            ArrayZ_ar_iter_m1  = ArrayZ_ar_iter
            del MatRhc_iter,Rlambda_iter,Rmeanhc_iter,RGammaP_iter,MatRGammaS_iter,Rtemp_iter

    #--- if ind_conv = 0, then iter_limit is reached without convergence
    if ind_conv == 0:
        MatReta_ar = MatReta_ar_iter                                    # MatReta_ar(nu,n_ar), MatReta_ar_iter(nu,n_ar)
        ArrayZ_arb = ArrayZ_ar_iter                                     # ArrayZ_ar(nu,nbmDMAP,nbMC),ArrayZ_ar_iter(nu,nbmDMAP,nbMC)
        iter_plotb = iter_limit
        if ind_display_screen == 1:
            print('------ No convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets')
            print(f'       iter_plot = {iter_plot}')
            print('        If convergence is not reached after a small number of iterations, decrease the value of alpha_relax1. \n')
            print('        If convergence is still not reached, decrease alpha_relax2 and/or increase iter_relax2. \n')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write(' --- No convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets  \n ')
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
            print('------ Convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets ')
            print(f'       iter_plot = {iter_plot}')
        if ind_print == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('      \n ')
                fidlisting.write('      \n ')
                fidlisting.write(' --- Convergence of the iteration algorithm in sub_solverInverse_constrainedByTargets   \n ') 
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
    plt.savefig(f'figure_sub_solverInverse_constrainedByTargets_{numfig}_Rerr.png')
    plt.close(h)

    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), RnormRlambda[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\Vert \lambda_{\iota}\Vert $', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)                                                                
    plt.ylabel(r'$\Vert \lambda_{\iota}\Vert $', fontsize=16)  
    numfig = numfig + 1
    plt.savefig(f'figure_sub_solverInverse_constrainedByTargets_{numfig}_RnormRlambda.png')
    plt.close(h)

    h = plt.figure()      
    plt.plot(range(1, iter_plot + 1), RcondGammaS[:iter_plot], 'b-')                                                 
    plt.title(r'Graph of function $\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16, weight='normal')                                         
    plt.xlabel(r'$\iota$', fontsize=16)  
    plt.ylabel(r'$\rm{cond} [\Gamma^{\prime\prime}(\lambda_{\iota})]$', fontsize=16)
    numfig = numfig + 1
    plt.savefig(f'figure_sub_solverInverse_constrainedByTargets_{numfig}_RcondGammaS.png')
    plt.close(h)

    #--- Print The relative norm of the extradiagonal term that as to be close to 0R
    #    and print Hmean_ar and diag(MatRHcov_ar)
    #                                                     
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
