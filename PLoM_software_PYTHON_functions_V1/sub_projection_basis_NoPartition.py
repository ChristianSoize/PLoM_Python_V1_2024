import numpy as np
import time
from sub_projection_basis_read_binary_file import sub_projection_basis_read_binary_file
from sub_projection_basis_read_text_file import sub_projection_basis_read_text_file
from sub_projection_basis_isotropic_kernel import sub_projection_basis_isotropic_kernel

def sub_projection_basis_NoPartition(nu, n_d, MatReta_d, ind_generator, mDP, nbmDMAP, ind_basis_type, 
                                     ind_file_type, ind_display_screen, ind_print, ind_plot, 
                                     ind_parallel, epsilonDIFFmin, step_epsilonDIFF, 
                                     iterlimit_epsilonDIFF, comp_ref):

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 2 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_projection_basis_NoPartition.py
    #  Subject      : loading and or constructing the projection basis
    #
    #  Publications: [1] C. Soize, R. Ghanem, Data-driven probability concentration and sampling on manifold, 
    #                       Journal of Computational Physics,  doi:10.1016/j.jcp.2016.05.044, 321, 242-258 (2016).
    #                [2] C. Soize, R. Ghanem, Probabilistic learning on manifolds, Foundations of Data Science, 
    #                       American  Institute of Mathematical Sciences (AIMS),  doi: 10.3934/fods.2020013, 2(3), 279-307 (2020). 
    #                [3] C. Soize, R. Ghanem, Probabilistic learning on manifolds (PLoM) with partition, International Journal for 
    #                       Numerical Methods in Engineering, doi: 10.1002/nme.6856, 123(1), 268-290 (2022).
    #                [4] C. Soize, R. Ghanem, Transient anisotropic kernel for probabilistic learning on manifolds, 
    #                       Computer Methods in Applied Mechanics and Engineering, pp.1-44 (2024).
    #
    # --- INPUTS 
    #        nu                  : dimension of random vector H = (H_1, ... H_nu)
    #        n_d                 : number of points in the training set for H
    #        MatReta_d(nu,n_d)   : n_d realizations of H   
    #        ind_generator       : 0 generator without using projection basis = dissipative-Hamiltonian-based MCMC
    #                            : 1 generator using the projection basis 
    #        mDP                 : maximum number of projection-basis vectors that are read on a file or that are generated
    #                              by the isotropic kernel, and which must be such that  nbmDMAP <= mDP <= n_d
    #        nbmDMAP             : dimension of the projection basis such that nbmDMAP <= mDP
    #        ind_basis_type      : 1, read nd,mDP,MatRg_mDP(nd,mDP) on a Binary File or a Text File  where nd should be equal to n_d
    #                            : 2, generation of the projection basis solving the eigenvalue problem related to the isotropic kernel
    #        ind_file_type       : used if ind_basis_type = 1
    #                            : 1, Binary File type and ind_basis_type = 1, with filename = 'ISDEprojectionBasisBIN.bin'
    #                            : 2, Text   File type and ind_basis_type = 1, with filename = 'ISDEprojectionBasisTXT.txt'    
    #                            --- parameters and variables controlling execution
    #        ind_display_screen  : 0, no display, if 1 display
    #        ind_print           : 0, no print, if 1 print
    #        ind_plot            : 0, no plot, if  1 plot
    #        ind_parallel        : 0, no parallel computation, if 1 parallel computation
    #                            --- if ind_basis_type = 2, then the following parameters are required for generating
    #                                   the projection basis by solving the eigenvalue problem related to the isotropic kernel:
    #                                   the smoothing parameter epsilonDIFF   is searched with an iteration algorithm              
    #        epsilonDIFFmin         :   epsilonDIFF is searched in interval [epsilonDIFFmin , +infty[                                    
    #        step_epsilonDIFF       :   step for searching the optimal value epsilonDIFF starting from epsilonDIFFmin
    #        iterlimit_epsilonDIFF  :   maximum number of the iteration algorithm for computing epsilonDIFF                              
    #        comp_ref               :   value in  [ 0.1 , 0.5 [  used for stopping the iteration algorithm.
    #                                   if comp =  Rlambda(nbmDMAP+1)/Rlambda(nbmDMAP) <= comp_ref, then algorithm is stopped
    #                                   The standard value for comp_ref is 0.2 
    #
    # --- OUTPUTS
    #        MatRg(n_d,nbmDMAP)  : matrix of the nbmDMAP projection basis vectors
    #        MatRa(n_d,nbmDMAP)  = MatRg*(MatRg.T @ MatRg).I 

    if ind_display_screen != 0 and ind_display_screen != 1:
        raise ValueError('STOP9 in sub_projection_basis_NoPartition: ind_display_screen must be equal to 0 or equal to 1')
    if ind_print != 0 and ind_print != 1:
        raise ValueError('STOP10 in sub_projection_basis_NoPartition: ind_print must be equal to 0 or equal to 1')
    if ind_plot != 0 and ind_plot != 1:
        raise ValueError('STOP11 in sub_projection_basis_NoPartition: ind_plot must be equal to 0 or equal to 1')
    if ind_parallel != 0 and ind_parallel != 1:
        raise ValueError('STOP12 in sub_projection_basis_NoPartition: ind_parallel must be equal to 0 or equal to 1')

    if ind_display_screen == 1:
        print('--- beginning Task5_ISDEProjectionBasis')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write(' ------ Task5_ISDEProjectionBasis \n ')
            fidlisting.write('      \n ')

    TimeStartISDEprojectionBasis = time.time()

    # ----------------------------------------------------------------------------------------------------------------------------------
    #                                    Check data, parameters, and initialization
    # ---------------------------------------------------------------------------------------------------------------------------------- 

    if nu > n_d or nu < 1 or n_d < 1:
        raise ValueError('STOP1 in sub_projection_basis_NoPartition: nu > n_d or nu < 1 or n_d < 1')

    nutemp, ndtemp = MatReta_d.shape  # MatReta_d(nu,n_d) 
    if nutemp != nu or ndtemp != n_d:
        raise ValueError('STOP2 in sub_projection_basis_NoPartition: the dimensions of MatReta_d are not consistent with nu and n_d')

    if ind_generator != 0 and ind_generator != 1:
        raise ValueError('STOP3 in sub_projection_basis_NoPartition: ind_generator must be equal to 0 or equal to 1')

    if ind_generator == 1:
        if mDP > n_d or mDP < 1:
            raise ValueError('STOP4 in sub_projection_basis_NoPartition: mDP > n_d or mDP < 1')
        if nbmDMAP > mDP or nbmDMAP < 1:
            raise ValueError('STOP5 in sub_projection_basis_NoPartition: nbmDMAP > mDP or nbmDMAP < 1')
        if ind_basis_type != 1 and ind_basis_type != 2:
            raise ValueError('STOP6 in sub_projection_basis_NoPartition: ind_basis_type must be equal to 1 or 2')        
        if ind_generator == 1 and ind_basis_type == 1:
            if ind_file_type != 1 and ind_file_type != 2:
                raise ValueError('STOP8 in sub_projection_basis_NoPartition: when ind_basis_type = 1, then ind_file_type must be equal to 1 or 2')

        if ind_basis_type == 2:  # generation of the projection basis solving the eigenvalue problem related to the isotropic kernel
            if nbmDMAP != nu + 1:  # in that case, we should have nbmDMAP = nu + 1
                raise ValueError('STOP13 in sub_projection_basis_NoPartition: when ind_basis_type = 2, we should have nbMDMAP = nu + 1')
        if ind_basis_type == 2:
            if not (0.1 <= comp_ref < 0.5):  #  comp_ref given by the user in [ 0.1 , 0.5 [
                raise ValueError('STOP14: for ind_basis_type = 2, comp_ref must be given by the user between 0.1 and 0.5')
            if epsilonDIFFmin <= 0:
                raise ValueError('STOP15: for ind_basis_type = 2, epsilonDIFFmin must be given by the user as a strictly positive real number')
            if step_epsilonDIFF <= 0:
                raise ValueError('STOP16: for ind_basis_type = 2, step_epsilonDIFF must be given by the user as a strictly positive real number')
            if iterlimit_epsilonDIFF < 1:
                raise ValueError('STOP17: for ind_basis_type = 2, iterlimit_epsilonDIFF must be given by the user as an integer larger than or equal to 1')

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(' ---  Parameters for the learning \n ')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' nu            = {nu:7d} \n ')
            fidlisting.write(f' n_d           = {n_d:7d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' ind_generator = {ind_generator:1d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f' ind_display_screen = {ind_display_screen:1d} \n ')
            fidlisting.write(f' ind_print          = {ind_print:1d} \n ')
            fidlisting.write(f' ind_plot           = {ind_plot:1d} \n ')
            fidlisting.write(f' ind_parallel       = {ind_parallel:1d} \n ')
            fidlisting.write('      \n ')
            if ind_generator == 0:
                nbmDMAP = n_d
                fidlisting.write(f' nbmDMAP       = {nbmDMAP:7d} \n ')
            if ind_generator == 1:
                fidlisting.write(f' mDP           = {mDP:7d} \n ')
                fidlisting.write(f' nbmDMAP       = {nbmDMAP:7d} \n ')
                fidlisting.write('      \n ')
                fidlisting.write(f' ind_basis_type     = {ind_basis_type:1d} \n ')
                if ind_basis_type == 1:
                    fidlisting.write(f' ind_file_type      = {ind_file_type:1d} \n ')
                fidlisting.write('      \n ')
                if ind_basis_type == 2:
                    fidlisting.write(f' epsilonDIFFmin         = {epsilonDIFFmin:14.7e} \n ')
                    fidlisting.write(f' step_epsilonDIFF       = {step_epsilonDIFF:14.7e} \n ')
                    fidlisting.write(f' iterlimit_epsilonDIFF  = {iterlimit_epsilonDIFF:7d} \n ')
                    fidlisting.write(f' comp_ref               = {comp_ref:5.1f} \n ')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')

    # ----------------------------------------------------------------------------------------------------------------------------------
    #     ind_generator = 0:  the projection basis is the canonical basis of R^{n_d}                                
    # ----------------------------------------------------------------------------------------------------------------------------------       

    if ind_generator == 0:  # the parameter nbmDMAP entered by the used is modified  
        MatRg = np.eye(n_d,n_d)     # MatRg(n_d,nbmDMAP) : matrix of the nbmDMAP projection basis vectors
        MatRa = np.eye(n_d,n_d)     # MatRa(n_d,nbmDMAP) = MatRg*(MatRg.T @ MatRg).I 

    # ----------------------------------------------------------------------------------------------------------------------------------
    #     ind_basis_type = 1:  read nd, mDP, MatRg_mDP(nd,mDP) on a Binary File or a Text File  where nd should be equal to n_d                                 
    # ---------------------------------------------------------------------------------------------------------------------------------- 

    if ind_generator == 1 and ind_basis_type == 1:

        # --- read Binary File
        if ind_file_type == 1:
            filename = 'ISDEprojectionBasisBIN.bin'
            MatRg_mDP = sub_projection_basis_read_binary_file(filename, n_d, mDP)
            MatRg = MatRg_mDP[:, :nbmDMAP]  # MatRg(n_d,nbmDMAP) matrix of the nbmDMAP projection basis vectors
            MatRa = np.linalg.solve(MatRg.T @ MatRg, MatRg.T).T  # MatRa(n_d,nbmDMAP): [a] =[g]([g]'[g])^(-1)  

        # --- read Text File
        if ind_file_type == 2:
            filename = 'ISDEprojectionBasisTXT.txt'
            MatRg_mDP = sub_projection_basis_read_text_file(filename, n_d, mDP)
            MatRg     = MatRg_mDP[:, :nbmDMAP]                                 # MatRg(n_d,nbmDMAP) matrix of the nbmDMAP projection basis vectors
            MatRa     = np.linalg.solve(MatRg.T @ MatRg, MatRg.T).T            # MatRa(n_d,nbmDMAP): [a] =[g]([g]'[g])^(-1)  

    # ----------------------------------------------------------------------------------------------------------------------------------                           
    #      ind_basis_type == 2: generation of the projection basis solving the eigenvalue problem related to the isotropic kernel:
    #                           MatRg(n_d,nbmDMAP): matrix of the nbmDMAP projection basis vectors
    #                           MatRa(n_d,nbmDMAP) = MatRg*(MatRg'*MatRg)^{-1} 
    # ---------------------------------------------------------------------------------------------------------------------------------- 

    if ind_generator == 1 and ind_basis_type == 2:
        MatRg = sub_projection_basis_isotropic_kernel(nu, n_d, MatReta_d, mDP, nbmDMAP, epsilonDIFFmin, step_epsilonDIFF, 
                                                      iterlimit_epsilonDIFF, comp_ref, ind_display_screen, ind_print, ind_plot)
        MatRa     = np.linalg.solve(MatRg.T @ MatRg, MatRg.T).T        # MatRa(n_d,nbmDMAP): [a] =[g]([g]'[g])^(-1)  

    ElapsedISDEprojectionBasis = time.time() - TimeStartISDEprojectionBasis

    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write('-------   Elapsed time for Task5_ISDEProjectionBasis \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'Elapsed Time   =  {ElapsedISDEprojectionBasis:10.2f}\n')
            fidlisting.write('      \n ')

    if ind_display_screen == 1:
        print('--- end Task5_ISDEProjectionBasis')

    return MatRg, MatRa
