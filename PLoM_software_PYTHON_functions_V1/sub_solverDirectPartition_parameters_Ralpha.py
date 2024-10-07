import numpy as np

def sub_solverDirectPartition_parameters_Ralpha(ind_constraints, iter_limit, alpha_relax1, iter_relax2, alpha_relax2):
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 2 July 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PlOM) 
    #  Function name: sub_solverDirectPartition_parameters_Ralpha.m
    #  Subject      : calculating and loading Ralpha_relax(iter_limit,1)
    #
    #--- INPUT  
    #        ind_constraints = 0 : no constraints concerning E{H] = 0 and E{H H'} = [I_nu]
    #                        = 1 : constraints E{H_j^2} = 1 for j =1,...,nu   
    #                        = 2 : constraints E{H] = 0 and E{H_j^2} = 1 for j =1,...,nu
    #                        = 3 : constraints E{H] = 0 and E{H H'} = [I_nu]  
    #        iter_limit      : maximum number of iterations used by the iterative algorithm to compute the Lagrange multipliers. 
    #                          A relaxation function  iter --> alpha_relax(iter)  controls the convergence of the iterative algorithm 
    #                          that is described by 3 parameters: alpha_relax1, iter_relax2, and alpha_relax2
    #        alpha_relax1    : value of alpha_relax for iter = 1  (for instance 0.001)
    #        iter_relax2     : value of iter (for instance, 20) such that  alpha_relax2 = alpha_relax(iter_relax2) 
    #                          if iter_relax2 = 1, then alpha_relax (iter) = alpha_relax2 for all iter >=1   
    #        alpha_relax2    : value of alpha_relax (for instance, 0.05) such that alpha_relax(iter >= iter_relax2) = apha_relax2
    #                          NOTE 1: If iter_relax2 = 1 , then Ralpha_relax(iter) = alpha_relax2 for all iter >=1
    #                          NOTE 2: If iter_relax2 >= 2, then  
    #                                  for iter >= 1 and for iter < iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax1 + (alpha_relax2 - alpha_relax1)*(iter-1)/(iter_relax2-1)
    #                                  for iter >= iter_relax2, we have:
    #                                      alpha_relax(iter) = alpha_relax2
    #                          NOTE 3: for decreasing the error err(iter), increase the value of iter_relax2
    #                          NOTE 4: if iteration algorithm dos not converge, decrease alpha_relax2 and/or increase iter_relax2
    #
    #--- OUTPUT
    #          Ralpha_relax(iter_limit): relaxation function for the iteration algorithm that computes the LaGrange multipliers
    
    #--- Generation of Ralpha_relax(iter_limit) 

    # No constraints
    if ind_constraints == 0:
        Ralpha_relax = np.array([])
        return Ralpha_relax

    # There are constraints
    if ind_constraints >= 1:
        Ralpha_relax = np.zeros(iter_limit)
        if iter_relax2 == 1:
            Ralpha_relax[:iter_limit] = alpha_relax2
        if iter_relax2 >= 2:
            for iter in range(1, iter_limit + 1):
                if iter >= 1 and iter < iter_relax2:
                    Ralpha_relax[iter - 1] = alpha_relax1 + (alpha_relax2 - alpha_relax1) * (iter - 1) / (iter_relax2 - 1)
                if iter >= iter_relax2:
                    Ralpha_relax[iter - 1] = alpha_relax2
        return Ralpha_relax
