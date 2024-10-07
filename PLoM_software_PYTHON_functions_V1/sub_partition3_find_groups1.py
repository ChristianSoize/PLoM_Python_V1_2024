import numpy as np
import gc
from joblib import Parallel, delayed
from sub_partition13_testINDEPr1r2 import sub_partition13_testINDEPr1r2

def sub_partition3_find_groups1(NKL,nr,MatRHexp,INDEPref,ind_parallel):

    # Copyright C. Soize 24 May 2024, revised 3 July 2024
    #
    # ---INPUTS 
    #          NKL              : dimension of random vector H
    #          nr               : number of independent realizations of random vector H
    #          MatRHexp(NKL,nr) : nr realizations of H = (H_1,...,H_NKL)
    #          INDEPref         : value of the mutual information to obtain the dependence of H_r1 with H_r2 used as follows.
    #                             Let INDEPr1r2 = i^nu(H_r1,H_r2) be the mutual information of random variables H_r1 and H_r2
    #                             One says that random variables H_r1 and H_r2 are DEPENDENT if INDEPr1r2 >= INDEPref.
    #          ind_parallel     : = 0 no parallel computing, = 1 parallel computing
    # --- OUTPUTS
    #           ngroup                   :  number of constructed independent groups  
    #           Igroup(ngroup)           :  vector Igroup(ngroup), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)
    #           mmax                     :  mmax = max_j mj for j = 1, ... , ngroup
    #           MatIgroup(ngroup,mmax)   :  MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj  
    #
    # --- METHOD 
    #     Constructing the groups using a graph approach:
    #     Step 1: computing the number of edges in the graph by analyzing the statistical dependence of the components of the random 
    #             vector H = (H_1,...,H_NKL) by testing the dependence 2 by 2. The test of the independence of two non-Gaussian normalized random 
    #             variables H_r1 et H_r2 is performed by using the MUTUAL INFORMATION criterion that is written as INDEPr1r2 = Sr1 + Sr2 - Sr1r2 
    #             with Sr1 = entropy of H_r1, Sr2 = entropy of H_r2, Sr1r2 =  entropy of (Hr1,Hr2) 
    #             The entropy that is a mathematical expectation of the log of a pdf is estimated with the Monte Carlo method by using the 
    #             same realizations that the one used for estimating the pdf by the Gaussian kernel method.
    #             The random variables H_r1 and H_r2 are assumed to be DEPENDENT if INDEPr1r2 > INDEPref.
    #
    #     Step 2: constructing the groups in exploring the common Nodes to the edges of the graph

    # --- STEP 1: constructing the adjacency matrix MatcurN 
    #             MatcurN(NKL,NKL): symmetric adjacency matrix such that MatcurN(r1,r2) = 1 if r1 and r2 are the two end nodes of an edge 
    
                                  #--- Matlab detailed algorithm for constructing the symmetric adjacency matrix MatcurN(NKL,NKL) 
                                  #    for a given level INDEPref  
                                  #    MatcurN   = zeros(NKL,NKL);
                                  #    for r1 = 1:NKL-1
                                  #        for r2 = r1+1:NKL           
                                  #               [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);                      
                                  #               if INDEPr1r2  > INDEPref              % H_r1 and H_r2 are dependent  
                                  #                  MatcurN(r1,r2)  = 1;
                                  #                  MatcurN(r2,r1)  = 1;
                                  #               end
                                  #        end
                                  #    end   
    npairs = NKL*(NKL - 1) // 2
    Indr1 = np.zeros(npairs,dtype=int)
    Indr2 = np.zeros(npairs,dtype=int)
    p = 0
    for r1 in range(1,NKL):
        for r2 in range(r1+1,NKL+1):
            p = p +1
            Indr1[p-1] = r1
            Indr2[p-1] = r2

    # --- Sequential computation
    if ind_parallel == 0:
        RINDEP = np.zeros(npairs)
        for p in range(npairs):
            r1 = Indr1[p]
            r2 = Indr2[p]
            INDEPr1r2 = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2)                      
            RINDEP[p] = INDEPr1r2

    # --- Parallel computation
    if ind_parallel == 1:
        def compute_INDEPr1r2(p,NKL,nr,MatRHexp,Indr1,Indr2):
            r1 = Indr1[p]
            r2 = Indr2[p]
            result = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2)
            return result

        # Exécution parallèle
        RINDEP_list = Parallel(n_jobs=-1)(delayed(compute_INDEPr1r2)(p,NKL,nr,MatRHexp,Indr1,Indr2) for p in range(npairs))
        RINDEP = np.array(RINDEP_list)
        del RINDEP_list
        gc.collect()
        
    MatcurN = np.zeros((NKL,NKL))
    for p in range(npairs):
        r1 = Indr1[p]
        r2 = Indr2[p] 
        INDEPr1r2 = RINDEP[p]
        if INDEPr1r2 > INDEPref:           # H_r1 and H_r2 are dependent  
            MatcurN[r1-1,r2-1] = 1
            MatcurN[r2-1,r1-1] = 1

    # --- STEP 2: Constructing the groups using a graph algorithm
    igroup = 0
    Igroup = np.zeros(NKL,dtype=int)
    MatIgroup = np.zeros((NKL,NKL),dtype=int)
    U = np.zeros(NKL,dtype=int)   # If U[r] == 0, then node r not treated; if U[r] == 1, then node r has been treated

    while np.any(U == 0):          # While there are nodes that have not been treated
        U0 = np.where(U == 0)[0]   # Find nodes that have not been treated
        x = U0[0]                  # Node used for starting the construction of a new group
        P = set()                  # List of the nodes to be analyzed, using a set to avoid duplicates
        V = set()                  # List of the nodes already analyzed
        RS = set([x])              # RS contains the nodes of the present group in construction
        igroup = igroup + 1

        P.add(x)
        while P:  # While P is not empty
            y = P.pop()  # Load a node and unstack P
            V.add(y)
            for z in range(NKL):                                  # Exploring all nodes z such that MatcurN[y, z] == 1 and z not in P or V
                if MatcurN[y, z] == 1 and z not in P.union(V):
                    P.add(z)                                      # Stack (P, z)
                    RS.add(z)                                     # z belongs to the subset
            
        m_igroup = len(RS)
        Igroup[igroup-1] = m_igroup                  # Store size of the current group
        RS_array = np.array(list(RS)) + 1
        MatIgroup[igroup-1,:m_igroup] = RS_array     # Increment each element in RS by 1 before storing
        U[list(RS)] = 1                              # All nodes in RS have been treated
        MatcurN[np.ix_(list(RS), list(RS))] = 0      # Set to zero the nodes belonging to the group igroup

    ngroup = igroup

    if ngroup < NKL:
        Igroup = Igroup[:ngroup]
        MatIgroup = MatIgroup[:ngroup, :]
        mmax = np.max(Igroup)
        MatIgroup = MatIgroup[:, :mmax]
    
    Igroup    = Igroup.astype(int)
    MatIgroup = MatIgroup.astype(int)
        
    if 'ngroup' not in locals() or 'Igroup' not in locals() or 'mmax' not in locals() or 'MatIgroup' not in locals():
        raise ValueError('STOP in sub_partition3_find_groups1: variable mmax does not exist. Remove the greatest value introduced in RINDEPref')

    return ngroup, Igroup, mmax, MatIgroup

