import numpy as np
import gc
from joblib import Parallel, delayed
from sub_partition13_testINDEPr1r2 import sub_partition13_testINDEPr1r2

def sub_partition4_find_groups2(NKL, nr, MatRHexp, INDEPref, ind_parallel):

    # Copyright C. Soize 24 May 2024
    
    #---INPUTS 
    #         NKL              : dimension of random vector H
    #         nr               : number of independent realizations of random vector H
    #         MatRHexp(NKL,nr) : nr realizations of H = (H_1,...,H_NKL)
    #         INDEPref         : value of the mutual information to obtain the dependence of H_r1 with H_r2 used as follows.
    #                            Let INDEPr1r2 = i^\nu(H_r1,H_r2) be the mutual information of random variables H_r1 and H_r2
    #                            One says that random variables H_r1 and H_r2 are DEPENDENT if INDEPr1r2 >= INDEPref.
    #         ind_parallel     : = 0 no parallel computing, = 1 parallel computing
    #--- OUTPUTS
    #          ngroup                  : number of constructed independent groups  
    #          Igroup(ngroup)          : vector Igroup(ngroup,1), mj = Igroup(j),  mj is the number of components of Y^j = (H_jr1,... ,H_jrmj)
    #          mmax                    : mmax = max_j mj for j = 1, ... , ngroup
    #          MatIgroup(ngroup,mmax)  : MatIgroup1(j,r) = rj, in which rj is the component of H in group j such that Y^j_r = H_jrj     
    #          nedge                   : number of pairs (r1,r2) for which H_r1 and H_r2 are dependent (number of edges in the graph)
    #          nindep                  : number of pairs (r1,r2) for which H_r1 and H_r2 are independent
    #          npair                   : total number of pairs (r1,r2)    = npairmax = NKL(NKL-1)/2
    #          MatPrintEdge(nedge,5)   : such that MatPrintEdge(edge,:)   = [edge  r1 r2 INDEPr1r2 INDEPref]
    #          MatPrintIndep(nindep,5) : such that MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPref]
    #          MatPrintPair(npair,5)   : such that MatPrintPair(pair,:)   = [pair  r1 r2 INDEPr1r2 INDEPref]
    #          RplotPair(npair)        : 1D array  RplotPair(pair) = INDEPr1r2 with pair=(r1,r2)
    #
    #--- METHOD 
    #
    #    Constructing the groups using a graph approach:
    #    Step 1: computing the number of edges in the graph by analyzing the statistical dependence of the components of the random 
    #            vector H = (H_1,...,H_NKL) by testing the dependence 2 by 2. The test of the independence of two non-Gaussian normalized random 
    #            variables H_r1 et H_r2 is performed by using the MUTUAL INFORMATION criterion that is written as INDEPr1r2 = Sr1 + Sr2 - Sr1r2 
    #            with Sr1 = entropy of H_r1, Sr2 = entropy of H_r2, Sr1r2 =  entropy of (Hr1,Hr2) 
    #            The entropy that is a mathematical expectation of the log of a pdf is estimated using the same realizations  that the one 
    #            used for estimating the pdf by the Gaussian kernel method.
    #            The random variables H_r1 and H_r2 are assumed to be DEPENDENT if INDEPr1r2 > INDEPref.
    #
    #    Step 2: constructing the groups in exploring the common Nodes to the edges of the graph
    #
    #------------------------------------------------------------------------------------------------------------------------------------------
    
    #--- STEP 1: constructing the symmetric adjacency matrix MatcurN(NKL,NKL) for a given level INDEPref
    #            MatcurN(NKL,NKL)       : symmetric adjacenty matrix such that MatcurN(r1,r2) = 1 if r1 and r2 are the two end nodes of an edge 
    #            MatPrintEdge(nedge,5)  : for print and plot
    #            MatPrintIndep(nindep,5): for print and plot
    #            MatPrintPair(npair,5)  : for print and plot 
    #            RplotPair(npairmax)  : for plot 

    npairmax      = NKL * (NKL - 1) // 2    
    MatcurN       = np.zeros((NKL, NKL))     # adjacency matrix
    RplotPair     = np.zeros(npairmax)       # stores mutual information values for pairs
    MatPrintEdge  = np.zeros((npairmax, 5))  # edges to be printed
    MatPrintIndep = np.zeros((npairmax, 5))  # independencies to be printed
    MatPrintPair  = np.zeros((npairmax, 5))  # pairs to be printed
     
    #--- Scalar sequence with matlab coding
    #    for r1 = 1:NKL-1
    #        for r2 = r1+1:NKL           
    #               [INDEPr1r2] = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2);
    #               pair = pair + 1;
    #               RplotPair(pair) = INDEPr1r2;
    #               MatPrintPair(pair,:) = [pair r1 r2 INDEPr1r2 INDEPref];               
    #               if INDEPr1r2  > INDEPref                   # H_r1 and H_r2 are dependent        
    #                  edge = edge + 1;
    #                  MatcurN(r1,r2)  = 1;
    #                  MatcurN(r2,r1)  = 1;
    #                  MatPrintEdge(edge,:) = [edge r1 r2 INDEPr1r2 INDEPref];                 
    #               end
    #               if INDEPr1r2  <= INDEPref        # H_r1 and H_r2 are independent, just loaded for printing                   
    #                  indep = indep + 1;
    #                  MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPref];                 
    #               end
    #        end
    #    end
        
    # Construct pairs of indices
    Indr1 = np.zeros(npairmax, dtype=int)
    Indr2 = np.zeros(npairmax, dtype=int)
    pair = 0
    for r1 in range(1,NKL):
        for r2 in range(r1 + 1, NKL+1):
            pair = pair + 1
            Indr1[pair-1] = r1
            Indr2[pair-1] = r2

    # Sequential computation
    if ind_parallel == 0:
        for pair in range(npairmax):
            r1 = Indr1[pair]
            r2 = Indr2[pair]
            INDEPr1r2 = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2)
            RplotPair[pair] = INDEPr1r2

    # Parallel computation
    if ind_parallel == 1:
        def compute_indep(pair,NKL,nr,MatRHexp,Indr1,Indr2):
            r1 = Indr1[pair]
            r2 = Indr2[pair]
            result = sub_partition13_testINDEPr1r2(NKL,nr,MatRHexp,r1,r2)
            return result
        
        RplotPair_list = Parallel(n_jobs=-1)(delayed(compute_indep)(pair,NKL,nr,MatRHexp,Indr1,Indr2) for pair in range(npairmax))
        RplotPair = np.array(RplotPair_list)
        del RplotPair_list
        gc.collect()

    # Populate the adjacency matrix and pair information
    edge = 0
    indep = 0
    for pair in range(npairmax):
        r1 = Indr1[pair]
        r2 = Indr2[pair]
        INDEPr1r2 = RplotPair[pair]
        MatPrintPair[pair,:] = [pair+1,r1,r2,INDEPr1r2,INDEPref]
        
        if INDEPr1r2 > INDEPref:  # Dependent pairs
            edge = edge + 1
            MatcurN[r1-1,r2-1] = 1
            MatcurN[r2-1,r1-1] = 1
            MatPrintEdge[edge-1,:] = [edge,r1,r2,INDEPr1r2,INDEPref]
        
        if INDEPr1r2 <= INDEPref:  # Independent pairs
            indep = indep + 1
            MatPrintIndep[indep-1,:] = [indep,r1,r2,INDEPr1r2,INDEPref]
    
    # Adjust the matrix sizes based on actual edges and independent pairs
    nedge  = edge
    nindep = indep
    npair  = npairmax
    if nedge < npairmax:
        MatPrintEdge = MatPrintEdge[:nedge, :]  # Adjusting the dimensions by keeping only the first 'nedge' rows
    if nindep < npairmax:
        MatPrintIndep = MatPrintIndep[:nindep, :]  # Adjusting the dimensions by keeping only the first 'nindep' rows

    # Step 2: Constructing groups using a graph algorithm
    igroup    = 0
    Igroup    = np.zeros(NKL,dtype=int)
    MatIgroup = np.zeros((NKL,NKL),dtype=int)
    U         = np.zeros(NKL,dtype=int)  # If U[r] == 0, then node r not treated; if U[r] == 1, then node r has been treated

    while np.any(U == 0):         # While there are nodes that have not been treated
        U0 = np.where(U == 0)[0]  # Find nodes that have not been treated
        x  = U0[0]                # Node used for starting the construction of a new group
        P  = [x]                  # Use a list to maintain order for P
        V  = set()                # List of the nodes already analyzed, using a set to avoid duplicates
        RS = [x]                  # Use a list to maintain order for RS
        igroup = igroup + 1
        
        while P:  # While P is not empty
            y = P.pop()  # Load a node and unstack P (LIFO behavior)
            if y not in V:
                V.add(y)
            for z in range(NKL):  # Explore neighbors of node y
                if MatcurN[y,z] == 1 and z not in P and z not in V:
                    P.append(z)   # Use append to maintain order in P
                    if z not in RS:
                        RS.append(z)  # Append z to RS if it's not already there
        
        m_igroup = len(RS)                       #  Number of columns in RS        
        Igroup[igroup-1] = m_igroup
        RS_array = np.array(RS) + 1  # Convert the ordered RS list to an array and increment by 1
        MatIgroup[igroup-1, :m_igroup] = RS_array  # Store RS in MatIgroup
        U[RS] = 1                          # All nodes in RS have been treated
        MatcurN[np.ix_(RS, RS)] = 0        # Set to zero the nodes belonging to the group igroup
    
    ngroup = igroup
    mmax = np.max(Igroup)
    if ngroup < NKL:
        Igroup = Igroup[:ngroup]
        MatIgroup = MatIgroup[:ngroup, :mmax]
    else:
        MatIgroup = MatIgroup[:, :mmax]
    return ngroup, Igroup, mmax, MatIgroup, nedge, nindep, npair, MatPrintEdge, MatPrintIndep, MatPrintPair, RplotPair

