import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   
import numpy as np

def sub_partition8_print_plot_groups2(INDEPopt, ngroup, Igroup, MatIgroup, nedge, MatPrintEdge, nindep, MatPrintIndep, npair, 
                                      MatPrintPair, RplotPair, ind_print, ind_plot, numfig):
    #
    # Copyright C. Soize 24 May 2024
    #
    # --- INPUTS
    # INDEPopt               : optimal value of INDEPref
    # ngroup                 : number of groups that are constructed
    # Igroup(ngroup)         : such that Igroup(j): number mj of the components of  Y^j = (H_r1,... ,H_rmj)
    # MatIgroup(ngroup,mmax) : such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_rj
    #                          with mmax = max_j Igroup(j)
    # nedge                  : number of pairs (r1,r2) for which H_r1 and H_r2 are dependent (number of edges in the graph)
    # MatPrintEdge(nedge,5)  : such that MatPrintEdge(edge,:) = [edge  r1 r2 INDEPr1r2 INDEPopt]
    # nindep                 : number of pairs (r1,r2) for which H_r1 and H_r2 are independent
    # MatPrintIndep(nindep,5): such that MatPrintIndep(indep,:) = [indep r1 r2 INDEPr1r2 INDEPopt]
    # npair                  : dimension of RplotPair
    # RplotPair(npair)       : such that RplotPair(pair)  = INDEPr1r2 with pair=(r1,r2)
    # npair                  : total number of pairs (r1,r2) = npairmax = NKL(NKL-1)/2
    # MatPrintPair(npair,5)  : such that MatPrintPair(pair,:) = [pair  r1 r2 INDEPr1r2 INDEPopt]
    # RplotPair(npair)       : such that RplotPair(pair) = INDEPr1r2 with pair=(r1,r2)
    # ind_print              : = 0 no print, = 1 print
    # ind_plot               : = 0 no plot,  = 1 plot
    # numfig                 : number of generated figures before executing this function

    # --- OUTPUT
    # numfig                 : number of generated figures after executing this function
   
    # --- Print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write('--------------- OPTIMAL PARTITION IN GROUPS OF INDEPENDENT RANDOM VECTORS ----------\n')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'Optimal value INDEPopt of INDEPref used for constructing the optimal partition = {INDEPopt:8.5f} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'ngroup  = {ngroup:7d} \n ')
            
            for j in range(ngroup):
                fidlisting.write('      \n ')
                PPrint = [j + 1] + MatIgroup[j, :Igroup[j]].tolist()  # MatIgroup contains only integers
                fidlisting.write(' '.join(f'{int(x):4d}' for x in PPrint) + '\n')  # Convert all entries to integers
            
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'nedge  = {nedge:7d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(' edge     r1    r2 INDEPr1r2 INDEPopt  \n ')
            fidlisting.write('      \n ')
            
            for edge in range(nedge):
                Print = MatPrintEdge[edge, :].tolist()  # First 3 columns: integers, last 2 columns: floats
                fidlisting.write(f' {int(Print[0]):4d} {int(Print[1]):6d} {int(Print[2]):6d} {Print[3]:8.5f} {Print[4]:8.5f} \n')  # Format accordingly
            
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'nindep  = {nindep:7d} \n ')
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write(' indep     r1    r2 INDEPr1r2 INDEPopt \n ')
            fidlisting.write('      \n ')
            
            for indep in range(nindep):
                Print = MatPrintIndep[indep, :].tolist()  # Same formatting as MatPrintEdge
                fidlisting.write(f' {int(Print[0]):4d} {int(Print[1]):6d} {int(Print[2]):6d} {Print[3]:8.5f} {Print[4]:8.5f} \n')
            
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
           
    # --- Plot
    if ind_plot == 1:        
        if nedge >= 2:
            h = plt.figure()
            plt.plot(np.arange(1, nedge+1), MatPrintEdge[:, 3], 'ob', np.arange(1, nedge+1), MatPrintEdge[:, 4], '-k')
            plt.title(f'Graph of $i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            plt.xlabel(f'pair $p=(r_1, r_2)$', fontweight='normal', fontsize=16)
            plt.ylabel(f'$i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            numfig = numfig + 1           
            plt.savefig(f'figure_PARTITION_{numfig}_i-H-iopt1.png')
            plt.close(h)   
                     
        if nindep >= 2:
            h = plt.figure()            
            plt.plot(np.arange(1, nindep+1), MatPrintIndep[:, 3], 'ob', np.arange(1, nindep+1), MatPrintIndep[:, 4], '-k')
            plt.title(f'Graph of $i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            plt.xlabel(f'pair $p=(r_1, r_2)$', fontweight='normal', fontsize=16)
            plt.ylabel(f'$i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PARTITION_{numfig}_i-H-iopt2.png')
            plt.close()

        if npair >= 2:
            h = plt.figure()
            plt.plot(np.arange(1, npair+1), MatPrintPair[:, 3], 'ob', np.arange(1, npair+1), MatPrintPair[:, 4], '-k')
            plt.title(f'Graph of $i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            plt.xlabel(f'pair $p=(r_1, r_2)$', fontweight='normal', fontsize=16)
            plt.ylabel(f'$i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PARTITION_{numfig}_i-H-iopt3.png')
            plt.close(h)

            h = plt.figure()
            plt.plot(np.arange(1, npair+1), RplotPair, 'ob')
            plt.title(f'Graph of $i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            plt.xlabel(f'pair $p=(r_1, r_2)$', fontweight='normal', fontsize=16)
            plt.ylabel(f'$i^{{\\nu}}(H^{{\\nu}}_{{r_1}}, H^{{\\nu}}_{{r_2}})$', fontweight='normal', fontsize=16)
            numfig = numfig + 1
            plt.savefig(f'figure_PARTITION_{numfig}_i-H-pair.png')
            plt.close(h)
    return numfig
