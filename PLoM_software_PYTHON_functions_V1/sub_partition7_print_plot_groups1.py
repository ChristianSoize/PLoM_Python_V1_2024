import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np

def sub_partition7_print_plot_groups1(INDEPopt, ngroup, Igroup, MatIgroup, npair, RplotPair, ind_print, ind_plot, numfig):
    #
    # Copyright C. Soize 24 May 2024 
    #
    #--- INPUTS
    #           INDEPopt               : optimal value of INDEPref
    #           ngroup                 : number of groups that are constructed
    #           Igroup(ngroup)         : such that Igroup(j): number mj of the components of  Y^j = (H_r1,... ,H_rmj)             
    #           MatIgroup(ngroup,mmax) : such that MatIgroup1(j,r) = rj : indice rj de H dans le groupe j tel que Y^j_r = H_rj  
    #                                    with mmax = max_j Igroup(j)
    #           npair                  : dimension of RplotPair
    #           RplotPair(npair)       : such that RplotPair(pair)  = INDEPr1r2 with pair=(r1,r2)
    #           ind_print              : = 0 no print, = 1 print
    #           ind_plot               : = 0 no plot,  = 1 plot
    #           numfig                 : number of generated figures before executing this function
    #--- OUTPUT  
    #           numfig                 : number of generated figures after executing this function

    #--- print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('      \n ')
            fidlisting.write('      \n ')
            fidlisting.write('--------------- OPTIMAL PARTITION IN GROUPS OF INDEPENDENT RANDOM VECTORS ----------')
            fidlisting.write('      \n ')    
            fidlisting.write('      \n ')
            fidlisting.write(f'Optimal value INDEPopt of INDEPref used for constructing the optimal partition = {INDEPopt:8.5f} \n ')
            fidlisting.write('      \n ')
            fidlisting.write(f'ngroup  = {ngroup:7d} \n ')
            for j in range(ngroup):
                fidlisting.write('      \n ')
                PPrint = [j + 1] + MatIgroup[j, :Igroup[j]].tolist()
                fidlisting.write(' '.join(f'{x:4d}' for x in PPrint) + '\n')
                
    #--- plot
    if ind_plot == 1:
        if npair >= 2:
            plt.figure()
            plt.plot(range(1, npair + 1), RplotPair, 'ob')
            plt.title(r'Values of $i^\nu(H^\nu_{r_1},H^\nu_{r_2})$ as a function of the pair $p = (r_1,r_2)$', 
                      fontweight='normal', fontsize=16)
            plt.xlabel(r'pair $p = (r_1,r_2)$', fontsize=16)
            plt.ylabel(r'$i^\nu(H^\nu_{r_1},H^\nu_{r_2})$', fontsize=16)
            numfig += 1
            plt.savefig(f'figure_PARTITION_{numfig}_INDGROUP.png')
            plt.close()
    return numfig
