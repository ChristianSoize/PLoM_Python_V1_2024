import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import numpy as np

def sub_polynomialChaosQWU_print_plot(nq_obs, nar_PCE, Indq_obs, MatRqq_ar0, MatRqq_PolChaos_ar0, 
                                      ind_display_screen, ind_print, ind_plot, ind_type):
    #----------------------------------------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 24 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_polynomialChaosQWU_print_plot
    #  Subject      : print and plot 

    #--- INPUT
    #        nq_obs                               : dimension of Qobs^0
    #        nar_PCE                              : number of realizations used for the PCE construction 
    #        Indq_obs(nq_obs)                     : observation number for Q
    #        MatRqq_ar0(nq_obs,nar_PCE)           : nar_PCE learned realizations for Qobs
    #        MatRqq_PolChaos_ar0(nq_obs,nar_PCE)) : polynomial chaos representation
    #        ind_display_screen                   : = 0 no display, = 1 display
    #        ind_print                            : = 0 no print,   = 1 print
    #        ind_plot                             : = 0 no plot,    = 1 plot
    #        ind_type                             : = 1 Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) 
    #                                             : = 2 Polynomial-chaos validation for MatRww_o(nw_obs,n_o)
    #                                             : = 3 Polynomial-chaos realization
    
    if ind_print == 1:
        if ind_type == 1:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n')
                fidlisting.write('Polynomial-chaos representation for MatRww_ar0(nw_obs,nar_PCE) --------------------------------------\n')
                fidlisting.write('\n')
        if ind_type == 2:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n')
                fidlisting.write('Polynomial-chaos validation for MatRww_o(nw_obs,n_o) ------------------------------------------------\n')
                fidlisting.write('\n')
        if ind_type == 3:
            with open('listing.txt', 'a+') as fidlisting:
                fidlisting.write('\n')
                fidlisting.write('Polynomial-chaos realization ------------------------------------------------------------------------\n')
                fidlisting.write('\n')

    #--- second-order moment
    MatRmom2QQ_ar0          = (MatRqq_ar0 @ MatRqq_ar0.T) / (nar_PCE - 1)                    # MatRmom2QQ_ar0(nq_obs,nq_obs)
    MatRmom2QQ_PolChaos_ar0 = (MatRqq_PolChaos_ar0 @ MatRqq_PolChaos_ar0.T) / (nar_PCE - 1)  # MatRmom2QQ_PolChaos_ar0(nq_obs,nq_obs)

    #--- mean value
    MatRmean = np.column_stack((np.mean(MatRqq_ar0, axis=1), np.mean(MatRqq_PolChaos_ar0, axis=1)))
    if ind_display_screen == 1:
        print('mean value: QQ_ar0 QQ_PolChaos_ar0')
        print(MatRmean)

    #--- standard deviation       
    MatRstd = np.column_stack((np.std(MatRqq_ar0, axis=1, ddof=1), np.std(MatRqq_PolChaos_ar0, axis=1,ddof=1)))
    if ind_display_screen == 1:
        print('standard deviation: QQ_ar0 QQ_PolChaos_ar0')
        print(MatRstd)

    #--- skewness  (power 3)      
    # MatRskew = np.column_stack((skew(MatRqq_ar0, axis=1), skew(MatRqq_PolChaos_ar0, axis=1)))
    # if ind_display_screen == 1:
    #     print('skewness: QQ_ar0 QQ_PolChaos_ar0')
    #     print(MatRskew)
    
    #--- kurtosis (power 4)      
    # MatRkurto = np.column_stack((kurtosis(MatRqq_ar0, axis=1), kurtosis(MatRqq_PolChaos_ar0, axis=1)))
    # if ind_display_screen == 1:
    #     print('kurtosis: QQ_ar0 QQ_PolChaos_ar0')
    #     print(MatRkurto)


    #---- print
    if ind_print == 1:
        with open('listing.txt', 'a+') as fidlisting:
            fidlisting.write('\n\n')
            fidlisting.write('Second-order moment matrix of MatRqq_ar0 \n')
            for i in range(nq_obs):
                Rprint = MatRmom2QQ_ar0[i, :]
                fidlisting.write(' '.join(f'{x:12.5e}' for x in Rprint) + '\n')
            fidlisting.write('\n\n')
            fidlisting.write('Second-order moment matrix of MatRqq_PolChaos_ar0 \n')
            for i in range(nq_obs):
                Rprint = MatRmom2QQ_PolChaos_ar0[i, :]
                fidlisting.write(' '.join(f'{x:12.5e}' for x in Rprint) + '\n')
            fidlisting.write('\n\n')
            fidlisting.write('--- Mean value: \n')
            fidlisting.write('                  mean(QQ_ar0)   mean(QQ_PolChaos_ar0) \n')
            fidlisting.write('\n')
            for i in range(nq_obs):
                fidlisting.write(f'             {MatRmean[i, 0]:14.7f} {MatRmean[i, 1]:14.7f}\n')
            fidlisting.write('\n\n')
            fidlisting.write('--- Standard deviation: \n')
            fidlisting.write('                  std(QQ_ar0)    std(QQ_PolChaos_ar0) \n')
            fidlisting.write('\n')
            for i in range(nq_obs):
                fidlisting.write(f'             {MatRstd[i, 0]:14.7f} {MatRstd[i, 1]:14.7f}\n')
            fidlisting.write('\n\n')
            # fidlisting.write('\n')
            # fidlisting.write('\n')
            # fidlisting.write('--- Skewness: \n')
            # fidlisting.write('                 skew(QQ_ar0)   skew(QQ_PolChaos_ar0) \n')
            # fidlisting.write('\n')
            # for i in range(nq_obs):
            #     fidlisting.write(f'             {MatRskew[i, 0]:14.7f} {MatRskew[i, 1]:14.7f}\n')
            # fidlisting.write('\n')
            # fidlisting.write('\n')
            # fidlisting.write('--- Kurtosis: \n')
            # fidlisting.write('                kurto(QQ_ar0)  kurto(QQ_PolChaos_ar0) \n')
            # fidlisting.write('\n')
            # for i in range(nq_obs):
            #     fidlisting.write(f'             {MatRkurto[i, 0]:14.7f} {MatRkurto[i, 1]:14.7f}\n')

    if ind_plot == 1:
        #--- plot histogram MatRqq_ar0(nq_obs,nar_PCE)
        for k in range(nq_obs):
            kobs = Indq_obs[k]   # Indq_obs(nq_obs)
            plt.figure()
            plt.hist(MatRqq_ar0[k, :], bins='auto')
            plt.title('Learning', fontsize=16)
            plt.xlabel('$q$', fontsize=16)
            plt.ylabel(f'${{\\rm{{histogram}}}}_{{Q_{{{kobs}}}}}(q)$', fontsize=16)
            plt.savefig(f'figure_histogram_Q{kobs}_learning.png')
            plt.close()

        #--- plot histogram MatRqq_PolChaos_ar0(nq_obs,nar_PCE)
        for k in range(nq_obs):
            kobs = Indq_obs[k]  # Indq_obs(nq_obs)
            plt.figure()
            plt.hist(MatRqq_PolChaos_ar0[k, :], bins='auto')
            if ind_type == 1:
                plt.title('Polynomial chaos representation', fontsize=16)
                plt.xlabel('$q$', fontsize=16)
                plt.ylabel(f'${{\\rm{{histogram}}}}_{{Q_{{{kobs}}}}}(q)$', fontsize=16)
                plt.savefig(f'figure_histogram_Q{kobs}_PCE_representation.png')
            elif ind_type == 2:
                plt.title('Polynomial chaos validation', fontsize=16)
                plt.xlabel('$q$', fontsize=16)
                plt.ylabel(f'${{\\rm{{histogram}}}}_{{Q_{{{kobs}}}}}(q)$', fontsize=16)
                plt.savefig(f'figure_histogram_Q{kobs}_PCE_validation.png')
            elif ind_type == 3:
                plt.title('Polynomial chaos realization', fontsize=16)
                plt.xlabel('$q$', fontsize=16)
                plt.ylabel(f'${{\\rm{{histogram}}}}_{{Q_{{{kobs}}}}}(q)$', fontsize=16)
                plt.savefig(f'figure_histogram_Q{kobs}_PCE_realization.png')
            plt.close()
