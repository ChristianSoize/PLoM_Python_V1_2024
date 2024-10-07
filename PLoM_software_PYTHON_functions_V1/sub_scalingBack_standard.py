import numpy as np

def sub_scalingBack_standard(nbx, nbsim, MatRx, Rbeta_scale, Ralpha_scale):

    # Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    # Subject:  
    #         For the standard scaling, back scaling: MatRxx from MatRx (inversion of function sub_scaling2_standard)   
    #         MatRxx = Ralpha_scale.*MatRx + repmat(Rbeta_scale,1,nbsim)  
    #
    #--- INPUTS
    #          nbx                     : dimension of random vector XX (unscaled) and X (scaled)
    #          nbsim                   : number of points in the training set for XX
    #          MatRx(nbx,nbsim)        : nbsim realizations of X
    #          Rbeta_scale(nbx);                     
    #          Ralpha_scale(nbx);   
    #--- OUTPUTS
    #          MatRxx(nbx,nbsim)        : nbsim realizations of XX

    # Performing the back scaling operation
    MatRxx = Ralpha_scale[:, np.newaxis] * MatRx + Rbeta_scale[:, np.newaxis]
    
    return MatRxx
