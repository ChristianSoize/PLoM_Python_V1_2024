import numpy as np

def sub_scaling_standard(nbx, nbsim, MatRxx):

    # Copyright: Christian Soize, Universite Gustave Eiffel, 25 May 2024
    #
    # Subject:
    #         For the standard scaling, compute Rbeta_scale(nbx,1), Ralpha_scale(nbx,1), Ralpham1_scale(nbx,1) for the scaling and the backscaling  
    #         MatRx  = Ralpham1_scale.*(MatRxx - repmat(Rbeta_scale,1,nbsim)) : scaling MatRxx in MatRx (Matlab instruction)  
    #         MatRxx = Ralpha_scale.*MatRx + repmat(Rbeta_scale,1,nbsim)      : back scaling MatRx in MatRxx  (Matlab instruction) 
    #
    #--- INPUTS
    #          nbx                     : dimension of random vector XX (unscaled) and X (scaled)
    #          nbsim                   : number of realizations for XX
    #          MatRxx(nbx,nbsim)       : nbsim realizations of XX
    #--- OUTPUTS
    #          MatRx(nbx,nbsim)        : nbsim realizations of X
    #          Rbeta_scale(nbx);                     
    #          Ralpha_scale(nbx);  
    #          Ralpham1_scale(nbx);
    
    Rmax           = np.max(MatRxx, axis=1)  # MatRxx(nbx,nbsim)
    Rmin           = np.min(MatRxx, axis=1)
    Rbeta_scale    = np.zeros(nbx)           # Rbeta_scale(nbx)
    Ralpha_scale   = np.zeros(nbx)           # Ralpha_scale(nbx)
    Ralpham1_scale = np.zeros(nbx)           # Ralpham1_scale(nbx)

    for k in range(nbx):
        if Rmax[k] - Rmin[k] != 0:
            Rbeta_scale[k]    = Rmin[k]
            Ralpha_scale[k]   = Rmax[k] - Rmin[k]
            Ralpham1_scale[k] = 1 / Ralpha_scale[k]
        else:
            Ralpha_scale[k]   = 1
            Ralpham1_scale[k] = 1
            Rbeta_scale[k]    = Rmin[k]

    MatRx = Ralpham1_scale[:,np.newaxis] * (MatRxx - Rbeta_scale[:,np.newaxis]) # MatRx(nbx,nbsim)
    return MatRx, Rbeta_scale, Ralpha_scale, Ralpham1_scale
