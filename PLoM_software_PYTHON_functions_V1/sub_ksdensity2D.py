import numpy as np

def sub_ksdensity2D(nsim, MatRdata, npoint, MatRpts, bw1, bw2):
    #----------------------------------------------------------------------------------------------------------
    #
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 08 September 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM)
    #  Function name: sub_ksdensity2D
    #  Subject      : compute the joint pdf of 2 real random variables X1 and X2
    #
    #---INPUTS
    #       nsim                     : number of realizations of X1 and X2 
    #       MatRdata (nsim, 2)       : nsim values of the realizations of X1 and X2
    #       npoint                   : number of points for direction 1 and 2 in the 2D grid 
    #                                  (total number in the 2D grid is npoint*npoint)
    #       MatRpts (npoint*npoint, 2): coordinates of the points in the 2D grid
    #       bw1, bw2                 : bandwidth for components 1 and 2
    #
    #---OUTPUTS
    #   Rpdf (npoint*npoint)      : value of the pdf at each point of the 2D grid

    # Coefficients for the Gaussian kernel
    coef0 = 1/(nsim*bw1*bw2*2*np.pi)
    coef1 = -1/(2*bw1*bw1)
    coef2 = -1/(2*bw2*bw2)
    
    # Initialize Rpdf with zeros
    Rpdf = np.zeros(npoint*npoint)

    # Loop over all points in the grid
    for i in range(npoint*npoint):
        x1i = MatRpts[i,0]  # x1 coordinate of the current grid point
        x2i = MatRpts[i,1]  # x2 coordinate of the current grid point
        
        # Squared differences between the point and data samples
        R1 = (x1i - MatRdata[:,0])**2
        R2 = (x2i - MatRdata[:,1])**2
        
        # Sum the contributions of the Gaussian kernel at this point
        sumi = np.sum(np.exp(coef1*R1 + coef2*R2))
        
        # Store the computed pdf value
        Rpdf[i] = coef0*sumi

    return Rpdf
