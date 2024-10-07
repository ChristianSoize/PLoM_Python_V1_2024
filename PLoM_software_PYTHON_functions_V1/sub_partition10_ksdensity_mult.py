import numpy as np

def sub_partition10_ksdensity_mult(n,nsim,dim,nexp,MatRxData,MatRxExp):

    # Copyright C. Soize 21 September 2024
    #
    # This function calculates the value, in nexp experimental vectors (or scalars if dim=1) x^1,...,x^nexp stored in 
    # the array MatRxExp(dim,nexp), of the joint probability density function of a random vector X of dimension dim, 
    # CONSIDERED AS THE MARGINAL DISTRIBUTION OF A RANDOM VECTOR OF DIMENSION n >= dim,
    # whose nsim realizations: X(theta_1),..., X(theta_nsim) are used to estimate the marginal probability density function 
    # using the Gaussian kernel method, and are stored in the array MatRxData(dim,nsim).   
    #
    #  Rhatsigma(j)       = empirical standard deviation of X_j estimated with MatRxData(j,:) = [X(theta_1),..., X(theta_nsim)]
    #  s                  = (4/(nsim*(2+n)))^(1/(n+4)), Silverman bandwidth for random vector of dimension n (and not ndim)
    #  Rsigma(j)          = Rhatsigma(j) * s   
    #  shss               = 1/sqrt(s2+(nsim-1)/nsim), coef correcteur sh et sh/s introduced by C. Soize
    #  sh                 = s*shss;    
    #  MatRxExpNorm(j,:)  = MatRxExp(j,:)/Rhatsigma(j);                       
    #  MatRxDataNorm(j,:) = MatRxData(j,:)/Rhatsigma(j);    
    #
    #---INPUT 
    #        MatRxData(dim,nsim) : nsim independent realizations used to estimate the pdf of random vector X(dim) 
    #        MatRxExp(dim,nexp)  : nexp experimental data of random vector X(dim)
    #
    #---OUTPUT
    #         Rpdf(nexp) : values of the pdf at points  Rx^1,...,Rx^nexp      
    print('bonjour0')
    print(dim)
    if dim == 1:
        # Case when dim (mj) is 1: both MatRxData and MatRxExp are 1D arrays
        Rhatsigma = np.std(MatRxData,ddof=1)                   # scalar for 1D
        s         = (4/((n + 2) * nsim))**(1/(n + 4))
        s2        = s*s
        shss      = 1/np.sqrt(s2 + (nsim - 1) / nsim)               # coef sh/s
        sh        = s*shss                                          # coef sh
        sh2       = sh*sh
        cons = 1/(np.sqrt(2*np.pi)*sh*Rhatsigma)
        
        MatRxDataNorm = MatRxData/Rhatsigma   # Normalize the 1D array
        MatRxExpNorm  = MatRxExp/Rhatsigma    # Normalize the 1D array
        print('bonjour dim 1')
        print(dim)  
        print(MatRxExp.shape) 
        print(MatRxData.shape)   
        print(Rhatsigma.shape)    
        # Rpdf for 1D case
        Rpdf     = np.zeros(nexp)        
        MatRexpo = np.zeros((nsim,nexp))                                   # MatRexpo(nsim,nexp) since dim = 1
        for alpha in range(nexp):    
            MatRexpo[:,alpha] = shss*MatRxDataNorm - MatRxExpNorm[alpha]   # MatRexpo(nsim,nexp), MatRxDataNorm(nsim), MatRxExpNorm(nexp)        MatRexpo = MatRbid                                                          # MatRexpo(nsim,nexp), MatRbid(nsim,nexp)
        MatRS = np.exp(-MatRexpo**2/(2*sh2))                               # MatRS(nsim,nexp), MatRexpo(nsim,nexp)
        for alpha in range(nexp):
            Rpdf[alpha] = cons*np.sum(MatRS[:,alpha])/nsim                 # Rpdf(nexp)
    else:
        # Case when dim > 1: MatRxData and MatRxExp are 2D arrays
        Rhatsigma = np.std(MatRxData,axis=1,ddof=1)                        # Rhatsigma(dim) is a 1D array and  MatRxData(dim,nsim) is a 2D array  
        s         = (4/((n + 2)*nsim))**(1/(n + 4))
        s2        = s*s
        shss      = 1/np.sqrt(s2 + (nsim - 1)/nsim)                        # coef sh/s
        sh        = s*shss                                                 # coef sh
        sh2       = sh*sh
        cons      = 1/(np.prod(np.sqrt(2*np.pi)*sh*Rhatsigma))
        
        MatRxDataNorm = np.zeros((dim, nsim))                              # MatRxDataNorm(dim,nsim),MatRxData(dim,nsim)                                     
        MatRxExpNorm  = np.zeros((dim, nexp))                              # MatRxExpNorm(dim,nexp),MatRxExp(dim,nexp)  
        print('bonjour dim 2')
        print(dim)
        print(nexp)         
        print(MatRxExpNorm.shape)   
        print(MatRxExp.shape) 
        print(Rhatsigma.shape)     
        for j in range(dim):                                                             
            MatRxExpNorm[j, :]  = MatRxExp[j, :] / Rhatsigma[j]            # MatRxExpNorm(dim,nexp),MatRxExp(dim,nexp)                    
            MatRxDataNorm[j, :] = MatRxData[j, :] / Rhatsigma[j]           # MatRxDataNorm(dim,nsim),MatRxData(dim,nsim)
        Rpdf = np.zeros(nexp)                                              # Rpdf(nexp)
        # Vectorial sequence for dim > 1
        MatRbid = np.zeros((dim,nsim,nexp))                                # MatRbid(dim,nsim,nexp),
        for alpha in range(nexp):
            MatRbid[:,:,alpha] = shss*MatRxDataNorm - np.tile(MatRxExpNorm[:,alpha].reshape(-1,1),(1,nsim))
        MatRexpo = np.transpose(MatRbid,(1,2,0))                           # MatRexpo(nsim,nexp,dim) = MatRbid(dim,nsim,nexp)
        MatRS    = np.exp(-np.sum(MatRexpo**2,axis=2)/(2*sh2))             # MatRS(nsim, nexp)
        for alpha in range(nexp):
            Rpdf[alpha] = cons * np.sum(MatRS[:, alpha]) / nsim            # Rpdf(nexp)
    
    return Rpdf
