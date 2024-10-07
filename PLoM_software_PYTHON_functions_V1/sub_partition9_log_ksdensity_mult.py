import numpy as np

def sub_partition9_log_ksdensity_mult(n,nsim,dim,nexp,MatRxData,MatRxExp):

    # Copyright C. Soize 20 September 2024
    
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
    #
    #---OUTPUT
    #          Rlogpdf(nexp) = values of the log pdf at points  Rx^1,...,Rx^nexp   
    #    

    if dim == 1:
        Rhatsigma = np.std(MatRxData,ddof=1)          # Rhatsigma is a scalar, MatRxData(nsim) is a 1D array
    else:
        Rhatsigma = np.std(MatRxData,axis=1,ddof=1)   # Rhatsigma(dim) is a 1D array and  MatRxData(dim,nsim) is a 2D array  

    s        = (4/((n+2)*nsim))**(1/(n+4))
    s2       = s*s
    shss     = 1/np.sqrt(s2 + (nsim-1)/nsim)          # coef sh/s
    sh       = s*shss                                 # coef sh
    sh2      = sh*sh
    log_cons = - np.sum(np.log(np.sqrt(2*np.pi) * sh * Rhatsigma))
    
    if dim == 1:                                      # 1D array
        MatRxDataNorm = MatRxData / Rhatsigma         # MatRxDataNorm(nsim) and MatRxData(nsim) are a 1D array, and Rhatsigma is a scalar 
        MatRxExpNorm  = MatRxExp / Rhatsigma          # MatRxExpNorm(nexp) and MatRxExp(nexp)  are a 1D array, and Rhatsigma is a scalar 
    else:                                             # 2D array        
        MatRxDataNorm = np.zeros((dim,nsim))          # MatRxDataNorm(dim,nsim),MatRxData(dim,nsim)
        MatRxExpNorm  = np.zeros((dim,nexp))          # MatRxExpNorm(dim,nexp),MatRxExp(dim,nexp)
        for j in range(dim):
            MatRxExpNorm[j, :]  = MatRxExp[j, :] / Rhatsigma[j]    # MatRxExpNorm(dim,nexp),MatRxExp(dim,nexp)
            MatRxDataNorm[j, :] = MatRxData[j, :] / Rhatsigma[j]   # MatRxDataNorm(dim,nsim),MatRxData(dim,nsim)
    
               #--- Scalar sequence using matalb instructions
               # for alpha = 1:nexp    
               #     for ell=1:nsim                                   
               #         norm2 = norm(shss*MatRxDataNorm(:,ell)-MatRxExpNorm(:,alpha) , 2);  % MatRxDataNorm(dim,nsim), MatRxExpNorm(dim,nexp)
               #         MatRS(ell,alpha) = exp(-0.5*norm2^2/sh2);                                       
               #     end 
               #     Rpdf(alpha)= cons*sum(MatRS(:,alpha))/nsim;                        % Rpdf(nexp,1)
               # end 
    if dim == 1:       # dim = 1, MatRbid is a 2D array with shape (nsim,nexp)
            MatRbid = np.zeros((nsim,nexp))                                             # MatRbid(nsim,nexp) since dim = 1
            for alpha in range(nexp):    
                MatRbid[:, alpha] = shss*MatRxDataNorm - MatRxExpNorm[alpha]            # MatRbid(nsim,nexp),MatRxDataNorm(nsim), MatRxExpNorm(nexp)                
            MatRexpo = MatRbid                                                          # No need to transpose as MatRbid is already (nsim,nexp)
            MatRS = np.exp(-MatRexpo**2/(2*sh2))                                        # MatRS(nsim,nexp),MatRexpo(nsim,nexp)
            Rlogpdf = log_cons + np.log(np.mean(MatRS,axis=0)).reshape(nexp,1)          # MatRS(nsim,nexp)
    else:              # dim > 1, MatRbid is a 3D array with shape (dim,nsim,nexp)
            MatRbid = np.zeros((dim,nsim,nexp))                                         # MatRbid(dim,nsim,nexp)           
            for alpha in range(nexp):                                                   # MatRxDataNorm(dim,nsim), MatRxExpNorm(dim,nexp)
                MatRbid[:, :, alpha] = shss * MatRxDataNorm - np.tile(MatRxExpNorm[:, alpha].reshape(-1,1),(1,nsim))  
            MatRexpo = np.transpose(MatRbid,(1,2,0))                                    # MatRexpo(nsim,nexp,dim),MatRbid(dim,nsim,nexp)
            MatRS    = np.exp(-(np.sum(MatRexpo**2,axis=2))/(2*sh2))                    # MatRS(nsim,nexp),MatRbid(dim,nsim,nexp)
            Rlogpdf  = log_cons + np.log(np.mean(MatRS,axis=0)).reshape(nexp,1)         # Rlogpdf(nexp,1), MatRS(nsim,nexp)

    return Rlogpdf
               