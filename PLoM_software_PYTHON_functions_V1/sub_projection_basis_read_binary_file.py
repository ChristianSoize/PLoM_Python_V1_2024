import numpy as np

def sub_projection_basis_read_binary_file(filename,n_d,mDP):

    # Copyright: Christian Soize, Université Gustave Eiffel, 02 June 2024
    #
    # Software     : Probabilistic Learning on Manifolds (PLoM) 
    # Function name: sub_projection_basis_read_binary_file.py
    # Subject      : read nd, mDP, MatRg_mDP(nd, mDP) on a Binary File where nd should be equal to n_d and mDP <= n_d
    
    # --- INPUTS
    #          filename           : file name of the type fileName = 'data.bin'
    #          n_d                : dimension of the realizations in the training dataset
    #          mDP                : maximum number of the projection basis vectors that are read on a binary file
    #
    # --- OUTPUTS
    #
    #          MatRg_mDP(nd,mDP)  : mDP vectors of the projection basis

    # --- Open the file in binary read mode
    try:
        with open(filename, 'rb') as file:
            # --- Read nd and mDPtemp
            nd = np.fromfile(file, dtype=np.int32, count=1)[0]
            mDPtemp = np.fromfile(file, dtype=np.int32, count=1)[0]

            # --- Checking data
            if nd != n_d:
                raise ValueError(f'STOP2 in sub_projection_basis_read_binary_file: the read dimension, nd, must be equal to n_d')
            if mDPtemp != mDP:
                raise ValueError(f'STOP3 in sub_projection_basis_read_binary_file: the read dimension, mDP, is not coherent with the given value of mDP ')

            # Read MatRg_mDP(nd, mDP)
            MatRg_mDP = np.fromfile(file, dtype=np.float64, count=nd * mDPtemp).reshape((nd, mDPtemp))

    except FileNotFoundError:
        raise FileNotFoundError(f'STOP1 in sub_projection_basis_read_binary_file: impossible to open the file {filename}')
    
    return MatRg_mDP
