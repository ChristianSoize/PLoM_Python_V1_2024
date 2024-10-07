def sub_projection_basis_read_text_file(filename, n_d, mDP):

    #  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_projection_basis_read_text_file
    #  Subject      : read nd,mDP,MatRg_mDP(n_d,mDP) on a Text File  where n_d should be equal to n_d and mDP <= n_d,  
   
    #--- INPUTS
    #          filename           : file name of the type fileName = 'data.txt'
    #          n_d                : dimension of the realizations in the training dataset
    #          mDP                : maximum number of the projection basis vectors that are read on a binary file
    #
    #--- OUTPUTS
    #
    #          MatRg_mDP(nd,mDP)  : mDP vectors of the projection basis

    #--- Open the file in text read mode
    try:
        fileID = open(filename, 'r')  # file name must be of the type fileName = 'data.txt'
    except IOError:
        raise Exception(f'STOP1 in sub_ISDE_projection_basis_read_text_file: impossible to open the file {filename}')

    #--- Read nd and mDPtemp
    nd      = int(fileID.readline().strip())
    mDPtemp = int(fileID.readline().strip())

    #--- Checking data
    if nd != n_d:
        fileID.close()
        raise ValueError('STOP2 in sub_ISDE_projection_basis_read_text_file: the read dimension, nd, must be equal to n_d')
    if mDPtemp != mDP:
        fileID.close()
        raise ValueError('STOP3 in sub_ISDE_projection_basis_read_text_file: the read dimension, mDP, is not coherent with the given value of mDP ')

    #--- Initialize MatRg_mDP
    MatRg_mDP = []

    #--- Read MatRg_mDP
    for i in range(n_d):
        line = fileID.readline().strip()
        values = [float(x) for x in line.split()]
        if len(values) != mDP:
            fileID.close()
            raise ValueError(f'STOP4 in sub_ISDE_projection_basis_read_text_file: number of values in line {i+1} does not match mDP')
        MatRg_mDP.append(values)

    # Close the file
    fileID.close()

    return MatRg_mDP
