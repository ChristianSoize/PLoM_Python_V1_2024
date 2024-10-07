import struct

def sub_projection_basis_write_binary_file(filename, n_d, mDP, MatRg_mDP):
    
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_projection_basis_write_binary_file
    #  Subject      : write n_d,mDP,MatRg_mDP(n_d,mDP) on a Binary File  filename of type 'data.bin'  

    # Open the file in binary write mode
    try:
        fileID = open(filename, 'wb')
    except IOError:
        raise Exception(f'STOP1 in sub_projection_basis_write_binary_file: impossible to open the file {filename}')
   
    # Write n_d and mDP
    fileID.write(struct.pack('i', n_d))
    fileID.write(struct.pack('i', mDP))
   
    # Write MatRg_mDP(n_d,mDP)
    for i in range(n_d):
        for j in range(mDP):
            fileID.write(struct.pack('d', MatRg_mDP[i][j]))
   
    # Close the file
    fileID.close()
