def sub_projection_basis_write_text_file(filename, n_d, mDP, MatRg_mDP):
    
    #  Copyright: Christian Soize, Universite Gustave Eiffel, 02 June 2024
    #
    #  Software     : Probabilistic Learning on Manifolds (PLoM) 
    #  Function name: sub_projection_basis_write_text_file
    #  Subject      : write n_d,mDP,MatRg_mDP(n_d,mDP) on a Text File  filename of type 'data.txt'  

    # Open the file in text write mode
    try:
        fileID = open(filename, 'w')
    except IOError:
        raise Exception(f'STOP1 in sub_projection_basis_write_text_file: impossible to open the file {filename}')
    
    # Write n_d and mDP
    fileID.write(f'{n_d}\n')
    fileID.write(f'{mDP}\n')
    
    # Write MatRg_mDP
    for i in range(n_d):
        line = ' '.join(f'{x:.15g}' for x in MatRg_mDP[i])
        fileID.write(f'{line}\n')
    
    # Close the file
    fileID.close()
