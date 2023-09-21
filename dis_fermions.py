import numpy as np
import scipy
from tqdm import tqdm


def generate_symmetric_matrix(N, seed=1234):
    # Generate random Gaussian numbers
    np.random.seed(seed)
    random_numbers = np.random.normal(0, np.sqrt(1 / N), size=(N, N))

    # Make the matrix symmetric
    symmetric_matrix = (random_numbers + random_numbers.T) / 2

    # Set diagonal elements to zero
    np.fill_diagonal(symmetric_matrix, 0)

    return symmetric_matrix


def try_to_load(file_name):
    import os

    if os.path.exists(file_name):
        eigvec = np.load(file_name)
        print("Loaded state from ", file_name)
        return True, eigvec
    else:
        return False, None


def generate_state_df(N, seed=1234):
    import netket as nk
    from netket import experimental as nkx
    from scipy.sparse.linalg import eigsh

    hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N // 2)

    filename = "disf_" + str(N) + "_" + str(seed) + ".npy"
    found, eigvec = try_to_load(filename)
    if found:
        return hi.all_states(), eigvec

    V1 = generate_symmetric_matrix(N, seed)
    V2 = generate_symmetric_matrix(N, seed + 1048 * seed)

    def c(site):
        return nkx.operator.fermion.destroy(hi, site)

    def cdag(site):
        return nkx.operator.fermion.create(hi, site)

    def nc(site):
        return nkx.operator.fermion.number(hi, site)

    print("Creating Hamiltonian...")
    H = 0.0

    for i in tqm(range(N)):
        for j in range(i + 1, N, 1):
            H += V1[i, j] * (cdag(i) * c(j) + cdag(j) * c(i))
            H += nc(i) * nc(j) * V2[i, j]

    print("Diagonalizing Hamiltonian...")
    sp_h = H.to_sparse().real

    def is_hermitian(matrix):
        conjugate_transpose = matrix.conj().transpose()
        return scipy.sparse.linalg.norm(matrix - conjugate_transpose) < 1.0e-10

    assert is_hermitian(sp_h)

    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    # print(eig_vals)
    np.save(filename, eig_vecs[:, 0])

    return hi.all_states(), eig_vecs[:, 0]


def generate_hamiltonian_df(N, seed=1234):
    import netket as nk
    from netket import experimental as nkx
    from scipy.sparse.linalg import eigsh

    hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N // 2)

    V1 = generate_symmetric_matrix(N, seed)
    V2 = generate_symmetric_matrix(N, seed + 1048 * seed)

    def c(site):
        return nkx.operator.fermion.destroy(hi, site)

    def cdag(site):
        return nkx.operator.fermion.create(hi, site)

    def nc(site):
        return nkx.operator.fermion.number(hi, site)

    print("Creating Hamiltonian...")
    H = 0.0

    for i in tqdm(range(N)):
        for j in range(i + 1, N, 1):
            H += V1[i, j] * (cdag(i) * c(j) + cdag(j) * c(i))
            H += nc(i) * nc(j) * V2[i, j]

    return hi, H
