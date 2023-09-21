import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh


def generate_symmetric_matrix(N, seed=1234):
    # Generate random Gaussian numbers
    np.random.seed(seed)
    random_numbers = np.random.normal(0, np.sqrt(1 / N), size=(N, N))

    # Make the matrix symmetric
    symmetric_matrix = (random_numbers + random_numbers.T) / 2

    return symmetric_matrix


def try_to_load(file_name):
    import os

    if os.path.exists(file_name):
        eigvec = np.load(file_name)
        print("# Loaded state from ", file_name)
        return True, eigvec
    else:
        return False, None


def generate_state_sk(N, seed=1234):
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    filename = "sk_" + str(N) + "_" + str(seed) + ".npy"
    found, eigvec = try_to_load(filename)
    if found:
        return hi.all_states(), eigvec

    Gamma = -1
    H = sum([Gamma * sigmax(hi, i) for i in range(N)])
    V = generate_symmetric_matrix(N, seed=seed)

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += sigmaz(hi, i) * sigmaz(hi, j) * V[i, j]

    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    np.save(filename, eig_vecs[:, 0])

    return hi.all_states(), eig_vecs[:, 0]


def generate_hamiltonian_sk(N, seed=1234):
    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    Gamma = -1
    H = sum([Gamma * sigmax(hi, i) for i in range(N)])
    V = generate_symmetric_matrix(N, seed=seed)

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += sigmaz(hi, i) * sigmaz(hi, j) * V[i, j]

    return hi, H
