import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz, sigmay
from scipy.sparse.linalg import eigsh

# https://arxiv.org/pdf/2110.00007.pdf


def generate_symmetric_matrix(N, seed=1234):
    # Generate random Gaussian numbers
    np.random.seed(seed)
    random_numbers = np.random.normal(0, 1, size=(N, N))

    # Make the matrix symmetric
    symmetric_matrix = (random_numbers + random_numbers.T) / 2

    # Set diagonal elements to zero
    np.fill_diagonal(symmetric_matrix, 0)

    return symmetric_matrix


def generate_state_sy(N, seed=1234):

    hi = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=0)

    H = 0.0j
    V = generate_symmetric_matrix(N) / np.sqrt(N)

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += sigmaz(hi, i) * sigmaz(hi, j) * V[i, j]
            H += sigmay(hi, i) * sigmay(hi, j) * V[i, j]
            H += sigmax(hi, i) * sigmax(hi, j) * V[i, j]

    sp_h = H.to_sparse().real

    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    print(eig_vals)

    return hi.all_states(), eig_vecs[:, 0]
