import sys

sys.path.append("../")

from dis_fermions import generate_state_df as generate_state
import numpy as np
import jax.numpy as jnp
import json
from slater import LogSlaterBfDeterminant
import netket.experimental as nkx
import flax
import netket as nk


def generate_symmetric_matrix(N, seed=1234):
    # Generate random Gaussian numbers
    np.random.seed(seed)
    random_numbers = np.random.normal(0, np.sqrt(1 / N), size=(N, N))

    # Make the matrix symmetric
    symmetric_matrix = (random_numbers + random_numbers.T) / 2

    # Set diagonal elements to zero
    np.fill_diagonal(symmetric_matrix, 0)

    return symmetric_matrix


def compute_energy(N, state, seed=1234):
    hi = nkx.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N // 2)

    V1 = generate_symmetric_matrix(N, seed)
    V2 = generate_symmetric_matrix(N, seed + 1048 * seed)

    def c(site):
        return nkx.operator.fermion.destroy(hi, site)

    def cdag(site):
        return nkx.operator.fermion.create(hi, site)

    def nc(site):
        return nkx.operator.fermion.number(hi, site)

    H = 0.0

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += V1[i, j] * (cdag(i) * c(j) + cdag(j) * c(i))

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += nc(i) * nc(j) * V2[i, j]

    state /= jnp.linalg.norm(state)

    return state.conj().T @ (H.to_sparse() @ state)


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


folder = "energy_params_disf_bf/"

N_values = [10, 12, 14, 16, 18, 20]
seeds = range(30)
alphas = [1 / 8, 1 / 4, 1 / 2, 1, 2]

n_Ns = len(N_values)
n_seeds = len(seeds)
n_alphas = len(alphas)

filename = "disf_bf_energy_errors.out"

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state(N, seed=seed)

        for k in range(n_alphas):
            alpha = alphas[k]
            key = str((N, seed, alpha))
            best_energy_errors = load_dict(filename)
            hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals=N, n_fermions=N // 2)
            model = LogSlaterBfDeterminant(
                hi, hidden_units=int(N * alpha), param_dtype=float
            )
            vstate = nk.vqs.MCState(
                sampler=nk.sampler.MetropolisLocal(hi), model=model, n_samples=1000
            )

            if not (key in best_energy_errors):
                with open(
                    folder + f"disf_bf_bestvar_N{N}_seed{seed}_alpha{alpha}.mpack", "rb"
                ) as file:
                    vstate.variables = flax.serialization.from_bytes(
                        vstate.variables, file.read()
                    )

                varstate = vstate.to_array()

                exact_energy = compute_energy(N, target_data, seed=seed)
                best_energy = compute_energy(N, varstate, seed=seed)

                best_energy_errors = load_dict(filename)
                best_energy_errors[key] = np.absolute(
                    (exact_energy - best_energy) / exact_energy
                )

                with open(filename, "w") as json_file:
                    json.dump(best_energy_errors, json_file)
