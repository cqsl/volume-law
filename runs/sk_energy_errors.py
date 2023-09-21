import sys

sys.path.append("../")

from sk import generate_state_sk as generate_state
import numpy as np
import jax.numpy as jnp
import json
import flax
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from simple_model import SimpleModel


def generate_symmetric_matrix(N, seed=1234):
    # Generate random Gaussian numbers
    np.random.seed(seed)
    random_numbers = np.random.normal(0, np.sqrt(1 / N), size=(N, N))

    # Make the matrix symmetric
    symmetric_matrix = (random_numbers + random_numbers.T) / 2

    return symmetric_matrix


def compute_energy(N, state, seed=1234):

    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    Gamma = -1
    H = sum([Gamma * sigmax(hi, i) for i in range(N)])
    V = generate_symmetric_matrix(N, seed=seed)

    for i in range(N):
        for j in range(i + 1, N, 1):
            H += sigmaz(hi, i) * sigmaz(hi, j) * V[i, j]

    state /= jnp.linalg.norm(state)

    return state.conj().T @ (H.to_sparse() @ state)


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


folder = "energy_params_sk/"

N_values = [10, 12, 14, 16, 18, 20]
seeds = range(20)
alphas = [1]

n_Ns = len(N_values)
n_seeds = len(seeds)
n_alphas = len(alphas)

filename = "sk_energy_errors.out"

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state(N, seed=seed)

        for k in range(n_alphas):
            alpha = alphas[k]
            key = str((N, seed, alpha))
            best_energy_errors = load_dict(filename)
            hi = nk.hilbert.Spin(s=1 / 2, N=N)
            model = SimpleModel(hidden_units=alpha * N)
            vstate = nk.vqs.MCState(
                sampler=nk.sampler.MetropolisLocal(hi), model=model, n_samples=1000
            )

            if not (key in best_energy_errors):

                with open(
                    folder + f"sk_bestvar_N{N}_seed{seed}_alpha{alpha}.mpack", "rb"
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
