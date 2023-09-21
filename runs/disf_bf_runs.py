import sys

sys.path.append("../")

from dis_fermions import generate_state_df as generate_state
from learning import train
import json
from slater import LogSlaterBfDeterminant
import netket.experimental as nkx
import flax


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


N_values = [10]
seeds = range(10)
alphas = [1]

n_Ns = len(N_values)
n_seeds = len(seeds)
n_alphas = len(alphas)

filename = "disf_bf_bestvar_.out"

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state(N, seed=seed)

        for k in range(n_alphas):
            alpha = alphas[k]
            key = str((N, seed, alpha))
            best_losses = load_dict(filename)
            hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals=N, n_fermions=N // 2)
            model = LogSlaterBfDeterminant(
                hi, hidden_units=N * alpha, param_dtype=float
            )

            if not (key in best_losses):
                best_loss_value, best_variables = train(
                    x_configs,
                    target_data,
                    model=model,
                    learning_rate=0.002,
                    num_epochs_overlap=1000,
                    verbose=True,
                    return_best_variables=True,
                )
                best_losses = load_dict(filename)
                best_losses[key] = best_loss_value
                print("# ", key, best_loss_value)

                with open(filename, "w") as json_file:
                    json.dump(best_losses, json_file)

                with open(
                    f"energy_params_disf_bf_final/disf_bf_bestvar_N{N}_seed{seed}_alpha{alpha}_.mpack",
                    "wb",
                ) as file:
                    file.write(flax.serialization.to_bytes(best_variables))
