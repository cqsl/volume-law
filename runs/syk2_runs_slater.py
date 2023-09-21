import sys

sys.path.append("../")

from syk import generate_state_syk2
from learning import train
import numpy as np
import json
from slater import SlaterDeterminant, LogSlaterDeterminant
from simple_model import SimpleModel
import netket.experimental as nkx


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


N_values = [14, 16, 18, 20]
seeds = range(10)

n_Ns = len(N_values)
n_seeds = len(seeds)

filename = "syk2_slater.out"

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state_syk2(N, seed=seed)

        key = str((N, seed))
        best_losses = load_dict(filename)
        # model=SimpleModel(hidden_units=2*N)
        hi = nkx.hilbert.SpinOrbitalFermions(n_orbitals=N, n_fermions=N // 2)
        model = LogSlaterDeterminant(hi, param_dtype=float)

        if not (key in best_losses):
            best_loss_value = train(
                x_configs,
                target_data,
                model=model,
                learning_rate=0.005,
                num_epochs_overlap=4000,
                verbose=True,
            )
            best_losses = load_dict(filename)
            best_losses[key] = best_loss_value
            print("# ", key, best_loss_value)

            with open(filename, "w") as json_file:
                json.dump(best_losses, json_file)
