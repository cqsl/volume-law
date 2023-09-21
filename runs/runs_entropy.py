import sys

sys.path.append("../")

from sk import generate_state_sk
from dis_fermions import generate_state_df
from renyin import renyin

import numpy as np
import json
import netket as nk
import jax.numpy as jnp


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


N_values = [10, 12, 14, 16, 18, 20]
seeds = range(40)

n_Ns = len(N_values)
n_seeds = len(seeds)

filename = "sk_entropy.out"
# filename='df_entropy.out'

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]

        if filename == "sk_entropy.out":
            x_configs, target_data = generate_state_sk(N, seed=seed)

        if filename == "df_entropy.out":
            x_configs, target_data_ = generate_state_df(N, seed=seed)
            target_data_ = jnp.array(target_data_)

            hi = nk.hilbert.Spin(s=1 / 2, N=N)
            x_configs = x_configs - 1
            idxs = hi.states_to_numbers(x_configs)
            idxs_ = jnp.arange(idxs.size)

            target_data = jnp.zeros(2**N)

            target_data = target_data.at[idxs].set(target_data_.at[idxs_].get())

        key = str((N, seed))
        entropies = load_dict(filename)

        if not (key in entropies):

            entropy_idx = 2
            subsys = np.arange(0, int(N / 2))
            entropy_value = renyin(entropy_idx, target_data, N, subsys)

            entropies = load_dict(filename)
            entropies[key] = entropy_value

            with open(filename, "w") as json_file:
                json.dump(entropies, json_file)
