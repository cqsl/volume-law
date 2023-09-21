import sys

sys.path.append("../")

from sk import generate_state_sk
from renyin import renyin

import numpy as np
import json


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


N_values = [10, 12, 14, 16, 18, 20]
seeds = range(10)

n_Ns = len(N_values)
n_seeds = len(seeds)

filename = "sk_entropy.out"

for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state_sk(N, seed=seed)

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
