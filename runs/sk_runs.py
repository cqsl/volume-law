import sys

sys.path.append("../")

from sk import generate_state_sk
from learning import train
import json
from simple_model import SimpleModel


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


N_values = [
    10,
    12,
]
seeds = range(10)
alphas = [1, 2]

n_Ns = len(N_values)
n_seeds = len(seeds)
n_alphas = len(alphas)

filename = "sk.out"


for i in range(n_Ns):
    N = N_values[i]
    for j in range(len(seeds)):
        seed = seeds[j]
        x_configs, target_data = generate_state_sk(N, seed=seed)

        for k in range(n_alphas):
            alpha = alphas[k]
            key = str((N, seed, alpha))
            best_losses = load_dict(filename)
            model = SimpleModel(hidden_units=alpha * N)

            if not (key in best_losses):
                best_loss_value = train(
                    x_configs,
                    target_data,
                    model=model,
                    learning_rate=0.001,
                    num_epochs_overlap=40000,
                    verbose=True,
                )
                best_losses = load_dict(filename)
                best_losses[key] = best_loss_value
                print("# ", key, best_loss_value)

                with open(filename, "w") as json_file:
                    json.dump(best_losses, json_file)
