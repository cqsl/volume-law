import sys

sys.path.append("../")

from sk import generate_state_sk
from learning import train
import json
import multiprocessing
from functools import partial


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def process_seed(best_losses, N, alphas, seed):
    x_configs, target_data = generate_state_sk(N, seed=seed)
    new_losses = {}
    n_alphas = len(alphas)
    for k in range(n_alphas):
        alpha = alphas[k]
        key = str((N, seed, alpha))
        if not (key in best_losses):
            best_loss_value = train(x_configs, target_data, alpha=alpha, verbose=False)
            new_losses[key] = best_loss_value
        else:
            best_loss_value = best_losses[key]

        print("# ", key, best_loss_value)

    return new_losses


def parallel_process_seeds(best_losses, N, alphas):
    process_seed_partial = partial(process_seed, best_losses, N, alphas)
    with multiprocessing.Pool() as pool:
        for b_losses in pool.map(process_seed_partial, seeds):
            best_losses.update(b_losses)

    return best_losses


if __name__ == "__main__":

    N_values = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    seeds = range(1, 10, 1)
    alphas = [1, 2]

    n_Ns = len(N_values)

    filename = "sk.out"
    best_losses = load_dict(filename)

    for i in range(n_Ns):
        N = N_values[i]
        best_losses.update(parallel_process_seeds(best_losses, N, alphas))
        with open(filename, "w") as json_file:
            json.dump(best_losses, json_file)
