import jax.numpy as jnp
from jax import random
from functools import partial
import jax
import optax
import numpy as np
from netket.jax import logsumexp_cplx
from tqdm import tqdm


def train(
    x_configs,
    target_data,
    model,
    learning_rate=0.01,
    num_epochs_overlap=50000,
    verbose=False,
    return_best_variables=False,
    return_best_entropy=False,
):

    if return_best_entropy:
        from renyin import renyin

    def overlap_loss(predictions, targets):
        predictions -= 0.5 * logsumexp_cplx(2.0 * predictions.real)
        loginfidelity = jnp.real(
            jnp.logaddexp(
                0,
                2.0 * logsumexp_cplx(jnp.conj(predictions) + targets).real
                + 1.0j * jnp.pi,
            )
        )
        return loginfidelity

    input_data = jnp.array(x_configs)
    target_data /= jnp.linalg.norm(target_data)
    target_data = jnp.log(target_data.astype(complex))

    rng = jax.random.PRNGKey(100)
    variables = model.init(rng, input_data)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables))
    if verbose:
        print(
            "# Number of parameters and size of hilbert:",
            num_params,
            input_data.shape[0],
        )

    @partial(jax.jit, static_argnames=["type"])
    def train_step(params, opt_state, input_data, target_data, type):
        def loss_fn(params):
            predictions = model.apply(params, input_data)
            if type == "overlap":
                return overlap_loss(predictions, target_data)

        loss_value, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_value

    # Define the optimizer using Optax
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(variables)

    type = "overlap"
    best_loss_value = jnp.inf

    if return_best_entropy:
        variables_history = []

    for epoch in tqdm(range(num_epochs_overlap)):

        variables, optimizer_state, loss_value = train_step(
            variables, optimizer_state, input_data, target_data, type
        )

        if not np.isnan(np.exp(loss_value)):
            best_loss_value = np.minimum(np.exp(loss_value), best_loss_value)
            best_loss_idx = jnp.argmin(loss_value)

        if epoch == 0:
            best_variables = variables

        if loss_value < best_loss_value:
            best_variables = variables

        if return_best_entropy:
            variables_history.append(variables)
            best_variables = variables_history[best_loss_idx]

        if epoch % 500 == 0 and verbose:
            print(f"# Epoch {epoch}, Loss: {np.exp(loss_value):.5e}")

    if return_best_entropy:
        entropy_idx = 2
        subsys = np.arange(0, int(N / 2))
        best_state = model.apply(best_variables, input_data)
        best_entropy = renyin(entropy_idx, best_state, N, subsys)
        return float(best_loss_value), best_entropy

    if return_best_variables:
        return float(best_loss_value), best_variables

    if return_best_variables == False and return_best_entropy == False:
        return float(best_loss_value)
