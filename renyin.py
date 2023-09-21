import qutip
import jax.numpy as jnp
import numpy as np


def renyin(n, data, n_comps, subsys):
    data /= jnp.linalg.norm(data)
    assert n != 1 and n != np.inf

    state_qutip = qutip.Qobj(np.array(data))
    state_qutip.dims = [[2] * n_comps, [1] * n_comps]
    mask = np.zeros(n_comps, dtype=bool)

    if len(subsys) == mask.size or len(subsys) == 0:
        return 0

    else:
        mask[subsys] = True
        rdm = state_qutip.ptrace(np.arange(n_comps)[mask])
        out = np.log2(np.trace(np.linalg.matrix_power(rdm, n))) / (1 - n)

        return np.absolute(out.real)
