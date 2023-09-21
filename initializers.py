import jax
import jax.numpy as jnp
from typing import Any

DType = Any


def random_rotation_matrix(dim, dtype=complex, key=jax.random.PRNGKey(0)):
    """
    Generate a random rotation matrix using Householder reflections.

    Args:
    - dim: Dimension of the matrix.
    - dtype: Data type of the matrix.
    - key: JAX PRNG key.

    Returns:
    - A random rotation matrix of shape (dim, dim) and the specified data type.
    """
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (dim, dim), dtype=dtype))
    return Q


def row_orthogonal_kernel_init(scale=0, dtype=complex):
    """
    Provides a initializer for a random matrix with orthogonal rows of shape (n, m), where m > n.
    The matrix is constructed by starting with the identity matrix padded with zeros and then applying random rotations
    along the last axis.

    Args:
    - scale: scaling factor.
    - dtype: Data type of the matrix.

    Returns:
    - A kernel initializer.
    """

    def _init(key, shape, dtype=dtype):
        assert len(shape) == 2
        n, m = shape
        if m <= n:
            raise ValueError(
                "Number of columns (m) must be greater than number of rows (n)."
            )
        identity_padded = jnp.hstack(
            (jnp.eye(n, dtype=dtype), jnp.zeros((n, m - n), dtype=dtype))
        )
        rotation_matrix = random_rotation_matrix(m, dtype, key)
        orthogonal_matrix = jnp.dot(identity_padded, rotation_matrix.T)

        return scale * orthogonal_matrix

    return _init
