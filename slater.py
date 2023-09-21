from functools import partial
from typing import Any, Sequence, Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.numpy.linalg import det

import netket.experimental as nkx
from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from initializers import row_orthogonal_kernel_init
from jax.nn.initializers import zeros, normal
from transformers import TransformerBf

PRNGKey = Any
Shape = Tuple[int, ...]
DType = Any
Array = Any


def _log_det(phi):
    log_det = jnp.linalg.slogdet(phi)
    return jnp.asarray(
        jnp.log(log_det[0].astype(complex)) + log_det[1].astype(complex)
    ).astype(complex)


class LogSlaterDeterminant(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    kernel_init: NNInitFunc = row_orthogonal_kernel_init(scale=1.0)
    param_dtype: DType = complex

    @nn.compact
    def __call__(self, n):
        assert self.hilbert.spin is None
        kernel = self.param(
            "Phi",
            self.kernel_init,
            (
                self.hilbert.n_fermions,
                self.hilbert.n_orbitals,
            ),
            self.param_dtype,
        )

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            phi = kernel[:, jnp.where(n, size=self.hilbert.n_fermions)[0]]
            return _log_det(phi)

        return log_sd(n)


class SlaterDeterminant(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    kernel_init: NNInitFunc = row_orthogonal_kernel_init(scale=1.0)
    param_dtype: DType = float

    @nn.compact
    def __call__(self, n):
        assert self.hilbert.spin is None
        kernel = self.param(
            "Phi",
            self.kernel_init,
            (
                self.hilbert.n_fermions,
                self.hilbert.n_orbitals,
            ),
            self.param_dtype,
        )

        @partial(jnp.vectorize, signature="(n)->()")
        def sd(n):
            phi = kernel[:, jnp.where(n, size=self.hilbert.n_fermions)[0]]
            return det(phi)

        return sd(n)


class LogSlaterBfDeterminant(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    hidden_units: int
    kernel_init: NNInitFunc = row_orthogonal_kernel_init(scale=1.0)
    param_dtype: DType = float

    @nn.compact
    def __call__(self, n):
        assert self.hilbert.spin is None

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            kernel = self.param(
                "Phi",
                self.kernel_init,
                (
                    self.hilbert.n_fermions,
                    self.hilbert.n_orbitals,
                ),
                self.param_dtype,
            )
            bf = nn.Dense(self.hidden_units, param_dtype=self.param_dtype)(n)
            bf = jax.nn.tanh(bf)
            bf = nn.Dense(
                self.hilbert.n_orbitals * self.hilbert.n_fermions,
                param_dtype=self.param_dtype,
            )(bf).reshape(kernel.shape)
            kernel += bf
            phi = kernel[:, jnp.where(n, size=self.hilbert.n_fermions)[0]]
            return _log_det(phi)

        return log_sd(n)
