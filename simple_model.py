import jax
from flax import linen as nn


# Define a simple feedforward neural network using Flax
class SimpleModel(nn.Module):
    hidden_units: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_units)
        self.dense2 = nn.Dense(2)  # Output should be a complex scalar

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.tanh(x)
        x = self.dense2(x)
        return x[:, 0] + 1.0j * x[:, 1]
