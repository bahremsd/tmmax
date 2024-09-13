import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
import sys # Import sys for EPSILON
from typing import Union # Type hints for function signatures

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon
