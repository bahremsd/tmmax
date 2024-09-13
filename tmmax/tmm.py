import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import vmap 
from typing import Union, List, Tuple, Text, Dict, Callable

from angle import _compute_layer_angles_single_wl_angle_point
from cascaded_matmul import _cascaded_matrix_multiplication
from data import interpolate_nk
from reflect_transmit import _compute_rt_one_wl, _create_phases_ts_rs, _calculate_transmittace_from_coeff


