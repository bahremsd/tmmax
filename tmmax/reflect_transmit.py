import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from typing import Union, List

from fresnel import _fresnel_s, _fresnel_p

def _compute_rt_at_interface_s(carry, concatenated_nk_list_theta):

    stacked_nk_list, stacked_layer_angles = concatenated_nk_list_theta

    carry_idx, carry_values = carry

    r_t_matrix = _fresnel_s(_first_layer_theta=stacked_layer_angles[0],   # Incident angle of the first layer
                            _second_layer_theta=stacked_layer_angles[1],  # Incident angle of the second layer
                            _first_layer_n=stacked_nk_list[0],            # Refractive index of the first layer
                            _second_layer_n=stacked_nk_list[1])           # Refractive index of the second layer

    carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)  # Set r,t coefficients at the current index
    carry_idx = carry_idx + 1

    return (carry_idx, carry_values), None
