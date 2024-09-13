import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from typing import Union, List

from fresnel import _fresnel_s, _fresnel_p

def _compute_rt_at_interface_s(carry, concatenated_nk_list_theta):
    """
    This function calculates the reflection (r) and transmission (t) coefficients 
    for s-polarization at the interface between two layers in a multilayer thin-film system. 
    It uses the Fresnel equations for s-polarized light. The function is designed to be used 
    in a JAX `lax.scan` loop, where it processes each interface iteratively.
    


    """

    # Unpack the concatenated list into refractive index list and angle list
    stacked_nk_list, stacked_layer_angles = concatenated_nk_list_theta
    # `stacked_nk_list`: contains the refractive indices of two consecutive layers at the interface
    # `stacked_layer_angles`: contains the angles of incidence for these two layers

    # Unpack the carry tuple
    carry_idx, carry_values = carry
    # `carry_idx`: current index in the process, starts from 0 and iterates over layer interfaces
    # `carry_values`: the array that stores the reflection and transmission coefficients

    # Compute the reflection and transmission coefficients using the Fresnel equations for s-polarization
    r_t_matrix = _fresnel_s(_first_layer_theta=stacked_layer_angles[0],   # Incident angle of the first layer
                            _second_layer_theta=stacked_layer_angles[1],  # Incident angle of the second layer
                            _first_layer_n=stacked_nk_list[0],            # Refractive index of the first layer
                            _second_layer_n=stacked_nk_list[1])           # Refractive index of the second layer
    # This line computes r and t coefficients between two consecutive layers 
    # based on their refractive indices and angles of incidence.

    # Store the computed r,t matrix in the `carry_values` array at the current index
    carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)  # Set r,t coefficients at the current index
    # The `carry_values.at[carry_idx, :].set(r_t_matrix)` updates the array at position `carry_idx` 
    # with the computed r,t coefficients.

    carry_idx = carry_idx + 1  # Move to the next index for the next iteration
    # The carry index is incremented to process the next layer interface in subsequent iterations.

    # Return the updated carry (with new index and r,t coefficients) and None for lax.scan compatibility
    return (carry_idx, carry_values), None
