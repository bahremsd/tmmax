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
    
    Args:
        carry: A tuple containing the index (carry_idx) and a matrix (carry_values) 
               where the reflection and transmission coefficients will be stored.
               - carry_idx (int): The current index, indicating which layer interface is being processed.
               - carry_values (array): An array to store the r,t coefficients for each interface.
        
        concatenated_nk_list_theta: A tuple containing two arrays:
               - stacked_nk_list (array): The refractive indices (n) of two consecutive layers at the interface.
               - stacked_layer_angles (array): The angles of incidence for the two consecutive layers.

    Returns:
        A tuple of:
            - Updated carry: The new index and updated matrix with the calculated r,t coefficients.
            - None: Required to match the JAX `lax.scan` interface, where a second argument is expected.
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



def _compute_rt_at_interface_p(carry, concatenated_nk_list_theta):
    """
    This function computes the reflection and transmission (r, t) coefficients at the interface between two layers
    for P-polarized light (parallel polarization). It uses the Fresnel equations to calculate these coefficients 
    based on the refractive indices and angles of incidence and refraction for the two layers.

    Args:
        carry: A tuple (carry_idx, carry_values) where:
            - carry_idx: The current index that keeps track of the layer.
            - carry_values: A matrix to store the computed reflection and transmission coefficients.
        
        concatenated_nk_list_theta: A tuple (stacked_nk_list, stacked_layer_angles) where:
            - stacked_nk_list: A list of refractive indices of the two consecutive layers.
            - stacked_layer_angles: A list of angles of incidence and refraction at the interface between the layers.

    Returns:
        A tuple:
            - Updated carry containing:
                - carry_idx incremented by 1.
                - carry_values with the newly computed r, t coefficients at the current interface.
            - None (This is used to maintain the structure of a functional-style loop but has no further use).
    """

    # Unpack the concatenated data into two variables: refractive indices (nk) and angles (theta)
    stacked_nk_list, stacked_layer_angles = concatenated_nk_list_theta  # Extract the refractive indices and angles from the input tuple
    carry_idx, carry_values = carry  # Unpack carry: carry_idx is the current index, carry_values stores r and t coefficients

    # Compute reflection (r) and transmission (t) coefficients at the interface using Fresnel equations for P-polarized light
    r_t_matrix = _fresnel_p(_first_layer_theta = stacked_layer_angles[0],  # Incident angle at the first layer
                              _second_layer_theta = stacked_layer_angles[1],  # Refraction angle at the second layer
                              _first_layer_n = stacked_nk_list[0],  # Refractive index of the first layer
                              _second_layer_n = stacked_nk_list[1])  # Refractive index of the second layer

    # Update carry_values by setting the r,t matrix at the current index (carry_idx)
    carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)  # Store the computed r,t matrix into the carry_values at the index 'carry_idx'

    carry_idx = carry_idx + 1  # Move to the next index for further iterations
    return (carry_idx, carry_values), None  # Return the updated carry with incremented index and updated r,t values, and None as a placeholder
