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


def _compute_rt_one_wl(nk_list: jnp.ndarray, layer_angles: jnp.ndarray,
                       wavelength: Union[float, jnp.ndarray], polarization: bool) -> jnp.ndarray:
    """
    Computes the reflectance and transmittance for a single wavelength 
    across multiple layers in a stack of materials. The computation 
    takes into account the refractive index of each layer, the angle of 
    incidence in each layer, the wavelength of the light, and the 
    polarization of the light.

    """

    # Initialize the state for `jax.lax.scan`. The first element (0) is a placeholder 
    # and won't be used. The second element is a 2D array of zeros to hold intermediate 
    # results, representing the reflectance and transmittance across layers.
    init_state = (0, jnp.zeros((len(nk_list) - 2, 2), dtype=jnp.float32))  # Initial state with an array of zeros
    # The shape of `jnp.zeros` is (num_layers - 2, 2) because we exclude the first 
    # and last layers, assuming they are boundary layers.

    # Stack the refractive indices (`nk_list`) for each adjacent pair of layers.
    # This creates a new array where each element contains a pair of adjacent refractive indices 
    # from `nk_list`, which will be used to compute the reflection and transmission at the interface 
    # between these two layers.
    stacked_nk_list = jnp.stack([nk_list[:-2], nk_list[1:-1]], axis=1)  # Stack the original and shifted inputs for processing in pairs
    # For example, if `nk_list` is [n1, n2, n3, n4], this will create pairs [(n1, n2), (n2, n3), (n3, n4)].

    # Similarly, stack the angles for adjacent layers.
    # The same logic applies to `layer_angles` as for `nk_list`. Each pair of adjacent layers 
    # will have an associated pair of angles.
    stacked_layer_angles = jnp.stack([layer_angles[:-2], layer_angles[1:-1]], axis=1)
    # This operation aligns the angles with the corresponding refractive indices.

    # Now we need to compute reflectance and transmittance for each interface. 
    # This can be done using `jax.lax.scan`, which efficiently loops over the stacked pairs 
    # of refractive indices and angles.

    # If the light is s-polarized (polarization = False), we call the function `_compute_rt_at_interface_s`.
    # This function calculates the reflection and transmission coefficients specifically for s-polarized light.
    if polarization == False:
        rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_s, init_state, (stacked_nk_list, stacked_layer_angles))  # s-polarization case
        # `jax.lax.scan` applies the function `_compute_rt_at_interface_s` to each pair of adjacent layers 
        # along with the corresponding angles. It processes this in a loop, accumulating the results.

    # If the light is p-polarized (polarization = True), we use `_compute_rt_at_interface_p` instead.
    # This function handles p-polarized light.
    elif polarization == True:
        rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_p, init_state, (stacked_nk_list, stacked_layer_angles))  # p-polarization case
        # The same process as above but with a function specific to p-polarized light.

    # Finally, return the computed reflectance and transmittance coefficients. 
    # The result is stored in `rt_one_wl[1]` (the second element of `rt_one_wl`), which corresponds 
    # to the reflectance and transmittance after all layers have been processed.
    return rt_one_wl[1]  # Return a 1D theta array for each layer
    # This output is the desired result: the reflectance and transmittance for the given wavelength.
