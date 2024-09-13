import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import vmap 
from typing import Union, List, Tuple, Text, Dict, Callable

from angle import _compute_layer_angles_single_wl_angle_point
from cascaded_matmul import _cascaded_matrix_multiplication
from data import interpolate_nk
from reflect_transmit import _compute_rt_one_wl, _create_phases_ts_rs, _calculate_transmittace_from_coeff


def _compute_kz_single_wl_angle_point(
    nk_list: jnp.ndarray,           # Array of complex refractive indices for different wavelengths
    layer_angles: Union[int, jnp.ndarray],  # Angle of incidence for each layer, can be a single angle or an array
    wavelength: Union[int, jnp.ndarray]     # Wavelengths corresponding to the refractive indices, can be a single wavelength or an array
) -> jnp.ndarray:                    # Returns an array of computed kz values for each wavelength and angle

    """
    Computes the z-component of the wave vector (kz) for a given set of refractive indices, layer angles, and wavelengths.


    """
    # 2 * jnp.pi * nk_list: Scales the refractive index to account for wavelength in radians
    # jnp.cos(layer_angles): Computes the cosine of the incident angle for each layer
    # / wavelength: Divides by wavelength to get the wave vector component in the z-direction
    # Calculate the z-component of the wave vector for each wavelength and angle
    return 2 * jnp.pi * nk_list * jnp.cos(layer_angles) / wavelength  