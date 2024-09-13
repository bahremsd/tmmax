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
    
    Args:
        nk_list (jnp.ndarray): A 1D array containing the refractive indices (n) or complex indices (n + ik) for different wavelengths.
        layer_angles (Union[int, jnp.ndarray]): A scalar or 1D array specifying the angle of incidence for each layer. It should be in radians.
        wavelength (Union[int, jnp.ndarray]): A scalar or 1D array of wavelengths corresponding to the refractive indices.

    Returns:
        jnp.ndarray: A 1D array of computed kz values, which represents the z-component of the wave vector for each wavelength and angle.
    """
    # 2 * jnp.pi * nk_list: Scales the refractive index to account for wavelength in radians
    # jnp.cos(layer_angles): Computes the cosine of the incident angle for each layer
    # / wavelength: Divides by wavelength to get the wave vector component in the z-direction
    # Calculate the z-component of the wave vector for each wavelength and angle
    return 2 * jnp.pi * nk_list * jnp.cos(layer_angles) / wavelength  


def _tmm_single_wl_angle_point(nk_functions: Dict[int, Callable], material_list: list[int],
                               thickness_list: jnp.ndarray, wavelength: Union[float, jnp.ndarray],
                               angle_of_incidence: Union[float, jnp.ndarray], polarization: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:


    def get_nk_values(wl):

        return jnp.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Get nk values for each material

    nk_list = get_nk_values(wavelength)  # Call get_nk_values to get refractive index values for all materials

    layer_angles = _compute_layer_angles_single_wl_angle_point(nk_list, angle_of_incidence, wavelength, polarization)

    kz = _compute_kz_single_wl_angle_point(nk_list, layer_angles, wavelength)

    layer_phases = kz * jnp.pad(thickness_list, (1), constant_values=0)

    rt = _compute_rt_one_wl(nk_list=nk_list, layer_angles=layer_angles, wavelength=wavelength, polarization=polarization)

    _phases_ts_rs = _create_phases_ts_rs(rt[1:,:], layer_phases[1:-1])

    tr_matrix = _cascaded_matrix_multiplication(_phases_ts_rs)

    tr_matrix = (1 / rt[0, 1]) * jnp.dot(jnp.array([[1, rt[0, 0]], [rt[0, 0], 1]]), tr_matrix)

    r = tr_matrix[1, 0] / tr_matrix[0, 0]
    t = 1 / tr_matrix[0, 0]

    R = jnp.abs(r) ** 2
    T = _calculate_transmittace_from_coeff(t, nk_list[0], nk_list[-1], angle_of_incidence, layer_angles[-1], polarization)


    return R, T
    # Return the reflectance and transmittance values
