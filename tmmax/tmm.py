import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import vmap 
from typing import Union, List, Tuple, Text, Dict, Callable

from .angle import _compute_layer_angles_single_wl_angle_point
from .cascaded_matmul import _cascaded_matrix_multiplication
from .data import interpolate_nk
from .reflect_transmit import _compute_rt_one_wl, _create_phases_ts_rs, _calculate_transmittace_from_coeff

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

    # Calculate the z-component of the wave vector for each wavelength and angle
    return 2 * jnp.pi * nk_list * jnp.cos(layer_angles) / wavelength  
    # 2 * jnp.pi * nk_list: Scales the refractive index to account for wavelength in radians
    # jnp.cos(layer_angles): Computes the cosine of the incident angle for each layer
    # / wavelength: Divides by wavelength to get the wave vector component in the z-direction


def _tmm_single_wl_angle_point(nk_functions: Dict[int, Callable], material_list: list[int],
                               thickness_list: jnp.ndarray, wavelength: Union[float, jnp.ndarray],
                               angle_of_incidence: Union[float, jnp.ndarray], polarization: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the reflectance (R) and transmittance (T) of a multi-layer optical film for a given wavelength
    and angle of incidence using the Transfer Matrix Method (TMM).

    Args:
        nk_functions (Dict[int, Callable]): Dictionary mapping material indices to functions that return 
                                           the complex refractive index (n + ik) for a given wavelength.
        material_list (list[int]): List of indices representing the order of materials in the stack.
        thickness_list (jnp.ndarray): Array of thicknesses for each layer in the stack.
        wavelength (Union[float, jnp.ndarray]): Wavelength(s) of light in the simulation.
        angle_of_incidence (Union[float, jnp.ndarray]): Angle of incidence in radians.
        polarization (bool): True for TM polarization, False for TE polarization.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Reflectance (R) and transmittance (T) of the optical stack.
    """

    def get_nk_values(wl):
        """
        Retrieves the complex refractive index values for each material at the given wavelength.

        Args:
            wl (Union[float, jnp.ndarray]): Wavelength or array of wavelengths.

        Returns:
            jnp.ndarray: Array of complex refractive index values for each material.
        """
        return jnp.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Get nk values for each material

    nk_list = get_nk_values(wavelength)  # Call get_nk_values to get refractive index values for all materials

    layer_angles = _compute_layer_angles_single_wl_angle_point(nk_list, angle_of_incidence, wavelength, polarization)
    # Compute the angles within each layer based on the refractive indices, incidence angle, and wavelength

    kz = _compute_kz_single_wl_angle_point(nk_list, layer_angles, wavelength)
    # Calculate the z-component of the wave vector for each layer

    layer_phases = kz * jnp.pad(thickness_list, (1), constant_values=0)
    # Compute the phase shifts in each layer by multiplying kz by the layer thicknesses
    # `jnp.pad(thickness_list, (1), constant_values=0)` adds a leading zero to the thickness_list

    rt = _compute_rt_one_wl(nk_list=nk_list, layer_angles=layer_angles, wavelength=wavelength, polarization=polarization)
    # Compute the reflection and transmission matrices for the wavelength

    _phases_ts_rs = _create_phases_ts_rs(rt[1:,:], layer_phases[1:-1])
    # Create a list of phase shift matrices from the transmission and reflection matrices and layer phases
    # Exclude the first and last reflection matrix, and the first and last phase shift values

    tr_matrix = _cascaded_matrix_multiplication(_phases_ts_rs)
    # Perform matrix multiplication to obtain the cascaded transfer matrix for the entire stack

    tr_matrix = (1 / rt[0, 1]) * jnp.dot(jnp.array([[1, rt[0, 0]], [rt[0, 0], 1]]), tr_matrix)
    # Normalize the transfer matrix and include the boundary conditions
    # `jnp.dot` multiplies the transfer matrix by the boundary conditions matrix

    r = tr_matrix[1, 0] / tr_matrix[0, 0]
    t = 1 / tr_matrix[0, 0]
    # Calculate the reflectance (r) and transmittance (t) from the transfer matrix
    # Reflectance is obtained by dividing the (1, 0) element by the (0, 0) element
    # Transmittance is obtained by taking the reciprocal of the (0, 0) element

    R = jnp.abs(r) ** 2
    T = _calculate_transmittace_from_coeff(t, nk_list[0], nk_list[-1], angle_of_incidence, layer_angles[-1], polarization)
    # Compute the reflectance (R) and transmittance (T) using their respective formulas
    # Reflectance R is the squared magnitude of r
    # Transmittance T is calculated using a function `_calculate_transmittace_from_coeff`

    return R, T
    # Return the reflectance and transmittance values



def tmm(material_list: List[str],
        thickness_list: jnp.ndarray,
        wavelength_arr: jnp.ndarray,
        angle_of_incidences: jnp.ndarray,
        polarization: Text) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform the Transfer Matrix Method (TMM) for multilayer thin films.
    
    Args:
        material_list (List[str]): A list of material names. Each material is identified by a string.
        thickness_list (jnp.ndarray): An array of thicknesses corresponding to each layer.
        wavelength_arr (jnp.ndarray): An array of wavelengths over which to perform the simulation.
        angle_of_incidences (jnp.ndarray): An array of angles of incidence.
        polarization (Text): The type of polarization ('s' for s-polarized or 'p' for p-polarized).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two JAX arrays. The first array represents the transmission coefficients, and the second array represents the reflection coefficients.
    """

    # Remove duplicate materials and create a unique set
    material_set = list(set(material_list))  # Create a unique list of materials
    # Create a mapping from material names to indices
    material_enum = {material: i for i, material in enumerate(material_set)}  # Map each material to an index
    # Convert the original material list to a list of indices
    material_list = [int(material_enum[material]) for material in material_list]  # Map materials to indices based on material_enum
    # Create a dictionary of interpolation functions for each material
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}  # Interpolate n and k for each material

    # Convert polarization type to a boolean flag
    if polarization == 's':
        polarization = False  # s-polarized light
    elif polarization == 'p':
        polarization = True  # p-polarized light
    else:
        raise TypeError("The polarization can be 's' or 'p', not the other parts. Correct it")  # Raise an error for invalid polarization input

    # Vectorize the _tmm_single_wl_angle_point function across wavelength and angle of incidence
    tmm_vectorized = vmap(vmap(_tmm_single_wl_angle_point, (None, None, None, 0, None, None)), (None, None, None, None, 0, None))  # Vectorize _tmm_single_wl_angle_point over wavelengths and angles

    # Apply the vectorized TMM function to the arrays
    result = tmm_vectorized(nk_funkcs, material_list, thickness_list, wavelength_arr, angle_of_incidences, polarization)  # Compute the TMM results

    # Return the computed result
    return result  # Tuple of transmission and reflection coefficients
