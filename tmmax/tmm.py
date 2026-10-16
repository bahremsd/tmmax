import jax.numpy as jnp # jax's numpy library we will use for all general mathematical operations
from jax import jit, vmap # Importing jit for JIT compilation and vmap for efficient vectorization
from jax import Array # Type definition for JAX arrays
from jax.typing import ArrayLike # JAX type hint for array-like objects (supports numpy, JAX arrays, etc.)
from typing import List, Tuple, Text
import numpy as np # just for np.any in the tmm function for checking input values

from .angle import compute_layer_angles
from .wavevector import compute_kz, compute_inc_layer_pass
from .cascaded_matmul import coh_cascaded_matrix_multiplication, incoh_cascaded_matrix_multiplication, compute_first_layer_matrix_coherent, compute_first_layer_matrix_incoherent
from .data import create_nk_list, material_distribution_to_set, create_data
from .reflect_transmit import compute_rt, calculate_reflectance_from_coeff, calculate_transmittace_from_coeff, compute_r_t_magnitudes_incoh

def find_coh_and_incoh_indices(coherency_list: ArrayLike):
    """
    This function determines the coherent and incoherent layers in an optical multilayer thin-film stack.
    It analyzes the `coherency_list` to identify consecutive coherent layers and incoherent layers,
    returning structured information about these layers for further processing in optical simulations.

    Arguments:
        coherency_list (ArrayLike): An array of integers (dtype=int32) representing the coherency states of the layers.
                                    The array has N elements for a multilayer stack with N layers, where:
                                    - 0 indicates a coherent layer.
                                    - 1 indicates an incoherent layer.

    Returns:
        1. coherent_groups (Array): A 2D array where each row specifies the start and end indices of consecutive coherent layers.
        2. incoherent_indices (Array): A 1D array containing indices of incoherent layers.
        3. coherency_indices (Array): A 1D array containing all coherent groups and incoherent layers, where:
            - Positive indices correspond to incoherent layers.
            - Negative indices correspond to coherent stacks, and their (absolute values - 1) indicate the index of the stack in coherent_groups.

    Example:
        Input: coherency_list = [0, 0, 0, 1, 0, 0, 1] for the multilayer thin film  [ Air | TiO2, SiO2, TiO2, Al2O3, SiO2, TiO2, SiO2 | Air ]
        Output: 
            coherent_groups = [[1, 3], [5, 6]]
            incoherent_indices = [0, 4, 7, 8]
            coherency_indices = [0, -1, 4, -2, 7, 8]
    """

    # Add padding to the coherency list with a layer of 1 on both sides for incoming and outgoing incoherent mediums
    padded_coherency_list = jnp.pad(coherency_list, pad_width=(1, 1), constant_values=1)

    # Compute the binary changes in the padded list to find transitions between 0 and 1
    changes = jnp.diff(padded_coherency_list, prepend=1 - padded_coherency_list[0])

    # Find the start indices of coherent groups (where changes equal -1)
    start_indices = jnp.where(changes == -1)[0]

    # Find the end indices of coherent groups (where changes equal 1, adjusted by -1)
    end_indices = jnp.where(changes == 1)[0] - 1

    # Remove the first element of end_indices to handle padding adjustments
    end_indices = jnp.delete(end_indices, 0)

    # Combine start and end indices to form coherent groups
    coherent_groups = jnp.stack([start_indices, end_indices], axis=1)

    # Identify indices of incoherent layers (where value is 1 in the padded list)
    incoherent_indices = jnp.where(padded_coherency_list == 1)[0]

    # Initialize coherency_indices with all incoherent layer indices
    coherency_indices = jnp.where(padded_coherency_list == 1)[0]

    # Insert coherent group leading indices into coherency_indices
    for idx, group in enumerate(coherent_groups):
        # Get the starting index of the current coherent group
        leading_index = group[0]

        # Find the position to insert the negative index in coherency_indices
        insertion_idx = jnp.searchsorted(coherency_indices, leading_index)

        # Insert the negative index corresponding to the current coherent group
        coherency_indices = jnp.insert(coherency_indices, insertion_idx, -idx - 1)

        # Remove the duplicated entry after insertion (adjusted by insertion_idx)
        coherency_indices = jnp.delete(coherency_indices, insertion_idx - 1)

    # Return the coherent groups, incoherent indices, and combined coherency indices
    return coherent_groups, incoherent_indices, coherency_indices

def tmm_coh_single_wl_angle_point(data: ArrayLike,
                                  material_distribution: ArrayLike,
                                  thickness_list: ArrayLike,
                                  wavelength: ArrayLike,
                                  angle_of_incidence: ArrayLike,
                                  polarization: ArrayLike) -> Array:
    """
    This function implements the JIT-ed version of Transfer Matrix Method (TMM) to 
    compute the reflectance (R) and transmittance (T) of a coherent multilayer thin-film 
    structure for a given wavelength, angle of incidence, and polarization.

    Arguments:
    ----------
    data: ArrayLike
        A 2D array where each row corresponds to the refractive index (n) and extinction 
        coefficient (k) data of a specific material. The first axis length is the number of 
        materials, and the second axis is the length of the nk vs wavelength data for each 
        material.

    material_distribution: ArrayLike
        An array of integers specifying the material index for each layer in the multilayer 
        structure, including the incoming and outgoing medium. Its length is N + 2, where 
        N is the number of thin-film layers.

    thickness_list: ArrayLike
        A 1D array of floats representing the physical thickness of each thin-film layer. 
        Its length is N, corresponding to the N layers in the multilayer structure.

    wavelength: ArrayLike
        The wavelength of the light wave (in the same units as the nk data). This is used 
        to calculate the optical properties for the specific wavelength.

    angle_of_incidence: ArrayLike
        The angle (in radians) of the incoming light with respect to the normal of the 
        multilayer thin-film surface.

    polarization: ArrayLike
        Specifies the polarization state of the light wave. If `False`, the light is 
        s-polarized (E field is perpendicular to the plane of incidence), and if `True`, 
        it is p-polarized (E field is parallel to the plane of incidence).

    Returns:
    --------
    Tuple of two scalars:
        R: Reflectance (a value between 0 and 1, representing the ratio of reflected power 
           to incident power).
        T: Transmittance (a value between 0 and 1, representing the ratio of transmitted 
           power to incident power).
    """

    # Create a list of refractive indices and extinction coefficients for each layer 
    # at the given wavelength using the provided material distribution.
    nk_list = create_nk_list(material_distribution, data, wavelength)
    
    # Compute the angles of light propagation within each layer based on Snell's Law
    # and the input angle of incidence.
    layer_angles = compute_layer_angles(angle_of_incidence, nk_list, polarization)
    
    # Compute the z-component of the wavevector (kz) in each layer, which is wavelength-dependent.
    kz = compute_kz(nk_list, layer_angles, wavelength)
    
    # Compute the phase change in each layer due to the thickness of the layer.
    # Excludes the incoming and outgoing media (index 0 and -1).
    layer_phases = jnp.multiply(kz.at[1:-1].get(), thickness_list)

    # Calculate reflection and transmission coefficients at each layer interface.
    # Excludes the incoming and outgoing media.
    rt = jnp.squeeze(compute_rt(nk_list = nk_list, angles = layer_angles, polarization = polarization))

    # Multiply transfer matrices for all internal layers to compute the combined matrix.
    # This represents the cumulative effect of reflections and transmissions.
    tr_matrix = coh_cascaded_matrix_multiplication(phases = layer_phases, rts = rt.at[1:,:].get())

    # Incorporate the reflection and transmission coefficients of the incoming medium
    # into the total transfer matrix.
    tr_matrix = jnp.multiply(jnp.true_divide(1, rt.at[0,1].get()), jnp.dot(jnp.array([[1, rt.at[0,0].get()], [rt.at[0,0].get(), 1]]), tr_matrix))

    # Extract the complex reflectance coefficient (r) from the total transfer matrix.
    r = jnp.true_divide(tr_matrix.at[1,0].get(), tr_matrix.at[0,0].get())

    # Extract the complex transmittance coefficient (t) from the total transfer matrix.
    t = jnp.true_divide(1, tr_matrix.at[0,0].get())

    # Compute the reflectance (R) from the reflectance coefficient.
    R = calculate_reflectance_from_coeff(r)

    # Compute the transmittance (T) from the transmittance coefficient, accounting for
    # impedance matching and angle-dependent factors in the incoming and outgoing media.
    T = calculate_transmittace_from_coeff(t, nk_list.at[0].get(), angle_of_incidence, nk_list.at[-1].get(), layer_angles.at[-1].get(), polarization) #calculate T
    
    # Return the computed reflectance and transmittance values as the output.
    return R, T


def tmm_incoh_single_wl_angle_point(data: ArrayLike,
                                    material_distribution: ArrayLike,
                                    thickness_list: ArrayLike,
                                    coherent_layer_indices_forward: ArrayLike,
                                    incoherent_layer_indices_forward: ArrayLike,
                                    coherency_indices_forward: ArrayLike,
                                    coherent_layer_indices_backward: ArrayLike,
                                    coherency_indices_backward: ArrayLike,
                                    wavelength: ArrayLike,
                                    angle_of_incidence: ArrayLike,
                                    polarization: ArrayLike) -> Array:
    """
    This function implements the Transfer Matrix Method (TMM) to compute the reflectance (R) 
    and transmittance (T) of a incoherent multilayer thin-film structure for a given wavelength, 
    angle of incidence, and polarization.

    Arguments:
    ----------
    data: ArrayLike
        A 2D array where each row corresponds to the refractive index (n) and extinction 
        coefficient (k) data of a specific material. The first axis length is the number of 
        materials, and the second axis is the length of the nk vs wavelength data for each 
        material.

    material_distribution: ArrayLike
        An array of integers specifying the material index for each layer in the multilayer 
        structure, including the incoming and outgoing medium. Its length is N + 2, where 
        N is the number of thin-film layers.

    thickness_list: ArrayLike
        A 1D array of floats representing the physical thickness of each thin-film layer. 
        Its length is N, corresponding to the N layers in the multilayer structure.

    coherent_layer_indices_forward: ArrayLike
        Indices of coherent layers for forward propagation.

    incoherent_layer_indices_forward: ArrayLike
        Indices of incoherent layers for forward propagation.

    coherency_indices_forward: ArrayLike
        Indices defining coherency conditions in forward direction.

    coherent_layer_indices_backward: ArrayLike
        Indices of coherent layers for backward propagation.

    coherency_indices_backward: ArrayLike
        Indices defining coherency conditions in backward direction.

    wavelength: ArrayLike
        The wavelength of the light wave (in the same units as the nk data). This is used 
        to calculate the optical properties for the specific wavelength.

    angle_of_incidence: ArrayLike
        The angle (in radians) of the incoming light with respect to the normal of the 
        multilayer thin-film surface.

    polarization: ArrayLike
        Specifies the polarization state of the light wave. If `False`, the light is 
        s-polarized (E field is perpendicular to the plane of incidence), and if `True`, 
        it is p-polarized (E field is parallel to the plane of incidence).

    Returns:
    --------
    Tuple of two scalars:
        R: Reflectance (a value between 0 and 1, representing the ratio of reflected power 
           to incident power).
        T: Transmittance (a value between 0 and 1, representing the ratio of transmitted 
           power to incident power).
    """

    # Create a list of refractive indices and extinction coefficients for each layer 
    # at the given wavelength using the provided material distribution.
    nk_list = create_nk_list(material_distribution, data, wavelength)
    
    # Compute the angles of light propagation within each layer based on Snell's Law
    # and the input angle of incidence.
    layer_angles = compute_layer_angles(angle_of_incidence, nk_list, polarization)
    
    # Compute the z-component of the wavevector (kz) in each layer, which is wavelength-dependent.
    kz = compute_kz(nk_list, layer_angles, wavelength)
    
    # Compute the phase change in each layer due to the thickness of the layer.
    # Excludes the incoming and outgoing media (index 0 and -1).
    layer_phases = jnp.multiply(kz.at[1:-1].get(), thickness_list)

    # Compute reflection and transmission coefficients at each interface
    rt_forward = jnp.squeeze(compute_rt(nk_list = nk_list, angles = layer_angles, polarization = polarization))
    rt_backward = jnp.squeeze(compute_rt(nk_list = jnp.flip(nk_list), angles = jnp.flip(layer_angles), polarization = polarization))

    # Compute magnitudes of reflection and transmission coefficients for forward propagation (not good for jit! update this func asap)
    forward_magnitudes = compute_r_t_magnitudes_incoh(coherent_layer_indices = coherent_layer_indices_forward,
                                                      coherency_indices = coherency_indices_forward,
                                                      rts = rt_forward,
                                                      layer_phases = layer_phases,
                                                      nk_list = nk_list,
                                                      layer_angles = layer_angles,
                                                      polarization = polarization)
    
    # Compute magnitudes of reflection and transmission coefficients for backward propagation (not good for jit! update this func asap)
    backward_magnitudes = compute_r_t_magnitudes_incoh(coherent_layer_indices = coherent_layer_indices_backward,
                                                       coherency_indices = coherency_indices_backward,
                                                       rts = rt_backward,
                                                       layer_phases = jnp.flip(layer_phases),
                                                       nk_list = jnp.flip(nk_list),
                                                       layer_angles = jnp.flip(layer_angles),
                                                       polarization = polarization)

    # Flip backward magnitudes to align with the correct propagation direction
    backward_magnitudes = jnp.flip(backward_magnitudes, axis=0)

    # Compute incoherent layer pass contributions
    layer_passes = compute_inc_layer_pass(incoherent_layer_indices_forward.at[1:-1].get(), layer_phases)

    # Compute the transfer matrix for the entire multilayer structure
    tr_matrix = incoh_cascaded_matrix_multiplication(forward_magnitudes = forward_magnitudes.at[1:,:].get(),
                                                     backward_magnitudes = backward_magnitudes.at[1:,:].get(),
                                                     layer_passes = layer_passes)

    # Compute the first layer matrix for the incoming medium
    first_layer_matrix = compute_first_layer_matrix_incoherent(r_forward = forward_magnitudes.at[0,0].get(),
                                                               t_forward = forward_magnitudes.at[0,1].get(),
                                                               r_backward = backward_magnitudes.at[0,0].get(),
                                                               t_backward = backward_magnitudes.at[0,1].get())

    # Multiply the first layer matrix with the transfer matrix
    tr_matrix = jnp.dot(first_layer_matrix, tr_matrix)

    # Compute reflectance (R) and transmittance (T) from the transfer matrix elements
    R = jnp.true_divide(tr_matrix.at[1,0].get(), tr_matrix.at[0,0].get())
    T = jnp.true_divide(1, tr_matrix.at[0,0].get())

    # Return computed reflectance and transmittance
    return R, T

@jit
def tmm_coh_single_wl_angle_point_jit(data: ArrayLike,
                                      material_distribution: ArrayLike,
                                      thickness_list: ArrayLike,
                                      wavelength: ArrayLike,
                                      angle_of_incidence: ArrayLike,
                                      polarization: ArrayLike) -> Array:
    """
    This function implements the JIT-ed version of Transfer Matrix Method (TMM) to 
    compute the reflectance (R) and transmittance (T) of a coherent multilayer thin-film 
    structure for a given wavelength, angle of incidence, and polarization.

    Arguments:
    ----------
    data: ArrayLike
        A 2D array where each row corresponds to the refractive index (n) and extinction 
        coefficient (k) data of a specific material. The first axis length is the number of 
        materials, and the second axis is the length of the nk vs wavelength data for each 
        material.

    material_distribution: ArrayLike
        An array of integers specifying the material index for each layer in the multilayer 
        structure, including the incoming and outgoing medium. Its length is N + 2, where 
        N is the number of thin-film layers.

    thickness_list: ArrayLike
        A 1D array of floats representing the physical thickness of each thin-film layer. 
        Its length is N, corresponding to the N layers in the multilayer structure.

    wavelength: ArrayLike
        The wavelength of the light wave (in the same units as the nk data). This is used 
        to calculate the optical properties for the specific wavelength.

    angle_of_incidence: ArrayLike
        The angle (in radians) of the incoming light with respect to the normal of the 
        multilayer thin-film surface.

    polarization: ArrayLike
        Specifies the polarization state of the light wave. If `False`, the light is 
        s-polarized (E field is perpendicular to the plane of incidence), and if `True`, 
        it is p-polarized (E field is parallel to the plane of incidence).

    Returns:
    --------
    Tuple of two scalars:
        R: Reflectance (a value between 0 and 1, representing the ratio of reflected power 
           to incident power).
        T: Transmittance (a value between 0 and 1, representing the ratio of transmitted 
           power to incident power).
    """

    # Create a list of refractive indices and extinction coefficients for each layer 
    # at the given wavelength using the provided material distribution.
    nk_list = create_nk_list(material_distribution, data, wavelength)
    
    # Compute the angles of light propagation within each layer based on Snell's Law
    # and the input angle of incidence.
    layer_angles = compute_layer_angles(angle_of_incidence, nk_list, polarization)
    
    # Compute the z-component of the wavevector (kz) in each layer, which is wavelength-dependent.
    kz = compute_kz(nk_list, layer_angles, wavelength)
    
    # Compute the phase change in each layer due to the thickness of the layer.
    # Excludes the incoming and outgoing media (index 0 and -1).
    layer_phases = jnp.multiply(kz.at[1:-1].get(), thickness_list)

    # Calculate reflection and transmission coefficients at each layer interface.
    # Excludes the incoming and outgoing media.
    rt = jnp.squeeze(compute_rt(nk_list = nk_list, angles = layer_angles, polarization = polarization))

    # Multiply transfer matrices for all internal layers to compute the combined matrix.
    # This represents the cumulative effect of reflections and transmissions.
    tr_matrix = coh_cascaded_matrix_multiplication(phases = layer_phases, rts = rt.at[1:,:].get())

    # Incorporate the reflection and transmission coefficients of the incoming medium
    # into the total transfer matrix.
    tr_matrix = jnp.multiply(jnp.true_divide(1, rt.at[0,1].get()), jnp.dot(jnp.array([[1, rt.at[0,0].get()], [rt.at[0,0].get(), 1]]), tr_matrix))

    # Extract the complex reflectance coefficient (r) from the total transfer matrix.
    r = jnp.true_divide(tr_matrix.at[1,0].get(), tr_matrix.at[0,0].get())

    # Extract the complex transmittance coefficient (t) from the total transfer matrix.
    t = jnp.true_divide(1, tr_matrix.at[0,0].get())

    # Compute the reflectance (R) from the reflectance coefficient.
    R = calculate_reflectance_from_coeff(r)

    # Compute the transmittance (T) from the transmittance coefficient, accounting for
    # impedance matching and angle-dependent factors in the incoming and outgoing media.
    T = calculate_transmittace_from_coeff(t, nk_list.at[0].get(), angle_of_incidence, nk_list.at[-1].get(), layer_angles.at[-1].get(), polarization) #calculate T
    
    # Return the computed reflectance and transmittance values as the output.
    return R, T

@jit
def vectorized_coh_tmm(data: ArrayLike, 
                       material_distribution: ArrayLike, 
                       thickness_list: ArrayLike, 
                       wavelengths: ArrayLike, 
                       angle_of_incidences: ArrayLike, 
                       polarization) -> Array:
    """
    This function is designed to vectorize the Transfer Matrix Method (TMM) calculations 
    over two axes: wavelength and angle of incidence. The function uses the `jax.vmap` 
    function to vectorize the `tmm_coh_single_wl_angle_point` function, which performs TMM 
    calculations for a single combination of wavelength and angle of incidence. The first 
    `vmap` maps over the wavelength axis, and the second `vmap` maps over the angle of 
    incidence axis. The other inputs (`data`, `material_distribution`, `thickness_list`, 
    and `polarization`) remain constant during the vectorized computation.

    Arguments:
    ----------
    data: ArrayLike
        A 2D array where each row corresponds to the refractive index (n) and extinction 
        coefficient (k) data of a specific material. The first axis length is the number of 
        materials, and the second axis is the length of the nk vs wavelength dataset for 
        each material.

    material_distribution: ArrayLike
        An array of integers specifying the material index for each layer in the multilayer 
        structure, including the incoming and outgoing medium. Its length is N + 2, where 
        N is the number of thin-film layers.

    thickness_list: ArrayLike
        A 1D array of floats representing the physical thickness of each thin-film layer. 
        Its length is N, corresponding to the N layers in the multilayer structure.

    wavelengths: ArrayLike
        The wavelengths of the light wave (in the same units as the nk data). This is used 
        to calculate the optical properties for the specific wavelength.

    angle_of_incidences: ArrayLike
        The angles (in radians) of the incoming light with respect to the normal of the 
        multilayer thin-film surface.

    polarization: ArrayLike
        Specifies the polarization state of the light wave. If `False`, the light is 
        s-polarized (perpendicular to the plane of incidence), and if `True`, it is 
        p-polarized (parallel to the plane of incidence).

    Returns:
    --------
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two arrays: 
            - The first array represents the reflection coefficients.
            - The second array represents the transmission coefficients.
            Both arrays have indices corresponding to angle of incidence (first index) and wavelength (second index).
    """

    # Use `vmap` to vectorize `tmm_coh_single_wl_angle_point` over wavelength and angle of incidence.
    # The first `vmap` applies vectorization across the angle_of_incidence axis (last dimension of the input array).
    # The second `vmap` applies vectorization across the wavelength axis (one level up in the input hierarchy).
    tmm_vmap = vmap(vmap(tmm_coh_single_wl_angle_point, (None, None, None, 0, None, None)), (None, None, None, None, 0, None))
    
    # Apply the vectorized function `tmm_vmap` to the input arguments and return the result.
    # `data`, `material_distribution`, `thickness_list`, and `polarization` remain constant during vectorized computation
    return tmm_vmap(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization)


def vectorized_incoh_tmm(data: ArrayLike,
                         material_distribution: ArrayLike,
                         thickness_list: ArrayLike,
                         coherent_layer_indices_forward: ArrayLike,
                         incoherent_layer_indices_forward: ArrayLike,
                         coherency_indices_forward: ArrayLike,
                         coherent_layer_indices_backward: ArrayLike,
                         coherency_indices_backward: ArrayLike,
                         wavelengths: ArrayLike,
                         angle_of_incidences: ArrayLike,
                         polarization: ArrayLike):
    """
    This function is designed to vectorize the Transfer Matrix Method (TMM) calculations 
    over two axes: wavelength and angle of incidence. The function uses the `jax.vmap` 
    function to vectorize the `tmm_coh_single_wl_angle_point` function, which performs TMM 
    calculations for a single combination of wavelength and angle of incidence. The first 
    `vmap` maps over the wavelength axis, and the second `vmap` maps over the angle of 
    incidence axis. The other inputs (`data`, `material_distribution`, `thickness_list`, 
    and `polarization`) remain constant during the vectorized computation.

    Arguments:
    ----------
    data: ArrayLike
        A 2D array where each row corresponds to the refractive index (n) and extinction 
        coefficient (k) data of a specific material. The first axis length is the number of 
        materials, and the second axis is the length of the nk vs wavelength dataset for 
        each material.

    material_distribution: ArrayLike
        An array of integers specifying the material index for each layer in the multilayer 
        structure, including the incoming and outgoing medium. Its length is N + 2, where 
        N is the number of thin-film layers.

    thickness_list: ArrayLike
        A 1D array of floats representing the physical thickness of each thin-film layer. 
        Its length is N, corresponding to the N layers in the multilayer structure.
        
    coherent_layer_indices_forward: ArrayLike
        A 1D array of indices representing the layers that are considered coherent in the 
        forward direction.

    incoherent_layer_indices_forward: ArrayLike
        A 1D array of indices representing the layers that are considered incoherent in the 
        forward direction. 

    coherency_indices_forward: ArrayLike
        A 1D array of indices specifying the layers where the coherence condition holds in the 
        forward direction.

    coherent_layer_indices_backward: ArrayLike
        A 1D array of indices representing the layers that are considered coherent in the backward
        direction.

    coherency_indices_backward: ArrayLike
        A 1D array of indices specifying the layers (coh or incoh) where the coherence condition 
        holds in the backward direction.

    wavelengths: ArrayLike
        The wavelengths of the light wave (in the same units as the nk data). This is used 
        to calculate the optical properties for the specific wavelength.

    angle_of_incidences: ArrayLike
        The angles (in radians) of the incoming light with respect to the normal of the 
        multilayer thin-film surface.

    polarization: ArrayLike
        Specifies the polarization state of the light wave. If `False`, the light is 
        s-polarized (perpendicular to the plane of incidence), and if `True`, it is 
        p-polarized (parallel to the plane of incidence).

    Returns:
    --------
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two arrays: 
            - The first array represents the reflection coefficients.
            - The second array represents the transmission coefficients.
            Both arrays have indices corresponding to angle of incidence (first index) and wavelength (second index).
    """

    # Use `vmap` to vectorize `tmm_incoh_single_wl_angle_point` over wavelength and angle of incidence.
    # The first `vmap` applies vectorization across the angle_of_incidence axis (last dimension of the input array).
    # The second `vmap` applies vectorization across the wavelength axis (one level up in the input hierarchy).
    tmm_vmap = vmap(vmap(tmm_incoh_single_wl_angle_point, 
                         (None, None, None, None, None, None, None, None, 0, None, None)), 
                         (None, None, None, None, None, None, None, None, None, 0, None))
    
    # Apply the vectorized function `tmm_vmap` to the input arguments and return the result.
    # `data`, `material_distribution`, `thickness_list`, and `polarization` remain constant during vectorized computation
    return tmm_vmap(data,
                    material_distribution,
                    thickness_list,
                    coherent_layer_indices_forward,
                    incoherent_layer_indices_forward,
                    coherency_indices_forward,
                    coherent_layer_indices_backward,
                    coherency_indices_backward,
                    wavelengths,
                    angle_of_incidences,
                    polarization)

def tmm_coh(material_list: List[str],
            thickness_list: ArrayLike,
            wavelength_arr: ArrayLike,
            angle_of_incidences: ArrayLike,
            polarization: Text) -> Tuple[Array, Array]:
    """
    Perform the Transfer Matrix Method (TMM) for coherent multilayer thin films.

    Arguments:
    ----------
        material_list (List[str]): A list of material names. Each material is identified by a string.
            Example: ["Air", "TiO2", "SiO2"]. These strings specify the materials in the multilayer thin film structure.
            The first and last materials in the list are typically assumed to be semi-infinite layers (e.g., Air, SiO2).
        
        thickness_list (jnp.ndarray): An array of thicknesses corresponding to the thin-film layers.
            Each element specifies the physical thickness of one layer. The order matches the materials in `material_list`.
            The thicknesses should be finite for thin-film layers and can be ignored for the semi-infinite layers.

        wavelength_arr (jnp.ndarray): An array of wavelengths (in micrometers or nanometers) for which the simulation is performed.
            This array allows users to specify the range of wavelengths for analyzing the optical properties of the multilayer.

        angle_of_incidences (jnp.ndarray): An array of angles of incidence (in degrees) at which the light interacts with the thin film structure.
            Each value represents a different incident angle, and the function calculates results for all specified angles.

        polarization (Text): The type of polarization of light, either "s" (s-polarization) or "p" (p-polarization).
            - "s" refers to perpendicular polarization with respect to the plane of incidence.
            - "p" refers to parallel polarization with respect to the plane of incidence.

    Returns:
    --------
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple of two JAX arrays:
            - The first array represents the reflection coefficients.
            - The second array represents the transmission coefficients.
            Both arrays have indices corresponding to angle of incidence (first index) and wavelength (second index).
    """
    
    # Convert the material list into a set and a material distribution array
    # The material set contains unique materials, and the distribution describes how materials are layered.
    material_set, material_distribution = material_distribution_to_set(material_list)
    
    # Create the required material data, such as refractive index and extinction coefficient, for each material in the set.
    # This function retrieves optical properties needed for TMM calculations.
    data = create_data(material_set)

    # Check the polarization input and convert it to a boolean JAX array.
    if polarization == 's':
        # For s-polarization, set the boolean flag to `False`.
        polarization = jnp.array([False], dtype=bool)
    elif polarization == 'p':
        # For p-polarization, set the boolean flag to `True`.
        polarization = jnp.array([True], dtype=bool)
    else:
        # Raise an error if the polarization input is invalid.
        raise TypeError("The polarization can be 's' or 'p', not other parts. Correct it")

    # Apply the vectorized TMM function, which computes transmission and reflection coefficients
    # over the specified wavelengths and angles of incidence.
    # The computation leverages JAX for high-performance array operations.
    result = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelength_arr, angle_of_incidences, polarization)

    # Return the calculated results, which include transmission and reflection coefficients.
    return result

def tmm(material_list: List[str],
        thickness_list: ArrayLike,
        wavelength_arr: ArrayLike,
        angle_of_incidences: ArrayLike,
        coherency_list: ArrayLike = None,
        polarization: Text = 's') -> Tuple[Array, Array]:
    """
    Perform the Transfer Matrix Method (TMM) for multilayer thin films.

    Arguments:
    ----------
        material_list (List[str]): A list of material names. Each material is identified by a string.
        thickness_list (ArrayLike): An array of thicknesses corresponding to each layer.
        wavelength_arr (ArrayLike): An array of wavelengths over which to perform the simulation.
        angle_of_incidences (ArrayLike): An array of angles of incidence.
        coherency_list (ArrayLike): An array that specifies the coherency (whether the layer is coherent or incoherent).
        polarization (Text): The type of polarization ('s' for s-polarized or 'p' for p-polarized).


    Returns:
    ----------
        Tuple[ArrayLike, ArrayLike]: A tuple containing two arrays: 
            - The first array represents the reflection coefficients.
            - The second array represents the transmission coefficients.
            Both arrays have indices corresponding to angle of incidence (first index) and wavelength (second index).
    """

    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------

    # --- material_list ---
    if not isinstance(material_list, list):
        raise TypeError("material_list must be a list of material names")
    if not material_list:
        raise ValueError("material_list cannot be empty")
    if not all(isinstance(m, str) for m in material_list):
        raise TypeError("all elements of material_list must be strings")

    # --- thickness_list ---
    if not isinstance(thickness_list, jnp.ndarray):
        raise TypeError("thickness_list must be a jax array")
    if thickness_list.shape[0] != (len(material_list)-2):
        raise ValueError("thickness_list length must match material_list length minus two")
    if np.any(thickness_list <= 0):
        raise ValueError("all thickness values must be positive and nonzero")

    # --- wavelength_arr ---
    if not isinstance(wavelength_arr, jnp.ndarray):
        raise TypeError("wavelength_arr must be a jax array")
    if np.any(wavelength_arr <= 0):
        raise ValueError("wavelength values must be positive")

    # --- angle_of_incidences ---
    if not isinstance(angle_of_incidences, jnp.ndarray):
        raise TypeError("angle_of_incidences must be a jax array")
    if np.any((angle_of_incidences < 0) | (angle_of_incidences > (jnp.pi/2))):
        raise ValueError("angles of incidence must be between 0 degree and 90 degree")

    # --- coherency_list ---
    if coherency_list is not None:
        if not isinstance(coherency_list, jnp.ndarray):
            raise TypeError("coherency_list must be a jnp.ndarray")
        if len(coherency_list) != len(thickness_list):
            raise ValueError("coherency list must have the same length as thickness list")
        if not all(x in [0, 1] for x in coherency_list.tolist()):
            raise ValueError("coherency_list can only contain 0 and 1")

    # --- polarization ---
    if polarization not in ('s', 'p'):
        raise ValueError("polarization must be s or p")

    # Convert the material list into a set and a material distribution array
    # The material set contains unique materials, and the distribution describes how materials are layered.
    material_set, material_distribution = material_distribution_to_set(material_list)

    # Create the required material data, such as refractive index and extinction coefficient, for each material in the set.
    # This function retrieves optical properties needed for TMM calculations.
    data = create_data(material_set)

    # Check the polarization input and convert it to a boolean JAX array.
    if polarization == 's':
        # For s-polarization, set the boolean flag to `False`.
        polarization = jnp.array([False], dtype=bool)
    elif polarization == 'p':
        # For p-polarization, set the boolean flag to `True`.
        polarization = jnp.array([True], dtype=bool)

    if coherency_list == None:
        # If the multilayer structure is fully coherent, use the vectorized coherent TMM function.
        result = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelength_arr, angle_of_incidences, polarization)
        # Return the result (tuple of reflection and transmission coefficients).
        return result

    # Identify coherent and incoherent layer indices for the forward direction.
    coherent_groups_forward, incoherent_indices_forward, coherency_indices_forward = find_coh_and_incoh_indices(coherency_list)
    
    # Identify coherent and incoherent layer indices for the backward direction (reverse of the coherency list).
    coherent_groups_backward, _, coherency_indices_backward = find_coh_and_incoh_indices(jnp.flip(coherency_list))

    # Check if the multilayer film is fully coherent (i.e., only 2 incoherent layers).
    if (len(incoherent_indices_forward) == 2):
        # If the multilayer structure is fully coherent, use the vectorized coherent TMM function.
        result = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelength_arr, angle_of_incidences, polarization)
        # Return the result (tuple of reflection and transmission coefficients).
        return result
    else:
        # If the multilayer structure is not fully coherent, use the vectorized incoherent TMM function.
        result = vectorized_incoh_tmm(data = data,
                                      material_distribution = material_distribution,
                                      thickness_list = thickness_list,
                                      coherent_layer_indices_forward = coherent_groups_forward,
                                      incoherent_layer_indices_forward = incoherent_indices_forward,
                                      coherency_indices_forward = coherency_indices_forward,
                                      coherent_layer_indices_backward = coherent_groups_backward,
                                      coherency_indices_backward = coherency_indices_backward,
                                      wavelengths = wavelength_arr,
                                      angle_of_incidences = angle_of_incidences,
                                      polarization = polarization)
        # Return the result (tuple of reflection and transmission coefficients).
        return result