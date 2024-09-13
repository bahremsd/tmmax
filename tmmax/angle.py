import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
import sys # Import sys for EPSILON
from typing import Union # Type hints for function signatures

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon

def is_propagating_wave(n: Union[float, jnp.ndarray], angle_of_incidence: Union[float, jnp.ndarray], polarization: bool) -> Union[float, jnp.ndarray]:
    """
    Determines whether a wave is forward-propagating through a multilayer thin film stack based on 
    the refractive index, angle of incidence, and polarization.

    
    Args:
    n (Union[float, jnp.ndarray]): Complex refractive index of the medium, which can be an array or scalar.
    angle_of_incidence (Union[float, jnp.ndarray]): Angle of incidence of the incoming wave in radians.
    polarization (bool): Polarization of the wave:
        - False for s-polarization (perpendicular to the plane of incidence).
        - True for p-polarization (parallel to the plane of incidence).

    
    Returns:
    Union[float, jnp.ndarray]: 
        - A positive value indicates forward propagation for both s and p polarizations.
        - A negative or zero value implies backward propagation or evanescent waves (non-propagating).

    
    The function evaluates whether the wave, given its angle, refractive index, and polarization, is a 
    forward-propagating wave (i.e., traveling from the front to the back of the stack). This is crucial 
    when calculating Snell's law in multilayer structures to ensure light is correctly entering or 
    exiting the stack.

    
    The check considers both real and complex values of the refractive index and angle, ensuring that the 
    light propagates within the correct angle range for physical interpretation.    
    """

    # Multiply the refractive index (n) by the cosine of the angle of incidence
    n_cos_theta = n * jnp.cos(angle_of_incidence)  # Compute n*cos(theta) for angle propagation

    def define_is_forward_if_bigger_than_eps(_):
        """Handle cases where the imaginary part of the refractive index is significant, i.e., 
        evanescent waves or lossy media."""
        is_forward_s = jnp.imag(n_cos_theta)  # Check if the wave decays exponentially in evanescent media
        is_forward_p = is_forward_s  # Both s- and p-polarizations have the same condition for evanescent decay
        return is_forward_s, is_forward_p  # Return the imaginary part for determining forward propagation

    def define_is_forward_if_smaller_than_eps(_):
        """Handle cases where the real part of the refractive index dominates, 
        indicating propagating waves."""
        is_forward_s = jnp.real(n_cos_theta)  # For s-polarization, check if Re[n * cos(theta)] > 0

        # For p-polarization, consider n * cos(theta*) where theta* is the complex conjugate of the angle
        n_cos_theta_star = n * jnp.cos(jnp.conj(angle_of_incidence))  # Calculate n * cos(conjugate(theta))
        is_forward_p = jnp.real(n_cos_theta_star)  # For p-polarization, check if Re[n * cos(theta*)] > 0

        return is_forward_s, is_forward_p  # Return real parts to determine forward propagation

    # Check whether the wave is evanescent or lossy by examining the imaginary part of n * cos(theta)
    condition = jnp.abs(jnp.imag(n_cos_theta)) > EPSILON * 1e2  # Set a threshold for significant imaginary part
    # Use conditional logic to handle different wave types based on whether the imaginary part is large
    is_forward_s, is_forward_p = jax.lax.cond(
        condition, 
        define_is_forward_if_bigger_than_eps,  # Handle evanescent/lossy cases
        define_is_forward_if_smaller_than_eps,  # Handle normal propagating waves
        None
    )

    # Return the result based on the polarization type
    if polarization is False:
        # For s-polarization, return whether the wave is forward-propagating
        return jnp.array([is_forward_s])  # s-polarization output as a single-element array
    elif polarization is True:
        # For p-polarization, return whether the wave is forward-propagating
        return jnp.array([is_forward_p])  # p-polarization output as a single-element array

def _compute_layer_angles_single_wl_angle_point(nk_list: jnp.ndarray,
                                                angle_of_incidence: Union[float, jnp.ndarray],
                                                wavelength: Union[float, jnp.ndarray],
                                                polarization: bool) -> jnp.ndarray:


    # Calculate the sine of the angles in the first layer using Snell's law
    # Here, we are computing sin(theta) for each layer using the ratio of the refractive index of the first layer 
    # to the refractive index of the current layer (Snell's Law).
    sin_theta = jnp.sin(angle_of_incidence) * nk_list[0] / nk_list  # Ratio ensures correct angle for each layer

    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is used here to calculate the inverse sine (arcsine) and is compatible with complex values if needed.
    theta_array = jnp.arcsin(sin_theta)  # Converts sin(theta) values back to theta (angle in radians)

    # Check if the wave is forward propagating or not by calculating its properties for the first and last layer.
    # is_propagating_wave returns a boolean array where True means the wave is propagating and False means evanescent.
    is_incoming_props = is_propagating_wave(nk_list[0], theta_array[0], polarization)  # First layer propagation check
    is_outgoing_props = is_propagating_wave(nk_list[-1], theta_array[-1], polarization)  # Last layer propagation check

    # If the wave is evanescent (non-propagating), update the angle by flipping it (subtracting from pi).
    def update_theta_arr_incoming(_):
        return theta_array.at[0].set(jnp.pi - theta_array[0])  # Flips the angle in the first layer if needed

    # Similarly for the outgoing wave in the last layer.
    def update_theta_arr_outgoing(_):
        return theta_array.at[-1].set(jnp.pi - theta_array[-1])  # Flips the angle in the last layer if needed

    # If the wave is propagating normally, return the theta_array unchanged.
    def return_unchanged_theta(_):
        return theta_array  # No angle flip if propagation is normal

    # Handle the evanescent and lossy cases by checking the incoming wave's properties.
    # If any wave in the first layer is non-propagating, the angle gets flipped.
    condition_incoming = jnp.any(is_incoming_props <= 0)  # Check if the incoming wave has an evanescent component
    condition_outgoing = jnp.any(is_outgoing_props <= 0)  # Check if the outgoing wave has an evanescent component

    # Conditionally update the theta_array based on whether the incoming wave is evanescent or not.
    # jax.lax.cond is used here to conditionally perform updates based on the given condition.
    theta_array = jax.lax.cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=None)  # Conditionally flip the angle for incoming wave
    theta_array = jax.lax.cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=None)  # Conditionally flip the angle for outgoing wave

    # Return the final angles of incidence (theta_array) for each layer, reflecting any necessary flips.
    return theta_array  # Final output: angles of incidence in each layer after applying Snell's law


