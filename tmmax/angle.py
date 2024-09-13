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
