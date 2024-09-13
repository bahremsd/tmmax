import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations


def _matmul(carry, phase_t_r):
    """
    Multiplies two complex matrices in a sequence.

    Args:
        carry (jax.numpy.ndarray): The accumulated product of the matrices so far.
                                   This is expected to be a 2x2 complex matrix.
        phase_t_r (jax.numpy.ndarray): A 3-element array where:
            - phase_t_r[0] represents the phase shift delta (a scalar).
            - phase_t_r[1] represents the transmission coefficient t or T (a scalar).
            - phase_t_r[2] represents the reflection coefficient r or R (a scalar).


    """
    # Create the diagonal phase matrix based on phase_t_r[0]
    # This matrix introduces a phase shift based on the delta value
    phase_matrix = jnp.array([[jnp.exp(-1j * phase_t_r[0]), 0],  # Matrix with phase shift for the first entry
                              [0, jnp.exp(1j * phase_t_r[0])]])  # Matrix with phase shift for the second entry

    # Create the matrix based on phase_t_r[1] and phase_t_r[2]
    # This matrix incorporates the transmission and reflection coefficients
    transmission_reflection_matrix = jnp.array([[1, phase_t_r[1]],  # Top row with transmission coefficient
                                               [phase_t_r[1], 1]])  # Bottom row with transmission coefficient

    # Compute the current matrix by multiplying the phase_matrix with the transmission_reflection_matrix
    # The multiplication is scaled by 1/phase_t_r[2] to account for the reflection coefficient
    mat = jnp.array(1 / phase_t_r[2]) * jnp.dot(phase_matrix, transmission_reflection_matrix)

    # Multiply the accumulated carry matrix with the current matrix
    # This updates the product with the new matrix
    result = jnp.dot(carry, mat)

    return result, None  # Return the updated matrix and None as a placeholder for jax.lax.scan
