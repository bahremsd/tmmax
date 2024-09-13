import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations


def _matmul(carry, phase_t_r):
    phase_matrix = jnp.array([[jnp.exp(-1j * phase_t_r[0]), 0], 
                              [0, jnp.exp(1j * phase_t_r[0])]]) 

    transmission_reflection_matrix = jnp.array([[1, phase_t_r[1]],  
                                               [phase_t_r[1], 1]])
    mat = jnp.array(1 / phase_t_r[2]) * jnp.dot(phase_matrix, transmission_reflection_matrix)

    result = jnp.dot(carry, mat)

    return result, None  
