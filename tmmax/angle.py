import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
import sys # Import sys for EPSILON
from typing import Union # Type hints for function signatures

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon

def is_propagating_wave(n: Union[float, jnp.ndarray], angle_of_incidence: Union[float, jnp.ndarray], polarization: bool) -> Union[float, jnp.ndarray]:


    n_cos_theta = n * jnp.cos(angle_of_incidence)  

    def define_is_forward_if_bigger_than_eps(_):

        is_forward_s = jnp.imag(n_cos_theta) 
        is_forward_p = is_forward_s  
        return is_forward_s, is_forward_p  

    def define_is_forward_if_smaller_than_eps(_):

        is_forward_s = jnp.real(n_cos_theta)  


        n_cos_theta_star = n * jnp.cos(jnp.conj(angle_of_incidence)) 
        is_forward_p = jnp.real(n_cos_theta_star)  

        return is_forward_s, is_forward_p 


    condition = jnp.abs(jnp.imag(n_cos_theta)) > EPSILON * 1e2 
    
    is_forward_s, is_forward_p = jax.lax.cond(
        condition, 
        define_is_forward_if_bigger_than_eps,  
        define_is_forward_if_smaller_than_eps, 
        None
    )


    if polarization is False:

        return jnp.array([is_forward_s]) 
    elif polarization is True:

        return jnp.array([is_forward_p])  
