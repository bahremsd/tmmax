from typing import Union, Tuple
import jax.numpy as jnp

def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], 
               _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], 
               _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:


    r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    

    t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))

    return jnp.array([r_s, t_s])
