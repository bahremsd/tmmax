from typing import Union, Tuple
import jax.numpy as jnp

def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], 
               _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], 
               _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function calculates the Fresnel reflection (r_s) and transmission (t_s) coefficients 
    for s-polarized light (electric field perpendicular to the plane of incidence) at the interface 
    between two materials. The inputs are the refractive indices and the angles of incidence and 
    refraction for the two layers.



    """
    
    # Calculate the reflection coefficient for s-polarized light using Fresnel's equations.
    # The formula: r_s = (n1 * cos(theta1) - n2 * cos(theta2)) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is reflected at the interface.
    r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Calculate the transmission coefficient for s-polarized light using Fresnel's equations.
    # The formula: t_s = 2 * n1 * cos(theta1) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is transmitted through the interface.
    t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Return the reflection and transmission coefficients as a JAX array
    return jnp.array([r_s, t_s])
