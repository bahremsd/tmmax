from functools import lru_cache # Importing lru_cache to cache the function results
import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import jit, device_put # Import JAX functions for JIT compilation
import numpy as np # Importing numpy lib for savetxt function for saving arrays to csv files
import os # Importing os to handle file paths
import pandas as pd # Importing pandas to handle CSV data
from typing import Union, Callable # Type hints for function signatures
import warnings # Importing the warnings module to handle warnings in the code

@lru_cache(maxsize=32)
def load_nk_data(material_name: str = '') -> Union[jnp.ndarray, None]:
    """
    Load the refractive index (n) and extinction coefficient (k) data for a given material: (n + 1j * k).

    This function fetches wavelength-dependent refractive index (n) and extinction coefficient (k) 
    data for a specified material. The data is read from a CSV file located in the 'nk_data/' directory. 
    The CSV file should be named after the material, e.g., 'Si.csv', and include three columns: wavelength (in micrometers), 
    refractive index (n), and extinction coefficient (k). These parameters are crucial for optical simulations, 
    allowing the user to work with materials' optical properties over a range of wavelengths.

    Args:
        material_name (str): The name of the material for which the data is to be loaded. 
                             This must not be an empty string, and the corresponding CSV file 
                             must exist in the 'nk_data/' directory.


    Raises:
        ValueError: If the material name is an empty string.
        FileNotFoundError: If the file for the given material does not exist in the 'nk_data/' folder.
        IOError: If there's an issue reading or parsing the file.
    """
    # Check that the material name is not an empty string
    if not material_name:  
        raise ValueError("Material name cannot be an empty string.")  # Raise an error if no material is provided

    # Construct the file path and check if the file exists
    file_path = os.getcwd() + "/" + os.path.join('nk_data', f'{material_name}.csv')  # Create the full path to the file
    if not os.path.exists(file_path):  
        # Raise an error if the file for the material does not exist
        raise FileNotFoundError(f"No data found for material '{material_name}' in 'nk_data/' folder (library database).")
    
    # Load the data from the CSV file
    try:
        # Load the CSV data as a JAX array (important for using JAX's functionality, like automatic differentiation)
        data = jnp.asarray(pd.read_csv(file_path, skiprows=1, header=None).values)  
    except Exception as e:
        # If an error occurs during file reading or conversion, raise an IOError
        raise IOError(f"An error occurred while loading data for '{material_name}': {e}")
    
    # Check if the file is empty or doesn't contain valid data
    if data.size == 0:  
        # Raise an error if the data array is empty or incorrectly formatted
        raise ValueError(f"The file for material '{material_name}' is empty or not in the expected format.")
    
    return data  # Return the loaded data as a JAX array
