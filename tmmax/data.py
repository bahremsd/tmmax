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

    Returns:
        jnp.ndarray: A 2D array containing the wavelength (first column), 
                     refractive index (n) (second column), and extinction coefficient (k) (third column).
                     Each row corresponds to a different wavelength.
                     
        None: If the function fails due to any raised exception or if the CSV file is empty, 
              it will return None.

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


def interpolate_1d(x: jnp.ndarray, y: jnp.ndarray) -> Callable[[float], float]:
    """
    Creates a 1D linear interpolation function based on the provided x and y arrays.
    
    This function returns a callable that performs linear interpolation on the input data points (x, y).
    Given an x value, it finds the corresponding y value by assuming a straight line between two closest points 
    in the x array and using the equation of the line. 
    
    Args:
        x (jnp.ndarray): Array of x values (independent variable). It must be sorted in ascending order.
        y (jnp.ndarray): Array of y values (dependent variable). It should have the same length as the x array.
    
    Returns:
        Callable[[float], float]: A function that, when provided with a single float x value, returns the corresponding
        interpolated float y value based on the linear interpolation.
    """
    
    @jit  # Just-In-Time compilation using JAX, speeds up the execution by compiling the function once.
    def interpolate(x_val: float) -> float:
        # Find the index where x_val would fit in x to maintain the sorted order
        idx = jnp.searchsorted(x, x_val, side='right') - 1
        # Ensure idx is within valid bounds (0 to len(x)-2) to avoid out-of-bounds errors
        idx = jnp.clip(idx, 0, x.shape[0] - 2)
        
        # Retrieve the two nearest x values, x_i and x_{i+1}, that surround x_val
        x_i, x_ip1 = x[idx], x[idx + 1]
        # Retrieve the corresponding y values, y_i and y_{i+1}, at those x positions
        y_i, y_ip1 = y[idx], y[idx + 1]
        
        # Calculate the slope of the line between (x_i, y_i) and (x_{i+1}, y_{i+1})
        slope = (y_ip1 - y_i) / (x_ip1 - x_i)
        
        # Interpolate the y value using the slope formula: y = y_i + slope * (x_val - x_i)
        return y_i + slope * (x_val - x_i)

    return interpolate  # Return the interpolation function to be used later


def interpolate_nk(material_name: str) -> Callable[[float], complex]:
    """
    Load the nk data for a given material and return a callable function that computes
    the complex refractive index for any wavelength.

    Args:
        material_name (str): Name of the material to load the nk data for.

    Returns:
        Callable[[float], complex]: A function that takes a wavelength (in meters) and 
                                    returns the complex refractive index.
    """
    nk_data = load_nk_data(material_name)  # Load the nk data for the specified material
    wavelength, refractive_index, extinction_coefficient = nk_data.T  # Transpose to get columns as variables

    # Interpolate refractive index and extinction coefficient
    compute_refractive_index = interpolate_1d(wavelength * 1e-6, refractive_index)  # Convert wavelength to meters for interpolation
    compute_extinction_coefficient = interpolate_1d(wavelength * 1e-6, extinction_coefficient)  # Convert wavelength to meters for interpolation

    @jit  # Just-in-time compile the function to optimize performance
    def compute_nk(wavelength: float) -> complex:
        """
        Compute the complex refractive index for a given wavelength.

        Args:
            wavelength (float): Wavelength in meters.

        Returns:
            complex: The complex refractive index, n + i*k, where n is the refractive index 
                     and k is the extinction coefficient.
        """
        n = compute_refractive_index(wavelength)  # Get the refractive index at the given wavelength
        k = compute_extinction_coefficient(wavelength)  # Get the extinction coefficient at the given wavelength
        return jnp.array(n + 1j * k)  # Combine n and k into a complex number and return

    return compute_nk  # Return the function that computes the complex refractive index


def add_material_to_nk_database(wavelength_arr, refractive_index_arr, extinction_coeff_arr, material_name=''):
    """
    Add material properties to the nk database by saving the data into a CSV file.

    This function validates and saves material properties such as wavelength, refractive index,
    and extinction coefficient into a CSV file. The file is named based on the provided material name.


    """
    
    # Validate input types
    # Check if all input arrays are of type jax.numpy.ndarray
    if not all(isinstance(arr, jnp.ndarray) for arr in [wavelength_arr, refractive_index_arr, extinction_coeff_arr]):
        raise TypeError("All input arrays must be of type jax.numpy.ndarray")

    # Ensure all arrays have the same length
    # Check if the length of refractive_index_arr and extinction_coeff_arr match wavelength_arr
    if not all(len(arr) == len(wavelength_arr) for arr in [refractive_index_arr, extinction_coeff_arr]):
        raise ValueError("All input arrays must have the same length")

    # Validate material name
    # Ensure that the material name is not an empty string
    if not material_name.strip():
        raise ValueError("Material name cannot be an empty string")

    # Check for extinction coefficients greater than 20
    # Warn and threshold extinction coefficients greater than 20 to 20
    if jnp.any(extinction_coeff_arr > 20):
        warnings.warn("Extinction coefficient being greater than 20 indicates that the material is almost opaque. "
                      "In the Transfer Matrix Method, to avoid the coefficients going to 0 and the gradient being zero, "
                      "extinction coefficients greater than 20 have been thresholded to 20.", UserWarning)
        extinction_coeff_arr = jnp.where(extinction_coeff_arr > 20, 20, extinction_coeff_arr)

    # Ensure the data is on the correct device
    # Move arrays to the appropriate device (e.g., GPU) for processing
    wavelength_arr, refractive_index_arr, extinction_coeff_arr = map(device_put, [wavelength_arr, refractive_index_arr, extinction_coeff_arr])

    # Combine the arrays into a single 2D array
    # Stack arrays as columns into a 2D array for saving
    data = jnp.column_stack((wavelength_arr, refractive_index_arr, extinction_coeff_arr))

    # Construct the file path
    # Create a file path for saving the data based on the material name
    path = os.path.join('nk_data', f'{material_name}.csv')
    
    # Save the file with a header
    # Convert the jax.numpy array to a numpy array for file saving and write to CSV
    np.savetxt(path, np.asarray(data), delimiter=',', header='wavelength_in_um,n,k', comments='')
    
    # Provide feedback on file creation
    # Inform the user whether the file was created or recreated successfully
    print(f"'{os.path.basename(path)}' {'recreated' if os.path.exists(path) else 'created'} successfully.")