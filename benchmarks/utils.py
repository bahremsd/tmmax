import numpy as np

from tmm import coh_tmm
from tmmax.data import interpolate_nk

from typing import List, Callable, Tuple, Union


speed_of_light = 299792458 # m/s

def get_nk_values(wl: float, nk_functions: List[Callable[[float], complex]], material_list: List[int]) -> np.ndarray:
    """
    This function retrieves the refractive index and extinction coefficient values 
    (represented as complex numbers) for a given wavelength `wl` from a list of materials.
    Each material has a corresponding function in the `nk_functions` list that, 
    when provided with the wavelength, returns the refractive index (n) and extinction 
    coefficient (k) as a complex number (n + j*k). The materials are indexed 
    by `material_list` to access their corresponding functions.

    Args:
    wl (float): The wavelength at which to compute the refractive index and extinction coefficient.
    nk_functions (List[Callable[[float], complex]]): A list of functions, each corresponding to 
    a material's refractive index and extinction coefficient. These functions take the wavelength 
    as input and return a complex number (n + j*k).
    material_list (List[int]): A list of indices, where each index corresponds to a material, 
    and is used to retrieve the respective function from the `nk_functions` list.

    Returns:
    np.ndarray: An array of complex numbers where each entry corresponds to the refractive index (n) 
    and extinction coefficient (k) of a material at the given wavelength `wl`.

    """
    
    return np.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Convert the resulting list to a NumPy array.


def tmm_coh_tmm_array(
    polarization: str, 
    material_list: List[str], 
    thickness_list: Union[np.ndarray, float], 
    angle_of_incidences: Union[np.ndarray, float], 
    wavelength_arr: Union[np.ndarray, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function calculates the reflection (R) and transmission (T) for a multilayer thin film stack 
    over arrays of wavelengths and angles of incidence using the coherent Transfer Matrix Method (TMM). 
    The polarization of the incident light is considered either "s" (TE) or "p" (TM). The function 
    interpolates the refractive index (n) and extinction coefficient (k) for the materials at given 
    wavelengths and applies TMM over the material layers for each wavelength and angle of incidence.
    
    Args:
        polarization (str): Polarization of the incident light. Either "s" (TE) or "p" (TM).
        material_list (List[str]): List of material names for each layer.
        thickness_list (Union[np.ndarray, float]): Array of thicknesses for each layer in nanometers.
        angle_of_incidences (Union[np.ndarray, float]): Array of angles of incidence in degrees.
        wavelength_arr (Union[np.ndarray, float]): Array of wavelengths in nanometers.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays for reflection (R) and transmission (T). 
        The shape of these arrays will be (len(wavelength_arr), len(angle_of_incidences)).
    """

    # Create a set of unique materials to avoid redundant interpolation # The list(set(...)) ensures unique materials.
    material_set = list(set(material_list))  
    
    # Assign each unique material an enumerated integer value to map it efficiently later # A dictionary with material names as keys and unique indices as values.
    material_enum = {material: i for i, material in enumerate(material_set)}
    
    # Replace material names in the list with their enumerated integer index using the dictionary created # Converts material names in material_list to their corresponding integer identifiers.
    material_list = [int(material_enum[material]) for material in material_list]
    
    # Create a dictionary mapping the enumerated material indices to their interpolated nk functions # Prepares interpolation functions for each unique material's refractive index data.
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}
    
    # Extend the thickness list by adding infinite boundaries for air above and below the stack # np.inf ensures that the first and last "layers" are considered infinitely thick (air layers).
    thickness_list = np.concatenate(([np.inf], thickness_list, [np.inf]), axis=None)
    
    # Initialize empty arrays for storing reflection (R) and transmission (T) results # The result arrays have dimensions len(wavelength_arr) x len(angle_of_incidences).
    R = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    T = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    
    # Nested loops to compute R and T for each combination of wavelength and angle of incidence # Outer loop: iterating over wavelengths; Inner loop: iterating over angles of incidence.
    for i in range(len(wavelength_arr)):
        for j in range(len(angle_of_incidences)):
            
            # Retrieve the refractive index (n) and extinction coefficient (k) for each material at the current wavelength # nk_list contains n and k for all materials at wavelength_arr[i].
            nk_list = get_nk_values(wavelength_arr[i], nk_funkcs, material_list)
            
            # Perform the coherent TMM calculation using the polarization, nk_list, thicknesses, and current angle/wavelength # The result is a dictionary containing 'R' and 'T'.
            result = coh_tmm(polarization, nk_list, thickness_list, angle_of_incidences[j], wavelength_arr[i])
            
            # Store the calculated reflection (R) and transmission (T) in the result arrays # Assign reflection and transmission values to the corresponding index in R and T arrays.
            R[i,j] = result['R']
            T[i,j] = result['T']

    # Return the final reflection (R) and transmission (T) arrays for all wavelengths and angles # These are 2D arrays, where each element corresponds to a specific wavelength and angle.
    return R, T

def generate_material_distribution_indices(N, low=0, high=10):
    """
    Generates an array of random integers with length N such that
    no two consecutive elements are the same.

    Args:
        N (int): Length of the array.
        low (int): Minimum value of the integers (inclusive).
        high (int): Maximum value of the integers (exclusive).

    Returns:
        numpy.ndarray: Array of random integers with no consecutive elements the same.
    """
    if N <= 0:
        raise ValueError("Array length N must be positive.")
    
    arr = np.zeros(N, dtype=int)
    arr[0] = np.random.randint(low, high)
    
    for i in range(1, N):
        prev = arr[i-1]
        new_val = prev
        while new_val == prev:
            new_val = np.random.randint(low, high)
        arr[i] = new_val

    return arr

def generate_material_list_with_air(index_array, material_list):
    """
    Takes an array of indices and a material list, and generates a list of materials
    based on the index array, with 'Air' concatenated at the start and end.

    Args:
        index_array (numpy.ndarray): Array of integers representing indices.
        material_list (list of str): List of material names (strings).

    Returns:
        list of str: List of materials with 'Air' at the start and end.
    """
    if not all(0 <= idx < len(material_list) for idx in index_array):
        raise ValueError("Index out of bounds for the material list.")
    
    # Generate the list of materials
    material_sequence = [material_list[idx] for idx in index_array]
    
    # Concatenate "Air" at the start and end
    return ["Air"] + material_sequence + ["Air"]