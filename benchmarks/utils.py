import numpy as np

from tmm import coh_tmm
from vtmm import tmm_rt
from tmmax.data import interpolate_nk

from typing import List, Callable


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


def tmm_coh_tmm_array(polarization, material_list, thickness_list, angle_of_incidences, wavelength_arr):
    material_set = list(set(material_list))
    material_enum = {material: i for i, material in enumerate(material_set)}
    material_list = [int(material_enum[material]) for material in material_list]
    nk_funkcs = {i: interpolate_nk(material) for i, material in enumerate(material_set)}
    thickness_list = np.concatenate(([np.inf], thickness_list, [np.inf]), axis=None)
    R = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    T = np.zeros((len(wavelength_arr), len(angle_of_incidences)), dtype=np.float64)
    
    for i in range(len(wavelength_arr)):
        for j in range(len(angle_of_incidences)):
            nk_list = get_nk_values(wavelength_arr[i], nk_funkcs, material_list)
            result = coh_tmm(polarization, nk_list, thickness_list, angle_of_incidences[j], wavelength_arr[i])
            R[i,j] = result['R']
            T[i,j] = result['T']

    return R,T
