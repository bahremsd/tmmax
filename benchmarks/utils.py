import numpy as np

from tmm import coh_tmm
from vtmm import tmm_rt
from tmmax.data import interpolate_nk

from typing import List, Callable


speed_of_light = 299792458 # m/s

def get_nk_values(wl: float, nk_functions: List[Callable[[float], complex]], material_list: List[int]) -> np.ndarray:

    return np.array([nk_functions[mat_idx](wl) for mat_idx in material_list])  # Convert the resulting list to a NumPy array.
