import numpy as np

from tmm import coh_tmm
from vtmm import tmm_rt
from tmmax.data import interpolate_nk

from typing import List, Callable


speed_of_light = 299792458 # m/s