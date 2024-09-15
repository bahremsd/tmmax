import numpy as np
import jax.numpy as jnp
import timeit
import matplotlib.pyplot as plt

# Importing utility functions for material and simulation setups
from utils import generate_material_distribution_indices
from utils import generate_material_list_with_air

from utils import tmm_coh_tmm_array
from utils import vtmm_tmm_rt_wl_theta
from tmmax.tmm import tmm

# Array for wavelength array lengths (number of wavelengths), ranging from 1 to 100, step size of 10
wl_arr_lengths = np.arange(1, 100, 10, dtype=int)

# Number of times each function is timed to get an average performance metric
timeit_repetition = 100

# Set of materials used for the simulation
material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]
polarization = "s"  # s-polarization (TE polarization)
angle_of_incidences = np.linspace(0, np.pi/2, 10)  # Incidence angles from 0 to 90 degrees

# Generate random material distribution indices and create a material list with air layers
indices = generate_material_distribution_indices(5, low=0, high=len(material_set))
material_list = generate_material_list_with_air(indices, material_set)

# Save the material distribution for later use
np.save("material_distribution_with_layer_num_5_wl_arr_exp.npy", material_list)

# Randomly generate a list of thicknesses for the layers, converted to meters
thickness_list = np.random.uniform(100, 500, 5) * 1e-9
np.save("thickness_list_with_layer_num_5_wl_arr_exp.npy", thickness_list)

# Lists to store execution times for each method
time_tmm = []
time_vtmm = []
time_tmmax = []

# Loop through different wavelength array lengths
for N in wl_arr_lengths:
    # Create a wavelength array from 500 nm to 1000 nm with N points
    wavelength_arr = jnp.linspace(500e-9, 1000e-9, N)
    
    # Measure the time for the TMM calculation with coherent layers
    t_tmm = timeit.timeit(lambda: tmm_coh_tmm_array(polarization, material_list, thickness_list, angle_of_incidences, wavelength_arr), 
                          number=timeit_repetition)
    
    # Measure the time for the VTMM calculation
    t_vtmm = timeit.timeit(lambda: vtmm_tmm_rt_wl_theta(polarization, wavelength_arr, angle_of_incidences, material_list, thickness_list), 
                           number=timeit_repetition)
    
    # Measure the time for the TMMax (a custom method, assumed similar to TMM) calculation
    t_tmmax = timeit.timeit(lambda: tmm(material_list=material_list, thickness_list=thickness_list, wavelength_arr=wavelength_arr, 
                                        angle_of_incidences=angle_of_incidences, polarization=polarization), 
                            number=timeit_repetition)
    
    # Append the measured times to the respective lists
    time_tmm.append(t_tmm)
    time_vtmm.append(t_vtmm)
    time_tmmax.append(t_tmmax)

# Save the measured times for further analysis
np.save("time_of_tmm_wl_arr_exp.npy", time_tmm)
np.save("time_of_vtmm_wl_arr_exp.npy", time_vtmm)
np.save("time_of_tmmax_wl_arr_exp.npy", time_tmmax)

# Plotting the execution times as a function of wavelength array lengths
plt.figure(figsize=(10, 6))
plt.plot(wl_arr_lengths, time_tmm, label="TMM", marker='o')
plt.plot(wl_arr_lengths, time_vtmm, label="VTMM", marker='s')
plt.plot(wl_arr_lengths, time_tmmax, label="TMMax", marker='^')

# Adding labels and title
plt.xlabel("Wavelength Array Length (N)")
plt.ylabel("Time (seconds)")
plt.title("Execution Time vs Wavelength Array Length")
plt.legend()

# Save the plot as a PNG file
plt.savefig("execution_time_vs_wavelength_array_length.png")
plt.show()  # Display the plot
