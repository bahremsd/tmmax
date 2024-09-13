import numpy as np
import jax.numpy as jnp
import timeit
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Importing utility functions for material index generation and TMM methods
from utils import generate_material_distribution_indices
from utils import generate_material_list_with_air
from utils import tmm_coh_tmm_array
from utils import vtmm_tmm_rt_wl_theta
from tmmax import tmm  # Importing the TMM function from tmmax module

# Number of layers to test from 2 to 50 (inclusive), using integer values
number_of_layers = np.arange(2, 51, dtype=int)

# Number of repetitions for the timeit function to get an average execution time
timeit_repetition = 100

# List of materials used in the simulations
material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]

# Polarization type, 's' polarization in this case
polarization = "s"

# Array of angles of incidence, linearly spaced from 0 to π/2, with 20 points
angle_of_incidences = np.linspace(0, np.pi/2, 20)

# Wavelength array from 500 nm to 1000 nm, linearly spaced with 20 points
wavelength_arr = jnp.linspace(500e-9, 1000e-9, 20)

# Lists to store execution times for each method
time_tmm = []
time_vtmm = []
time_tmmax = []

# Loop through different numbers of layers (from 2 to 50)
for N in number_of_layers:
    # Generate random material distribution indices and the corresponding material list with air layers
    indices = generate_material_distribution_indices(N, low=0, high=len(material_set))
    material_list = generate_material_list_with_air(indices, material_set)
    
    # Save the material distribution list as a .npy file for reference
    np.save(f"material_distribution_with_layer_num_{N}.npy", material_list)
    
    # Randomly generate thicknesses for each layer between 100 nm and 500 nm
    thickness_list = np.random.uniform(100, 500, N) * 1e-9
    np.save(f"thickness_list_with_layer_num_{N}.npy", thickness_list)  # Save the thickness list

    # Measure execution time for the standard TMM method and append to the time list
    t_tmm = timeit.timeit(lambda: tmm_coh_tmm_array(polarization, material_list, thickness_list, 
                                                     angle_of_incidences, wavelength_arr), 
                          number=timeit_repetition)
    time_tmm.append(t_tmm)

    # Measure execution time for the VTMM method and append to the time list
    t_vtmm = timeit.timeit(lambda: vtmm_tmm_rt_wl_theta(polarization, wavelength_arr, 
                                                        angle_of_incidences, material_list, 
                                                        thickness_list), 
                           number=timeit_repetition)
    time_vtmm.append(t_vtmm)

    # Measure execution time for the TMMax method and append to the time list
    t_tmmax = timeit.timeit(lambda: tmm(material_list=material_list, thickness_list=thickness_list, 
                                        wavelength_arr=wavelength_arr, angle_of_incidences=angle_of_incidences, 
                                        polarization=polarization), 
                            number=timeit_repetition)
    time_tmmax.append(t_tmmax)

# Save the time measurements for each method into .npy files
np.save("time_of_tmm_layersize_exp.npy", time_tmm)
np.save("time_of_vtmm_layersize_exp.npy", time_vtmm)
np.save("time_of_tmmax_layersize_exp.npy", time_tmmax)

# Plotting the execution times for each method with respect to the number of layers
plt.figure(figsize=(10, 6))

# Plot times for each method
plt.plot(number_of_layers, time_tmm, label="TMM Time", marker="o")
plt.plot(number_of_layers, time_vtmm, label="VTMM Time", marker="s")
plt.plot(number_of_layers, time_tmmax, label="TMMax Time", marker="^")

# Adding labels and title
plt.xlabel("Number of Layers")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs. Number of Layers")

# Adding a legend to distinguish the methods
plt.legend()

# Save the plot as a .png file
plt.savefig("execution_time_vs_layers.png")