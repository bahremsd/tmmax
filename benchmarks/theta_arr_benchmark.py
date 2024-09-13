import numpy as np
import jax.numpy as jnp
import timeit
import matplotlib.pyplot as plt

# Import necessary functions for simulation
from utils import generate_material_distribution_indices
from utils import generate_material_list_with_air
from utils import tmm_coh_tmm_array
from utils import vtmm_tmm_rt_wl_theta
from tmmax import tmm

# Define the array of lengths for the angle of incidence array
angle_arr_lengths = np.arange(1, 100, 10, dtype=int)

# Number of repetitions for the timeit function
timeit_repetition = 100

# List of materials to be used in the thin-film stack
material_set = ["SiO2", "TiO2", "MgF2", "MgO", "SiO", "Al2O3", "CdS"]

# Polarization type for the simulation ('s' or 'p')
polarization = "s"

# Wavelength array spanning from 500 nm to 1000 nm (in 10 steps)
wavelength_arr = jnp.linspace(500e-9, 1000e-9, 10)

# Generate random material distribution indices and corresponding material list (with air as the boundary layers)
indices = generate_material_distribution_indices(5, low=0, high=len(material_set))
material_list = generate_material_list_with_air(indices, material_set)

# Save the material list to a numpy file
np.save("material_distribution_with_layer_num_5_angle_arr_exp.npy", material_list)

# Generate random thicknesses for the 5-layer thin film, with thickness between 100 nm and 500 nm
thickness_list = np.random.uniform(100, 500, 5) * 1e-9

# Save the thickness list to a numpy file
np.save("thickness_list_with_layer_num_5_angle_arr_exp.npy", thickness_list)

# Initialize lists to store execution times for different TMM methods
time_tmm = []
time_vtmm = []
time_tmmax = []

# Loop through each angle array length and time the TMM computations
for N in angle_arr_lengths:
    # Generate angle of incidences array with N points between 0 and pi/2
    angle_of_incidences = np.linspace(0, np.pi/2, N)

    # Time the coherent TMM array calculation and store the result
    t_tmm = timeit.timeit(
        lambda: tmm_coh_tmm_array(polarization, material_list, thickness_list, angle_of_incidences, wavelength_arr), 
        number=timeit_repetition
    )
    
    # Time the vtmm_tmm_rt_wl_theta calculation and store the result
    t_vtmm = timeit.timeit(
        lambda: vtmm_tmm_rt_wl_theta(polarization, wavelength_arr, angle_of_incidences, material_list, thickness_list), 
        number=timeit_repetition
    )
    
    # Time the TMM computation using the tmmax library and store the result
    t_tmmax = timeit.timeit(
        lambda: tmm(material_list=material_list, thickness_list=thickness_list, wavelength_arr=wavelength_arr, angle_of_incidences=angle_of_incidences, polarization=polarization), 
        number=timeit_repetition
    )

    # Append the times to respective lists
    time_tmm.append(t_tmm)
    time_vtmm.append(t_vtmm)
    time_tmmax.append(t_tmmax)

# Save the execution times to numpy files for further analysis
np.save("time_of_tmm_angle_arr_exp.npy", time_tmm)
np.save("time_of_vtmm_angle_arr_exp.npy", time_vtmm)
np.save("time_of_tmmax_angle_arr_exp.npy", time_tmmax)

# ================== PLOTTING RESULTS ==================

# Convert the lists to numpy arrays for easier manipulation
time_tmm = np.array(time_tmm)
time_vtmm = np.array(time_vtmm)
time_tmmax = np.array(time_tmmax)

# Plot the execution times against the angle array lengths
plt.figure(figsize=(10, 6))

# Plot each method's timing with different markers
plt.plot(angle_arr_lengths, time_tmm, 'o-', label="TMM Coherent", color="blue")
plt.plot(angle_arr_lengths, time_vtmm, 's-', label="VTMM", color="green")
plt.plot(angle_arr_lengths, time_tmmax, '^-', label="TMMax", color="red")

# Labeling the plot
plt.xlabel("Angle of Incidence Array Length", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.title("Execution Time vs Angle of Incidence Array Length", fontsize=16)

# Adding a legend to distinguish the methods
plt.legend(loc="upper left")

# Save the plot to a file
plt.savefig("tmm_execution_time_vs_angle_length.png")

