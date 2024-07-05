import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

import matplotlib.pyplot as plt

from constants import *
from generalConstants import AMS_directory, Project_directory
from MiscFunctions import quiver_data_to_segments, set_axes_equal
import numpy as np
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix


quiver_length = 0.3 * R_E
quiver_widths = 1


data_dir = Project_directory + '/0_GeneratedData/LTT_Data'
current_data_set = f'/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False'
target_combination = 'omega_x_30.0_omega_y_30.0_omega_z_0.0.dat'

#state_history_array = np.genfromtxt(data_dir + f'{current_data_set}/states_history/state_history_{target_combination}')
#dependent_variable_history_array = np.genfromtxt(data_dir + f'{current_data_set}/dependent_variable_history/dependent_variable_history_{target_combination}')
state_history_array = np.genfromtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_1_year.dat')
dependent_variable_history_array = np.genfromtxt('/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_1_year.dat')

t_days = (state_history_array[:, 0] - state_history_array[0, 0]) / (3600 * 24)

t_hours = (state_history_array[:, 0] - state_history_array[0, 0]) / 3600    # hours
x_J2000 = state_history_array[:, 1]
y_J2000 = state_history_array[:, 2]
z_J2000 = state_history_array[:, 3]
vx_J2000 = state_history_array[:, 4]
vy_J2000 = state_history_array[:, 5]
vz_J2000 = state_history_array[:, 6]
quaternions_inertial_to_body_fixed_vector = state_history_array[:, 7:11]
omega_x = state_history_array[:, 11]
omega_y = state_history_array[:, 12]
omega_z = state_history_array[:, 13]

# Extract dependent variables
t_dependent_variables_hours = (dependent_variable_history_array[:, 0]-dependent_variable_history_array[0, 0])/3600
keplerian_state = dependent_variable_history_array[:, 1:7]
received_irradiance_shadow_function = dependent_variable_history_array[:, 7]
spacecraft_srp_acceleration_vector = dependent_variable_history_array[:, 8:11]
spacecraft_srp_torque_vector = dependent_variable_history_array[:, 11:14]
spacecraft_sun_relative_position = dependent_variable_history_array[:, 14:17]
earth_sun_relative_position = dependent_variable_history_array[:, 17:20]
spacecraft_total_torque_norm = dependent_variable_history_array[:, 20]
vanes_x_rotations = np.rad2deg(dependent_variable_history_array[:, 21:25])  # Note: this might need to be changed; is there a way to make this automatic?
vanes_y_rotations = np.rad2deg(dependent_variable_history_array[:, 25:29])  # Note: this might need to be changed; is there a way to make this automatic?
optimal_torques= dependent_variable_history_array[:, 29:32]
vane_torques = dependent_variable_history_array[:, 32:35]

plt.figure()
plt.plot(t_days, dependent_variable_history_array[:, 1])
plt.xlabel("time")
plt.ylabel("semi-major axis")

plt.figure()
plt.plot(t_days, dependent_variable_history_array[:, 7])
plt.xlabel("time")
plt.ylabel("received irradiance")

fig = plt.figure()
ax_orbit = fig.add_subplot(projection='3d')

ax_orbit.scatter(x_J2000, y_J2000, z_J2000)
# earth representation
u_E, v_E = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x_E = R_E * np.cos(u_E) * np.sin(v_E)
y_E = R_E * np.sin(u_E) * np.sin(v_E)
z_E = R_E * np.cos(v_E)
ax_orbit.plot_wireframe(x_E, y_E, z_E, color="b", label="Earth")

# Real spacecraft orbit
previous_spacecraft_position = ax_orbit.plot([], [], [], c="b", alpha=0.3, label="Spacecraft position history")[0]
current_spacecraft_position = ax_orbit.scatter(x_J2000[0], y_J2000[0], z_J2000[0], c="k", label="Spacecraft")
ax_orbit.set_xlabel("X [m]")
ax_orbit.set_ylabel("Y [m]")
ax_orbit.set_zlabel("Z [m]")

# Sun rays
absolute_minimum = min([min(x_J2000), min(y_J2000), min(z_J2000)])
absolute_maximum = max([max(x_J2000), max(y_J2000), max(z_J2000)])
xgrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
ygrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
zgrid = np.linspace(absolute_minimum * 1.1, absolute_maximum * 1.1, 5)
Xg, Yg, Zg = np.meshgrid(xgrid, ygrid, zgrid)

# Define solar rays vector field
u = -np.ones_like(Xg) * earth_sun_relative_position[0, 0]  # Constant x-component
v = -np.ones_like(Yg) * earth_sun_relative_position[0, 1]  # Constant y-component
w = -np.ones_like(Zg) * earth_sun_relative_position[0, 2]  # Constant z-component

# Plot the sun rays and the sail normal vector
sun_rays = ax_orbit.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5,
                           linewidth=quiver_widths, length=quiver_length)
R_BI = quaternion_entries_to_rotation_matrix(quaternions_inertial_to_body_fixed_vector[0, :].T)
R_IB = np.linalg.inv(R_BI)
current_sail_normal = ax_orbit.quiver(x_J2000[0], y_J2000[0], z_J2000[0], R_IB[0, 2], R_IB[1, 2], R_IB[2, 2],
                                      color="k", normalize=True, linewidth=quiver_widths, length=quiver_length)
current_sail_srp_acceleration = ax_orbit.quiver(x_J2000[0], y_J2000[0], z_J2000[0],
                                                spacecraft_srp_acceleration_vector[0, 0],
                                                spacecraft_srp_acceleration_vector[0, 1],
                                                spacecraft_srp_acceleration_vector[0, 2],
                                                color="g", normalize=True, linewidth=quiver_widths,
                                                length=quiver_length, label="SRP acceleration")
set_axes_equal(ax_orbit)

plt.show()