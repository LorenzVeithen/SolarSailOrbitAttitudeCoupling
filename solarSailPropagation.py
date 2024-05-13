import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
from constants import *

from MiscFunctions import set_axes_equal, axisRotation
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime


# Define solar sail
# Boom points
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0, 0],
                   [boom_length, 0, 0],
                   [0, boom_length, 0],
                   [0, boom_attachment_point, 0]])

panel2 = np.array([[0, -boom_attachment_point, 0],
                    [0, -boom_length, 0],
                    [boom_length, 0, 0],
                    [boom_attachment_point, 0, 0]])

panel3 = np.array([[-boom_attachment_point, 0, 0],
                   [-boom_length, 0, 0],
                   [0, -boom_length, 0],
                   [0, -boom_attachment_point, 0]])

panel4 = np.array([[0, boom_attachment_point, 0],
                    [0, boom_length, 0],
                    [-boom_length, 0, 0],
                    [-boom_attachment_point, 0, 0]])

wings_coordinates_list = [panel1, panel2, panel3, panel4]
vanes_coordinates_list = []
wings_optical_properties = [np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])] * 4
vanes_optical_properties = []

acs_object = sail_attitude_control_systems("None", boom_list)
sail = sail_craft("ACS3",
                  len(wings_coordinates_list),
                  len(vanes_coordinates_list),
                  wings_coordinates_list,
                  vanes_coordinates_list,
                  wings_optical_properties,
                  vanes_optical_properties,
                  sail_I,
                  sail_mass,
                  sail_mass_without_wings,
                  sail_nominal_CoM,
                  sail_material_areal_density,
                  sail_material_areal_density,
                  acs_object)

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
simulation_end_epoch = DateTime(2024, 6, 1, 9).epoch()

# Initial states
initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=398600441500000.0,
    semi_major_axis=a_0,
    eccentricity=e_0,
    inclination=i_0,
    argument_of_periapsis=w_0,
    longitude_of_ascending_node=raan_0,
    true_anomaly=theta_0,
)
initial_rotational_state = np.concatenate((rotation_matrix_to_quaternion_entries(np.eye(3)), np.array([0., 0., 0.])))

sailProp = sailCoupledDynamicsProblem(sail,
               initial_translational_state,
               initial_rotational_state,
               simulation_start_epoch,
               simulation_end_epoch)
dependent_variables = sailProp.define_dependent_variables()
bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
termination_settings, integrator_settings = sailProp.define_numerical_environment()
acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, vehicle_target_settings)
combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models, torque_models, dependent_variables)
t0 = time.time()
states_array, dependent_variable_array = sailProp.run_sim(bodies, combined_propagator_settings)
t1 = time.time()

print(t1-t0)
## Plot Kepler elements as a function of time
time_hours = (dependent_variable_array[:,0] - dependent_variable_array[0,0])/3600
kepler_elements = dependent_variable_array[:,1:7]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()

## Shadow parameter as a function of time, to check whether eclipses indeed take place or not and solar pressure acceleration norm
acc_vec_SRP = dependent_variable_array[:, 8:11]
acc_norm = np.sqrt(dependent_variable_array[:, 8]**2 + dependent_variable_array[:, 9]**2 + dependent_variable_array[:, 10]**2)
plt.figure()
plt.plot(time_hours, dependent_variable_array[:, 7], label="Shadow factor")
plt.plot(time_hours, acc_norm/acc0, label="Normalised SRP acceleration norm")
plt.legend()
plt.grid(True)

## Compare the relative position of the Sun and of the SRP acceleration
r_sc_sun = dependent_variable_array[:, 12:15]
r_sc_sun_norm = np.sqrt(r_sc_sun[:, 0]**2 + r_sc_sun[:, 1]**2 + r_sc_sun[:, 2]**2)
plt.figure()
plt.grid(True)
normalised_acc_vec_SRP = acc_vec_SRP[:, 0]
normalised_acc_vec_SRP[acc_norm != 0] = normalised_acc_vec_SRP[acc_norm != 0]/acc_norm[acc_norm != 0]
normalised_acc_vec_SRP[acc_norm == 0] = 0

plt.plot(time_hours, normalised_acc_vec_SRP, label=r'$a_{SRP}$ Inertial X-component')
plt.plot(time_hours, r_sc_sun[:, 0]/r_sc_sun_norm, label=r"$r_{s\odot}$ Inertial X-component")
plt.legend()

## Plot the positional state history
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'ACS3 trajectory around Earth')
points = np.array([states_array[:, 1], states_array[:, 2], states_array[:, 3]]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

cmap=plt.get_cmap('coolwarm')
colors=cmap(dependent_variable_array[:, 7])

for ii in range(len(dependent_variable_array[:, 7])-1):
    segii = segments[ii]
    lii, = ax.plot(segii[:, 0], segii[:, 1], segii[:, 2], color=colors[ii], linewidth=2)

    lii.set_solid_capstyle('round')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = R_E * np.cos(u)*np.sin(v)
y = R_E * np.sin(u)*np.sin(v)
z = R_E * np.cos(v)
ax.plot_wireframe(x, y, z, color="r")
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
set_axes_equal(ax)
#print(dependent_variable_array[:, -4])



# 313 rotation

psi_list = dependent_variable_array[:, -1]
theta_list = dependent_variable_array[:, -2]
phi_list = dependent_variable_array[:, -3]

plt.figure()
plt.plot(time_hours, psi_list, label="psi")
plt.plot(time_hours, theta_list, label="theta")
plt.plot(time_hours, phi_list, label="phi")
plt.legend()
plt.show()
'''
M_list = []
for i in range(psi_list):
    M = axisRotation(-phi_list[i], 3) * axisRotation(-theta_list[i], 1) * axisRotation(-psi_list[i], 3)
    M_list.append(M)

M0 = M_list[0]
Bx, By, Bz = zip(*M)
'''