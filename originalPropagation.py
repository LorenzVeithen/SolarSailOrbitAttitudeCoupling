import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")
# Load standard modules
import numpy as np
import matplotlib.pyplot as plt

from constants import *

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from MiscFunctions import set_axes_equal, axisRotation
from attitudeControllersClass import *

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# integrator settings
initial_step_size = 1.0
control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12)
validation_settings = propagation_setup.integrator.step_size_validation(1E-5, 1E3)

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
simulation_end_epoch = DateTime(2024, 6, 1, 9).epoch()

# Create default body settings for "Earth"
bodies_to_create = ["Earth", "Sun"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Add vehicle object to system of bodies
body_settings.add_empty_settings("ACS3")
#sail_rigid_prop = spacecraft_rigid_body_properties(systemOfBodies=bodies)
body_settings.get("ACS3").rigid_body_settings = environment_setup.rigid_body.custom_time_dependent_rigid_body_properties(mass_function=spacecraft_mass,
                                                                                                                         center_of_mass_function=spacecraft_center_of_mass,
                                                                                                                         inertia_tensor_function=spacecraft_mass_moment_of_inertia)

#
occulting_bodies_dict = dict()
occulting_bodies_dict[ "Sun" ] = [ "Earth" ]

# Written such that it can be extended in the future
panel_geom_1 = environment_setup.vehicle_systems.time_varying_panel_geometry(   surface_normal_function=panel_surface_normal,
                                                                                position_vector_function=panel_position_vector,
                                                                                area_function=panel_area,
                                                                                frame_orientation="")
reflection_law = environment_setup.radiation_pressure.solar_sail_optical_body_panel_reflection(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
#reflection_law = environment_setup.radiation_pressure.specular_diffuse_body_panel_reflection(1, 0, False)
panel1 = environment_setup.vehicle_systems.body_panel_settings(panel_type_id="Sail", panel_reflection_law=reflection_law, panel_geometry=panel_geom_1)
panelled_body = environment_setup.vehicle_systems.full_panelled_body_settings(panel_settings=[panel1])  # No rotational models for the panels
body_settings.get('ACS3').vehicle_shape_settings = panelled_body

vehicle_target_settings = environment_setup.radiation_pressure.panelled_radiation_target(occulting_bodies_dict)


constant_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# create rotation model settings and assign to body settings of "Earth"
body_settings.get("ACS3").rotation_model_settings = environment_setup.rotation_model.constant_rotation_model(global_frame_orientation,
                                                                                                "VehicleFixed",
                                                                                                constant_orientation )
# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)


environment_setup.add_radiation_pressure_target_model(
    bodies, "ACS3", vehicle_target_settings)

# Define bodies that are propagated
bodies_to_propagate = ["ACS3"]

# Define central bodies of propagation
central_bodies = ["Earth"]

# General simulation settings
# Create termination settings
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch, terminate_exactly_on_final_condition = True)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
    initial_time_step = initial_step_size,
    coefficient_set = propagation_setup.integrator.rkf_45,
    step_size_control_settings = control_settings,
    step_size_validation_settings = validation_settings )

# Translational dynamics settings
# Define accelerations acting on ACS3
acceleration_settings_acs3 = dict(
    Earth=[propagation_setup.acceleration.point_mass_gravity()],
    Sun=[propagation_setup.acceleration.radiation_pressure()]
)

acceleration_settings = {"ACS3": acceleration_settings_acs3}
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Initial translational state
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=bodies.get("Earth").gravitational_parameter,
    semi_major_axis=a_0,
    eccentricity=e_0,
    inclination=i_0,
    argument_of_periapsis=w_0,
    longitude_of_ascending_node=raan_0,
    true_anomaly=theta_0,
)

# Create propagation settings
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    propagation_setup.propagator.gauss_modified_equinoctial
)

# Rotational dynamics settings
torque_settings_on_sail = dict(Sun=[propagation_setup.torque.radiation_pressure()])  # dict(Earth = [propagation_setup.torque.second_degree_gravitational()])
torque_settings = {'ACS3': torque_settings_on_sail}
torque_model = propagation_setup.create_torque_models(bodies, torque_settings, bodies_to_propagate)

initial_rotational_state = np.concatenate((rotation_matrix_to_quaternion_entries(np.eye(3)), np.array([0., 0., 0.])))

rotational_propagator_settings = propagation_setup.propagator.rotational( torque_model,
                                                                          bodies_to_propagate,
                                                                          initial_rotational_state,
                                                                          simulation_start_epoch,
                                                                          integrator_settings,
                                                                          termination_settings)

# Combined dynamics
# DEPENDENT VARIABLES
dependent_variables = [ propagation_setup.dependent_variable.keplerian_state('ACS3', 'Earth'),
                        propagation_setup.dependent_variable.received_irradiance_shadow_function("ACS3", "Sun"),
                        propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type, "ACS3", "Sun"),
                        propagation_setup.dependent_variable.relative_position("ACS3", "Sun"),
                        propagation_setup.dependent_variable.central_body_fixed_spherical_position('Earth', 'ACS3'),
                        propagation_setup.dependent_variable.single_torque_norm(propagation_setup.torque.radiation_pressure_type, "ACS3", "Sun"),
                        propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles('ACS3'),
                      ]

# MULTI-TYPE PROPAGATOR
propagator_list = [translational_propagator_settings, rotational_propagator_settings]
combined_propagator_settings = propagation_setup.propagator.multitype( propagator_list,
                                                                       integrator_settings,
                                                                       simulation_start_epoch,
                                                                       termination_settings,
                                                                       output_variables = dependent_variables)

# Create simulation object and propagate the dynamics
simulator = numerical_simulation.create_dynamics_simulator(bodies, combined_propagator_settings)
state_history = simulator.state_history
dependent_variable_history = simulator.dependent_variable_history

# Extract the resulting state history and convert it to a ndarray
states_array = result2array(state_history)
dependent_variable_array = result2array(dependent_variable_history)

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
print(dependent_variable_array[:, -4])



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