import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
from constants import *
from generalConstants import AMS_directory, Project_directory

from MiscFunctions import set_axes_equal
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.interface import spice
from tudatpy.kernel.interface import spice_interface

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()

# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants, include_shadow=False, sim_start_epoch=simulation_start_epoch)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    "double_ideal_optical_model",
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties,
                                    torque_allocation_problem_objective_function_weights=[2./3., 1./3.])

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
sail.set_desired_sail_body_frame_inertial_rotational_velocity(np.array([0., 0., 0.]))

# Initial states
initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=398600441500000.0,
    semi_major_axis=a_0,
    eccentricity=e_0,
    inclination=i_0,
    argument_of_periapsis=w_0,
    longitude_of_ascending_node=raan_0,
    true_anomaly=theta_0)

# Random initial orientation just to try
spice.load_standard_kernels()
constant_cartesian_position_Sun = spice_interface.get_body_cartesian_state_at_epoch('Sun',
                                                                                 'Earth',
                                                                                 'J2000',
                                                                                 'NONE',
                                                                                    simulation_start_epoch)[:3]
new_z = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)
new_y = np.cross(np.array([0, 1, 0]), new_z)/np.linalg.norm(np.cross(np.array([0, 1, 0]), new_z))
new_x = np.cross(new_y, new_z)/np.linalg.norm(np.cross(new_y, new_z))
inertial_to_body_initial = np.zeros((3, 3))
inertial_to_body_initial[:, 0] = new_x
inertial_to_body_initial[:, 1] = new_y
inertial_to_body_initial[:, 2] = new_z

#inertial_to_body_initial = np.dot(np.dot(R.from_euler('y', 0, degrees=True).as_matrix(), R.from_euler('x', 0, degrees=True).as_matrix()), R.from_euler('z', 0, degrees=True).as_matrix())
initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
initial_rotational_velocity = np.array([5 * 2 * np.pi / 3600., 5 * 2 * np.pi / 3600, 5 * 2 * np.pi / 3600])
initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

sailProp = sailCoupledDynamicsProblem(sail,
               initial_translational_state,
               initial_rotational_state,
               simulation_start_epoch,
               simulation_end_epoch)

dependent_variables = sailProp.define_dependent_variables(acs_object)
bodies, vehicle_target_settings = sailProp.define_simulation_bodies(reduced_ephemeris_model_boolean=False)
sail.setBodies(bodies)
termination_settings, integrator_settings = sailProp.define_numerical_environment()
acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models, torque_models, dependent_variables)
t0 = time.time()
state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
t1 = time.time()

rotations_per_hour = initial_rotational_velocity * 3600/(2*np.pi)
sailProp.write_results_to_file(state_history,
                               Project_directory + f'/0_GeneratedData/PropagationData/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}_test.dat',
                               dependent_variable_history,
                               Project_directory + f'/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}_test.dat')

print(t1-t0)