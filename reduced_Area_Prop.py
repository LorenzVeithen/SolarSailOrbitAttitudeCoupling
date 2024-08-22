import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
#from constants import *
from generalConstants import *

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

from scipy.spatial.transform import Rotation as R
from generalConstants import R_E

# initial orbit
a_0 = R_E + 1000e3           # [m ]initial spacecraft semi-major axis
e_0 = 4.03294322e-03         # [-] initial spacecraft eccentricity
i_0 = np.deg2rad(98.0131)    # [deg] initial spacecraft inclination
w_0 = np.deg2rad(120.0)      # [deg] initial spacecraft argument of pericentre
raan_0 = np.deg2rad(27.0)    # [deg] initial spacecraft RAAN
theta_0 = np.deg2rad(275.0)  # [deg] initial spacecraft true anomaly

# Sail characteristics - using ACS3 as baseline for initial testing
sail_mass = 16  # kg
sail_mass_without_wings = 15.66     # kg
sail_I = np.zeros((3, 3))
sail_I[0, 0] = 10.5
sail_I[1, 1] = 10.5
sail_I[2, 2] = 21
sail_nominal_CoM = np.array([0., 0., 0.])
sail_material_areal_density = 0.00425   # kg/m^2
vane_mechanical_rotation_limits = ([-np.pi, -np.pi], [np.pi, np.pi])

# Sail shape
boom_length = 4.586066535     # m
boom_attachment_point = 0.64    # m
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0., 0.],
                   [boom_length, 0., 0.],
                   [0., boom_length, 0.],
                   [0., boom_attachment_point, 0.]])

panel2 = np.array([[0., -boom_attachment_point, 0.],
                    [0., -boom_length, 0.],
                    [boom_length, 0., 0.],
                    [boom_attachment_point, 0., 0.]])

panel3 = np.array([[-boom_attachment_point, 0., 0.],
                   [-boom_length, 0., 0.],
                   [0., -boom_length, 0.],
                   [0., -boom_attachment_point, 0.]])

panel4 = np.array([[0., boom_attachment_point, 0.],
                    [0., boom_length, 0.],
                    [-boom_length, 0., 0.],
                    [-boom_attachment_point, 0., 0.]])

wings_coordinates_list = [panel1, panel2, panel3, panel4]
wings_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1, 1])] * 4
wings_rotation_matrices_list = [R.from_euler('z', -45., degrees=True).as_matrix(),
                                R.from_euler('z', -135., degrees=True).as_matrix(),
                                R.from_euler('z', -225., degrees=True).as_matrix(),
                                R.from_euler('z', -315., degrees=True).as_matrix()]

vane_angle = np.deg2rad(30.)
vane_side_length = 0.5
vanes_rotation_matrices_list = [
                                ]   # R.from_euler('z', 45., degrees=True).as_matrix()

vanes_origin_list = [
                     ]  # np.array([np.cos(np.pi/4) * boom_length/np.sqrt(2), np.sin(np.pi/4) * boom_length/np.sqrt(2), 0.])

vanes_coordinates_list = []
for i in range(len(vanes_origin_list)):
    current_vane_coords_body_frame_coords = vanes_origin_list[i]
    current_vane_rotation_matrix_body_to_vane = vanes_rotation_matrices_list[i]
    current_vane_rotation_matrix_vane_to_body = current_vane_rotation_matrix_body_to_vane

    second_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, -np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    third_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    current_vane_coords_body_frame_coords = np.vstack((current_vane_coords_body_frame_coords, second_point_body_frame, third_point_body_frame))
    vanes_coordinates_list.append(current_vane_coords_body_frame_coords)

vanes_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1., 1.])] * len(vanes_origin_list)

vanes_rotational_dof = np.array([])   #

algorithm_constants = {}
algorithm_constants["tol_vane_angle_determination_start_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination"] = 1e-3

algorithm_constants["tol_torque_allocation_problem_constraint"] = 1e-7
algorithm_constants["tol_torque_allocation_problem_objective"] = 1e-4
algorithm_constants["tol_torque_allocation_problem_x"] = 1e-3

algorithm_constants["max_rotational_velocity_orientation_change_update_vane_angles_degrees"] = 8  # [deg]
algorithm_constants["max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees"] = 7  # [deg]
algorithm_constants["max_relative_change_in_rotational_velocity_magnitude"] = 0.1  # [-]

# a bit of a different nature of analysis, still produce the data but do a bit differently
# use the things selected from before for that one especially
algorithm_constants["max_vane_torque_orientation_error"] = 15.  # [deg]     - go to DIRECT algorithm
algorithm_constants["max_vane_torque_relative_magnitude_error"] = 0.25  # [-]

# purely design, not really tuning parameters
algorithm_constants["sigmoid_scaling_parameter"] = 3        # [-] but is related to the rate of change of the vane angles
algorithm_constants["sigmoid_time_shift_parameter"] = 4     # [s]
algorithm_constants["vane_controller_shut_down_rotational_velocity_tolerance"] = 0.1
algorithm_constants["torque_allocation_problem_target_weight"] = 2/3

algorithm_constants["max_vane_torque_orientation_error"] = 15.  # [deg]     - go to DIRECT algorithm
algorithm_constants["max_vane_torque_relative_magnitude_error"] = 0.25  # [-]

wings_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1, 1])] * 4
tag = 'reduced_area_LTT'

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()

# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("None", boom_list, sail_I, algorithm_constants,
                                           include_shadow=False,
                                           sim_start_epoch=simulation_start_epoch)
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
#inertial_to_body_initial = np.dot(np.dot(inertial_to_body_initial, R.from_euler('x', 45., degrees=True).as_matrix()), R.from_euler('y', 0., degrees=True).as_matrix())    # rotate by 45 deg around x

# actually is body to inertial but it works so don't touch it
#inertial_to_body_initial = np.dot(np.dot(R.from_euler('y', 0, degrees=True).as_matrix(), R.from_euler('x', 0, degrees=True).as_matrix()), R.from_euler('z', 0, degrees=True).as_matrix())
initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
initial_rotational_velocity = np.array([0 * 2 * np.pi / 3600., 0 * 2 * np.pi / 3600, 0 * 2 * np.pi / 3600])
initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

sun_dir = constant_cartesian_position_Sun / np.linalg.norm(constant_cartesian_position_Sun)

sailProp = sailCoupledDynamicsProblem(sail,
               initial_translational_state,
               initial_rotational_state,
               simulation_start_epoch,
               simulation_end_epoch)

dependent_variables = sailProp.define_dependent_variables(acs_object)
bodies, vehicle_target_settings = sailProp.define_simulation_bodies(reduced_ephemeris_model_boolean=False)
sail.setBodies(bodies)
termination_settings, integrator_settings = sailProp.define_numerical_environment(
    control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),
    validation_settings=propagation_setup.integrator.step_size_validation(1E-5, 100))
acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models, torque_models, dependent_variables)
t0 = time.time()
state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
t1 = time.time()

rotations_per_hour = initial_rotational_velocity * 3600/(2*np.pi)
sailProp.write_results_to_file(state_history,
                               Project_directory + f'/0_GeneratedData/PropagationData/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}_{tag}.dat',
                               dependent_variable_history,
                               Project_directory + f'/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}_{tag}.dat')

print(t1-t0)