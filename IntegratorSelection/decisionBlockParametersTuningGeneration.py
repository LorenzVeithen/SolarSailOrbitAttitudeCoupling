import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
from integratorSelectionSailModel import *
import os

from MiscFunctions import set_axes_equal
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.data import save2txt

save_dir = integrator_selection_data_directory + "/decisionBlockTuning/"

#
algorithm_constants = {}
algorithm_constants["tol_vane_angle_determination_start_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination"] = 1e-7

algorithm_constants["tol_torque_allocation_problem_constraint"] = 1e-7
algorithm_constants["tol_torque_allocation_problem_objective"] = 0
algorithm_constants["tol_torque_allocation_problem_x"] = 1e-4

algorithm_constants["max_rotational_velocity_orientation_change_update_vane_angles_degrees"] = 0.01  # [deg]
algorithm_constants["max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees"] = 0.01  # [deg]
algorithm_constants["max_relative_change_in_rotational_velocity_magnitude"] = 0.05  # [-]

algorithm_constants["max_vane_torque_orientation_error"] = 15.  # [deg]     - go to DIRECT algorithm
algorithm_constants["max_vane_torque_relative_magnitude_error"] = 0.25  # [-]

algorithm_constants["sigmoid_scaling_parameter"] = 3        # [-] but is related to the rate of change of the vane angles
algorithm_constants["sigmoid_time_shift_parameter"] = 4     # [s]
algorithm_constants["vane_controller_shut_down_rotational_velocity_tolerance"] = 0.1



chosen_integrator = propagation_setup.integrator.rkf_56
chosen_propagator = propagation_setup.propagator.gauss_modified_equinoctial
chosen_tolerance = 1e-12

tolerances_to_tune = {"tol_vane_angle_determination_start_golden_section": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],    # This is a tolerance on the vane angle determination optimisation objective value... Not big physical meaning
                      "tol_vane_angle_determination_golden_section": [],        # not really important as thisjust tries to get the angle inside the envelope - just do a test independently
                      "tol_vane_angle_determination": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],

                      "tol_torque_allocation_problem_constraint": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                                                                   1e-4, 1e-3, 1e-2, 1e-1],
                      "tol_torque_allocation_problem_objective": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                                                                  1e-4, 1e-3, 1e-2],
                      "tol_torque_allocation_problem_x": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],

                      "max_rotational_velocity_orientation_change_update_vane_angles_degrees": [0.1, 0.25, 0.5, 1, 2.5,
                                                                                                5, 10, 15, 20, 25],
                      "max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees": [0.1, 0.25, 0.5,
                                                                                                       1, 2.5, 5, 10,
                                                                                                       15, 20, 25],
                      "max_relative_change_in_rotational_velocity_magnitude": [],   # definitely not driving, not sure if it is even worth investigating

                      "max_vane_torque_orientation_error": [0.1, 1, 5, 10, 20, 30, 40, 50],
                      "max_vane_torque_relative_magnitude_error": []}

# TODO: to be finished
for integrator in integrator_list:
    print(f"------integrator: {integrator} ------")
    for j, tolerance in enumerate(tolerances_list):
        print(f"------tolerance: {tolerance} ------")
        # Set simulation start and end epochs
        simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
        simulation_end_epoch = DateTime(2024, 6, 1, 10).epoch()

        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants, include_shadow=False)
        acs_object.set_vane_characteristics(vanes_coordinates_list,
                                            vanes_origin_list,
                                            vanes_rotation_matrices_list,
                                            0,
                                            np.array([0, 0, 0]),
                                            0.0045,
                                            vanes_rotational_dof,
                                            vane_has_ideal_model,
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
        inertial_to_body_initial = np.dot(np.dot(R.from_euler('y', 0, degrees=True).as_matrix(), R.from_euler('x', 0, degrees=True).as_matrix()), R.from_euler('z', 0, degrees=True).as_matrix())
        initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
        initial_rotational_velocity = np.array([5 * 2 * np.pi / 3600., 5 * 2 * np.pi / 3600, 5 * 2 * np.pi / 3600])
        initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

        sailProp = sailCoupledDynamicsProblem(sail,
                       initial_translational_state,
                       initial_rotational_state,
                       simulation_start_epoch,
                       simulation_end_epoch)

        dependent_variables = sailProp.define_dependent_variables(acs_object)
        bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
        sail.setBodies(bodies)
        termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=integrator,
                                                                                          control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(chosen_tolerance, chosen_tolerance))
        acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
        combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings,
                                                                   acceleration_models, torque_models,
                                                                   dependent_variables,
                                                                   selected_propagator_=chosen_propagator)
        t0 = time.time()
        state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
        t1 = time.time()

        rotations_per_hour = initial_rotational_velocity * 3600/(2*np.pi)
        if (not os.path.exists(save_dir + f'{str(integrator).split(".")[-1]}')):
            os.mkdir(save_dir + f'{str(integrator).split(".")[-1]}')

        if (not os.path.exists(save_dir + f'{str(integrator).split(".")[-1]}/tol_{tolerance}')):
            os.mkdir(save_dir + f'{str(integrator).split(".")[-1]}/tol_{tolerance}')
        current_complete_dir = save_dir + f'{str(integrator).split(".")[-1]}/tol_{tolerance}/'

        sailProp.write_results_to_file(state_history,
                                       current_complete_dir + f'state_history.dat',
                                       dependent_variable_history,
                                       current_complete_dir + f'dependent_variable_history.dat')

        dict_to_write = {'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
        dict_to_write['Propagation run successfully'] = propagation_outcome
        dict_to_write['Propagator'] = str(chosen_propagator).split(".")[-1]
        dict_to_write['tolerance'] = str(tolerance)
        save2txt(dict_to_write, current_complete_dir + f'ancillary_simulation_info.txt')
        print(t1-t0)



