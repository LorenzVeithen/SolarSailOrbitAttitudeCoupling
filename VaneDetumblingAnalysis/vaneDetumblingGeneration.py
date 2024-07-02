import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules
from vaneDetumbling_ACS3Model import *
from generalConstants import Project_directory, R_E
import numpy as np
import itertools

from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.data import save2txt
from MiscFunctions import chunks
from multiprocessing import Process
import time
from mpi4py import MPI
import os

def runDetumblingAnalysis(selected_combinations, vanes_optical_properties, vane_has_ideal_model_bool, sma_0, ecc_0, i_0_deg, save_directory, overwrite_previous=False, include_shadow_bool=False):
    # Set simulation start and end epochs
    simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
    simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()  # 30 days into the future but the simulation will likely finish way earlier

    initial_sma = sma_0
    initial_ecc = ecc_0
    intial_inc = np.deg2rad(i_0_deg)

    # sort the combinations by magnitude to start with the easiest
    temp_sort_array = np.empty((len(selected_combinations), 2), dtype=object)
    for si in range(len(selected_combinations)):
        temp_sort_array[si, 0] = selected_combinations[si]
        temp_sort_array[si, 1] = np.sqrt(selected_combinations[si][0]**2 + selected_combinations[si][1]**2 + selected_combinations[si][2]**2)
    sorted_temp_sort_array = temp_sort_array[np.argsort(temp_sort_array[:, 1])]
    selected_combinations = sorted_temp_sort_array[:, 0]

    for counter, combination in enumerate(selected_combinations):
        print(f"--- running {combination}, {100 * ((counter+1)/len(selected_combinations))}% ---")

        # initial rotational state
        inertial_to_body_initial = np.dot(np.dot(R.from_euler('y', 0, degrees=True).as_matrix(), R.from_euler('x', 0, degrees=True).as_matrix()), R.from_euler('z', 0, degrees=True).as_matrix())
        initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
        initial_rotational_velocity = np.array([combination[0] * 2 * np.pi / 3600., combination[1] * 2 * np.pi / 3600, combination[2] * 2 * np.pi / 3600])
        initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file) and overwrite_previous==False):
            # if the file exists, skip this propagation
            continue

        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants, include_shadow=include_shadow_bool)
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
                                            torque_allocation_problem_objective_function_weights=[2. / 3., 1. / 3.])

        sail = sail_craft("ACS3",
                          len(wings_coordinates_list),
                          len(vanes_origin_list),
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
            semi_major_axis=initial_sma,
            eccentricity=initial_ecc,
            inclination=intial_inc,
            argument_of_periapsis=w_0,
            longitude_of_ascending_node=raan_0,
            true_anomaly=theta_0)

        sailProp = sailCoupledDynamicsProblem(sail,
                       initial_translational_state,
                       initial_rotational_state,
                       simulation_start_epoch,
                       simulation_end_epoch)

        dependent_variables = sailProp.define_dependent_variables(acs_object)
        bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
        sail.setBodies(bodies)
        termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                                                                          control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),)
        acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
        combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models,
                                                                   torque_models, dependent_variables,
                                                                   selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                                                                   output_frequency_in_seconds=1)

        t0 = time.time()
        state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
        t1 = time.time()

        save2txt(state_history, save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
        save2txt(dependent_variable_history, save_directory + f'/dependent_variable_history/dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')

        print(f'{combination}: {t1-t0}')
