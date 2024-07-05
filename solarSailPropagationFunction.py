import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules

from generalConstants import Project_directory, R_E, ACS3_opt_model_coeffs_set, double_ideal_opt_model_coeffs_set, single_ideal_opt_model_coeffs_set
import numpy as np

from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem
from scipy.spatial.transform import Rotation as R
from MiscFunctions import divide_list

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.data import save2txt
import time
import os

def runPropagationAnalysis(all_combinations,
                          optical_model_mode_str,
                          orbital_mode,
                          rank,
                          num_processes,
                          overwrite_previous=False,
                          include_shadow_bool=False,
                          run_mode='vane_detumbling',
                          output_frequency_in_seconds_=1):

    # import different models depending on the mode considered
    if (run_mode == 'vane_detumbling'):
        import VaneDetumblingAnalysis.vaneDetumbling_ACS3Model as sail_model
        acs_mode = 'vanes'
        keplerian_bool = False
        selected_wings_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
    elif (run_mode == 'LTT'):
        import LongTermTumblingAnalysis.longTermTumbling_ACS3Model as sail_model
        acs_mode = 'None'
        keplerian_bool = False
        selected_vanes_optical_properties = []
    elif (run_mode == 'keplerian_vane_detumbling' or run_mode == 'keplerian_LTT'):
        if (run_mode == 'keplerian_vane_detumbling'):
            import VaneDetumblingAnalysis.vaneDetumbling_ACS3Model as sail_model
        elif (run_mode == 'keplerian_LTT'):
            import LongTermTumblingAnalysis.longTermTumbling_ACS3Model as sail_model

        acs_mode = 'None'
        keplerian_bool = True
        all_combinations = [(0, 0, 0)]
        selected_wings_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(
            sail_model.wings_coordinates_list)
        selected_vanes_optical_properties = []
    else:
        raise Exception("Unknown run mode.")

    # case considered
    eccentricities = [sail_model.e_0, 0.3, 0.6]
    inclinations_deg = [np.rad2deg(sail_model.i_0), 45.0, 0.0]
    sma = ['LEO', 'MEO', 'GEO']
    sma_ecc_inc_combinations = [[sma[0], eccentricities[0], inclinations_deg[0]],  # like previous: currently comb_0
                                [sma[1], eccentricities[0], inclinations_deg[0]],  # like previous: currently comb_1
                                [sma[1], eccentricities[1], inclinations_deg[0]],  # like previous: currently comb_2
                                [sma[2], eccentricities[0], inclinations_deg[0]],  # like previous: currently comb_3
                                [sma[0], eccentricities[0], inclinations_deg[1]],
                                [sma[0], eccentricities[0], inclinations_deg[2]],
                                [sma[1], eccentricities[0], inclinations_deg[1]],
                                [sma[1], eccentricities[0], inclinations_deg[2]],
                                [sma[2], eccentricities[0], inclinations_deg[1]],
                                [sma[2], eccentricities[0], inclinations_deg[2]]]
    # add a GTO to the lot ? The amount of data is slowly getting out of hand... rip rip rip

    sma_mode = sma_ecc_inc_combinations[orbital_mode][0]
    ecc = sma_ecc_inc_combinations[orbital_mode][1]
    inc = sma_ecc_inc_combinations[orbital_mode][2]

    if (sma_mode == 'LEO'):
        initial_sma = sail_model.a_0
    elif (sma_mode == 'MEO'):
        initial_sma = R_E + 10000e3  # m
    elif (sma_mode == 'GEO'):
        initial_sma = R_E + 36000e3  # m

    initial_ecc = ecc
    initial_inc = np.deg2rad(inc)

    # get directory and correct optical properties
    if (optical_model_mode_str == "ACS3_optical_model"):
        if (run_mode == 'vane_detumbling' or run_mode == 'keplerian_vane_detumbling'):
            selected_vanes_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif (run_mode == 'LTT'):
            selected_wings_optical_properties = [np.array(ACS3_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc))}/NoAsymetry_data_ACS3_opt_model_shadow_{bool(include_shadow_bool)}'
    elif (optical_model_mode_str == "double_ideal_optical_model"):
        if (run_mode == 'vane_detumbling' or run_mode == 'keplerian_vane_detumbling'):
            selected_vanes_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif (run_mode == 'LTT'):
            selected_wings_optical_properties = [np.array(double_ideal_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc))}/NoAsymetry_data_double_ideal_opt_model_shadow_{bool(include_shadow_bool)}'
    elif (optical_model_mode_str == "single_ideal_optical_model"):
        if (run_mode == 'vane_detumbling' or run_mode == 'keplerian_vane_detumbling'):
            selected_vanes_optical_properties = [np.array(single_ideal_opt_model_coeffs_set)] * len(sail_model.vanes_coordinates_list)
        elif (run_mode == 'LTT'):
            selected_wings_optical_properties = [np.array(single_ideal_opt_model_coeffs_set)] * len(sail_model.wings_coordinates_list)
        save_sub_dir = f'{sma_mode}_ecc_{np.round(ecc, 1)}_inc_{np.round(np.rad2deg(initial_inc), 1)}/NoAsymetry_data_single_ideal_opt_model_shadow_{bool(include_shadow_bool)}'
    else:
        raise Exception("Unrecognised optical model mode in detumbling propagation")

    if (not os.path.exists(sail_model.analysis_save_data_dir + f'/{save_sub_dir}') and rank == 0):
        os.makedirs(sail_model.analysis_save_data_dir + f'/{save_sub_dir}/states_history')
        os.makedirs(sail_model.analysis_save_data_dir + f'/{save_sub_dir}/dependent_variable_history')
    save_directory = sail_model.analysis_save_data_dir + f'/{save_sub_dir}'

    if (keplerian_bool == False):
        # remove combinations which have already been done
        if (not overwrite_previous):
            new_combs = []
            for comb in all_combinations:
                initial_rotational_velocity = np.array(
                    [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
                rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
                tentative_file = save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
                if (os.path.isfile(tentative_file)):
                    # if the file exists, skip this propagation
                    continue
                else:
                    new_combs.append(comb)
            all_combinations = new_combs

        # cut into the number of parallel processes and take the required chunk
        chunks_list = divide_list(all_combinations, num_processes)
        selected_combinations = chunks_list[rank]
    else:
        selected_combinations = all_combinations

    # Set simulation start and end epochs
    simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
    simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()  # 30 days into the future but the simulation will likely finish way earlier

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
        if (os.path.isfile(tentative_file) and overwrite_previous==False and keplerian_bool==False):
            # if the file exists, skip this propagation
            continue

        # Define solar sail - see constants file
        acs_object = sail_attitude_control_systems(acs_mode, sail_model.boom_list, sail_model.sail_I, sail_model.algorithm_constants, include_shadow=include_shadow_bool)
        if (acs_mode == 'vanes' or run_mode=='keplerian_vane_detumbling'):
            acs_object.set_vane_characteristics(sail_model.vanes_coordinates_list,
                                                sail_model.vanes_origin_list,
                                                sail_model.vanes_rotation_matrices_list,
                                                0,
                                                np.array([0, 0, 0]),
                                                0.0045,
                                                sail_model.vanes_rotational_dof,
                                                optical_model_mode_str,
                                                sail_model.wings_coordinates_list,
                                                sail_model.vane_mechanical_rotation_limits,
                                                selected_vanes_optical_properties,
                                                torque_allocation_problem_objective_function_weights=[2. / 3., 1. / 3.])

        sail = sail_craft("ACS3",
                          len(sail_model.wings_coordinates_list),
                          len(sail_model.vanes_origin_list),
                          sail_model.wings_coordinates_list,
                          sail_model.vanes_coordinates_list,
                          selected_wings_optical_properties,
                          selected_vanes_optical_properties,
                          sail_model.sail_I,
                          sail_model.sail_mass,
                          sail_model.sail_mass_without_wings,
                          sail_model.sail_nominal_CoM,
                          sail_model.sail_material_areal_density,
                          sail_model.sail_material_areal_density,
                          acs_object)
        sail.set_desired_sail_body_frame_inertial_rotational_velocity(np.array([0., 0., 0.]))

        # Initial states
        initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=398600441500000.0,
            semi_major_axis=initial_sma,
            eccentricity=initial_ecc,
            inclination=initial_inc,
            argument_of_periapsis=sail_model.w_0,
            longitude_of_ascending_node=sail_model.raan_0,
            true_anomaly=sail_model.theta_0)

        sailProp = sailCoupledDynamicsProblem(sail,
                       initial_translational_state,
                       initial_rotational_state,
                       simulation_start_epoch,
                       simulation_end_epoch)

        dependent_variables = sailProp.define_dependent_variables(acs_object, keplerian_bool=keplerian_bool)
        bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
        sail.setBodies(bodies)
        termination_settings, integrator_settings = sailProp.define_numerical_environment(integrator_coefficient_set=propagation_setup.integrator.rkf_56,
                                                                                          control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12, 1.0E-12),)
        acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings,
                                                                                   keplerian_bool=keplerian_bool)
        combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models,
                                                                   torque_models, dependent_variables,
                                                                   selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                                                                   output_frequency_in_seconds=output_frequency_in_seconds_)

        t0 = time.time()
        state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
        t1 = time.time()

        if (keplerian_bool):
            save2txt(state_history,
                     save_directory + f'/keplerian_orbit_state_history.dat')
            save2txt(dependent_variable_history,
                     save_directory + f'/keplerian_orbit_dependent_variable_history.dat')
        else:
            save2txt(state_history, save_directory + f'/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')
            save2txt(dependent_variable_history, save_directory + f'/dependent_variable_history/dependent_variable_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat')


        print(f'{combination}: {t1-t0}')
