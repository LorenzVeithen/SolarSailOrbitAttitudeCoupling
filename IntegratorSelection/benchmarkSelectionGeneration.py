import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
from integratorSelectionSailModel import *

from MiscFunctions import set_axes_equal
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup

algorithm_constants = {}
algorithm_constants["tol_vane_angle_determination_start_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination_golden_section"] = 1e-3
algorithm_constants["tol_vane_angle_determination"] = 1e-4

algorithm_constants["tol_torque_allocation_problem_constraint"] = 1e-7
algorithm_constants["tol_torque_allocation_problem_objective"] = 0#1e-7
algorithm_constants["tol_torque_allocation_problem_x"] = 1e-4

algorithm_constants["max_rotational_velocity_orientation_change_update_vane_angles_degrees"] = 5  # [deg]
algorithm_constants["max_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees"] = 5  # [deg]
algorithm_constants["max_relative_change_in_rotational_velocity_magnitude"] = 0.05  # [-]

algorithm_constants["max_vane_torque_orientation_error"] = 15.  # [deg]
algorithm_constants["max_vane_torque_relative_magnitude_error"] = 0.25  # [-]

algorithm_constants["sigmoid_scaling_parameter"] = 3        # [-] but is related to the rate of change of the vane angles
algorithm_constants["sigmoid_time_shift_parameter"] = 4     # [s]
algorithm_constants["vane_controller_shut_down_rotational_velocity_tolerance"] = 0.1

benchmark_time_steps = [2**(-5), 2**(-6)]   #2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0, 2**(-1), 2**(-2), 2**(-3), 2**(-4), , ... , 2**(-7), 2**(-8)
# Note that a faster propagation would probably require smaller time steps, but the benchmark is only used on a single
# propagation to make the choices
for dt in benchmark_time_steps:
    print(f"------Benchmark Selection: {dt} s------")
    # Set simulation start and end epochs
    simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
    simulation_end_epoch = DateTime(2024, 6, 1, 10).epoch()

    # Define solar sail - see constants file
    acs_object = sail_attitude_control_systems("vane_benchmark_test", boom_list, sail_I, algorithm_constants, include_shadow=False, sim_start_epoch=simulation_start_epoch)
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
    termination_settings, integrator_settings = sailProp.define_numerical_environment(initial_time_step=dt, benchmark_bool=True)
    acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object, vehicle_target_settings)
    combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings,
                                                               acceleration_models, torque_models, dependent_variables,
                                                               selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial)
    t0 = time.time()
    state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(bodies, combined_propagator_settings)
    t1 = time.time()

    rotations_per_hour = initial_rotational_velocity * 3600/(2*np.pi)
    sailProp.write_results_to_file(state_history,
                                   f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/IntegratorSelection/BenchmarkSelection/MEE/state_history_benchmark_dt_{dt}.dat',
                                   dependent_variable_history,
                                   f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/IntegratorSelection/BenchmarkSelection/MEE/dependent_variable_history_benchmark_dt_{dt}.dat')

    print(t1-t0)



