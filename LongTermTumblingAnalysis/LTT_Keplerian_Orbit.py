import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)

# Load standard modules
from longTermTumbling_ACS3Model import *
from generalConstants import Project_directory
import numpy as np

from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem
from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.data import save2txt
import time

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2024, 6, 1, 0).epoch()
simulation_end_epoch = DateTime(2024, 6, 30, 0).epoch()  # 30 days into the future

# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("None", boom_list, sail_I, include_shadow=False)

sail = sail_craft("ACS3",
                  len(wings_coordinates_list),
                  0,
                  wings_coordinates_list,
                  [],
                  wings_optical_properties,
                  [],
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
inertial_to_body_initial = np.dot(
    np.dot(R.from_euler('y', 0, degrees=True).as_matrix(), R.from_euler('x', 0, degrees=True).as_matrix()),
    R.from_euler('z', 0, degrees=True).as_matrix())
initial_quaternions = rotation_matrix_to_quaternion_entries(inertial_to_body_initial)
initial_rotational_velocity = np.array([0 * 2 * np.pi / 3600., 0 * 2 * np.pi / 3600, 0 * 2 * np.pi / 3600])
initial_rotational_state = np.concatenate((initial_quaternions, initial_rotational_velocity))

sailProp = sailCoupledDynamicsProblem(sail,
                                      initial_translational_state,
                                      initial_rotational_state,
                                      simulation_start_epoch,
                                      simulation_end_epoch)

dependent_variables = sailProp.define_dependent_variables(acs_object, keplerian_bool=True)
bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
sail.setBodies(bodies)
termination_settings, integrator_settings = sailProp.define_numerical_environment(
    integrator_coefficient_set=propagation_setup.integrator.rkf_56,
    control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-12,
                                                                                                 1.0E-12), )
acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, acs_object,
                                                                           vehicle_target_settings,
                                                                           keplerian_bool=True)

combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings,
                                                           acceleration_models,
                                                           torque_models, dependent_variables,
                                                           selected_propagator_=propagation_setup.propagator.gauss_modified_equinoctial,
                                                           output_frequency_in_seconds=10.0)
t0 = time.time()
state_history, states_array, dependent_variable_history, dependent_variable_array, number_of_function_evaluations, propagation_outcome = sailProp.run_sim(
    bodies, combined_propagator_settings)
t1 = time.time()

thinner_state_history = {}
thinner_dependent_variable_history = {}
previous_time_update = 0

# only store every 10 seconds to avoid too large files
for t_id, time_s in enumerate(states_array[:, 0]):
    if (time_s - previous_time_update > 10.):
        thinner_state_history[time_s] = states_array[t_id, 1:]
        thinner_dependent_variable_history[time_s] = dependent_variable_array[t_id, 1:]
        previous_time_update = time_s

rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
save2txt(thinner_state_history,
         LTT_save_data_dir + f'/keplerian_orbit_state_history.dat')
save2txt(thinner_dependent_variable_history,
         LTT_save_data_dir + f'/keplerian_orbit_dependent_variable_history.dat')

print(t1 - t0)