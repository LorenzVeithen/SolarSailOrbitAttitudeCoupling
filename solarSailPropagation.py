import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt
import time
from constants import *

from MiscFunctions import set_axes_equal
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
from dynamicsSim import sailCoupledDynamicsProblem

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.astro import element_conversion
from tudatpy.astro.time_conversion import DateTime


# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list)
acs_object.set_vane_characteristics(vanes_coordinates_list, vane_origin_list, vanes_rotation_matrices_list, 0,
                                    np.array([0, 0, 0]), 0.0045)

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
simulation_end_epoch = DateTime(2024, 6, 2, 0).epoch()

# Initial states
initial_translational_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=398600441500000.0,
    semi_major_axis=a_0,
    eccentricity=e_0,
    inclination=i_0,
    argument_of_periapsis=w_0,
    longitude_of_ascending_node=raan_0,
    true_anomaly=theta_0)

#inertial_to_body_initial = R.from_euler('y', 90, degrees=True).as_matrix()
initial_rotational_state = np.concatenate((rotation_matrix_to_quaternion_entries(np.eye(3)), np.array([0., 0., 0.])))

sailProp = sailCoupledDynamicsProblem(sail,
               initial_translational_state,
               initial_rotational_state,
               simulation_start_epoch,
               simulation_end_epoch)

dependent_variables = sailProp.define_dependent_variables(acs_object)
bodies, vehicle_target_settings = sailProp.define_simulation_bodies()
sail.setBodies(bodies)
termination_settings, integrator_settings = sailProp.define_numerical_environment()
acceleration_models, torque_models = sailProp.define_dynamical_environment(bodies, vehicle_target_settings)
combined_propagator_settings = sailProp.define_propagators(integrator_settings, termination_settings, acceleration_models, torque_models, dependent_variables)
t0 = time.time()
state_history, states_array, dependent_variable_history, dependent_variable_array = sailProp.run_sim(bodies, combined_propagator_settings)
t1 = time.time()
sailProp.write_results_to_file(state_history,
                               'PropagationData/state_history.dat',
                               dependent_variable_history,
                               'PropagationData/dependent_variable_history.dat')

print(t1-t0)



