from constants import *
from MiscFunctions import *
from attitudeControllersClass import sail_attitude_control_systems
from time import time

# these should be moved to global constants
include_shadow = False
solar_irradiance = 1400
alpha_s_deg = 34
beta_s_deg = -123

torque_allocation_problem_objective_function_weights = [1, 0]

# sunlight vector
sunlight_vector_body_frame = np.array([np.sin(np.deg2rad(alpha_s_deg)) * np.cos(np.deg2rad(beta_s_deg)),
                            np.sin(np.deg2rad(alpha_s_deg)) * np.sin(np.deg2rad(beta_s_deg)),
                            -np.cos(np.deg2rad(alpha_s_deg))])   # In the body reference frame
sunlight_vector_body_frame = sunlight_vector_body_frame/np.linalg.norm(sunlight_vector_body_frame)

# attitude control system object
acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants, include_shadow=include_shadow)
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
                                    vanes_optical_properties)

vane_angles_bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
previous_vanes_torque = [None]
target_torque = np.array([10., 0., 0.])
solar_irradiance_W = 1400
t0 = time()

initial_guess = np.array([[-75.29949703, -106.63008688], [156.1042524, 50.86419753], [-167.40740741, 18.03383631], [140.173754, -159.74394147]])

v_angles, vane_torques, optimal_torque = acs_object.vane_system_angles_from_desired_torque(acs_object,
                                                                vane_angles_bounds,
                                                                target_torque,
                                                                previous_vanes_torque,
                                                                sunlight_vector_body_frame,
                                                                np.deg2rad(initial_guess))
print(time()-t0)
print(np.rad2deg(v_angles))
print(optimal_torque)
print(vane_torques)

print("torque directions")
print(target_torque/np.linalg.norm(target_torque))
print(vane_torques.sum(axis=0)/np.linalg.norm(vane_torques.sum(axis=0)))