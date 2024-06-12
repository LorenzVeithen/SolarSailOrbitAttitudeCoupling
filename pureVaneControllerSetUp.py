from constants import *
from MiscFunctions import *
from attitudeControllersClass import sail_attitude_control_systems
from vaneControllerMethods import vane_system_angles_from_desired_torque

# these should be moved to global constants
include_shadow = False
solar_irradiance = 1400
alpha_s_deg = -139
beta_s_deg = 98

torque_allocation_problem_objective_function_weights = [1, 0]

# sunlight vector
sunlight_vector_body_frame = np.array([np.sin(np.deg2rad(alpha_s_deg)) * np.cos(np.deg2rad(beta_s_deg)),
                            np.sin(np.deg2rad(alpha_s_deg)) * np.sin(np.deg2rad(beta_s_deg)),
                            -np.cos(np.deg2rad(alpha_s_deg))])   # In the body reference frame
sunlight_vector_body_frame = sunlight_vector_body_frame/np.linalg.norm(sunlight_vector_body_frame)

# attitude control system object
acs_object = sail_attitude_control_systems("vanes", boom_list, include_shadow=include_shadow)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    vane_has_ideal_model,
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits)

vane_angles_bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
previous_vanes_torque = None
target_torque = np.array([10., 0., 0.])
solar_irradiance_W = 1400
v_angles, vane_torques = vane_system_angles_from_desired_torque(acs_object, vane_angles_bounds, target_torque, previous_vanes_torque, sunlight_vector_body_frame)
print(np.rad2deg(v_angles))
print(vane_torques.sum(axis=0))

print("torque directions")
print(target_torque/np.linalg.norm(target_torque))
print(vane_torques.sum(axis=0)/np.linalg.norm(vane_torques.sum(axis=0)))