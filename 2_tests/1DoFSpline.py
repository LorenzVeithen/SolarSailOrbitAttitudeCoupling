from constants import *
from MiscFunctions import *
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from vaneControllerMethods import vaneTorqueAllocationProblem, vaneAnglesAllocationProblem
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model
from AMSDerivation.truncatedEllipseCoefficientsFunctions import ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model, ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model
import matplotlib.pyplot as plt
import numpy as np
from vaneControllerMethods import cart_to_pol, get_ellipse_pts
from generalConstants import Project_directory

color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
line_style_loop = ["-", "--", "-.", ":", "-", "--"]
# attitude control system object
vane_optical_model_str = "ACS3_optical_model"
#vanes_optical_properties = [np.array([0.0, 0.0, 1, 1, 0.0, 0.0, 2/3, 2/3, 0.03, 0.6])] * 4
vanes_optical_properties = [np.array([0.1, 0.57, 0.74, 0.23, 0.16, 0.2, 2/3, 2/3, 0.03, 0.6])] * 4

include_shadow = False
# feasible torque ellipse coefficients from pre-computed functions
if (vane_optical_model_str == "double_ideal_optical_model"):
    vane_has_ideal_model = True
    if (include_shadow):
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_double_ideal_optical_model()
    else:
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_double_ideal_optical_model()
elif (vane_optical_model_str == "single_ideal_optical_model"):
    vane_has_ideal_model = True
    if (include_shadow):
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_single_ideal_optical_model()
    else:
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_single_ideal_optical_model()
elif (vane_optical_model_str == "ACS3_optical_model"):
    vane_has_ideal_model = False
    if (include_shadow):
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_TRUE_ACS3_optical_model()
    else:
        ellipse_coefficient_functions_list = ellipse_truncated_coefficients_function_shadow_FALSE_ACS3_optical_model()
elif (vane_optical_model_str == "AMS_Derivation"):
    vane_has_ideal_model = False
    ellipse_coefficient_functions_list = []

else:
    raise Exception("Requested set of ellipse coefficients have not been explicitly implemented yet")

acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    vane_optical_model_str,
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties)

# sail object
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

vaneAngleProblem = vaneAnglesAllocationProblem(1,
                                               ([-np.pi, -np.pi], [np.pi, np.pi]),
                                               10,
                                               wings_coordinates_list,
                                               acs_object,
                                               include_shadow=include_shadow)

torque_allocation_problem_objective_function_weights = [2/3, 1/3]
target_torque = np.array([0.0, 0., 5.])
previous_torque_allocation_solution = np.array([0] * 12)

# sunlight vector
alpha_s_deg_list = [0, 60, 120, -60, -120, 180]
beta_s_deg_list = [0, 60, 120, 60, 120, 0]

for case_id in range(2):
    plt.figure()
    for c_id, (alpha_s_deg, beta_s_deg) in enumerate(zip(alpha_s_deg_list, beta_s_deg_list)):

        # get 2-DoFs ellipse
        A = ellipse_coefficient_functions_list[0](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))
        B = ellipse_coefficient_functions_list[1](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))
        C = ellipse_coefficient_functions_list[2](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))
        D = ellipse_coefficient_functions_list[3](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))
        E = ellipse_coefficient_functions_list[4](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))
        F = ellipse_coefficient_functions_list[5](np.deg2rad(alpha_s_deg), np.deg2rad(beta_s_deg))

        params = cart_to_pol((A, B, C, D, E, F))
        Ty_ellipse, Tz_ellipse = get_ellipse_pts(params, npts=250)

        sunlight_vector_body_frame = np.array([np.sin(np.deg2rad(alpha_s_deg)) * np.cos(np.deg2rad(beta_s_deg)),
                                    np.sin(np.deg2rad(alpha_s_deg)) * np.sin(np.deg2rad(beta_s_deg)),
                                    -np.cos(np.deg2rad(alpha_s_deg))])   # In the body reference frame
        sunlight_vector_body_frame = sunlight_vector_body_frame/np.linalg.norm(sunlight_vector_body_frame)


        # vane torque allocation problem
        vaneAngleProblem.update_vane_angle_determination_algorithm(target_torque, sunlight_vector_body_frame,
                                                                   vane_variable_optical_properties=True,
                                                                   vane_optical_properties_list=vanes_optical_properties)

        tap = vaneTorqueAllocationProblem(acs_object,
                                          wings_coordinates_list,
                                          vane_has_ideal_model,
                                          include_shadow,
                                          ellipse_coefficient_functions_list,
                                          vanes_optical_properties,
                                          w1=torque_allocation_problem_objective_function_weights[0],
                                          w2=torque_allocation_problem_objective_function_weights[1],
                                          num_shadow_mesh_nodes=10)

        tap.set_desired_torque(target_torque, previous_torque_allocation_solution)
        tap.set_attaignable_moment_set_ellipses(sunlight_vector_body_frame)
        splines_list = tap.reducedDOFConstraintSplines(vaneAngleProblem, case=case_id)

        sx = splines_list[0][0]
        sy = splines_list[1][0]
        sz = splines_list[2][0]

        alpha_range = sx.t
        free_vane_angle = np.linspace(-np.pi, np.pi, 1000)
        Tx = sx(free_vane_angle)
        Ty = sy(free_vane_angle)
        Tz = sz(free_vane_angle)

        plt.plot(Ty, Tz, label=r'$\alpha_{s, \mathcal{B}}$' + f'= {round(alpha_s_deg, 1)}°, ' + r'$\beta_{s, \mathcal{B}}$' + f'= {round(beta_s_deg, 1)}°',
                 color=color_list[c_id],
                 linestyle=line_style_loop[c_id])
        plt.scatter(sy.c[::10], sz.c[::10],
                    marker='o',
                    color=color_list[c_id],
                    linestyle=line_style_loop[c_id])
        #plt.plot(Ty_ellipse, Tz_ellipse,
        #            color=color_list[c_id],
        #            linestyle="-.")

    plt.xlabel(r"$\tilde{T}_{y, \mathcal{B}}$ [-]", fontsize=14)
    plt.ylabel(r"$\tilde{T}_{z, \mathcal{B}}$ [-]", fontsize=14)
    plt.grid(True)
    plt.legend(ncol=1, loc='lower right')
    plt.savefig(Project_directory + f'/0_FinalPlots/Misc/1DoF_AMS_{case_id}_sh_{include_shadow}',
                dpi=600,
                bbox_inches='tight')
plt.show()

