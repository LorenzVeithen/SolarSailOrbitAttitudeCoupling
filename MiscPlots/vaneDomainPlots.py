import numpy as np
import matplotlib.pyplot as plt
from vaneControllerMethods import vaneAnglesAllocationProblem
from constants import *
from attitudeControllersClass import sail_attitude_control_systems
from matplotlib import cm

line_style_loop = ["-", "--", "-.", ":", "-", "--", "-."]
color_loop = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
alpha_s_list = [0, 60, 120, 180, 240, 300]
beta_s_list = [0, 0, 0, 0, 0, 0]
plt.figure()
for k, (alpha_s, beta_s) in enumerate(zip(alpha_s_list, beta_s_list)):
    print(k)
    acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
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

    vaap = vaneAnglesAllocationProblem(0,
                                       [(-np.pi, np.pi), [-np.pi, np.pi]],
                                       10,
                                       wings_coordinates_list,
                                       acs_object,
                                       include_shadow=True)

    alpha_s_rad = np.deg2rad(alpha_s)
    beta_s_rad = np.deg2rad(beta_s)
    n_s = np.array([np.sin(alpha_s_rad) * np.cos(beta_s_rad),
                    np.sin(alpha_s_rad) * np.sin(beta_s_rad),
                    -np.cos(alpha_s_rad)])  # In the body reference frame
    vaap.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), n_s, vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)

    theta_v_deg_list = np.linspace(-180, 180, 360)
    phi_v_deg_list = np.linspace(-180, 180, 360)
    theta_matrix = np.zeros((len(theta_v_deg_list), len(phi_v_deg_list)))
    phi_matrix = np.zeros((len(theta_v_deg_list), len(phi_v_deg_list)))
    fitness_matrix = np.zeros((len(theta_v_deg_list), len(phi_v_deg_list)))
    line_determination_matrix = np.zeros((len(theta_v_deg_list), len(phi_v_deg_list)))
    for i, theta_v_deg in enumerate(theta_v_deg_list):
        for j, phi_v_deg in enumerate(phi_v_deg_list):
            theta_v_rad = np.deg2rad(theta_v_deg)
            phi_v_rad = np.deg2rad(phi_v_deg)

            theta_matrix[i, j] = theta_v_deg
            phi_matrix[i, j] = phi_v_deg

            fit = vaap.fitness([theta_v_rad, phi_v_rad])[0]
            if (fit > 1e20):
                fitness_matrix[i, j] = 1e20
                line_determination_matrix[i, j] = 1
            else:
                fitness_matrix[i, j] = fit
                line_determination_matrix[i, j] = 0

    lower_bound_phi = []
    upper_bound_phi = []
    for i, theta_v in enumerate(theta_v_deg_list):
        current_theta_line = line_determination_matrix[i, :]
        current_line_diffs = abs(np.diff(current_theta_line))
        lower_bound_phi.append(phi_matrix[i, np.where(current_line_diffs == 1)[0][0]])
        upper_bound_phi.append(phi_matrix[i, np.where(current_line_diffs == 1)[0][1]])

    plt.plot(theta_v_deg_list, lower_bound_phi, linestyle='--', color=color_loop[k])
    plt.plot(theta_v_deg_list, upper_bound_phi, linestyle='-', color=color_loop[k])

plt.grid(True)
for k, (alpha_s, beta_s) in enumerate(zip(alpha_s_list, beta_s_list)):
    plt.plot([], [], linewidth=7, linestyle='-', color=color_loop[k],
             label=fr'$\alpha_s = {alpha_s}$° $\beta_s = {beta_s}$°')
plt.plot([], [], linestyle="-", label='upper bounds', color='k')
plt.plot([], [], linestyle="--", label='lower bounds', color='k')
plt.legend(ncol=2)
plt.xlabel(r'$\theta_{v}$ [deg]', fontsize=14)
plt.ylabel(r'$\phi_{v}$  [deg]', fontsize=14)
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc/VaneAllocationAngleDomainShadow.png",
            bbox_inches='tight', dpi=1200)
plt.show()