# Determine the attainable moment set of a given vane

from constants import *
import numpy as np
from scipy.optimize import fsolve, minimize

vane_inner_angle = (np.pi - (2 * vane_angle))/2
h_vane = np.cos(vane_inner_angle) * vane_side_length
b_vane = 2 * np.sin(vane_inner_angle) * vane_side_length
A_vane = h_vane * b_vane/2
c = 299792458   # m/s

vane_coordinates = vanes_coordinates_list[0]
alpha_front = vanes_optical_properties[0][0]
alpha_back = vanes_optical_properties[0][1]
rho_s_front = vanes_optical_properties[0][2]
rho_s_back = vanes_optical_properties[0][3]
rho_d_front = vanes_optical_properties[0][4]
rho_d_back = vanes_optical_properties[0][5]
B_front = vanes_optical_properties[0][6]
B_back = vanes_optical_properties[0][7]
emissivity_front = vanes_optical_properties[0][8]
emissivity_back = vanes_optical_properties[0][9]

vanes_origin = vanes_origin_list[0]
R_BV = vanes_rotation_matrices_list[0]
R_VB = np.linalg.inv(R_BV)
absorption_reemission_ratio = (emissivity_back * B_back - emissivity_front * B_front)/(emissivity_back + emissivity_front)
n_s = np.array([0, 0, 1])
n_s = n_s/np.linalg.norm(n_s)
n_s_incoming = -n_s
n_sail = np.array([0, 0, 1])
n_sail = n_s/np.linalg.norm(n_sail)
W = 1400    # W / m^2 - roughly
initial_centroid = np.array([(2 * h_vane/3), 0, 0])

def shadow_function(alpha_1, alpha_2):
    R_vane_rotation = np.array([[np.cos(alpha_2), np.sin(alpha_1), np.sin(alpha_2) * np.cos(alpha_1)],
                                [0, np.cos(alpha_1), -np.sin(alpha_1)],
                                [-np.sin(alpha_2), np.sin(alpha_1), np.cos(alpha_2) * np.cos(alpha_1)]])
    # n = np.array([np.cos(alpha_1) * np.sin(alpha_2), -np.sin(alpha_1), np.cos(alpha_1) * np.cos(alpha_2)])
    n_vane = np.dot(R_vane_rotation, np.array([0, 0, 1]))
    n_vane = n_vane / np.linalg.norm(n_vane)
    c_theta = np.dot(n_vane, n_s)

    for vane_point in vane_coordinates:
        rotated_point = np.dot(R_vane_rotation, vane_point)
        current_rotated_point_body_frame = np.dot(R_BV, rotated_point) + vanes_origin
        current_rotated_point_projected_on_sail_body_frame = current_rotated_point_body_frame - (np.dot(current_rotated_point_body_frame, n_sail)/np.dot(n_s_incoming, n_sail)) * n_s_incoming


    return False

# Make a convex hull from point cloud of the sail coordinates