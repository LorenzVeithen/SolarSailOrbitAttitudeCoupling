import numpy as np
from scipy.spatial.transform import Rotation as R
from generalConstants import R_E, Project_directory

# data directory for integrator analysis
integrator_selection_data_directory = Project_directory + '/0_GeneratedData/IntegratorSelection_Data'

# initial orbit
a_0 = R_E + 1000e3           # [m ]initial spacecraft semi-major axis
e_0 = 4.03294322e-03         # [-] initial spacecraft eccentricity
i_0 = np.deg2rad(98.0131)    # [deg] initial spacecraft inclination
w_0 = np.deg2rad(120.0)      # [deg] initial spacecraft argument of pericentre
raan_0 = np.deg2rad(27.0)    # [deg] initial spacecraft RAAN
theta_0 = np.deg2rad(275.0)  # [deg] initial spacecraft true anomaly

# Sail characteristics - using ACS3 as baseline for initial testing
sail_mass = 16  # kg
sail_mass_without_wings = 15.66     # kg
sail_I = np.zeros((3, 3))
sail_I[0, 0] = 10.5
sail_I[1, 1] = 10.5
sail_I[2, 2] = 21
sail_nominal_CoM = np.array([0., 0., 0.])
sail_material_areal_density = 0.00425   # kg/m^2
vane_mechanical_rotation_limits = ([-np.pi, -np.pi], [np.pi, np.pi])

# Sail shape
boom_length = 7.     # m
boom_attachment_point = 0.64    # m
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0., 0.],
                   [boom_length, 0., 0.],
                   [0., boom_length, 0.],
                   [0., boom_attachment_point, 0.]])

panel2 = np.array([[0., -boom_attachment_point, 0.],
                    [0., -boom_length, 0.],
                    [boom_length, 0., 0.],
                    [boom_attachment_point, 0., 0.]])

panel3 = np.array([[-boom_attachment_point, 0., 0.],
                   [-boom_length, 0., 0.],
                   [0., -boom_length, 0.],
                   [0., -boom_attachment_point, 0.]])

panel4 = np.array([[0., boom_attachment_point, 0.],
                    [0., boom_length, 0.],
                    [-boom_length, 0., 0.],
                    [-boom_attachment_point, 0., 0.]])

wings_coordinates_list = [panel1, panel2, panel3, panel4]
wings_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1, 1])] * 4
wings_rotation_matrices_list = [R.from_euler('z', -45., degrees=True).as_matrix(),
                                R.from_euler('z', -135., degrees=True).as_matrix(),
                                R.from_euler('z', -225., degrees=True).as_matrix(),
                                R.from_euler('z', -315., degrees=True).as_matrix()]

vane_angle = np.deg2rad(30.)
vane_side_length = 0.5
vanes_rotation_matrices_list = [R.from_euler('z', 90., degrees=True).as_matrix(),
                                R.from_euler('z', 0., degrees=True).as_matrix(),
                                R.from_euler('z', 270., degrees=True).as_matrix(),
                                R.from_euler('z', 180., degrees=True).as_matrix(),
                                ]   # R.from_euler('z', 45., degrees=True).as_matrix()

vanes_origin_list = [np.array([0., boom_length, 0.]),
                     np.array([boom_length, 0., 0.]),
                     np.array([0, -boom_length, 0.]),
                     np.array([-boom_length, 0., 0.]),
                     ]  # np.array([np.cos(np.pi/4) * boom_length/np.sqrt(2), np.sin(np.pi/4) * boom_length/np.sqrt(2), 0.])

vanes_coordinates_list = []
for i in range(len(vanes_origin_list)):
    current_vane_coords_body_frame_coords = vanes_origin_list[i]
    current_vane_rotation_matrix_body_to_vane = vanes_rotation_matrices_list[i]
    current_vane_rotation_matrix_vane_to_body = current_vane_rotation_matrix_body_to_vane

    second_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, -np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    third_point_body_frame = (np.dot(current_vane_rotation_matrix_vane_to_body,
                                     np.array([np.sin(vane_angle) * vane_side_length, np.cos(vane_angle) * vane_side_length, 0]))
                               + current_vane_coords_body_frame_coords)

    current_vane_coords_body_frame_coords = np.vstack((current_vane_coords_body_frame_coords, second_point_body_frame, third_point_body_frame))
    vanes_coordinates_list.append(current_vane_coords_body_frame_coords)

vane_has_ideal_model = True
vanes_optical_properties = [np.array([0., 0., 1., 1., 0., 0., 2/3, 2/3, 1., 1.])] * len(vanes_origin_list)

vanes_rotational_dof = np.array([[True, True], [True, True], [True, True], [True, True]])   #
