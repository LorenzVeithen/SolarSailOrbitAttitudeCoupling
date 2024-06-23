import numpy as np
from scipy.spatial.transform import Rotation as R
from generalConstants import R_E, Project_directory

# LTT save data
LTT_save_data_dir = Project_directory + f'/0_GeneratedData/LTT_Data/'

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
wings_optical_properties = [np.array([0.1, 0.57, 0.74, 0.23, 0.16, 0.2, 2/3, 2/3, 0.03, 0.6])] * 4
wings_rotation_matrices_list = [R.from_euler('z', -45., degrees=True).as_matrix(),
                                R.from_euler('z', -135., degrees=True).as_matrix(),
                                R.from_euler('z', -225., degrees=True).as_matrix(),
                                R.from_euler('z', -315., degrees=True).as_matrix()]
