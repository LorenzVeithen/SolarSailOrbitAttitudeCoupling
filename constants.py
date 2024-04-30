import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ideas:
## Make the full code object-oriented for the different characteristics of the mission / spacecraft

R_E = 6371e3
# initial orbit
a_0 = R_E + 1000e3        # [m ]initial spacecraft semi-major axis
e_0 = 4.03294322e-03         # [-] initial spacecraft eccentricity
i_0 = np.deg2rad(98.0131)    # [deg] initial spacecraft inclination
w_0 = np.deg2rad(120.0)      # [deg] initial spacecraft argument of pericentre
raan_0 = np.deg2rad(27.0)    # [deg] initial spacecraft RAAN
theta_0 = np.deg2rad(275.0)  # [deg] initial spacecraft true anomaly

# Sail characteristics - using ACS3 as baseline for initial testing
sail_mass = 16  # kg
sail_I = np.zeros((3, 3))
sail_I[0, 0] = 10.5
sail_I[1, 1] = 10.5
sail_I[2, 2] = 21
CoM_offset = np.array([0, 0, 0.05])

# For initial testing, cannonball radiation pressure characteristics
sail_cannonball_reference_area_radiation = 20 * 4  # m^2
sail_cannonball_radiation_pressure_coefficient = 1.8  # [-]

single_panel_area = 20  # m^2 area of a single panel
n_panels = 4  # [-] number of panels

boom_length = 7  # m
# Sail performance metrics
acc0 = 0.045 * 1E-3   # m/s/s characteristic sail acceleration
