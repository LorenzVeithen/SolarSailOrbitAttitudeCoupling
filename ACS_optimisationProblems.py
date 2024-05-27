import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from constants import *
from MiscFunctions import *
from ACS_dynamicalModels import vane_dynamical_model
import pygmo as pg

class vaneAnglesAllocationProblem:
    # Solving the angle allocation problem of the vanes for a single vane
    def __init__(self, vane_id, bounds, vane_edge_mesh_nodes, sail_craft_object, acs_object, include_shadow=True):
        self.vane_id = vane_id
        self.bounds = bounds    # [[-np.pi, np.pi], [-np.pi, np.pi]]
        self.vane_target_torque = None
        self.sun_direction_body_frame = None
        self.sail_craft = sail_craft_object
        self.include_shadow = include_shadow
        self.hull = None

        self.R_BV = acs_object.vane_reference_frame_rotation_matrix_list[vane_id]
        self.vane_origin = acs_object.vane_reference_frame_origin_list[vane_id]
        self.vane_nominal_coordinates = acs_object.vane_panels_coordinates_list[vane_id]

        # Vane optical properties
        self.alpha_front = None
        self.alpha_back = None
        self.rho_s_front = None
        self.rho_s_back = None
        self.rho_d_front = None
        self.rho_d_back = None
        self.B_front = None
        self.B_back = None
        self.emissivity_front = None
        self.emissivity_back = None
        self.absorption_reemission_ratio = None

        # Mesh the vane edges for shadow determination
        all_meshed_points = np.array([[0, 0, 0]])
        for i in range(1, np.shape(self.vane_nominal_coordinates)[0] + 1):
            if (i == np.shape(self.vane_nominal_coordinates)[0]):
                delta_vec = self.vane_nominal_coordinates[0, :] - self.vane_nominal_coordinates[i - 1, :]
            else:
                delta_vec = self.vane_nominal_coordinates[i, :] - self.vane_nominal_coordinates[i - 1, :]
            meshed_points = np.zeros((vane_edge_mesh_nodes + 1, 3))

            for j in range(vane_edge_mesh_nodes+1):
                meshed_points[j, :] = self.vane_nominal_coordinates[i - 1, :] + j * delta_vec / vane_edge_mesh_nodes
            all_meshed_points = np.vstack((all_meshed_points, meshed_points))
        self.meshed_vane_coordinates = all_meshed_points

    def get_name(self):
        return "Vane angles allocation problem"

    def fitness(self, x):
        return [(1/3) * np.sum(((self.single_vane_torque(x)
                                - self.vane_target_torque)/vane_angles_allocation_scaling_factor)**2)]

    def single_vane_torque(self, x):
        rotated_points_body_frame = vane_dynamical_model([np.rad2deg(x[0])],
                                                         [np.rad2deg(x[1])],
                                                         1,
                                                         [self.vane_origin],
                                                         [self.meshed_vane_coordinates],
                                                         [self.R_BV])[0]

        centroid_body_frame, vane_area, surface_normal_body_frame = compute_panel_geometrical_properties(
            rotated_points_body_frame)
        c_theta = np.dot(surface_normal_body_frame, self.sun_direction_body_frame)

        # Get the vane torque according to the optical model
        if (c_theta >= 0):  # the front is exposed
            f = (W * vane_area * abs(c_theta) / c) * ((
              self.alpha_front * self.absorption_reemission_ratio - 2 * self.rho_s_front * c_theta - self.rho_d_front * self.B_front) * surface_normal_body_frame + (
              self.alpha_front + self.rho_d_front) * self.sun_direction_body_frame)
        else:
            f = (W * vane_area * abs(c_theta) / c) * ((
              self.alpha_back * self.absorption_reemission_ratio - 2 * self.rho_s_back * c_theta + self.rho_d_back * self.B_back) * surface_normal_body_frame + (
              self.alpha_back + self.rho_d_back) * self.sun_direction_body_frame)

        force_on_vane_body_reference_frame = np.dot(self.R_BV, f)
        torque_on_body_from_vane = np.cross(centroid_body_frame, force_on_vane_body_reference_frame)

        result = torque_on_body_from_vane
        if (self.include_shadow):
            shadow_bool = self.vane_shadow(rotated_points_body_frame[2:-1, :],
                                           self.hull)  # In practice the hull would be updated at each iteration of the propagation
            if (shadow_bool):
                result = np.array([1e23, 1e23, 1e23])  # Penalty if in shadow
        return result
    def get_bounds(self):
        return self.bounds

    def update_vane_angle_determination_algorithm(self, Td, n_s, vane_variable_optical_properties=False):
        # Target torque
        self.vane_target_torque = Td

        # Sun direction in the body frame
        self.sun_direction_body_frame = n_s

        # Spacecraft shadow hull
        vstack_stacking = np.array([[0, 0, 0]])
        for wing in self.sail_craft.sail_wings_coordinates:
            vstack_stacking = np.vstack((vstack_stacking, wing))
        relative_sun_vector_same_shape = np.zeros(np.shape(vstack_stacking[1:, :]))
        relative_sun_vector_same_shape[:, :3] = self.sun_direction_body_frame
        total_hull = np.vstack((vstack_stacking[1:, :] + relative_sun_vector_same_shape * 2,
                                vstack_stacking[1:, :] - relative_sun_vector_same_shape * 2))

        if (abs(n_s[2]) < 1e-15): # Coplanar hull, no real shadow
            self.sunlight_is_in_sail_plane = True
            self.hull = total_hull
        else:
            self.sunlight_is_in_sail_plane = False
            self.hull = Delaunay(total_hull)

        self.total_hull = total_hull
        self.all_wing_points = vstack_stacking[1:, :]

        # Surface optical properties
        if (vane_variable_optical_properties):
            vane_optical_properties = self.sail_craft.get_ith_panel_optical_properties(self.vane_id, "Vane")
            self.alpha_front = vane_optical_properties[0]
            self.alpha_back = vane_optical_properties[1]
            self.rho_s_front = vane_optical_properties[2]
            self.rho_s_back = vane_optical_properties[3]
            self.rho_d_front = vane_optical_properties[4]
            self.rho_d_back = vane_optical_properties[5]
            self.B_front = vane_optical_properties[6]
            self.B_back = vane_optical_properties[7]
            self.emissivity_front = vane_optical_properties[8]
            self.emissivity_back = vane_optical_properties[9]
            self.absorption_reemission_ratio = (self.emissivity_back * self.B_back - self.emissivity_front * self.B_front) / (
                                                       self.emissivity_back + self.emissivity_front)
        return True

    def set_shadow_bool(self, sb):
        self.include_shadow = sb
        return True

    def vane_shadow(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if (self.sunlight_is_in_sail_plane):
            return False
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return any(hull.find_simplex(p) >= 0)

    def plot_shadow_hull(self):
        """
        Simple 3D plot of the shadow hull
        :return: ax, the axis object permitting to plot other items in the figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.total_hull[:, 0], self.total_hull[:, 1], self.total_hull[:, 2])
        ax.scatter(self.all_wing_points[:, 0], self.all_wing_points[:, 1], self.all_wing_points[:, 2])
        for simplex in self.hull.simplices:
            plt.plot(self.total_hull[simplex, 0], self.total_hull[simplex, 1], self.total_hull[simplex, 2], 'k-')
        return ax

class torqueAllocationProblem:

    def __init__(self, acs_object):
        self.acs_object = acs_object    # Object with all vane characteristics
        self.previous_torque = None
        self.current_torque = None
        self.desired_torque = None

        # Determine if the degrees of freedom and position of each vane allows a moment around the x, y, or z axis
        # Make an array of booleans with True if the related torque should be equal to zero

        self.eq_constraints_vane_capabilities = np.array([False] * (self.acs_object.number_of_vanes * 3))
        for i in range(self.acs_object.number_of_vanes):
            if (not self.acs_object.vanes_rotational_dof_booleans[i][0]):
                self.eq_constraints_vane_capabilities[i * 3 + 2] = True      # Body torque around Z is not possible

            if (not self.acs_object.vanes_rotational_dof_booleans[i][1]):
                self.eq_constraints_vane_capabilities[i * 3] = True      # Body torque around X is not possible
                self.eq_constraints_vane_capabilities[i * 3 + 1] = True  # Body torque around Y is not possible
            else:
                # A torque around X, Y or both is possible based on the DoF.
                # Check with the vane orientation (position will be taken care of with the attaignable moment set
                idx = np.where(abs(abs(np.linalg.inv(self.acs_object.vane_reference_frame_rotation_matrix_list[i])[:, 1]) - 1) < 1e-15)[0]
                if (len(idx) !=0):
                    if (idx[0] == 0): self.eq_constraints_vane_capabilities[i * 3 + 1] = True       # Torque fully around X, no torque around Y
                    elif (idx[0] == 1): self.eq_constraints_vane_capabilities[i * 3] = True         # Torque fully around Y, no torque around X
                    # Else, no constraint


    def fitness(self, x):
        # x is 3 * self.acs_object.number_of_vanes long array with the x-y-z body torques for each vane in the same
        # order as the rest of the vane lists
        self.current_torque = x.reshape((self.acs_object.number_of_vanes, 3)).sum(axis=0)

        eq_constraint_list = []
        # The total torque needs to be the desired torque
        eq_constraint_list.append((self.current_torque - self.desired_torque)[0])  # = 0
        eq_constraint_list.append((self.current_torque - self.desired_torque)[1])  # = 0
        eq_constraint_list.append((self.current_torque - self.desired_torque)[2])  # = 0

        # Degree of freedom limitations and constraints based on the orientation of the vanes
        eq_constraint_list = eq_constraint_list + list(x[self.eq_constraints_vane_capabilities])
        return [np.linalg.norm(self.current_torque - self.previous_torque)**2] + eq_constraint_list

    def get_bounds(self):
        return ([-1] * (self.acs_object.number_of_vanes * 3), [1] * (self.acs_object.number_of_vanes * 3))   # Some bounds but these should not interfere with the rest

    def get_nec(self):
        return 3 + len(self.eq_constraints_vane_capabilities[self.eq_constraints_vane_capabilities == True])    # This is the number of equality constraints. The constraints themselves are specified in the fitness function

    def get_nic(self):
        return 0    # This is the number of inequality constraints. The constraints themselves are specified in the fitness function
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def set_desired_torque(self, desired_torque, previous_torque):
        """
        et les docstrings wesh? comment la pouponne va comprendre sinon?(déjà qu'elle écoute rien)

        :param Td: s
        :return:
        """
        self.previous_torque = previous_torque
        self.desired_torque = desired_torque
        return True