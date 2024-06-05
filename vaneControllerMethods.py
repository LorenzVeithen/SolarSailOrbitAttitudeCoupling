import numpy as np
import matplotlib
from numba import jit
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pygmo as pg

from constants import *
from ACS_dynamicalModels import vane_dynamical_model
from MiscFunctions import *
import itertools

#matplotlib.pyplot.switch_backend('Agg')

class vaneAnglesAllocationProblem:
    # Solving the angle allocation problem of the vanes for a single vane
    def __init__(self, vane_id, bounds, vane_edge_mesh_nodes, sail_craft_object, acs_object, include_shadow=True):
        self.vane_id = vane_id
        self.bounds = bounds    # [[-np.pi, np.pi], [-np.pi, np.pi]]
        self.vane_target_torque = None
        self.unit_direction_vane_target_torque = None
        self.sun_direction_body_frame = None
        self.sail_craft = sail_craft_object
        self.include_shadow = include_shadow
        self.hull = None

        self.R_BV = np.linalg.inv(acs_object.vane_reference_frame_rotation_matrix_list[vane_id])

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

    def fitness(self, x):
        """

        :param x: variables to be optimised. x[0]: rotation around vane x-axis in radians; x[1]: rotation around vane
        y-axis in radiants; x[2] scaling factor of target torque, to be maximised
        :return: list containing the objective function and equality constraints
        """
        torque_x = self.single_vane_torque(x)
        obj = [(1/3) * np.sum(((torque_x - self.vane_target_torque))**2)]
        return obj

    def get_bounds(self):
        """
        Function returning the bounds on the variables to be optimised.
        :return: self.bounds, a list of two lists: the lower and upper bounds on each variable to be optimised
        """
        return self.bounds

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0

    def get_name(self):
        return "Vane angles allocation problem"


    def single_vane_torque(self, x):
        rotated_points_body_frame = vane_dynamical_model([np.rad2deg(x[0])],
                                                         [np.rad2deg(x[1])],
                                                         1,
                                                         [self.vane_origin],
                                                         [self.meshed_vane_coordinates],
                                                         [self.R_BV])[0]

        centroid_body_frame, vane_area, surface_normal_body_frame = compute_panel_geometrical_properties(
            rotated_points_body_frame)  # This is all in the body frame
        c_theta = np.dot(surface_normal_body_frame, -self.sun_direction_body_frame)/(np.linalg.norm(surface_normal_body_frame) * np.linalg.norm(-self.sun_direction_body_frame))
        # Get the vane torque according to the optical model, in the body frame
        if (c_theta >= 0):  # the front is exposed
            # W * vane_area/ c_sol *
            f = (abs(c_theta)) * ((
              self.alpha_front * self.absorption_reemission_ratio - 2 * self.rho_s_front * c_theta - self.rho_d_front * self.B_front) * surface_normal_body_frame + (
              self.alpha_front + self.rho_d_front) * -self.sun_direction_body_frame)
        else:
            # W * vane_area/ c_sol *
            f = (abs(c_theta)) * ((
              self.alpha_back * self.absorption_reemission_ratio - 2 * self.rho_s_back * c_theta + self.rho_d_back * self.B_back) * surface_normal_body_frame + (
              self.alpha_back + self.rho_d_back) * -self.sun_direction_body_frame)

        force_on_vane_body_frame = f
        torque_on_body_from_vane = (1/np.linalg.norm(self.vane_origin)) * np.cross(centroid_body_frame, force_on_vane_body_frame)

        result = torque_on_body_from_vane
        if (self.include_shadow):
            shadow_bool = self.vane_shadow(rotated_points_body_frame[2:-1, :],
                                           self.hull)  # In practice the hull would be updated at each iteration of the propagation
            if (shadow_bool):
                result = np.array([1e23, 1e23, 1e23])  # Penalty if in shadow
        return result   # Non-dimensional, physical torque is given when multiplicating by norm(r_arm) * W * vane_area/c_sol

    def update_vane_angle_determination_algorithm(self, Td, n_s, vane_variable_optical_properties=False):
        """
        Function permitting to update the properties of the class object at each iteration of the propagation to target
        different target torques under different sunlight conditions or vane optical properties.
        :param Td: np.array[(1, 3)], target torque in Nm at the time of evaluating the optimisation.
        :param n_s: np.array[(1, 3)], unit vector of the sunlight direction in the body fixed frame.
        :param vane_variable_optical_properties: boolean, indicating if the optical properties of the vanes can change
        as a function of time.
        :return: True, if the process was completed successfully
        """
        # Target torque
        self.vane_target_torque = Td
        self.unit_direction_vane_target_torque = self.vane_target_torque / np.linalg.norm(self.vane_target_torque)

        # Sun direction in the body frame
        self.sun_direction_body_frame = n_s

        # Spacecraft shadow hull
        vstack_stacking = np.array([[0, 0, 0]])
        for wing in self.sail_craft.sail_wings_coordinates:
            vstack_stacking = np.vstack((vstack_stacking, wing))
        relative_sun_vector_same_shape = np.zeros(np.shape(vstack_stacking[1:, :]))
        relative_sun_vector_same_shape[:, :3] = self.sun_direction_body_frame
        total_hull = np.vstack((vstack_stacking[1:, :] + relative_sun_vector_same_shape * 20,
                                vstack_stacking[1:, :] - relative_sun_vector_same_shape * 20))

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

class vaneTorqueAllocationProblem:
    def __init__(self, acs_object, sail_object, vane_has_ideal_model, include_shadow, num_points_ellipse_constraint=1000):
        self.acs_object = acs_object    # Object with all vane characteristics
        self.sail_object = sail_object
        self.vane_has_ideal_model = vane_has_ideal_model
        self.include_shadow = include_shadow
        self.previous_torque = None
        self.current_torque = None
        self.desired_torque = None
        self.num_points_ellipse_constraint = num_points_ellipse_constraint

        # Determine if the degrees of freedom and position of each vane allows a moment around the x, y, or z axis
        # Make an array of booleans with True if the related torque should be equal to zero

        self.eq_constraints_vane_capabilities = np.array([0] * (self.acs_object.number_of_vanes * 3))
        for i in range(self.acs_object.number_of_vanes):
            if (not self.acs_object.vanes_rotational_dof_booleans[i][0]):       # DoF constraints
                self.eq_constraints_vane_capabilities[i * 3 + 2] = 2            # Body torque around Z is not possible

            if (not self.acs_object.vanes_rotational_dof_booleans[i][1]):       # DoF constraints
                # Only certain combinations of torques in Z and X/Y are feasible
                self.eq_constraints_vane_capabilities[i * 3] = 2                # X
                self.eq_constraints_vane_capabilities[i * 3 + 1] = 2            # Y
            else:
                # A torque around X, Y or both is possible based on the DoF.
                # Check with the vane orientation (position will be taken care of with the attaignable moment set
                idx = np.where(abs(abs(np.linalg.inv(self.acs_object.vane_reference_frame_rotation_matrix_list[i])[:, 1]) - 1) < 1e-15)[0]
                if (len(idx) !=0):
                    if (idx[0] == 0): self.eq_constraints_vane_capabilities[i * 3 + 1] = 1       # Torque fully around X, no torque around Y
                    elif (idx[0] == 1): self.eq_constraints_vane_capabilities[i * 3] = 1         # Torque fully around Y, no torque around X
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
        eq_constraint_list = eq_constraint_list + list(x[self.eq_constraints_vane_capabilities==1])


        # Ellipse inequality constraints
        ineq_constraint_list = []
        for i, (vane_AMS_coeff_x, vane_AMS_coeff_y)  in enumerate(zip(self.vanes_AMS_coefficients_x, self.vanes_AMS_coefficients_y)):
            # Get torque on the current vane
            T_current_vane = x.reshape((self.acs_object.number_of_vanes), 3)[i, :]
            Tx = T_current_vane[0]
            Ty = T_current_vane[1]
            Tz = T_current_vane[2]

            if (vane_AMS_coeff_x != None and self.eq_constraints_vane_capabilities[i * 3] != 1):
                if (self.eq_constraints_vane_capabilities[i * 3] == 0):  # is an inequality constraint
                    ineq_constraint_list.append(np.dot(np.array([[Tx[0] ** 2, Tx[0] * Tz[1], Tz[1] ** 2, Tx[0], Tz[1], 1]]), vane_AMS_coeff_x))
                elif (self.eq_constraints_vane_capabilities[i * 3] == 2):  # is an equality constraint
                    eq_constraint_list.append(np.dot(np.array([[Tx[0] ** 2, Tx[0] * Tz[1], Tz[1] ** 2, Tx[0], Tz[1], 1]]), vane_AMS_coeff_x))

            if (vane_AMS_coeff_y != None and self.eq_constraints_vane_capabilities[i * 3 + 1] != 1):
                if (self.eq_constraints_vane_capabilities[i * 3 + 1] == 0):  # is an inequality constraint
                    ineq_constraint_list.append(np.dot(np.array([[Ty[0] ** 2, Ty[0] * Tz[1], Tz[1] ** 2, Ty[0], Tz[1], 1]]), vane_AMS_coeff_y))
                elif (self.eq_constraints_vane_capabilities[i * 3 + 1] == 2):  # is an equality constraint
                    eq_constraint_list.append(np.dot(np.array([[Ty[0] ** 2, Ty[0] * Tz[1], Tz[1] ** 2, Ty[0], Tz[1], 1]]), vane_AMS_coeff_y))

        return [np.linalg.norm(self.current_torque - self.previous_torque)**2] + eq_constraint_list

    def get_bounds(self):
        return ([-1] * (self.acs_object.number_of_vanes * 3), [1] * (self.acs_object.number_of_vanes * 3))

    def get_nec(self):
        return 3 + len(self.eq_constraints_vane_capabilities[self.eq_constraints_vane_capabilities == True])

    def get_nic(self):
        return 0 #self.num_points_ellipse_constraint
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)   #TODO: investigate using more accurate gradient computations

    def set_desired_torque(self, desired_torque, previous_torque):
        """
        et les docstrings wesh? comment la pouponne va comprendre sinon?(déjà qu'elle écoute rien)

        :param Td: s
        :return:
        """
        self.previous_torque = previous_torque
        self.desired_torque = desired_torque
        return True
    def set_attaignable_moment_set_ellipses(self, vanes_AMS_coefficients_x, vanes_AMS_coefficients_y, n_s):
        # Use pre-computed ellipses
        self.vanes_AMS_coefficients_x = vanes_AMS_coefficients_x
        self.vanes_AMS_coefficients_y = vanes_AMS_coefficients_y

        # If the degrees of freedom of the vanes are limited, compute relevant constraints
        for i, vane_rotational_dof_booleans in enumerate(self.acs_object.vanes_rotational_dof_booleans):
            vaneAngleProblem = vaneAnglesAllocationProblem(i,
                                                           ([-np.pi, -np.pi], [np.pi, np.pi]),
                                                           10,
                                                           self.sail_object,
                                                           self.acs_object,
                                                           include_shadow=self.include_shadow)
            if (vane_rotational_dof_booleans[0] == False):
                # alpha_1 is zero
                if (self.vane_has_ideal_model==True): # ideal model

                    sx_list = [lambda a: a]
                    sy_list = [lambda a: 0]     #   Tz
            if (vane_rotational_dof_booleans[1] == False):
                pass
        return True
    def check_inequality_sign(self, weights):
        inf_point = np.array([1e23, 1e23])
        D_inf = np.array(
            [[inf_point[0] ** 2, inf_point[0] * inf_point[1], inf_point[1] ** 2, inf_point[0], inf_point[1], 1]])
        return np.dot(D_inf, weights)[0] > 0  # return True if infinity is outside, False if it is inside


class constrainedEllipseCoefficientsProblem:
    def __init__(self, convex_hull_points):
        self.convex_hull_points = convex_hull_points
        self.A0 = None

        # Vector of inequality constraints
        D = np.array([0, 0, 0, 0, 0, 0])
        for xp in convex_hull_points[0:, :]:
            D = np.vstack((D, np.array([xp[0] ** 2, xp[0] * xp[1], xp[1] ** 2, xp[0], xp[1], 1])))
        self.D = D[1:, :]

    def fitness(self, x):
        [a, b, c, d, e, f] = x
        if (self.check_inequality_sign(x)):
            x *= 1  # all rows < 0
        else:
            x *= -1  # all rows < 0
        ineq_constraint = list(np.dot(-self.D, x))  # all rows < 0
        A = self.ellipse_area(x)
        self.previous_A = A
        obj = -A
        return [obj] + [a + c - 1] + (ineq_constraint + [b ** 2 - 4 * a * c])

    def ellipse_area(self, x):
        [a, b, c, d, e, f] = x
        if ((b ** 2 - 4 * a * c) < 0):
            x0, y0, ap, bp, e, phi = cart_to_pol([a, b, c, d, e, f])
            A = np.pi * ap * bp
        else:
            A = self.previous_A
        return A

    def get_bounds(self):
        return (
        [-20, -21, -22, -23, -24, -25], [20, 21, 22, 23, 24, 25])  # Generic bounds as they are necessary for pygmo

    def get_nic(self):
        return np.shape(self.D)[0] + 1

    def get_nec(self):
        return 1

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def sef_initial_area(self, a0):
        self.A0 = a0

    def check_inequality_sign(self, current_weights):
        inf_point = np.array([1e23, 1e23])
        D_inf = np.array(
            [[inf_point[0] ** 2, inf_point[0] * inf_point[1], inf_point[1] ** 2, inf_point[0], inf_point[1], 1]])
        return np.dot(D_inf, current_weights)[0] > 0  # return True if infinity is outside, False if it is inside


def generate_AMS_data(vane_id, vaneAngleProblem, current_sun_angle_alpha_deg, current_sun_angle_beta_deg, alpha_1_range, alpha_2_range, optical_model_str ="Ideal_model", savefig=True, savedat=False, shadow_computation=0):
    # shadow computation; 0: without shadow effects, 1: with shadow effects, 2: both
    # TODO: WATCH OUT CHANGE FILENAME WRT OPTICAL MODEL USED
    xlabels = ["Tx", "Tx", "Ty"]
    ylabels = ["Ty", "Tz", "Tz"]

    alpha_1_range_grid, alpha_2_range_grid = np.meshgrid(alpha_1_range, alpha_2_range)
    flattened_alpha_1_range_grid = alpha_1_range_grid.reshape(-1)
    flattened_alpha_2_range_grid = alpha_2_range_grid.reshape(-1)

    # Current sun angles
    alpha_s_deg = np.deg2rad(current_sun_angle_alpha_deg)
    beta_s_deg = np.deg2rad(current_sun_angle_beta_deg)

    # Compute sun vector in body frame
    n_s = np.array([np.sin(alpha_s_deg) * np.cos(beta_s_deg),
                    np.sin(alpha_s_deg) * np.sin(beta_s_deg),
                    -np.cos(alpha_s_deg)])   # In the body reference frame

    vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), n_s,
                                                               vane_variable_optical_properties=False)  # and the next time you can put False
    print(f'vane_id={vane_id}, alpha_s_deg={round(np.rad2deg(alpha_s_deg), 1)}, beta_s_deg={round(np.rad2deg(beta_s_deg), 1)}')
    if (savefig): current_figs = [plt.figure(1), plt.figure(2), plt.figure(3)]
    if (shadow_computation==2):
        shadow_l = [0, 1]
    else:
        shadow_l = [shadow_computation]

    returned_arrays = []
    for m in shadow_l:
        if (m==1):
            SHADOW_BOOL = True
            current_fig_label = "With Shadow"
        elif (m==0):
            SHADOW_BOOL = False
            current_fig_label = "No shadow"
        else:
            raise Exception("Error. Wrong shadow condition in generate_AMS_data.")

        vaneAngleProblem.set_shadow_bool(SHADOW_BOOL)
        T_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Tx_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Ty_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        Tz_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        C_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
        for i, alpha_1 in enumerate(alpha_1_range):
            for j, alpha_2 in enumerate(alpha_2_range):
                T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2])
                T_[i, j] = np.linalg.norm(T)
                Tx_[i, j], Ty_[i, j], Tz_[i, j], C_[i, j] = T[0], T[1], T[2], np.sum(T ** 2) * (1 / 3)

        # Remove the shadow points, if any
        BT = T_
        BT[BT < 1e20], BT[BT > 1e20] = 0, 1
        if (SHADOW_BOOL):
            T_[T_ > 1e20], Tx_[Tx_ > 1e20], Ty_[Ty_ > 1e20], Tz_[Tz_ > 1e20], C_[C_ > 1e20]  = None, None, None, None, None
        flattened_BT, flattened_Tx, flattened_Ty, flattened_Tz  = BT.reshape(-1), Tx_.reshape(-1), Ty_.reshape(-1), Tz_.reshape(-1)

        # Write data to file for further processing
        array_to_save = np.stack([flattened_alpha_1_range_grid, flattened_alpha_2_range_grid,
                                  alpha_s_deg * np.ones(np.shape(flattened_alpha_1_range_grid)),
                                  beta_s_deg * np.ones(np.shape(flattened_alpha_1_range_grid)), flattened_BT,
                                  flattened_Tx, flattened_Ty, flattened_Tz], axis=1)
        returned_arrays.append(array_to_save)
        if (savedat):
            np.savetxt(
                f"./AMS/Datasets/{optical_model_str}/vane_{vane_id}/AMS_alpha_{round(np.rad2deg(alpha_s_deg), 1)}_beta_{round(np.rad2deg(beta_s_deg), 1)}_shadow_{str(SHADOW_BOOL)}.csv",
                array_to_save, delimiter=",",
                header='alpha_1, alpha_2, alpha_sun, beta_sun, Shadow_bool, Tx, Ty, Tz')

        if (savefig):
            # Plot the scatter-data
            plt.figure(1)
            plt.scatter(flattened_Tx, flattened_Ty, s=1, label=current_fig_label)
            plt.figure(2)
            plt.scatter(flattened_Tx, flattened_Tz, s=1, label=current_fig_label)
            plt.figure(3)
            plt.scatter(flattened_Ty, flattened_Tz, s=1, label=current_fig_label)

    if (savefig):
        for i in range(1, 4):
            plt.figure(i)
            plt.title(f'vane {vane_id}: alpha_s_deg={round(np.rad2deg(alpha_s_deg), 1)}, beta_s_deg={round(np.rad2deg(beta_s_deg), 1)}')
            plt.xlabel(xlabels[i-1])
            plt.ylabel(ylabels[i-1])
            plt.legend(loc='lower left')
            plt.savefig(f'./AMS/Plots/{optical_model_str}/vane_{vane_id}/plot_{i}/AMS_{i}_alpha_{round(np.rad2deg(alpha_s_deg), 1)}_beta_{round(np.rad2deg(beta_s_deg), 1)}.png')
            plt.close(current_figs[i-1])
    return returned_arrays, shadow_l    # shadow_l returned to indicate the shadow condition used in the calculation

def plot_and_files_wrapper_generate_AMS_data(vane_id,
                              vaneAngleProblem,
                              sun_angle_alpha_range,
                              sun_angle_beta_range,
                              alpha_1_range,
                              alpha_2_range,
                              optical_model_str = "Ideal_model",
                              savefig=True,
                              savedat=False,
                              shadow_computation=0):

    for current_angle_alpha in sun_angle_alpha_range:
        for current_angle_beta in sun_angle_beta_range:
            generate_AMS_data(vane_id, vaneAngleProblem, current_angle_alpha, current_angle_beta, alpha_1_range,
                              alpha_2_range, optical_model_str, savefig, savedat, shadow_computation)

"""
    Functions used in the vane controllers
"""
def fit_2d_ellipse(x, y):
    """
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

@jit(nopython=True)
def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        print(a, b, c, d, f, g)
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def fourierSumFunction(x, *bargs, order=4):
    """

    :param x: input array of length 2.
    :param b: coefficients array of length 1 + 2 * (order-1) + 2 * (order-1) + 4 * (len(combs) - 2 * (order-1)))
    :param order=4: order of the Fourier polynomial.
    :return:
    """
    comb_range = int(order)
    lst = list(range(comb_range))
    combinations = []
    for it in itertools.product(lst, repeat=2):
        combinations.append(list(it))
    combinations = combinations[1:]

    alpha_s = x[:, 0]
    beta_s = x[:, 1]

    b_ind = 0
    res = 1 * bargs[b_ind]; b_ind += 1
    for c in combinations:
        if (c[0] == 0):
            res += (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
        elif (c[1] == 0):
            res += (np.sin(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
        else:
            res += (np.sin(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.sin(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
    return res

def fourierSeriesFunction(x, *bargs, order_n=4, order_m=4):
    alpha_s = x[:, 0]
    beta_s = x[:, 1]
    b_ind = 0
    res = bargs[b_ind]; b_ind += 1
    for i in range(1, order_n + 1):
        for j in range(1, order_m + 1):
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
    return res

def combinedFourierFitFunction(x, *bargs, order=4, order_n=4, order_m=4):
    comb_range = int(order)
    lst = list(range(comb_range))
    combinations = []
    for it in itertools.product(lst, repeat=2):
        combinations.append(list(it))
    combinations = combinations[1:]

    alpha_s = x[:, 0]
    beta_s = x[:, 1]

    b_ind = 0
    res = 1 * bargs[b_ind]; b_ind += 1
    for c in combinations:
        if (c[0] == 0):
            res += (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
        elif (c[1] == 0):
            res += (np.sin(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * bargs[b_ind]; b_ind += 1
        else:
            res += (np.sin(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.sin(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.sin(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1
            res += (np.cos(alpha_s) ** c[0]) * (np.cos(beta_s) ** c[1]) * bargs[b_ind]; b_ind += 1

    for i in range(1, order_n + 1):
        for j in range(1, order_m + 1):
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.sin(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.sin(j * beta_s); b_ind += 1
            res += bargs[b_ind] * np.cos(i * alpha_s) * np.cos(j * beta_s); b_ind += 1
    return res