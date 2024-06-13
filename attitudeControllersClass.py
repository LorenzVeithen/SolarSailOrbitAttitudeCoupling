from constants import sail_mass, sail_I, sail_nominal_CoM
from scipy.spatial.transform import Rotation as R
import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point, compute_panel_geometrical_properties
from  ACS_dynamicalModels import vane_dynamical_model, shifted_panel_dynamical_model, sliding_mass_dynamical_model
from numba import jit
from vaneControllerMethods import buildEllipseCoefficientFunctions, ellipseCoefficientFunction, vaneTorqueAllocationProblem, vane_system_angles_from_desired_torque
from constants import AMS_directory, sail_I
from constants import c_sol, W
from constants import tol_rotational_velocity_orientation_change_update_vane_angles_degrees, tol_relative_change_in_rotational_velocity_magnitude
from constants import tol_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees
class sail_attitude_control_systems:

    def __init__(self, ACS_system, booms_coordinates_list, spacecraft_inertia_tensor=sail_I, include_shadow=False, sail_craft_name="ACS3"):
        # General
        self.sail_attitude_control_system = ACS_system          # String defining the ACS to be used
        self.bool_mass_based_controller = None                  # Boolean indicating if the ACS concept is mass-based. TODO: should this be depracated?
        self.ACS_mass = 0                                       # [kg] Total mass of the ACS. Initialised to zero and each instance of set_... adds to this sum
        self.ACS_CoM = None                                     # [m] Body-fixed center of mass of the total ACS. Initialised to the center of the spacecraft.
        self.include_shadow = include_shadow
        self.sail_craft_name = sail_craft_name
        self.spacecraft_inertia_tensor = spacecraft_inertia_tensor

        # Booms characteristics
        self.number_of_booms = len(booms_coordinates_list)      # [] Number of booms in the sail.
        self.booms_coordinates_list = booms_coordinates_list    # [m] List of 2x3 arrays (first row is boom origin, second row is boom tip). Assuming straight booms.

        # Vanes
        self.number_of_vanes = 0                                # [] Number of vanes of the ACS.
        self.vane_panels_coordinates_list = None                # [m] num_of_vanes long list of (num_of_vanes x 3) arrays of the coordinates of the polygons defining the vanes of the ACS.
        self.vane_reference_frame_origin_list = None            # [m] num_of_vanes long list of (1x3) arrays of the coordinates of the vane coordinate frame origins, around which the vane rotations are defined.
        self.vane_reference_frame_rotation_matrix_list = None   # num_of_vanes long list of (3x3) rotation matrices from the body fixed frame to the vane fixed frame.
        self.vane_material_areal_density = None
        self.vanes_rotational_dof_booleans = None               # num_of_vanes long list of lists of booleans [True, True] stating the rotational degree of freedom of each vane. 0: x and 1:y in vane coordinate frames
        self.vanes_areas_list = None
        self.latest_updated_vane_torques = [None]
        self.latest_updated_optimal_torque_allocation = [None]
        self.vane_mechanical_rotation_limits = None
        self.latest_updated_vane_angles = [[None]]
        self.body_fixed_rotational_velocity_at_last_vane_angle_update = [None]
        self.body_fixed_sunlight_vector_at_last_angle_update = None

        # Shifted wings (will also include tilted wings later)
        self.number_of_wings = 0                                # Number of wings in the sail.
        self.wings_number_of_points = 0
        self.wings_coordinates_list = None                      # [m] num_of_wings long list of (num_of_wings x 3) arrays of the coordinates of the polygons defining the wings of the ACS.
        self.wings_areas_list = None                            # [m^2] number_of_wings long list of the area of each wing panel.
        self.wings_reference_frame_rotation_matrix_list = None  # num_of_wings long list of (3x3) rotation matrices from from the body fixed frame to the wing fixed frame.
        self.retain_wings_area_bool = None                      # Bool indicating if the area of the panels should be conserved (pure translation of the panel).
        self.max_wings_inwards_translations_list = None         # num_of_wings long list of the maximum inward wing translation (vertical in the wing reference frame).
        self.point_to_boom_belonging_list = None

        # Sliding mass
        self.number_of_sliding_masses = 0
        self.sliding_masses_list = None                         # [kg] number_of_booms long list of the
        self.sliding_mass_extreme_positions_list = None         # [m] Extreme positions of the booms. Sliding masses are assumed to move in straight lines (along booms) TODO: incomplete
        self.sliding_mass_system_is_accross_two_booms = None    # TODO
        self.sliding_mass_unit_direction = None                 # TODO

        #
        self.number_of_gimballed_masses = 0
        self.gimball_mass = None

        # Reflectivity
        self.number_of_reflectivity_devices = 0

        # Summation variables
        self.ACS_CoM_stationary_components = np.array([0, 0, 0])
        self.actuator_states = {}
        self.actuator_states["vane_rotation_x_default"] = np.zeros((self.number_of_vanes, 1))
        self.actuator_states["vane_rotation_y_default"] = np.zeros((self.number_of_vanes, 1))
        self.actuator_states["sliding_masses_body_frame_positions_default"] = np.zeros((3 * self.number_of_sliding_masses, 1))
        self.actuator_states["gimballed_masses_body_frame_positions_default"] = np.zeros((3 * self.number_of_gimballed_masses, 1))
        self.actuator_states["wings_reflectivity_devices_values_default"] = np.zeros((self.number_of_reflectivity_devices, 1))
        self.actuator_states["wings_positions_default"] = np.zeros((3 * self.wings_number_of_points, 1))

        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]
        self.actuator_states["sliding_masses_body_frame_positions"] = self.actuator_states["sliding_masses_body_frame_positions_default"]
        self.actuator_states["gimballed_masses_body_frame_positions"] = self.actuator_states["gimballed_masses_body_frame_positions_default"]
        self.actuator_states["wings_reflectivity_devices_values"] = self.actuator_states["wings_reflectivity_devices_values_default"]
        self.actuator_states["wings_positions"] = self.actuator_states["wings_positions_default"]

    def computeBodyFrameTorqueForDetumbling(self, bodies, tau_max, desired_rotational_velocity_vector=np.array([0, 0, 0]), rotational_velocity_tolerance_rotations_per_hour=0.1, timeToPassivateACS=0):
        """
        Function computing the required torque for detumbling the spacecraft to rest. For a time-independent attitude
        control system, this function can be evaluated a single time.

        :param bodies:  tudatpy.kernel.numerical_simulation.environment.SystemOfBodies object containing the information
        on the bodies present in the TUDAT simulation.
        :param tau_max: Maximum input torque of the ACS at a given time.
        :param desired_rotational_velocity_vector=np.array([0, 0, 0]): desired final rotational velocity vector.
        :param rotational_velocity_tolerance=0.1: tolerance on the magnitude of the largest absolute  value of
        the components of the rotational velocity vector. Detumbling torque then becomes zero.
        :param timeToPassivateACS=0: Estimated time to passivate the attitude control system, to avoid a discontinuous
        actuator control.
        :return tau_star: the optimal control torque.

        References:
        Aghili, F. (2009). Time-optimal detumbling control of spacecraft. Journal of guidance, control, and dynamics,
        32(5), 1671-1675.
        """
        body_fixed_angular_velocity_vector = bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity
        if (np.linalg.norm(desired_rotational_velocity_vector - np.array([0, 0, 0]))>1e-15):
            if (len(desired_rotational_velocity_vector[desired_rotational_velocity_vector > 0]) > 1):
                raise Exception("The desired final rotational velocity vector in " +
                                "computeBodyFrameTorqueForDetumblingToNonZeroFinalVelocity " +
                                "has more than one non-zero element. Spin-stabilised spacecraft should be about an " +
                                "Eigen-axis.")
            elif (np.count_nonzero(
                    self.spacecraft_inertia_tensor - np.diag(np.diagonal(self.spacecraft_inertia_tensor))) != 0):
                raise Exception("computeBodyFrameTorqueForDetumblingToNonZeroFinalVelocity is only valid for " +
                                " axisymmetric spacecrafts.")
        omega_tilted = body_fixed_angular_velocity_vector - desired_rotational_velocity_vector

        if (max(abs(omega_tilted)) * 3600 / (2 * np.pi) < rotational_velocity_tolerance_rotations_per_hour):
            return np.array([0, 0, 0])

        sail_craft_inertia_tensor = self.spacecraft_inertia_tensor

        inertiaTensorTimesAngularVelocity = np.dot(sail_craft_inertia_tensor, omega_tilted)
        predictedTimeToRest = np.linalg.norm(inertiaTensorTimesAngularVelocity)/tau_max

        if ((predictedTimeToRest < timeToPassivateACS) and (timeToPassivateACS != 0)):
            tau_target = (timeToPassivateACS - predictedTimeToRest)/timeToPassivateACS  # Linearly decreasing the torque applied such that the ACS is turned OFF smoothly
        else:
            tau_target = tau_max
        tau_star = - (inertiaTensorTimesAngularVelocity/np.linalg.norm(inertiaTensorTimesAngularVelocity)) * tau_target
        return tau_star.reshape(-1, 1)

    def attitude_control(self, bodies, desired_sail_body_frame_inertial_rotational_velocity):
        # Returns an empty array if nothing has changed
        wings_coordinates = []
        wings_optical_properties = []
        vanes_coordinates = []
        vanes_optical_properties = []

        moving_masses_CoM_components = np.array([0, 0, 0])
        moving_masses_positions = {}
        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]
        self.actuator_states["sliding_masses_body_frame_positions"] = self.actuator_states["sliding_masses_body_frame_positions_default"]
        self.actuator_states["gimballed_masses_body_frame_positions"] = self.actuator_states["gimballed_masses_body_frame_positions_default"]
        self.actuator_states["wings_reflectivity_devices_values"] = self.actuator_states["wings_reflectivity_devices_values_default"]
        self.actuator_states["wings_positions"] = self.actuator_states["wings_positions_default"]
        if (bodies != None):
            match self.sail_attitude_control_system:
                case "gimball_mass":
                    self.__pure_gimball_mass(bodies, desired_sail_body_frame_inertial_rotational_velocity)
                    moving_masses_CoM_components = np.zeros([0, 0, 0])
                    moving_masses_positions["gimball_mass"] = np.array([0, 0, 0], dtype="float64")
                case "vanes":
                    # Here comes the controller of the vanes, which will give the rotations around the x and y axis in the
                    # vane coordinate frame
                    sunlight_vector_inertial_frame = (bodies.get_body(self.sail_craft_name).position - bodies.get_body(
                        "Sun").position) / np.linalg.norm(
                        bodies.get_body(self.sail_craft_name).position - bodies.get_body("Sun").position)
                    R_IB = bodies.get_body(self.sail_craft_name).inertial_to_body_fixed_frame
                    sunlight_vector_body_frame = np.dot(R_IB, sunlight_vector_inertial_frame)

                    if (self.body_fixed_rotational_velocity_at_last_vane_angle_update[0] != None):
                        # Check how much the rotational velocity vector orientation has changed
                        change_in_rotational_velocity_orientation_rad = np.arccos(np.dot(bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity,
                                                                           self.body_fixed_rotational_velocity_at_last_vane_angle_update)/
                                                                     (np.linalg.norm(bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity)
                                                                      * np.linalg.norm(self.body_fixed_rotational_velocity_at_last_vane_angle_update)))

                        # Check how much the rotational velocity vector magnitude has changed
                        relative_change_in_rotational_velocity_magnitude = ((np.linalg.norm(bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity)
                                                                            -np.linalg.norm(self.body_fixed_rotational_velocity_at_last_vane_angle_update))/
                                                                            np.linalg.norm(self.body_fixed_rotational_velocity_at_last_vane_angle_update))

                        # Check how much the sunlight vector in the body frame has changed
                        change_in_body_fixed_sunlight_vector_orientation_rad = np.arccos(np.dot(sunlight_vector_body_frame,
                                                                           self.body_fixed_sunlight_vector_at_last_angle_update)/
                                                                     (np.linalg.norm(sunlight_vector_body_frame)
                                                                      * np.linalg.norm(self.body_fixed_sunlight_vector_at_last_angle_update)))
                    else:
                        # dummy values to force update
                        change_in_rotational_velocity_orientation_rad = 10
                        relative_change_in_rotational_velocity_magnitude = 2
                        change_in_body_fixed_sunlight_vector_orientation_rad = -1

                    if (np.rad2deg(change_in_rotational_velocity_orientation_rad) > tol_rotational_velocity_orientation_change_update_vane_angles_degrees
                        or np.rad2deg(change_in_rotational_velocity_orientation_rad) < 0
                        or np.rad2deg(change_in_body_fixed_sunlight_vector_orientation_rad) > tol_sunlight_vector_body_frame_orientation_change_update_vane_angles_degrees
                        or np.rad2deg(change_in_body_fixed_sunlight_vector_orientation_rad) < 0
                        or relative_change_in_rotational_velocity_magnitude > tol_relative_change_in_rotational_velocity_magnitude):
                        current_solar_irradiance = W     #TODO: extract solar irradiance from the body object
                        required_body_torque = self.computeBodyFrameTorqueForDetumbling(bodies,
                                                                                        5e-5,
                                                                                        desired_rotational_velocity_vector=desired_sail_body_frame_inertial_rotational_velocity,
                                                                                        rotational_velocity_tolerance_rotations_per_hour=0.1,
                                                                                        timeToPassivateACS=0)
                        required_body_torque = required_body_torque.reshape(-1)/(current_solar_irradiance/c_sol)

                        if (np.linalg.norm(required_body_torque) > 1e-15):
                            previous_optimal_torque = self.latest_updated_optimal_torque_allocation if (self.latest_updated_optimal_torque_allocation[0]==None) else self.latest_updated_optimal_torque_allocation

                            controller_vane_angles, vane_torques, optimal_torque_allocation = vane_system_angles_from_desired_torque(self,
                                                                                                            self.vane_mechanical_rotation_limits,
                                                                                                            required_body_torque,
                                                                                                            previous_optimal_torque,
                                                                                                            sunlight_vector_body_frame,
                                                                                                            initial_vane_angles_guess_rad=self.latest_updated_vane_angles)

                        else:
                            controller_vane_angles = np.zeros((self.number_of_vanes, 2))
                            vane_torques = np.zeros((self.number_of_vanes, 3))
                            optimal_torque_allocation = np.zeros((self.number_of_vanes, 3))
                            previous_optimal_torque = self.latest_updated_vane_torques

                        #if (previous_optimal_torque[0] != None):
                        #    print("Difference between previous and new torque")
                        #    print(vane_torques.reshape(-1)-previous_optimal_torque)
                        vane_torques = vane_torques * current_solar_irradiance / c_sol
                        #print(optimal_torque_allocation)
                        print(f"required torque:{required_body_torque}")
                        print(f"optimal torque:{optimal_torque_allocation.sum(axis=0)}")
                        print(f"vane torque: {vane_torques.sum(axis=0) * (current_solar_irradiance / c_sol)**-1}")
                        print(f"rotations per hour {bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity * 3600 / (2 * np.pi)}")

                        self.latest_updated_vane_torques = vane_torques.reshape(-1)
                        self.latest_updated_vane_angles = controller_vane_angles
                        self.latest_updated_optimal_torque_allocation = optimal_torque_allocation.reshape(-1)
                        self.body_fixed_rotational_velocity_at_last_vane_angle_update = bodies.get_body(self.sail_craft_name).body_fixed_angular_velocity
                        self.body_fixed_sunlight_vector_at_last_angle_update = sunlight_vector_body_frame

                    #print( self.latest_updated_vane_angles)
                    vane_x_rotation_degrees, vane_y_rotation_degrees = np.rad2deg(self.latest_updated_vane_angles[:, 0]),  np.rad2deg(self.latest_updated_vane_angles[:, 1])
                    self.actuator_states["vane_rotation_x"] = np.deg2rad(vane_x_rotation_degrees.reshape(-1, 1))
                    self.actuator_states["vane_rotation_y"] = np.deg2rad(vane_y_rotation_degrees.reshape(-1, 1))
                    vanes_coordinates = self.__vane_dynamics(vane_x_rotation_degrees, vane_y_rotation_degrees)
                case "shifted_wings":
                    wing_shifts_list = [[-0.4, -0.4, -0.4, -0.4],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 0]]
                    wings_coordinates = self.__shifted_panel_dynamics(wing_shifts_list)
                case "sliding_masses":

                    sliding_masses_CoM, sliding_masses_positions_body_fixed_frame_list = self.__sliding_mass_dynamics([-3.5, -4])
                    moving_masses_CoM_components = sliding_masses_CoM * sum(self.sliding_masses_list)

                    moving_masses_positions["sliding_masses"] = sliding_masses_positions_body_fixed_frame_list
                case "None":
                    # No attitude control system - the spacecraft remains inert
                    pass
                case "test":
                    pass
                case _:
                    raise Exception("Selected ACS not available yet.")

        self.compute_attitude_system_center_of_mass(vanes_coordinates, moving_masses_CoM_components)
        # the attitude-control algorithm should give the output
        panels_coordinates, panels_optical_properties = {}, {}
        panels_coordinates["wings"] = wings_coordinates
        panels_coordinates["vanes"] = vanes_coordinates
        panels_optical_properties["wings"] = wings_optical_properties
        panels_optical_properties["vanes"] = vanes_optical_properties
        return panels_coordinates, panels_optical_properties, self.ACS_CoM, moving_masses_positions  # TODO: Be careful to link self.ACS_CoM with the outgoing variable here

    def compute_attitude_system_center_of_mass(self, vanes_coordinates, moving_masses_CoM_components):
        # Compute the complete ACS center of mass (excluding the components due to the wings)
        if (self.ACS_mass == 0):
            self.ACS_CoM = np.array([0, 0, 0])
        else:
            ACS_CoM = np.array([0, 0, 0], dtype="float64")
            ACS_CoM += self.ACS_CoM_stationary_components
            ACS_CoM += moving_masses_CoM_components
            if (len(vanes_coordinates) != 0):
                for i in range(len(vanes_coordinates)):
                    vane_centroid, vane_area, _ = compute_panel_geometrical_properties(vanes_coordinates[i])
                    ACS_CoM += vane_centroid * (vane_area * self.vane_material_areal_density)

            self.ACS_CoM = ACS_CoM / self.ACS_mass
        return True

    # Controllers
    def __pure_gimball_mass(self, current_sail_state, desired_sail_state):
        return

    # ACS characteristics inputs and dynamics
    def set_vane_characteristics(self, vanes_coordinates_list,
                                 vanes_reference_frame_origin_list,
                                 vanes_reference_frame_rotation_matrix_list,
                                 stationary_system_components_mass,
                                 stationary_system_system_CoM,
                                 vanes_material_areal_density,
                                 vanes_rotational_dof_booleans,
                                 vane_has_ideal_model,
                                 wings_coordinates_list,
                                 vane_mechanical_rotation_bounds,
                                 torque_allocation_problem_objective_function_weights=[1, 0],
                                 directory_feasibility_ellipse_coefficients=f'{AMS_directory}/Datasets/Ideal_model/vane_1/dominantFitTerms'):
        """
        Function setting the characteristics of the ACS vanes actuator.
        Should be called a single time.
        :param vanes_coordinates_list:
        :param vanes_reference_frame_origin_list:
        :param vanes_reference_frame_rotation_matrix_list:
        :param stationary_system_components_mass:
        :param stationary_system_system_CoM:
        :param vanes_material_areal_density:
        :return: True if the process was completed successfully
        """
        self.ACS_mass += stationary_system_components_mass  # WITHOUT VANES, which are taken into account below
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.number_of_vanes = len(vanes_reference_frame_origin_list)
        self.vane_panels_coordinates_list = vanes_coordinates_list
        self.vane_reference_frame_origin_list = vanes_reference_frame_origin_list
        self.vane_reference_frame_rotation_matrix_list = vanes_reference_frame_rotation_matrix_list
        self.vanes_rotational_dof_booleans = vanes_rotational_dof_booleans
        self.actuator_states["vane_rotation_x_default"] = np.zeros((self.number_of_vanes, 1))
        self.actuator_states["vane_rotation_y_default"] = np.zeros((self.number_of_vanes, 1))
        self.vane_mechanical_rotation_limits = [(vane_mechanical_rotation_bounds[0][i], vane_mechanical_rotation_bounds[1][i]) for i in
                                       range(len(vane_mechanical_rotation_bounds[0]))]

        # Determine vane component of the ACS mass
        vanes_areas = []
        for i in range(len(self.vane_panels_coordinates_list)):
            _, vane_area, _ = compute_panel_geometrical_properties(self.vane_panels_coordinates_list[i])
            vanes_areas.append(vane_area)
        self.vanes_areas_list = vanes_areas
        self.vane_material_areal_density = vanes_material_areal_density
        self.ACS_mass = sum(vanes_areas) * vanes_material_areal_density

        # Determine if a vane is on a boom
        self.vane_is_aligned_on_body_axis = [False] * self.number_of_vanes
        for vane_id, vane in enumerate(self.vane_panels_coordinates_list):
            vane_attachment_point = vane[0, :]  # as per convention
            if (np.shape(np.nonzero(vane_attachment_point)[0])[0]==1):
                self.vane_is_aligned_on_body_axis[vane_id] = True
        self.vane_is_aligned_on_body_axis = np.array(self.vane_is_aligned_on_body_axis)

        # feasible torque ellipse coefficients
        ellipse_coefficient_functions_list = []
        for i in range(6):
            filename = f'{directory_feasibility_ellipse_coefficients}/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{str(self.include_shadow)}.txt'
            built_function = buildEllipseCoefficientFunctions(filename)
            ellipse_coefficient_functions_list.append(
                lambda aps, bes, f=built_function: ellipseCoefficientFunction(aps, bes, f))

        # vane torque allocation problem
        self.vane_torque_allocation_problem_object = vaneTorqueAllocationProblem(self,
                                                                  wings_coordinates_list,
                                                                  vane_has_ideal_model,
                                                                  self.include_shadow,
                                                                  ellipse_coefficient_functions_list,
                                                                  w1=torque_allocation_problem_objective_function_weights[0],
                                                                  w2=torque_allocation_problem_objective_function_weights[1])

        return True

    def __vane_dynamics(self, rotation_x_deg, rotation_y_deg):
        # Get the vane panel coordinates as a result of the rotation
        # Based on the initial vane position and orientation in the body frame
        if (not all(np.rad2deg(self.vane_mechanical_rotation_limits[0][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[0][1]) for angle in rotation_x_deg)
                or not all(np.rad2deg(self.vane_mechanical_rotation_limits[1][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) for angle in rotation_y_deg)):
            print(all(np.rad2deg(self.vane_mechanical_rotation_limits[0][0]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[0][1]) for angle in rotation_x_deg))
            print(all(np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) <= angle <= np.rad2deg(self.vane_mechanical_rotation_limits[1][1]) for angle in rotation_y_deg))
            raise Exception("Requested vane deflection is not permitted:" + f"x-rotation={rotation_x_deg} degrees and y-rotation={rotation_y_deg} degrees.")

        if (self.vane_reference_frame_origin_list == None
                or self.vane_panels_coordinates_list == None
                or self.vane_reference_frame_rotation_matrix_list == None
                or self.number_of_vanes == 0):
            raise Exception("Vane characteristics have not been set by the user.")

        new_vane_coordinates = vane_dynamical_model(rotation_x_deg,
                                                    rotation_y_deg,
                                                    self.number_of_vanes,
                                                    self.vane_reference_frame_origin_list,
                                                    self.vane_panels_coordinates_list,
                                                    self.vane_reference_frame_rotation_matrix_list)
        return new_vane_coordinates

    def set_shifted_panel_characteristics(self, wings_coordinates_list, wings_areas_list, wings_reference_frame_rotation_matrix_list, keep_constant_area, stationary_system_components_mass, stationary_system_system_CoM):
        """
        Function setting the characteristics of the shifted panels actuator.
        Should be called a single time.
        :param wings_coordinates_list:
        :param wings_areas_list:
        :param wings_reference_frame_rotation_matrix_list:
        :param keep_constant_area:
        :param stationary_system_components_mass:
        :param stationary_system_system_CoM:
        :return: True if the process was completed successfully
        """
        self.ACS_mass += stationary_system_components_mass                      # Includes only the mechanisms, the wings are already elsewhere
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.wings_coordinates_list = wings_coordinates_list
        self.number_of_wings = len(self.wings_coordinates_list)
        self.retain_wings_area_bool = keep_constant_area
        self.wings_areas_list = wings_areas_list
        self.wings_reference_frame_rotation_matrix_list = wings_reference_frame_rotation_matrix_list

        #TODO: Could be broken, check
        horizontal_stack = np.array([])
        for wing in self.wings_coordinates_list:
            self.wings_number_of_points += np.shape(wing)[0]
            for i in range(np.shape(wing)[0]):
                horizontal_stack = np.hstack((horizontal_stack, wing[i, :]))
        self.actuator_states["wings_positions_default"] = horizontal_stack.reshape(-1, 1)   # Make vertical again

        if (keep_constant_area):
            # Check that the sail geometry makes sense: the vertical space between the wings edge and the boom should be non-zero
            # Determine the maximum vertical translation of each panel (should all be equal)
            # There are some constraints on the defined geometry: need to have space between the boom and the attachment point to allow this move (see image in thesis document)
            # Check the necessary constraints on the geometry of the sail itself to be sure that it makes sense
            self.max_wings_inwards_translations_list = []
            for i, wing_coords in enumerate(wings_coordinates_list):
                wing_maximum_negative_vertical_displacement = -1e10
                for point in wing_coords[:, :3]:
                    # Determine the closest point on each point
                    distance_to_booms_list = []
                    for boom in self.booms_coordinates_list:
                        boom_origin = boom[0, :]     # Boom origin
                        boom_tip = boom[1, :]        # Boom tip
                        closest_point = closest_point_on_a_segment_to_a_third_point(boom_origin, boom_tip, point)
                        distance = np.linalg.norm(point-closest_point)
                        distance_to_booms_list.append(distance)

                    # Determine closest boom
                    closest_boom_index = distance_to_booms_list.index(min(distance_to_booms_list))  # Both booms could be valid for a single point in the 3 attachments case, but this does not matter too much

                    # Determine the maximum translation along the Y-axis of the wing frame based on the geometry
                    ## Get point coordinates in wing frame
                    point_coordinates_wing_reference_frame = np.matmul(np.linalg.inv(
                        self.wings_reference_frame_rotation_matrix_list[i]),
                        point)

                    boom_origin_coordinates_wing_reference_frame = np.matmul(np.linalg.inv(
                        self.wings_reference_frame_rotation_matrix_list[i]),
                        self.booms_coordinates_list[closest_boom_index][0, :])

                    boom_tip_coordinates_wing_reference_frame = np.matmul(
                        np.linalg.inv(self.wings_reference_frame_rotation_matrix_list[i]),
                        self.booms_coordinates_list[closest_boom_index][1, :])

                    point_x_wing_reference_frame = point_coordinates_wing_reference_frame[0]
                    point_y_wing_reference_frame = point_coordinates_wing_reference_frame[1]

                    # Line equation of the selected boom in the wing reference frame
                    boom_dy = boom_tip_coordinates_wing_reference_frame[1]-boom_origin_coordinates_wing_reference_frame[1]
                    boom_dx = boom_tip_coordinates_wing_reference_frame[0]-boom_origin_coordinates_wing_reference_frame[0]

                    a = boom_dy/boom_dx
                    b = boom_tip_coordinates_wing_reference_frame[1] - a * boom_tip_coordinates_wing_reference_frame[0]
                    point_maximum_negative_vertical_displacement = (a * point_x_wing_reference_frame + b) - point_y_wing_reference_frame
                    if (point_maximum_negative_vertical_displacement >= 0):
                        print(point_maximum_negative_vertical_displacement)
                        raise Exception("Maximum negative vertical displacement is positive or equal to zero")
                    wing_maximum_negative_vertical_displacement = max(point_maximum_negative_vertical_displacement, wing_maximum_negative_vertical_displacement)
                self.max_wings_inwards_translations_list.append(wing_maximum_negative_vertical_displacement)

        if (not keep_constant_area):
            # Determine which points are on which booms: check that the dot product between position vectors of the attachment point and the boom tip is positive and smaller than the boom length squared
            # Do it only once in either case
            point_to_boom_belonging_list = []
            for i in range(self.number_of_wings):
                current_panel_coordinates = self.wings_coordinates_list[i]
                point_to_boom_belonging = []
                for point in current_panel_coordinates[:, :3]:
                    found_boom = False
                    for j, boom in enumerate(self.booms_coordinates_list):
                        point_position_vector_with_respect_to_boom_origin = point - boom[0, :]
                        boom_tip_position_vector_with_respect_to_boom_origin = boom[1, :] - boom[0, :]

                        if (np.linalg.norm(np.cross(point_position_vector_with_respect_to_boom_origin,
                                                    boom_tip_position_vector_with_respect_to_boom_origin)) < 1e-15          # Check that the points are aligned
                                and np.dot(point_position_vector_with_respect_to_boom_origin,
                                           boom_tip_position_vector_with_respect_to_boom_origin) > 0                        # Check that you are on the correct side of the infinite line
                                and np.linalg.norm(point_position_vector_with_respect_to_boom_origin) <= np.linalg.norm(
                                    boom_tip_position_vector_with_respect_to_boom_origin)):                                 # Check that you are not beyond the line end
                            # The point is on the selected boom
                            point_to_boom_belonging.append(j)    # list index of the boom to which the point belongs
                            found_boom = True
                            break  # Can go to the next point
                    if (not found_boom): point_to_boom_belonging.append(None)  # The point was not found to belong to any boom
                if (all(point_to_boom_belonging) == None):
                    print(f"Warning. Boom {i} does not have any attachment point on the booms.")
                point_to_boom_belonging_list.append(point_to_boom_belonging)
            self.point_to_boom_belonging_list = point_to_boom_belonging_list
        return True

    def __shifted_panel_dynamics(self, wings_shifts_list):
        # panel_shifts_list: [[del_p1p1, del_p1p2, ...], [del_p2p1, del_p2p2, ...], ...] in meters along the boom
        wing_coordinates_list = shifted_panel_dynamical_model(wings_shifts_list,
                                  self.number_of_wings,
                                  self.wings_coordinates_list,
                                  self.wings_reference_frame_rotation_matrix_list,
                                  self.retain_wings_area_bool,
                                  self.point_to_boom_belonging_list,
                                  self.max_wings_inwards_translations_list,
                                  self.booms_coordinates_list)
        return wing_coordinates_list


    def set_sliding_masses_characteristics(self, sliding_masses_list, stationary_system_components_mass, stationary_system_system_CoM, sliding_mass_system_type=0):
        self.ACS_mass += stationary_system_components_mass + sum(sliding_masses_list)
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.bool_mass_based_controller = True
        self.sliding_masses_list = sliding_masses_list      # [m1, m2, m3, m4] in the same order as the booms or [m1, m2] if the system type is 1. (for standard cross sail)
        self.number_of_sliding_masses = len(sliding_masses_list)
        self.actuator_states["sliding_masses_body_frame_positions_default"] = np.zeros((3 * (self.number_of_sliding_masses ), 1))

        if (sliding_mass_system_type == 0):
            # 1 mass per boom (2 for a nominal square configuration)
            self.sliding_mass_extreme_positions_list = self.booms_coordinates_list
            self.sliding_mass_system_is_accross_two_booms = [False] * len(self.booms_coordinates_list)

        elif (sliding_mass_system_type == 1):
            # 1 mass per aligned boom (2 for a nominal square configuration)
            ## Determine which booms are aligned with each other
            ## Assume that no more than 2 booms are aligned with each other (would not make sense otherwise
            aligned_booms_list = []
            boom_availability = [True] * self.number_of_booms
            for i in range(self.number_of_booms):
                boom_1_vector = self.booms_coordinates_list[i][1, :] - self.booms_coordinates_list[i][0, :]
                if (boom_availability[i]): # Only consider the element if it was not found as aligned before
                    for j in range(i+1, self.number_of_booms):
                        if (boom_availability[j]):
                            boom_2_vector = self.booms_coordinates_list[j][1, :] -self.booms_coordinates_list[j][0, :]
                            if (np.linalg.norm(np.cross(boom_1_vector, boom_2_vector)) < 1e-15):
                                # Store that the two booms are linked
                                aligned_booms_list.append((i, j))

                                # Reduce these booms to zero length to show that they are done
                                #temporary_boom_list[i][0, :] = temporary_boom_list[i][1, :]
                                #temporary_boom_list[j][0, :] = temporary_boom_list[j][1, :]
                                boom_availability[i] = False
                                boom_availability[j] = False

            # Determine booms which are not aligned with others
            isolated_booms_list = [i for i, val in enumerate(boom_availability) if val]
            independent_sliding_mass_systems = aligned_booms_list
            if (len(isolated_booms_list)!=0): independent_sliding_mass_systems += isolated_booms_list

            # Loop through the independent sliding masses and determine the end points
            self.sliding_mass_extreme_positions_list = [None] * len(independent_sliding_mass_systems)
            self.sliding_mass_system_is_accross_two_booms = []
            for sliding_mass in independent_sliding_mass_systems:
                if (type(sliding_mass)==tuple):
                    self.sliding_mass_system_is_accross_two_booms.append(True)
                    # These are combined booms, find the end points. Convention: points in quadrant I and IV are the "tip" of the direction vector
                    first_point_is_tip = False  # First assume that the first point is not the tip
                    first_point_x = self.booms_coordinates_list[sliding_mass[0]][1][0]
                    first_point_y = self.booms_coordinates_list[sliding_mass[0]][1][1]
                    if (first_point_x == 0):
                        if (first_point_y > 0): first_point_is_tip = True
                    elif (first_point_y == 0):
                        if (first_point_x > 0): first_point_is_tip = True
                    elif (first_point_y/first_point_x > 0):
                        first_point_is_tip = True

                    if (first_point_is_tip):
                        origin = self.booms_coordinates_list[sliding_mass[1]][1]
                        tip = self.booms_coordinates_list[sliding_mass[0]][1]
                    else:
                        origin = self.booms_coordinates_list[sliding_mass[0]][1]
                        tip = self.booms_coordinates_list[sliding_mass[1]][1]
                    self.sliding_mass_extreme_positions_list[sliding_mass[0]] = np.array([origin, tip])
                else:
                    # individual boom, the end points of the mass movement are the coordinates of the origin and tip of the boom
                    self.sliding_mass_extreme_positions_list[sliding_mass] = self.booms_coordinates_list[sliding_mass]
                    self.sliding_mass_system_is_accross_two_booms.append(False)

        self.sliding_mass_unit_direction = []
        for sm in self.sliding_mass_extreme_positions_list:
            sm_vector = sm[1, :] - sm[0, :]
            sm_vector_unit = sm_vector / np.linalg.norm(sm_vector)
            self.sliding_mass_unit_direction.append(sm_vector_unit)
        return True

    def __sliding_mass_dynamics(self, displacement_from_boom_origin_list):
        if (self.sliding_masses_list == None or self.sliding_mass_extreme_positions_list == None
            or self.sliding_mass_system_is_accross_two_booms == None or self.sliding_mass_unit_direction == None):
            raise Exception("Error. Sliding mass system was not propertly initialised.")

        return sliding_mass_dynamical_model(displacement_from_boom_origin_list,
                                         self.sliding_masses_list,
                                         self.sliding_mass_extreme_positions_list,
                                         self.sliding_mass_system_is_accross_two_booms,
                                         self.sliding_mass_unit_direction)



    def is_mass_based(self):
        return self.bool_mass_based_controller

    def set_gimball_mass_chateristics(self, mass_of_gimbaled_ballast):
        self.bool_mass_based_controller = True
        self.gimbaled_mass = mass_of_gimbaled_ballast
        return True

    def get_attitude_system_mass(self):
        return self.ACS_mass

    def initialise_actuator_states_dictionary(self):
        self.actuator_states["vane_rotation_x"] = self.actuator_states["vane_rotation_x_default"]
        self.actuator_states["vane_rotation_y"] = self.actuator_states["vane_rotation_y_default"]
        self.actuator_states["sliding_masses_body_frame_positions"] = self.actuator_states["sliding_masses_body_frame_positions_default"]
        self.actuator_states["gimballed_masses_body_frame_positions"] = self.actuator_states["gimballed_masses_body_frame_positions_default"]
        self.actuator_states["wings_reflectivity_devices_values"] = self.actuator_states["wings_reflectivity_devices_values_default"]
        self.actuator_states["wings_positions"] = self.actuator_states["wings_positions_default"]
        return True
    def get_attitude_control_system_actuators_states(self):
        # convert the actuator states variable to something compatible with the dependent_variables history
        keys_list = ["vane_rotation_x", "vane_rotation_y",
                     "sliding_masses_body_frame_positions", "gimballed_masses_body_frame_positions",
                     "wings_reflectivity_devices_values", "wings_positions"]
        dependent_variable_array = np.array([[0]])
        for key in keys_list:
            if key in self.actuator_states.keys():
                dependent_variable_array = np.vstack((dependent_variable_array, self.actuator_states[key]))
        dependent_variable_array = dependent_variable_array[1:, 0]
        return dependent_variable_array