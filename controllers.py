from constants import sail_mass, sail_I, sail_nominal_CoM
from scipy.spatial.transform import Rotation as R
import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point, compute_panel_geometrical_properties


class sail_attitude_control_systems:

    def __init__(self, ACS_system, booms_coordinates_list):
        # General
        self.sail_attitude_control_system = ACS_system          # String defining the ACS to be used
        self.bool_mass_based_controller = None                  # Boolean indicating if the ACS concept is mass-based. TODO: should this be depracated?
        self.ACS_mass = 0                                       # [kg] Total mass of the ACS. Initialised to zero and each instance of set_... adds to this sum
        self.ACS_CoM = None                                     # [m] Body-fixed center of mass of the total ACS. Initialised to the center of the spacecraft.

        # Booms characteristics. TODO: this could be made more general because it is used in most systems considered
        self.number_of_booms = len(booms_coordinates_list)      # [] Number of booms in the sail.
        self.booms_coordinates_list = booms_coordinates_list    # [m] List of 2x3 arrays (first row is boom origin, second row is boom tip). Assuming straight booms.

        # Vanes
        self.number_of_vanes = None                             # [] Number of vanes of the ACS.
        self.vane_panels_coordinates_list = None                # [m] num_of_vanes long list of (num_of_vanes x 3) arrays of the coordinates of the polygons defining the vanes of the ACS.
        self.vane_reference_frame_origin_list = None            # [m] num_of_vanes long list of (1x3) arrays of the coordinates of the vane coordinate frame origins, around which the vane rotations are defined.
        self.vane_reference_frame_rotation_matrix_list = None   # num_of_vanes long list of (3x3) rotation matrices from from the body fixed frame to the vane fixed frame.
        self.vane_material_areal_density = None

        # Shifted wings
        self.number_of_wings = None                             # Number of wings in the sail.
        self.wings_coordinates_list = None                      # [m] num_of_wings long list of (num_of_vanes x 3) arrays of the coordinates of the polygons defining the wings of the ACS.
        self.wings_areas_list = None                            # [m^2] number_of_wings long list of the area of each wing panel.
        self.wings_reference_frame_rotation_matrix_list = None  # num_of_wings long list of (3x3) rotation matrices from from the body fixed frame to the wing fixed frame.
        self.retain_wings_area_bool = None                      # Bool indicating if the area of the panels should be conserved (pure translation of the panel).
        self.max_wings_inwards_translations_list = None         # num_of_wings long list of the maximum inward wing translation (vertical in the wing reference frame).

        # Sliding mass
        self.sliding_masses_list = None                         # [kg] number_of_booms long list of the
        self.sliding_mass_extreme_positions_list = None         # [m] Extreme positions of the booms. Sliding masses are assumed to move in straight lines (along booms) TODO: incomplete
        self.sliding_mass_system_is_accross_two_booms = None    # TODO
        self.sliding_mass_unit_direction = None                 # TODO

        #
        self.gimball_mass = None

        # Summation variables
        self.ACS_CoM_stationary_components = np.array([0, 0, 0])

    def attitude_control(self, bodies, desired_sail_state):
        # Returns an empty array if nothing has changed
        wings_coordinates = []
        wings_optical_properties = []
        vanes_coordinates = []
        vanes_optical_properties = []
        ACS_CoM = np.array([0, 0, 0])
        thrust_levels = []

        moving_masses_CoM_components = np.array([0, 0, 0])
        moving_masses_positions = {}
        match self.sail_attitude_control_system:
            case "gimball_mass":
                self.__pure_gimball_mass(bodies, desired_sail_state)
                moving_masses_CoM_components = np.zeros([0, 0, 0])
                moving_masses_positions["gimball_mass"] = np.array([0, 0, 0], dtype="float64")
            case "vanes":
                vanes_coordinates = self.__vane_dynamics([-20, -20, -20, -20], [-45, -45, -45, -45])
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
                # No attitude control system
                pass    # TODO: is this really good practice to keep a pass statement
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

    def __pure_vane_controller(self, desired_sail_state):

        return

    # ACS characteristics inputs and dynamics
    def set_vane_characteristics(self, vanes_coordinates_list, vane_reference_frame_origin_list, vanes_reference_frame_rotation_matrix_list, stationary_system_components_mass, stationary_system_system_CoM, vane_material_areal_density):
        self.ACS_mass += stationary_system_components_mass  # WITHOUT VANES, which are taken into account below
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.number_of_vanes = len(vane_reference_frame_origin_list)
        self.vane_panels_coordinates_list = vanes_coordinates_list
        self.vane_reference_frame_origin_list = vane_reference_frame_origin_list
        self.vane_reference_frame_rotation_matrix_list = vanes_reference_frame_rotation_matrix_list
        vanes_areas = []
        for i in range(len(self.vane_panels_coordinates_list)):
            _, vane_area, _ = compute_panel_geometrical_properties(self.vane_panels_coordinates_list[i])
            vanes_areas.append(vane_area)
        self.vane_material_areal_density = vane_material_areal_density
        self.ACS_mass = sum(vanes_areas) * vane_material_areal_density
        return True

    def __vane_dynamics(self, rotation_x_deg, rotation_y_deg):
        # Get the vane panel coordinates as a result of the rotation
        # Based on the initial vane position and orientation in the body frame
        if (not all(-90 <= angle <= 90 for angle in rotation_x_deg) or not all(-90 <= angle <= 90 for angle in rotation_y_deg)):
            raise Exception("Requested vane deflection is not permitted:" + f"x-rotation={rotation_x_deg} degrees and y-rotation={rotation_y_deg} degrees.")

        if (self.vane_reference_frame_origin_list == None
                or self.vane_panels_coordinates_list == None
                or self.vane_reference_frame_rotation_matrix_list == None):
            raise Exception("Vane characteristics have not been set by the user.")

        new_vane_coordinates = []
        for i in range(self.number_of_vanes):    # For each vane
            current_vane_origin = self.vane_reference_frame_origin_list[i]
            current_vane_coordinates = self.vane_panels_coordinates_list[i]
            current_vane_frame_rotation_matrix = self.vane_reference_frame_rotation_matrix_list[i]

            rotated_vane_coordinates = np.zeros(np.shape(current_vane_coordinates))
            for j in range(len(current_vane_coordinates[:, 0])):    # For each coordinate of the panel
                # Get the panel coordinate points in the vane-centered coordinate system
                current_vane_coordinate_vane_reference_frame = np.matmul(np.linalg.inv(current_vane_frame_rotation_matrix), current_vane_coordinates[j, :] - current_vane_origin)
                # Now rotate along the vane-fixed x-axis and then y-axis
                Rx = R.from_euler('x', rotation_x_deg[i], degrees=True).as_matrix()
                Ry = R.from_euler('y', rotation_y_deg[i], degrees=True).as_matrix()
                vane_rotation_matrix = np.matmul(Ry, Rx)
                current_vane_coordinate_rotated_vane_reference_frame = np.matmul(vane_rotation_matrix, current_vane_coordinate_vane_reference_frame)

                # Convert back to the body fixed reference frame
                current_vane_coordinate_rotated_body_fixed_reference_frame = np.matmul(current_vane_frame_rotation_matrix, current_vane_coordinate_rotated_vane_reference_frame) + current_vane_origin
                rotated_vane_coordinates[j, :] = current_vane_coordinate_rotated_body_fixed_reference_frame
            new_vane_coordinates.append(rotated_vane_coordinates)
        return new_vane_coordinates

    def set_shifted_panel_characteristics(self, wings_coordinates_list, wings_areas_list, wings_reference_frame_rotation_matrix_list, keep_constant_area, stationary_system_components_mass, stationary_system_system_CoM):
        self.ACS_mass += stationary_system_components_mass                      # Includes only the mechanisms, the wings are already elsewhere
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.wings_coordinates_list = wings_coordinates_list
        self.number_of_wings = len(self.wings_coordinates_list)
        self.retain_wings_area_bool = keep_constant_area
        self.wings_areas_list = wings_areas_list
        self.wings_reference_frame_rotation_matrix_list = wings_reference_frame_rotation_matrix_list

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
        wing_coordinates_list = []
        for i in range(self.number_of_wings):
            current_wing_coordinates = self.wings_coordinates_list[i]
            current_wing_shifts = wings_shifts_list[i]
            new_current_panel_coordinates = np.zeros(np.shape(current_wing_coordinates))
            current_wing_reference_frame_rotation_matrix = self.wings_reference_frame_rotation_matrix_list[i]
            if (not self.retain_wings_area_bool): current_wing_boom_belongings = self.point_to_boom_belonging_list[i]
            for j, point in enumerate(current_wing_coordinates[:, :3]):
                if (self.retain_wings_area_bool):
                    # Here, the panel is just shifted without any shape deformation. The shift is made along the Y-axis of the considered quadrant
                    # The tether-spool system dictates the movement
                    if (not all_equal(current_wing_shifts)):
                        raise Exception("Inconsistent inputs for the shifted panels with constant area. All shifts need to be equal.")
                    elif (current_wing_shifts[0] < self.max_wings_inwards_translations_list[i]):
                        raise Exception("Requested shift is larger than allowable by the define geometry or positive (only negative are permitted). "
                                        + f"requested shift: {current_wing_shifts[0]}, maximum negative shift: {self.max_wings_inwards_translations_list[i]}")
                    else:
                        # Rotate to the quadrant reference frame
                        point_coordinates_wing_reference_frame = np.matmul(np.linalg.inv(current_wing_reference_frame_rotation_matrix), point)  # Get the position vector in the wing reference frame
                        translated_point_coordinates_wing_reference_frame = point_coordinates_wing_reference_frame + current_wing_shifts[j] * np.array([0, 1, 0])               # Get the translated point in the wing reference frame
                        new_point_coordinates_body_fixed_frame = np.matmul(current_wing_reference_frame_rotation_matrix, translated_point_coordinates_wing_reference_frame)     # Rotate back to body fixed reference frame
                else:
                    # Just shift the panels according to the inputs assuming that the material is extensible enough (simplifying assumption to avoid melting one's brain)
                    # More general implementation but less realistic implementation for most cases
                    related_boom = current_wing_boom_belongings[j]
                    if (related_boom != None):  #  Only do a shift if the attachment point belongs to a boom, not otherwise
                        boom_vector = self.booms_coordinates_list[related_boom][1, :] - self.booms_coordinates_list[related_boom][0, :]
                        boom_vector_unit = boom_vector/np.linalg.norm(boom_vector)  # Could change to do it a single time and find it in a list
                        new_point_coordinates_body_fixed_frame = point + boom_vector_unit * current_wing_shifts[j]  # Applying the panel shift
                        if (np.linalg.norm(new_point_coordinates_body_fixed_frame) > np.linalg.norm(boom_vector)):
                            raise Exception("Error. Wing shifted beyond the boom length")
                    else:
                        new_point_coordinates_body_fixed_frame = point
                new_current_panel_coordinates[j, :] = new_point_coordinates_body_fixed_frame
            wing_coordinates_list.append(new_current_panel_coordinates)
        return wing_coordinates_list


    def set_sliding_masses_characteristics(self, sliding_masses_list, stationary_system_components_mass, stationary_system_system_CoM, sliding_mass_system_type=0):
        self.ACS_mass += stationary_system_components_mass + sum(sliding_masses_list)
        self.ACS_CoM_stationary_components += stationary_system_components_mass * stationary_system_system_CoM
        self.bool_mass_based_controller = True
        self.sliding_masses_list = sliding_masses_list      # [m1, m2, m3, m4] in the same order as the booms or [m1, m2] if the system type is 1. (for standard cross sail)

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

        # The displacement should be around the origin of the boom for an independent one, and wrt to the middle of the boom if it is an aligned one
        sliding_mass_system_CoM = np.array([0, 0, 0], dtype="float64")
        sliding_masses_positions_body_fixed_frame_list = []
        for i, current_mass in enumerate(self.sliding_masses_list):
            current_unit_direction = self.sliding_mass_unit_direction[i]
            current_displacement = displacement_from_boom_origin_list[i]
            current_extreme_positions = self.sliding_mass_extreme_positions_list[i]
            if ((not self.sliding_mass_system_is_accross_two_booms[i]) and current_displacement<0):
                raise Exception("Error. Negative displacement for single-directional sliding mass.")

            if (self.sliding_mass_system_is_accross_two_booms[i]):
                current_sliding_mass_origin_body_fixed_frame = (current_extreme_positions[0, :] + current_extreme_positions[1, :])/2
            else:
                current_sliding_mass_origin_body_fixed_frame = current_extreme_positions[0, :]

            current_mass_position_body_fixed_frame = current_sliding_mass_origin_body_fixed_frame + current_displacement * current_unit_direction

            # Check that it is still within bounds
            current_displacement_norm = np.linalg.norm(current_mass_position_body_fixed_frame - current_sliding_mass_origin_body_fixed_frame)
            if ((current_displacement_norm > max(np.linalg.norm(current_sliding_mass_origin_body_fixed_frame - current_extreme_positions[1, :]),
                                                 np.linalg.norm(current_sliding_mass_origin_body_fixed_frame - current_extreme_positions[0, :])))): #TODO: correct because this is wrong
                raise Exception("Error. The requested displacement is larger than the sliding mass system capabilities.")
            sliding_mass_system_CoM += current_mass_position_body_fixed_frame * current_mass
            sliding_masses_positions_body_fixed_frame_list.append(current_mass_position_body_fixed_frame)

        return sliding_mass_system_CoM/sum(self.sliding_masses_list), sliding_masses_positions_body_fixed_frame_list

    def is_mass_based(self):
        return self.bool_mass_based_controller

    def set_gimball_mass_chateristics(self, mass_of_gimbaled_ballast):
        self.bool_mass_based_controller = True
        self.gimbaled_mass = mass_of_gimbaled_ballast
        return True

    def get_attitude_system_mass(self):
        return self.ACS_mass

class sail_craft:
    def __init__(self,
                 num_wings,
                 num_vanes,
                 initial_wings_coordinates_body_frame,
                 initial_vanes_coordinates_body_frame,
                 initial_wings_optical_properties,
                 initial_vanes_optical_properties,
                 initial_inertia_tensor_body_frame,
                 sail_mass_without_ACS,
                 spacecraft_mass_without_sail,
                 sail_CoM_without_ACS,
                 sail_material_areal_density,
                 vane_material_areal_density,
                 attitude_control_object):

        # Link to other classes
        self.bodies = None
        self.attitude_control_system = attitude_control_object                          # The ACS class object to obtain all control mechanisms

        # Non-varying sail characteristics
        self.sail_num_wings = num_wings
        self.sail_num_vanes = num_vanes
        self.sail_mass_without_attitude_control_system = sail_mass_without_ACS
        self.spacecraft_mass_without_sail = spacecraft_mass_without_sail
        self.attitude_control_system_mass = attitude_control_object.get_attitude_system_mass()
        self.sail_mass = sail_mass_without_ACS + self.attitude_control_system_mass   # Without mass-based attitude control system,
        self.sail_center_of_mass_body_fixed_position_without_ACS = sail_CoM_without_ACS
        self.desired_sail_state = None
        self.sail_material_areal_density = sail_material_areal_density
        self.vane_material_areal_density = sail_material_areal_density

        # Time-varying variables
        ## Panels
        self.sail_wings_coordinates = initial_wings_coordinates_body_frame            # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_wings_optical_properties = initial_wings_optical_properties         # num_panels x 10 array of panel surface properties
        self.sail_wings_areas = np.zeros(num_wings)                                    # List of panel areas
        self.sail_wings_centroids = [None] * num_wings                                # List of panel centroids
        self.sail_wings_surface_normals = [None] * num_wings                          # List of panel surface normal

        ## Vanes
        self.sail_vanes_coordinates = initial_vanes_coordinates_body_frame              # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_vanes_optical_properties = initial_vanes_optical_properties           # num_panels x 10 array of panel surface properties
        self.sail_vanes_areas = np.zeros(num_vanes)                                     # List of panel areas
        self.sail_vanes_centroids = [None] * num_vanes                                  # List of panel centroids
        self.sail_vanes_surface_normals = [None] * num_vanes                            # List of panel surface normal

        ## Gimball and moving masses systems
        self.sail_attitude_control_system_center_of_mass = sail_mass_without_ACS        # Mass of the ACS as an independent system, intialised at the sail CoM without ACS - TODO: check that this makes sense, and if the CoM is updated fast enough for this to be "allowed"
        self.moving_masses_positions_dict = None

        ## General spacecraft
        self.sail_center_of_mass_position = sail_CoM_without_ACS                        # 3D vector - initialised to value without ACS
        self.sail_inertia_tensor = initial_inertia_tensor_body_frame                    # 3x3 tensor

        # Initialising some properties
        self.compute_reflective_panels_properties(panel_id_list=list(range(self.sail_num_wings)), vanes_id_list=list(range(self.sail_num_vanes)))  # Compute the individual panel properties

        # Total vane mass
        vane_mass = 0
        for i in range(self.sail_num_vanes):
            vane_mass += self.sail_vanes_areas[i] * self.sail_material_areal_density
        self.sail_vanes_total_mass = vane_mass

        # Bookkeeping variables
        self.current_time = -1

    def compute_reflective_panels_properties(self, panel_id_list, vanes_id_list):  # should enable the possibility of only recomputing a specific panel
        # For each panel and vane, obtain coordinates of panels and compute the panel area
        for j, id_list in enumerate([panel_id_list, vanes_id_list]):
            if (j==0):
                coordinates_list_pointer = self.sail_wings_coordinates
                centroids_list_pointer = self.sail_wings_centroids
                areas_list_pointer = self.sail_wings_areas
                surface_normals_list_pointer = self.sail_wings_surface_normals
            else:
                coordinates_list_pointer = self.sail_vanes_coordinates
                centroids_list_pointer = self.sail_vanes_centroids
                areas_list_pointer = self.sail_vanes_areas
                surface_normals_list_pointer = self.sail_vanes_surface_normals

            for p_id in id_list:
                current_panel_centroid, current_panel_area, current_panel_surface_normal = compute_panel_geometrical_properties(coordinates_list_pointer[p_id])

                # Assign to correct lists
                centroids_list_pointer[p_id] = current_panel_centroid
                areas_list_pointer[p_id] = current_panel_area
                surface_normals_list_pointer[p_id] = current_panel_surface_normal
        return 0

    def compute_sail_center_of_mass(self, ACS_center_of_mass):
        self.compute_reflective_panels_properties(list(range(self.sail_num_wings)), list(range(self.sail_num_vanes)))
        summation = np.array([0, 0, 0], dtype="float64")

        # Fixed bus and booms
        summation += self.spacecraft_mass_without_sail * self.sail_center_of_mass_body_fixed_position_without_ACS

        # Compute contribution of panels
        for i in range(self.sail_num_wings):
            summation += self.sail_wings_centroids[i] * (self.sail_wings_areas[i] * self.sail_material_areal_density)

        ## Contribution of vanes - already taking into account in the ACS component
        #for i in range(self.sail_num_vanes):
        #    summation += self.sail_vanes_centroids[i] * (self.sail_vanes_areas[i] * self.vane_material_areal_density)

        summation += self.attitude_control_system_mass * ACS_center_of_mass
        return summation/self.sail_mass

    def compute_sail_inertia_tensor(self):  # TODO: handle both moving masses and moving panels
        return self.sail_inertia_tensor

    # Control the sail craft
    def sail_attitude_control_system(self, t):
        # Calls the ACS, updating the spacecraft properties before sending it to the propagation
        if (t != self.current_time):
            # Call all ACS controllers here to change the spacecraft state and control the orbit
            panels_coordinates, panels_optical_properties, ACS_CoM, moving_masses_positions = self.attitude_control_system.attitude_control(self.bodies, self.desired_sail_state)    # Find way to include the current state and the desired one
            self.sail_wings_coordinates = panels_coordinates["wings"] if (len(panels_coordinates["wings"]) != 0) else self.sail_wings_coordinates
            self.sail_vanes_coordinates = panels_coordinates["vanes"] if (len(panels_coordinates["vanes"]) != 0) else self.sail_vanes_coordinates
            self.sail_wings_optical_properties = panels_optical_properties["wings"] if (len(panels_optical_properties["wings"]) != 0) else self.sail_wings_optical_properties
            self.sail_vanes_optical_properties = panels_optical_properties["vanes"] if (len(panels_optical_properties["vanes"]) != 0) else self.sail_vanes_optical_properties

            # Recompute the necessary spacecraft properties based on the new configuration
            self.compute_reflective_panels_properties(list(range(self.sail_num_wings)), list(range(self.sail_num_vanes)))
            self.sail_center_of_mass_position = self.compute_sail_center_of_mass(ACS_CoM)
            self.moving_masses_positions_dict = moving_masses_positions
            self.current_time = t
        return 0

    # Pass body object to the class
    def setBodies(self, bodies):
        # Sets the bodies necessary to call the controller
        # This should be called right after the creation of the bodies, and the class is not fully defined until then
        self.bodies = bodies
        return 0

    # Pass the desired state to the class
    def setDesiredState(self, desired_state):
        self.desired_sail_state = desired_state
        return 0

    # Get rigid body properties
    def get_sail_mass(self, t):
        self.sail_attitude_control_system(t)
        return self.sail_mass

    def get_sail_center_of_mass(self, t):
        self.sail_attitude_control_system(t)
        return self.sail_center_of_mass_position

    def get_sail_inertia_tensor(self, t):
        self.sail_attitude_control_system(t)
        return self.sail_inertia_tensor

    # Get moving masses positions
    def get_sail_moving_masses_positions(self, t):
        self.sail_attitude_control_system(t)
        return self.moving_masses_positions_dict

    def get_number_of_wings(self):
        return self.sail_num_wings

    def get_number_of_vanes(self):
        return self.sail_num_vanes

    # Get specific panel properties
    def get_ith_panel_surface_normal(self, panel_id, panel_type=""):
        #self.sail_attitude_control_system(0)    # TODO: update to remove time dependence?
        if (panel_type == "Vane"):
            return self.sail_vanes_surface_normals[panel_id]
        else:
            return self.sail_wings_surface_normals[panel_id]

    def get_ith_panel_area(self, panel_id, panel_type=""):
        #self.sail_attitude_control_system(0)    # TODO: update to remove time dependence?
        if (panel_type == "Vane"):
            return self.sail_vanes_areas[panel_id]
        else:
            return self.sail_wings_areas[panel_id]

    def get_ith_panel_centroid(self, panel_id, panel_type=""):
        #self.sail_attitude_control_system(0)    # TODO: update to remove time dependence?
        if (panel_type == "Vane"):
            return self.sail_vanes_centroids[panel_id]
        else:
            return self.sail_wings_centroids[panel_id]

    def get_ith_panel_optical_properties(self, panel_id, panel_type=""):
        #self.sail_attitude_control_system(0)    # TODO: update to remove time dependence?
        if (panel_type == "Vane"):
            return self.sail_vanes_optical_properties[panel_id]
        else:
            return self.sail_wings_optical_properties[panel_id]

    def get_ith_panel_coordinates(self, panel_id, panel_type=""):
        #self.sail_attitude_control_system(0)    # TODO: update to remove time dependence?
        if (panel_type == "Vane"):
            return self.sail_vanes_coordinates[panel_id]
        else:
            return self.sail_wings_coordinates[panel_id]

def spacecraft_mass(t):
    return sail_mass

def spacecraft_center_of_mass(t):
    return sail_nominal_CoM

def spacecraft_mass_moment_of_inertia(t):
    return sail_I

def panel_surface_normal():
    return np.array([[0], [0], [1]], dtype="float64")

def panel_position_vector():
    return np.array([[0], [0], [0]], dtype="float64")

def panel_area():
    return 20.0