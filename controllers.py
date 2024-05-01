from constants import sail_mass, sail_I
from scipy.spatial.transform import Rotation as R
import numpy as np
from MiscFunctions import all_equal, closest_point_on_a_segment_to_a_third_point


class sail_attitude_control_systems:

    def __init__(self, ACS_system, default_panel_coordinates, ):
        self.sail_attitude_control_system = ACS_system
        self.sail_default_panel_coordinates = default_panel_coordinates            # Default coordinates of the panels, when no control has been applied
        self.vane_reference_frame_origin = None
        self.vane_panels_coordinates = None
        self.wings_coordinates_list = None
        self.vane_reference_frame_rotation_matrix = None
        self.sliding_masses = [None, None]
        self.gimball_mass = None
        self.bool_mass_based_controller = True if (ACS_system=="...") else False
        self.number_of_vanes = None
        self.number_of_wings = None
        self.boom_tips_coordinates_list = None


    def attitude_control(self, bodies, desired_sail_state):
        # Returns an empty array if nothing has changed
        thrust_levels = []
        panel_optical_properties = []
        moving_mass_position = []
        panel_coordinates = []
        vane_coordinates = []
        match self.sail_attitude_control_system:
            case "gimball_mass":
                self.__pure_gimball_mass(bodies, desired_sail_state)
            case "vanes":
                pass
            case "test":
                #vane_coordinates = self.__vane_dynamics([90, 90, 90, 90], [-90, -90, -90, -90])
                wing_shifts_list = [[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]]
                panel_coordinates = self.__shifted_panel_dynamics(wing_shifts_list)
            case _:
                raise Exception("Selected ACS not available yet.")

        # the attitude-control algorithm should give the output
        return panel_coordinates, vane_coordinates, panel_optical_properties, moving_mass_position, thrust_levels

    def __pure_gimball_mass(self, current_sail_state, desired_sail_state):

        return

    def __pure_vane_controller(self, desired_sail_state):

        return

    def set_vane_characteristics(self, vane_reference_frame_origin_list, vanes_coordinates_list, vanes_reference_frame_rotation_matrix_list):
        # vanes_coordinates_list: num_vanes long list of num_coordinates_per_vane x 3 array, first point is the vane attachement point in body fixed frame
        # vanes_orientations: num_vanes long list of 3x3 rotation matrices from the body fixed frame to the vane fixed frame
        self.vane_reference_frame_origin = vane_reference_frame_origin_list  # Maybe change to avoid stupid conventions
        self.vane_panels_coordinates = vanes_coordinates_list
        self.vane_reference_frame_rotation_matrix = vanes_reference_frame_rotation_matrix_list
        self.number_of_vanes = len(vane_reference_frame_origin_list)
        return True

    def __vane_dynamics(self, rotation_x_deg, rotation_y_deg):
        # Get the vane panel coordinates as a result of the rotation
        # Based on the initial vane position and orientation in the body frame
        if (not all(-90 <= angle <= 90 for angle in rotation_x_deg) or not all(-90 <= angle <= 90 for angle in rotation_y_deg)):
            raise Exception("Requested vane deflection is not permitted:" + f"x-rotation={rotation_x_deg} degrees and y-rotation={rotation_y_deg} degrees.")

        if (self.vane_reference_frame_origin == None
                or self.vane_panels_coordinates == None
                or self.vane_reference_frame_rotation_matrix == None):
            raise Exception("Vane characteristics have not been set by the user.")

        new_vane_coordinates = []
        for i in range(self.number_of_vanes):    # For each vane
            current_vane_origin = self.vane_reference_frame_origin[i]
            current_vane_coordinates = self.vane_panels_coordinates[i]
            current_vane_frame_rotation_matrix = self.vane_reference_frame_rotation_matrix[i]

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

    def set_shifted_panel_characteristics(self, wings_coordinates_list, wings_areas_list, wings_reference_frame_rotation_matrix_list, boom_coordinates_list, keep_constant_area):
        self.wings_coordinates_list = wings_coordinates_list       # The initial panel coordinates, in default state, not updated ones that move
        self.boom_coordinates_list = boom_coordinates_list    # Assuming that the booms are straight only and going out from the origin. list 2x3 array (top is boom origin, bottom is boom tip)
        self.number_of_wings = len(self.wings_coordinates_list)
        self.wings_areas_list = wings_areas_list
        self.wings_reference_frame_rotation_matrix_list = wings_reference_frame_rotation_matrix_list
        self.keep_constant_area = keep_constant_area               # Bool indicating if the area of the panels should be conserved (pure translation of the panel)

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
                    for boom in self.boom_coordinates_list:
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
                        boom_coordinates_list[closest_boom_index][0, :])

                    boom_tip_coordinates_wing_reference_frame = np.matmul(
                        np.linalg.inv(self.wings_reference_frame_rotation_matrix_list[i]),
                        boom_coordinates_list[closest_boom_index][1, :])

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
                    for j, boom in enumerate(self.boom_coordinates_list):
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
            if (not self.keep_constant_area): current_wing_boom_belongings = self.point_to_boom_belonging_list[i]
            for j, point in enumerate(current_wing_coordinates[:, :3]):
                if (self.keep_constant_area):
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
                        boom_vector = self.boom_coordinates_list[related_boom][1, :]-self.boom_coordinates_list[related_boom][0, :]
                        boom_vector_unit = boom_vector/np.linalg.norm(boom_vector)
                        new_point_coordinates_body_fixed_frame = point + boom_vector_unit * current_wing_shifts[j]  # Applying the panel shift
                        if (np.linalg.norm(new_point_coordinates_body_fixed_frame) > np.linalg.norm(boom_vector)):
                            raise Exception("Error. Wing shifted beyond the boom length")
                    else:
                        new_point_coordinates_body_fixed_frame = point
                new_current_panel_coordinates[j, :] = new_point_coordinates_body_fixed_frame
            wing_coordinates_list.append(new_current_panel_coordinates)
        return wing_coordinates_list


    def is_mass_based(self):
        return self.bool_mass_based_controller

    def set_gimball_mass_chateristics(self, mass_of_gimbaled_ballast):
        self.gimbaled_mass = mass_of_gimbaled_ballast
        return True

    def set_sliding_masses_characteristics(self):
        pass

    def get_attitude_system_mass(self):
        if (self.is_mass_based()):
            if ((all(self.sliding_masses) != None) and (self.gimball_mass != None)):
                return sum(self.sliding_masses) + self.gimball_mass
            elif (all(self.sliding_masses) != None):
                return sum(self.sliding_masses)
            elif (self.gimball_mass != None):
                return self.gimball_mass
        else:
            return 0                    # to be expanded


class sail_craft:
    def __init__(self,
                 num_panels,
                 num_vanes,
                 initial_panels_coordinates_body_frame,
                 initial_vanes_coordinates_body_frame,
                 initial_panels_optical_properties,
                 initial_vanes_optical_properties,
                 initial_inertia_tensor_body_frame,
                 sail_mass_without_ACS,
                 spacecraft_mass_without_sail,
                 sail_CoM_without_ACS,
                 sail_material_areal_density,
                 attitude_control_object):

        # Link to other classes
        self.bodies = None
        self.attitude_control_system = attitude_control_object                          # The ACS class object to obtain all control mechanisms

        # Non-varying sail characteristics
        self.sail_num_panels = num_panels
        self.sail_num_vanes = num_vanes
        self.sail_mass_without_attitude_control_system = sail_mass_without_ACS
        self.spacecraft_mass_without_sail = spacecraft_mass_without_sail
        self.attitude_conctrol_system_mass = attitude_control_object.get_attitude_system_mass()
        self.sail_mass = sail_mass_without_ACS + self.attitude_conctrol_system_mass   # Without mass-based attitude control system,
        self.sail_center_of_mass_body_fixed_position_without_ACS = sail_CoM_without_ACS
        self.desired_sail_state = None
        self.sail_material_areal_density = sail_material_areal_density

        # Time-varying variables
        ## Panels
        self.sail_panels_coordinates = initial_panels_coordinates_body_frame            # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_panels_optical_properties = initial_panels_optical_properties         # num_panels x 10 array of panel surface properties
        self.sail_panels_areas = np.zeros(num_panels)                                   # List of panel areas
        self.sail_panels_centroids = [None] * num_panels                                # List of panel centroids
        self.sail_panels_surface_normals = [None] * num_panels                          # List of panel surface normal

        ## Vanes
        self.sail_vanes_coordinates = initial_vanes_coordinates_body_frame              # List of num_points x 3 arrays - assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_vanes_optical_properties = initial_vanes_optical_properties           # num_panels x 10 array of panel surface properties
        self.sail_vanes_areas = np.zeros(num_vanes)                                     # List of panel areas
        self.sail_vanes_centroids = [None] * num_vanes                                  # List of panel centroids
        self.sail_vanes_surface_normals = [None] * num_vanes                            # List of panel surface normal

        ## Gimball and moving masses systems
        self.sail_attitude_control_system_center_of_mass = sail_mass_without_ACS        # Mass of the ACS as an independent system, intialised at the sail CoM without ACS - TODO: check that this makes sense, and if the CoM is updated fast enough for this to be "allowed"

        ## General spacecraft
        self.sail_center_of_mass_position = sail_CoM_without_ACS                        # 3D vector - initialised to value without ACS
        self.sail_inertia_tensor = initial_inertia_tensor_body_frame                    # 3x3 tensor

        # Initialising some properties
        self.compute_reflective_panels_properties(panel_id_list=list(range(self.sail_num_panels)), vanes_id_list=list(range(self.sail_num_vanes)))  # Compute the individual panel properties

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
                coordinates_list_pointer = self.sail_panels_coordinates
                centroids_list_pointer = self.sail_panels_centroids
                areas_list_pointer = self.sail_panels_areas
                surface_normals_list_pointer = self.sail_panels_surface_normals
            else:
                coordinates_list_pointer = self.sail_vanes_coordinates
                centroids_list_pointer = self.sail_vanes_centroids
                areas_list_pointer = self.sail_vanes_areas
                surface_normals_list_pointer = self.sail_vanes_surface_normals

            for p_id in id_list:
                current_panel_coordinates = coordinates_list_pointer[p_id]
                number_of_attachment_points = len(current_panel_coordinates[:, 0])

                # Compute centroid
                current_panel_centroid = np.array([np.sum(current_panel_coordinates[:, 0])/number_of_attachment_points,
                                          np.sum(current_panel_coordinates[:, 1])/number_of_attachment_points,
                                          np.sum(current_panel_coordinates[:, 2])/number_of_attachment_points])

                # Compute area
                ref_point_area_calculation = current_panel_coordinates[0, :]
                current_panel_coordinates_wrt_ref_point = current_panel_coordinates - ref_point_area_calculation
                current_panel_coordinates_wrt_ref_point = current_panel_coordinates_wrt_ref_point[1:, :]
                cross_product_sum = np.array([0, 0, 0], dtype='float64')
                for i in range(number_of_attachment_points-2):
                    cross_product_sum += np.cross(current_panel_coordinates_wrt_ref_point[i], current_panel_coordinates_wrt_ref_point[i+1])
                current_panel_area = (1/2) * np.linalg.norm(cross_product_sum)

                # Compute surface normal
                current_panel_surface_normal = cross_product_sum/np.linalg.norm(cross_product_sum)  # Stokes theorem

                # Assign to correct lists
                centroids_list_pointer[p_id] = current_panel_centroid
                areas_list_pointer[p_id] = current_panel_area
                surface_normals_list_pointer[p_id] = current_panel_surface_normal
        return 0

    def compute_sail_center_of_mass(self):
        self.compute_reflective_panels_properties(list(range(self.sail_num_panels)), list(range(self.sail_num_vanes)))
        sum = 0

        # Fixed bus and booms
        sum += self.spacecraft_mass_without_sail * self.sail_center_of_mass_body_fixed_position_without_ACS

        # Compute contribution of panels
        for i in range(self.sail_num_panels):
            sum += self.sail_panels_centroids[i] * (self.sail_panels_areas[i] * self.sail_material_areal_density)

        # Contribution of vanes
        for i in range(self.sail_num_vanes):
            sum += self.sail_vanes_centroids[i] * (self.sail_vanes_areas[i] * self.sail_material_areal_density)


        sum += (self.attitude_conctrol_system_mass-self.sail_vanes_total_mass) * self.sail_attitude_control_system_center_of_mass
        return sum/self.sail_mass

    def compute_sail_inertia_tensor(self):  # TODO: handle both moving masses and moving panels
        return self.sail_inertia_tensor

    # Control the sail craft
    def sail_attitude_control_system(self, t):
        # Calls the ACS, updating the spacecraft properties before sending it to the propagation
        if (t != self.current_time):
            # Call all ACS controllers here to change the spacecraft state and control the orbit
            panels_coordinates, vanes_coordinates, panels_optical_properties, moving_mass_position, thrust_levels = self.attitude_control_system.attitude_control(self.bodies, self.desired_sail_state)    # Find way to include the current state and the desired one
            self.sail_panels_coordinates = panels_coordinates if (len(panels_coordinates) != 0) else self.sail_panels_coordinates
            self.sail_vanes_coordinates = vanes_coordinates if (len(vanes_coordinates) != 0) else self.sail_vanes_coordinates
            self.sail_panels_optical_properties = panels_optical_properties if (len(panels_optical_properties) !=0) else self.sail_panels_optical_properties
            # TODO: take care of the vanes optical properties, the moving masses and the thrust levels (and any other which will come in

            self.compute_reflective_panels_properties(list(range(self.sail_num_panels)), list(range(self.sail_num_vanes)))        # Recompute the panel properties after the coordinates have been updated
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

    # Get specific panel properties
    def get_ith_panel_surface_normal(self, t, panel_id, panel_type=""):
        self.sail_attitude_control_system(t)
        if (panel_type == "Vane"):
            return self.sail_vanes_surface_normals[panel_id]
        else:
            return self.sail_panels_surface_normals[panel_id]

    def get_ith_panel_area(self, t, panel_id, panel_type=""):
        self.sail_attitude_control_system(t)
        if (panel_type == "Vane"):
            return self.sail_vanes_areas[panel_id]
        else:
            return self.sail_panels_areas[panel_id]

    def get_ith_panel_centroid(self, t, panel_id, panel_type=""):
        self.sail_attitude_control_system(t)
        if (panel_type == "Vane"):
            return self.sail_vanes_centroids[panel_id]
        else:
            return self.sail_panels_centroids[panel_id]

    def get_ith_panel_optical_properties(self, t, panel_id, panel_type=""):
        self.sail_attitude_control_system(t)
        if (panel_type == "Vane"):
            return self.sail_vanes_optical_properties[panel_id]
        else:
            return self.sail_panels_optical_properties[panel_id]

    def get_ith_panel_coordinates(self, t, panel_id, panel_type=""):
        self.sail_attitude_control_system(t)
        if (panel_type == "Vane"):
            return self.sail_vanes_coordinates[panel_id]
        else:
            return self.sail_panels_coordinates[panel_id]

def spacecraft_mass(t):
    return sail_mass

def spacecraft_center_of_mass(t):
    return np.array([0, 0, 0])

def spacecraft_mass_moment_of_inertia(t):
    return sail_I