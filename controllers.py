from constants import sail_mass, sail_I
import numpy as np


class sail_attitude_control_systems():

    def __init__(self, ACS_system):
        self.sail_attitude_control_system = ACS_system
        self.vane_attachment_positions = None
        self.vane_panels_coordinates = None
        self.vane_panels_orientations = None
    def attitude_control(self, bodies, desired_sail_state):

        match self.sail_attitude_control_system:
            case "gimball_mass":
                self.gimball_mass(bodies, desired_sail_state)
            case _:
                print("ACS not available yet")
        CoM_shift = np.zeros(3)
        vane_deflection = []
        panel_reflectivity_change = []
        panel_shift = []
        panel_twist = []
        return CoM_shift, vane_deflection, panel_reflectivity_change, panel_shift, panel_twist

    def gimball_mass(self, current_sail_state, desired_sail_state):

        return

    def set_vane_characteristics(self, vanes_coordinates_list, vanes_orientations):
        # vanes_coordinates_list: num_vanes long list of num_coordinates_per_vane x 3 array, first point is the vane attachement point in body fixed frame
        # vanes_orientations: num_vanes long list of 3x3 rotation matrices from the body fixed frame to the vane fixed frame
        self.vane_attachment_positions = vanes_coordinates_list[0][0, :]
        self.vane_panels_coordinates = vanes_coordinates_list
        self.vane_panels_orientations = vanes_orientations
        return 0

    def __vane_dynamics(self, rotation_x_deg, rotation_y_deg):
        # Get the vane panel coordinates as a result of the rotation
        # Based on the initial vane position and orientation in the body frame
        if ( not (-90 <  rotation_x_deg < 90) or not (-90 <  rotation_y_deg < 90)):
            print("Requested vane deflection is not permitted:" + f"x-rotation={rotation_x_deg} degrees and y-rotation={rotation_y_deg} degrees")
            return False

        if (self.vane_attachment_positions == None
                or self.vane_panels_coordinates == None
                or self.vane_panels_orientations == None):
            print("Vane positions have not been set by the user")
            return False

        for i in range(len(rotation_x_deg)):
            # For each vane
            for j in range(len(self.vane_panels_coordinates[i][:, 0])):
                # For each coordinate of the panel

                # Get the panel coordinate points in the vane-fixed coordinate system
                current_coord = self.vane_panels_coordinates[i][j, :] - self.vane_attachment_positions





        return 0


class sail_craft(sail_attitude_control_systems):
    def __init__(self,
                 num_panels,
                 total_mass,
                 initial_CoM_position_body_frame,
                 initial_inertia_tensor_body_frame,
                 initial_panel_coordinates_body_frame,
                 initial_pabel_optical_properties,
                 ACS_system):
        self.sail_num_panels = num_panels
        self.sail_mass = total_mass
        self.sail_center_of_mass_position = initial_CoM_position_body_frame             # 3D vector
        self.sail_inertia_tensor = initial_inertia_tensor_body_frame                    # 3x3 tensor
        self.sail_panel_coordinates = initial_panel_coordinates_body_frame              # List of num_points x 3 arrays- Assuming points {p_i}, 0<=i<n, forms a counterclockwise polygon,
        self.sail_panels_areas = np.zeros(num_panels)                                   # List of panel areas
        self.sail_panels_centroids = np.zeros(num_panels)                               # List of panel centroids
        self.sail_panels_surface_normals = [None] * num_panels                          # List of panel surface normal
        self.sail_panels_properties = initial_pabel_optical_properties                  # num_panels x 10 array of panel surface properties
        self.compute_panel_properties(panel_id_list=list(range(self.sail_num_panels)))  # Compute the individual panel properties
        self.current_time = None
        self.bodies = None
        super(sail_attitude_control_systems).__init__(ACS_system)                       # Initialise the ACS parent class

    def compute_panel_properties(self, panel_id_list):  # should enable the possibility of only recomputing a specific panel
        # For each panel, obtain coordinates of panels and compute the panel area
        for p_id in panel_id_list:
            current_panel_coordinates = self.sail_panel_coordinates[p_id]
            number_of_attachment_points = len(current_panel_coordinates[:, 0])

            # Compute centroid
            current_panel_centroid = np.array([np.sum(current_panel_coordinates[:, 0])/number_of_attachment_points,
                                      np.sum(current_panel_coordinates[:, 1])/number_of_attachment_points,
                                      np.sum(current_panel_coordinates[:, 2])/number_of_attachment_points])

            # Compute area
            ref_point_area_calculation = current_panel_coordinates[0, :]
            current_panel_coordinates_wrt_ref_point = current_panel_coordinates - ref_point_area_calculation
            current_panel_coordinates_wrt_ref_point = current_panel_coordinates_wrt_ref_point[1:, :]
            cross_product_sum = np.array([0, 0, 0])
            for i in range(number_of_attachment_points-2):
                cross_product_sum += np.cross(current_panel_coordinates_wrt_ref_point[i], current_panel_coordinates_wrt_ref_point[i+1])
            current_panel_area = (1/2) * np.linalg.norm(cross_product_sum)

            # Compute surface normal
            cross_product_sum /= np.linalg.norm(cross_product_sum)  # Stokes theorem

            # Mass moment of inertia


            self.sail_panels_centroids[p_id] = current_panel_centroid
            self.sail_panels_areas[p_id] = current_panel_area
            self.sail_panels_surface_normals[p_id] = cross_product_sum
        return 0

    # Control the sail craft
    def sail_attitude_control_system(self, t):
        # Calls the ACS, updating the spacecraft properties before sending it to the propagation
        if (t != self.current_time):
            # Call all ACS controllers here to change the spacecraft state and control the orbit
            sail_attitude_control_systems.attitude_control(self.bodies, self.desired_sail_state)    # Find way to include the current state and the desired one
            # Access the spacecraft state (translation and rotational) and determine the course of action
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
    def get_ith_panel_surface_normal(self, t, panel_id):
        self.sail_attitude_control_system(t)
        return self.sail_panels_surface_normals[panel_id]

    def get_ith_panel_area(self, t, panel_id):
        self.sail_attitude_control_system(t)
        return self.sail_panels_areas[panel_id]

    def get_ith_panel_centroid(self, t, panel_id):
        self.sail_attitude_control_system(t)
        return self.sail_panels_centroids[panel_id]

    def get_ith_panel_optical_properties(self, t, panel_id):
        self.sail_attitude_control_system(t)
        return self.sail_panels_properties[panel_id]

def spacecraft_mass(t):
    return sail_mass

def spacecraft_center_of_mass(t):
    return np.array([0, 0, 0])

def spacecraft_mass_moment_of_inertia(t):
    return sail_I