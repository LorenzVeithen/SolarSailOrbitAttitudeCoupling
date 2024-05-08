import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from controllers import sail_craft, sail_attitude_control_systems
from constants import sail_mass, sail_I, boom_length
from scipy.spatial.transform import Rotation as R
from MiscFunctions import compute_panel_geometrical_properties

boom_attachment_point = 0.64

VANES_BOOL = False
SHIFTED_PANELS_BOOL = True
keep_area = False

SLIDING_MASS_BOOL = False

# Boom points
boom1 = np.array([[0, 0, 0], [0, boom_length, 0]])
boom2 = np.array([[0, 0, 0], [boom_length, 0, 0]])
boom3 = np.array([[0, 0, 0], [0, -boom_length, 0]])
boom4 = np.array([[0, 0, 0], [-boom_length, 0, 0]])
boom_list = [boom1, boom2, boom3, boom4]

panel1 = np.array([[boom_attachment_point, 0, 0],
                   [boom_length, 0, 0],
                   [0, boom_length, 0],
                   [0, boom_attachment_point, 0]])

panel2 = np.array([[0, -boom_attachment_point, 0],
                    [0, -boom_length, 0],
                    [boom_length, 0, 0],
                    [boom_attachment_point, 0, 0]])

panel3 = np.array([[-boom_attachment_point, 0, 0],
                   [-boom_length, 0, 0],
                   [0, -boom_length, 0],
                   [0, -boom_attachment_point, 0]])

panel4 = np.array([[0, boom_attachment_point, 0],
                    [0, boom_length, 0],
                    [-boom_length, 0, 0],
                    [-boom_attachment_point, 0, 0]])

panel_coordinates_list = [panel1, panel2, panel3, panel4]
panels_optical_properties = [np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])] * 4

# vanes
vane_angle = np.deg2rad(30)             # rad
vane_side_length = 0.5                  # m
vane1 = np.array([[0, boom_length, 0],
                  [vane_side_length*np.cos(vane_angle), boom_length+vane_side_length*np.sin(vane_angle), 0],
                  [-vane_side_length*np.cos(vane_angle), boom_length+vane_side_length*np.sin(vane_angle), 0]])

vane2 = np.array([[boom_length, 0, 0],
                  [boom_length + vane_side_length*np.sin(vane_angle), -vane_side_length*np.cos(vane_angle), 0],
                  [boom_length + vane_side_length*np.sin(vane_angle), vane_side_length*np.cos(vane_angle), 0]])

vane3 = np.array([[0, -boom_length, 0],
                  [-vane_side_length*np.cos(vane_angle), -boom_length-vane_side_length*np.sin(vane_angle), 0],
                  [vane_side_length*np.cos(vane_angle), -boom_length-vane_side_length*np.sin(vane_angle), 0]])

vane4 = np.array([[-boom_length, 0, 0],
                  [-boom_length - vane_side_length*np.sin(vane_angle), vane_side_length*np.cos(vane_angle), 0],
                  [-boom_length - vane_side_length*np.sin(vane_angle), -vane_side_length*np.cos(vane_angle), 0]])

vane_coordinates_list = [vane1, vane2, vane3, vane4]
vanes_optical_properties = [np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])] * 4


vane_origin_list = [vane[0, :] for vane in vane_coordinates_list]
vanes_rotation_matrices_list = [R.from_euler('z', 90, degrees=True).as_matrix(),
                                R.from_euler('z', 0, degrees=True).as_matrix(),
                                R.from_euler('z', 270, degrees=True).as_matrix(),
                                R.from_euler('z', 180, degrees=True).as_matrix()]

wings_rotation_matrices_list = [R.from_euler('z', -45, degrees=True).as_matrix(),
                                R.from_euler('z', -135, degrees=True).as_matrix(),
                                R.from_euler('z', -225, degrees=True).as_matrix(),
                                R.from_euler('z', -315, degrees=True).as_matrix()]

if (SHIFTED_PANELS_BOOL):
    if (keep_area == True):
        # Change panel coordinates to test shifted panels implementation, by ensuring that the panels can be shifted
        for i, wing_coords in enumerate(panel_coordinates_list):
            new_points = np.zeros(np.shape(wing_coords))
            for j, point in enumerate(wing_coords):
                point_wing_frame = np.matmul(np.linalg.inv(wings_rotation_matrices_list[i]), point)
                point_wing_frame += np.array([0, 0.5, 0])
                new_point = np.matmul(wings_rotation_matrices_list[i], point_wing_frame)
                new_points[j, :] = new_point
            panel_coordinates_list[i] = new_points

if (VANES_BOOL):
    acs_object = sail_attitude_control_systems("vanes", boom_list)
    acs_object.set_vane_characteristics(vane_coordinates_list, vane_origin_list, vanes_rotation_matrices_list, 0, np.array([0, 0, 0]), 0.0045)

elif(SHIFTED_PANELS_BOOL):
    acs_object = sail_attitude_control_systems("shifted_wings", boom_list)
    wing_area_list = []
    for i in range(len(panel_coordinates_list)):
        _, wing_area, _ = compute_panel_geometrical_properties(panel_coordinates_list[i])
        wing_area_list.append(wing_area)
    acs_object.set_shifted_panel_characteristics(panel_coordinates_list, wing_area_list, wings_rotation_matrices_list,
                                                 keep_area, 0, np.array([0, 0, 0]))
elif(SLIDING_MASS_BOOL):
    acs_object = sail_attitude_control_systems("sliding_masses", boom_list)
    acs_object.set_sliding_masses_characteristics([10, 10], 0, np.array([0, 0, 0]), 1)



if (VANES_BOOL):
    sail = sail_craft(4, 4, panel_coordinates_list, vane_coordinates_list, panels_optical_properties, vanes_optical_properties,
                      sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)
elif(SHIFTED_PANELS_BOOL):
    sail = sail_craft(4, 0, panel_coordinates_list, [], [], vanes_optical_properties,
                  sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)
elif(SLIDING_MASS_BOOL):
    sail = sail_craft(4, 0, panel_coordinates_list, [], [], [],
                      sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, 0.00425, acs_object)

wing_area_list = [sail.get_ith_panel_area(-1, i, "Sail") for i in range(4)]
CoM = sail.get_sail_center_of_mass(0)
moving_masses_positions = sail.get_sail_moving_masses_positions(0)

panel_coordinates_list = [sail.get_ith_panel_coordinates(0, i, "Sail") for i in range(4)]
if VANES_BOOL: vane_coordinates_list = [sail.get_ith_panel_coordinates(0, i, "Vane") for i in range(4)]

# Plot booms
fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)
for boom in boom_list:
    ax.plot([boom[0][0], boom[1][0]], [boom[0][1], boom[1][1]],zs=[boom[0][2], boom[1][2]], color="k")

ax.add_collection3d(Poly3DCollection(panel_coordinates_list, alpha=0.5, zorder=0))
if VANES_BOOL: ax.add_collection3d(Poly3DCollection(vane_coordinates_list, alpha=0.5, zorder=0, color="g"))

vstack_centroid_surface_normal = np.array([0, 0, 0, 0, 0, 0])
if (VANES_BOOL): j_max = 2
else: j_max = 1
for j in range(j_max):
    if (j > 0):
        tp = "Vane"
    else:
        tp = "Sail"
    for i in range(4):
        panel_surface_normal_vector = sail.get_ith_panel_surface_normal(0, i, tp)
        panel_centroid = sail.get_ith_panel_centroid(0, i, tp)
        hstack_centroid_surface_normal = np.hstack((panel_centroid, panel_surface_normal_vector))
        vstack_centroid_surface_normal = np.vstack((vstack_centroid_surface_normal, hstack_centroid_surface_normal))

ax.quiver(vstack_centroid_surface_normal[:, 0],
        vstack_centroid_surface_normal[:, 1],
        vstack_centroid_surface_normal[:, 2],
        vstack_centroid_surface_normal[:, 3],
        vstack_centroid_surface_normal[:, 4],
        vstack_centroid_surface_normal[:, 5],
        color='r', arrow_length_ratio=0.1, zorder=20)

ax.set_box_aspect([1,1,1])
ax.set_xlim([-12, 12])  # set x-axis limits from 0 to 2
ax.set_ylim([-12, 12])  # set y-axis limits from 0 to 4
ax.set_zlim([-12, 12])  # set z-axis limits from 0 to 6
ax.set_proj_type('ortho')
ax.scatter(CoM[0], CoM[1], CoM[2], color="b", s=3)

for key in moving_masses_positions.keys():
    masses_positions_list = moving_masses_positions[key]
    for mass_position in masses_positions_list:
        # TODO: maybe ensure that the mass is non-zero
        ax.scatter(mass_position[0], mass_position[1], mass_position[2], color="magenta", s=7)

plt.show()