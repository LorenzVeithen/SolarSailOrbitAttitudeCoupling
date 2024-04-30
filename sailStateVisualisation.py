import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from controllers import sail_craft, sail_attitude_control_systems
from constants import sail_mass, sail_I, boom_length
from scipy.spatial.transform import Rotation as R

boom_attachment_point = 0.64
# Plot booms
fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)

ax.plot([0, 0], [0, boom_length],zs=[0, 0], color="k")
ax.plot([0, 0], [0, -boom_length],zs=[0, 0], color="k")
ax.plot([0, boom_length], [0, 0],zs=[0, 0], color="k")
ax.plot([0, -boom_length], [0, 0],zs=[0, 0], color="k")

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
vane_angle = np.deg2rad(30)         #rad
vane_side_length = 0.5               #m
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
acs_object = sail_attitude_control_systems("test", panel_coordinates_list)
acs_object.set_vane_characteristics(vane_origin_list, vane_coordinates_list, vanes_rotation_matrices_list)
sail = sail_craft(4, 4, panel_coordinates_list, vane_coordinates_list, panels_optical_properties, vanes_optical_properties,
                  sail_I, 16, 15.66, np.array([0, 0, 0.05]), 0.00425, acs_object)

CoM = sail.get_sail_center_of_mass(0)

panel_coordinates_list = [sail.get_ith_panel_coordinates(0, i, "Sail") for i in range(4)]
vane_coordinates_list = [sail.get_ith_panel_coordinates(0, i, "Vane") for i in range(4)]
ax.add_collection3d(Poly3DCollection(panel_coordinates_list, alpha=0.5, zorder=0))
ax.add_collection3d(Poly3DCollection(vane_coordinates_list, alpha=0.5, zorder=0, color="g"))

vstack_centroid_surface_normal = np.array([0, 0, 0, 0, 0, 0])
for j in range(2):
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
plt.show()