import numpy as np
from itertools import groupby
from numba import jit

@jit(nopython=True, cache=True)
def compute_panel_geometrical_properties(panel_coordinates):
    current_panel_coordinates = panel_coordinates
    number_of_attachment_points = len(current_panel_coordinates[:, 0])

    # Compute centroid
    current_panel_centroid = np.array([np.sum(current_panel_coordinates[:, 0]) / number_of_attachment_points,
                                       np.sum(current_panel_coordinates[:, 1]) / number_of_attachment_points,
                                       np.sum(current_panel_coordinates[:, 2]) / number_of_attachment_points])

    # Compute area
    ref_point_area_calculation = current_panel_coordinates[0, :]
    current_panel_coordinates_wrt_ref_point = current_panel_coordinates - ref_point_area_calculation
    current_panel_coordinates_wrt_ref_point = current_panel_coordinates_wrt_ref_point[1:, :]
    cross_product_sum = np.array([0, 0, 0], dtype='float64')
    for i in range(number_of_attachment_points - 2):
        cross_product_sum += np.cross(current_panel_coordinates_wrt_ref_point[i],
                                      current_panel_coordinates_wrt_ref_point[i + 1])
    current_panel_area = (1 / 2) * np.linalg.norm(cross_product_sum)

    # Compute surface normal
    current_panel_surface_normal = cross_product_sum / np.linalg.norm(cross_product_sum)  # Stokes theorem
    return current_panel_centroid, current_panel_area, current_panel_surface_normal

def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

def closest_point_on_a_segment_to_a_third_point(p1, p2, p3):
    if (all(p3[i] == p1[i] for i in range(len(p3)))):
        return p1
    elif (all(p3[i] == p2[i] for i in range(len(p3)))):
        return p2
    else:
        d = (p2-p1)/np.linalg.norm((p2-p1))
        w = p3-p1
        magnitude = np.dot(w, d)
        p4 = d * np.dot(w, d)
        if (magnitude < 0):
            return p1
        elif (magnitude > 0 and np.linalg.norm(p4-p1) > np.linalg.norm(p2-p1)):
            return p2
        else:
            return p4

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def set_axes_equal(ax):  # from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return

def Rx(theta):
    '''
    Rotation matrix around the x axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the x axis
    '''
    return np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
    '''
    Rotation matrix around the y axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the y axis
    '''
    return np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    '''
    Rotation matrix around the z axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the z axis
    '''
    return np.matrix([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def axisRotation(theta, axis=0):
    if (axis==0):
        return Rx(theta)
    elif (axis==1):
        return Ry(theta)
    elif (axis==2):
        return Rz(theta)

def centeroidnp_2D(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


