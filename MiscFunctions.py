import numpy as np
from itertools import groupby
from numba import jit

@jit(nopython=True, cache=True)
def compute_panel_geometrical_properties(panel_coordinates):
    # Number of attachment points
    number_of_attachment_points = np.shape(panel_coordinates)[0]

    # Compute centroid
    current_panel_centroid = np.zeros(3)
    for i in range(number_of_attachment_points):
        current_panel_centroid[0] += panel_coordinates[i, 0]
        current_panel_centroid[1] += panel_coordinates[i, 1]
        current_panel_centroid[2] += panel_coordinates[i, 2]
    current_panel_centroid /= number_of_attachment_points

    # Compute area using vectorized operations
    ref_point_area_calculation = panel_coordinates[0, :]
    current_panel_coordinates_wrt_ref_point = np.zeros((number_of_attachment_points - 1, 3))
    for i in range(1, number_of_attachment_points):
        current_panel_coordinates_wrt_ref_point[i - 1, 0] = panel_coordinates[i, 0] - ref_point_area_calculation[0]
        current_panel_coordinates_wrt_ref_point[i - 1, 1] = panel_coordinates[i, 1] - ref_point_area_calculation[1]
        current_panel_coordinates_wrt_ref_point[i - 1, 2] = panel_coordinates[i, 2] - ref_point_area_calculation[2]

    # Vectorized cross product sum
    cross_product_sum = np.zeros(3)
    for i in range(number_of_attachment_points - 2):
        u = current_panel_coordinates_wrt_ref_point[i]
        v = current_panel_coordinates_wrt_ref_point[i + 1]
        cross_product_sum[0] += u[1] * v[2] - u[2] * v[1]
        cross_product_sum[1] += u[2] * v[0] - u[0] * v[2]
        cross_product_sum[2] += u[0] * v[1] - u[1] * v[0]

    current_panel_area = 0.5 * np.sqrt(cross_product_sum[0]**2 + cross_product_sum[1]**2 + cross_product_sum[2]**2)

    # Compute surface normal
    norm = np.sqrt(cross_product_sum[0]**2 + cross_product_sum[1]**2 + cross_product_sum[2]**2)
    current_panel_surface_normal = cross_product_sum / norm

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

def lists_of_arrays_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for arr1, arr2 in zip(list1, list2):
        if (not np.array_equal(arr1, arr2) and np.amax(abs(arr2-arr1))>1e-15):
            return False

    return True

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

def Rx_matrix(theta):
    '''
    Rotation matrix around the x axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the x axis
    '''
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def Ry_matrix(theta):
    '''
    Rotation matrix around the y axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the y axis
    '''
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def Rz_matrix(theta):
    '''
    Rotation matrix around the z axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the z axis
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


