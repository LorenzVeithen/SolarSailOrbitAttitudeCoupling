import numpy as np
from itertools import groupby
from numba import jit
from scipy.linalg import lu

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

def find_linearly_independent_rows(matrix):
    # Perform LU decomposition
    P, L, U = lu(matrix)

    # Identify the pivot rows
    # The pivots are the rows in the original matrix that are linearly independent
    pivot_rows = np.where(np.abs(U.diagonal()) > 1e-10)[0]

    # Extract the linearly independent rows
    independent_rows = matrix[pivot_rows, :]

    return pivot_rows, independent_rows

def bring_inside_bounds(original: float | np.ndarray, lower_bound: float,
                        upper_bound: float, include: str = 'lower') -> float | np.ndarray:

    """This function brings a number inside the given bounds, assuming the interval defined by the bounds can periodically extend the whole real line (e.g. an angle of 9$\pi$ is equivalent to an angle of $\pi$ and at the same time equivalent to an angle of $-\pi$). If a (multidimensional) array is passed, the operation is performed on all its entries. It returns the same object and of the same dimension as it was given.
    *Note:* This function's support of arrays is limited to one-dimensional and two-dimensional arrays.

    Parameters
    ----------
    original: float | np.ndarray
        The original number or array of numbers.

    lower_bound: float
        The lower bound of the periodic interval.

    upper_bound: float
        The upper bound of the periodic interval.

    include: str
        The bound that is to be kept. It can be 'upper' or 'lower'. Anything else will result in an error.

    Returns
    -------
    float | np.array
        The number of array of numbers, all inside the interval.

    """

    if include not in ['upper', 'lower']:
        raise ValueError('(bring_inside_bounds): Invalid value for argument "include". Only "upper" and "lower" are allowed. Provided: ' + include)

    if type(original) in [float, np.float32, np.float64, np.float128]:
        to_return = bring_inside_bounds_scalar(original, lower_bound, upper_bound, include)
    else:
        dim_num = len(original.shape)

        if dim_num == 1:
            to_return = bring_inside_bounds_single_dim(original, lower_bound, upper_bound, include)
        elif dim_num == 2:
            to_return = bring_inside_bounds_double_dim(original, lower_bound, upper_bound, include)
        else:
            raise ValueError('(bring_inside_bounds): Invalid input array.')

    return to_return


def bring_inside_bounds_single_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    """This function brings the entries of a one-dimensional array inside the given bounds, assuming the interval defined by the bounds can periodically extend the whole real line (e.g. an angle of 9$\pi$ is equivalent to an angle of $\pi$ and at the same time equivalent to an angle of $-\pi$). It returns another one-dimensional array.

    Parameters
    ----------
    original: np.ndarray
        The original array.

    lower_bound: float
        The lower bound of the periodic interval.

    upper_bound: float
        The upper bound of the periodic interval.

    include: str
        The bound that is to be kept. It can be 'upper' or 'lower'. Anything else will result in an error.

    Returns
    -------
    np.array
        The array of numbers, all inside the interval.

    """

    new = np.zeros_like(original)
    for idx in range(len(new)):
        new[idx] = bring_inside_bounds_scalar(original[idx], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_double_dim(original: np.ndarray, lower_bound: float,
                                   upper_bound: float, include: str = 'lower') -> np.ndarray:

    """This function brings the entries of a two-dimensional array inside the given bounds, assuming the interval defined by the bounds can periodically extend the whole real line (e.g. an angle of 9$\pi$ is equivalent to an angle of $\pi$ and at the same time equivalent to an angle of $-\pi$). It returns another two-dimensional array.

    Parameters
    ----------
    original: np.ndarray
        The original array.

    lower_bound: float
        The lower bound of the periodic interval.

    upper_bound: float
        The upper bound of the periodic interval.

    include: str
        The bound that is to be kept. It can be 'upper' or 'lower'. Anything else will result in an error.

    Returns
    -------
    np.array
        The array of numbers, all inside the interval.

    """

    lengths = original.shape
    new = np.zeros_like(original)
    for idx0 in range(lengths[0]):
        for idx1 in range(lengths[1]):
            new[idx0, idx1] = bring_inside_bounds_scalar(original[idx0, idx1], lower_bound, upper_bound, include)

    return new


def bring_inside_bounds_scalar(original: float, lower_bound: float,
                               upper_bound: float, include: str = 'lower') -> float:

    """This function brings a scalar inside the given bounds, assuming the interval defined by the bounds can periodically extend the whole real line (e.g. an angle of 9$\pi$ is equivalent to an angle of $\pi$ and at the same time equivalent to an angle of $-\pi$). It returns another scalar.

    Parameters
    ----------
    original: float
        The original number.

    lower_bound: float
        The lower bound of the periodic interval.

    upper_bound: float
        The upper bound of the periodic interval.

    include: str
        The bound that is to be kept. It can be 'upper' or 'lower'. Anything else will result in an error.

    Returns
    -------
    float
        The number, now inside the interval.

    """

    # EXPLAIN THINGS HERE. MAKE CLEAR WHAT VARIABLES REPRESENT.

    if original == upper_bound or original == lower_bound:
        if include == 'lower':
            return lower_bound
        else:
            return upper_bound

    if lower_bound < original < upper_bound:
        return original

    center = (upper_bound + lower_bound) / 2.0

    if original < lower_bound:
        reflect = True
    else:
        reflect = False

    if reflect:
        original = 2.0 * center - original

    dividend = original - lower_bound
    divisor = upper_bound - lower_bound
    remainder = dividend % divisor
    new = lower_bound + remainder

    if reflect: new = 2.0 * center - new

    if new == lower_bound and include == 'upper':
        new = upper_bound
    if new == upper_bound and include == 'lower':
        new = lower_bound

    return new

def small_singular_value_entries(M, threshold=1e-10):
    # https://stackoverflow.com/questions/18452633/how-do-i-associate-which-singular-value-corresponds-to-what-entry
    U, s, V = np.linalg.svd(M)
    S = np.zeros(M.shape,dtype = np.float64)
    m = min(M.shape)
    S[:m,:m] = np.diag(s)
    Sp = S.copy()
    for m in range(0,m):
      Sp[m,m] = 1.0/Sp[m,m] if Sp[m,m] != 0 else 0
    Vs = np.matrix(V).getH()
    Vs_significant = Vs[np.where(s>threshold)]

    significant_var_index = np.where(np.sum(np.abs(Vs_significant),axis = 0) > threshold)[1]
    return significant_var_index

def sun_angles_from_sunlight_vector(R_VB, n_s):
    n_s_vane_frame = np.dot(R_VB, n_s)
    n_s_vane_frame /= np.linalg.norm(n_s_vane_frame)
    if (abs((abs(n_s_vane_frame[2]) - 1)) < 1e-15):
        current_alpha_s = np.arccos(-n_s_vane_frame[2])
        current_beta_s = 0  # beta does not matter
    else:
        current_beta_s = np.arctan2(n_s_vane_frame[1], n_s_vane_frame[0])
        if (abs(np.cos(current_beta_s)) >= 1e-15):
            current_alpha_s = np.arctan2(n_s_vane_frame[0] / np.cos(current_beta_s), -n_s_vane_frame[2])
        else:
            current_alpha_s = np.arctan2(n_s_vane_frame[1] / np.sin(current_beta_s), -n_s_vane_frame[2])

    return current_alpha_s, current_beta_s

def special_round(vec, decimals=3):
    """
    Rounding function for numbers between -1 and 1
    Reference:
    https://stackoverflow.com/questions/70377163/python-how-to-round-numbers-smaller-than-1-adaptively-with-specified-precision
    :param vec: 1D numpy array
    :return: rounded 1D numpy array
    """
    exponents = np.floor(np.log10(np.abs(vec))).astype(int)
    return np.stack([np.round(v, decimals=-e+decimals) for v, e in zip(vec, exponents)])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def divide_list(lst, n):
    """
    Divide a list into n roughly equal parts.

    Parameters:
    lst (list): The list to be divided.
    n (int): The number of parts to divide the list into.

    Returns:
    list of lists: A list containing n sublists with the divided elements.
    """
    # Determine the size of each part
    avg = len(lst) / float(n)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out




