import matplotlib.pyplot as plt
import numpy as np
from generalConstants import AMS_directory
from vaneControllerMethods import buildEllipseCoefficientFunctions, ellipseCoefficientFunction, cart_to_pol
from matplotlib import cm
from scipy.optimize import golden

coefficients_labels = ['A', 'B', 'C', 'D', 'E', 'F']
sh_comp = 0     # 0 if no shadow
COMPUTE_SCALING = False
scaling = 1e-2      # necessary if COMPUTE_SCALING is False, disregarded otherwise
def truncated_fourier_ellipse(relative_magnitude, plot_bool=False):
    """

    :param relative_magnitude: maximum relative magnitude between the most significant and least significant term of the
    truncated fourier fit.
    :param plot_bool: boolean, True to plot related figures
    :return:
    """
    ellipse_coefficient_functions_list = []
    the_id_relevance_list = []
    for i in range(6):
        filename = f'{AMS_directory}/Datasets/Ideal_model/vane_1/dominantFitTerms/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{bool(sh_comp)}.txt'
        average_magnitude = np.genfromtxt(filename, delimiter=',')[:, 1]
        id_relevance = np.where(abs(average_magnitude) < max(average_magnitude) * relative_magnitude)[0]
        if (len(id_relevance) == 0):
            id_relevance = [-1]
        the_id_relevance = id_relevance[0]
        the_id_relevance_list.append(the_id_relevance)
        built_function = buildEllipseCoefficientFunctions(filename, number_of_terms=the_id_relevance+1)
        ellipse_coefficient_functions_list.append(
            lambda aps, bes, f=built_function: ellipseCoefficientFunction(aps, bes, f))

    test_data_file = f'{AMS_directory}/Datasets/Ideal_model/vane_1/ellipseCoefficients/AMS_ellipse_coefficients_shadow_{bool(sh_comp)}_hull_TyTz_test_data.csv'
    test_data = np.genfromtxt(test_data_file, delimiter=',')

    #test_data = test_data[3200:]
    x = test_data
    n = 100  # number of rows
    idx = np.random.choice(len(x), n, replace = False)
    test_data = np.array([x[i] for i in idx])

    alpha_s_rad_list = test_data[:, 0]
    beta_s_rad_list = test_data[:, 1]

    fourier_ellipse_coefficients = np.zeros_like(test_data)
    fourier_ellipse_coefficients[:, 0] = alpha_s_rad_list
    fourier_ellipse_coefficients[:, 1] = beta_s_rad_list

    for i, (alpha_s_rad, beta_s_rad) in enumerate(zip(alpha_s_rad_list, beta_s_rad_list)):
        if (i % 100 == 0):
            print(i)

        for j in range(6):
            fourier_ellipse_coefficients[i, 2 + j] = ellipse_coefficient_functions_list[j](alpha_s_rad, beta_s_rad)

    area_diff_rel = []
    for i in range(len(fourier_ellipse_coefficients[:, 0])):
        a, b, c, d, e, f = fourier_ellipse_coefficients[i, 2:]
        _, _, ap, bp, _, _ = cart_to_pol(fourier_ellipse_coefficients[i, 2:])
        A_fourier = np.pi * ap * bp
        _, _, ap, bp, _, _ = cart_to_pol(test_data[i, 2:])
        A_real = np.pi * ap * bp
        area_diff_rel.append(100 * (A_fourier - A_real) / A_real)
    area_diff_rel = np.array(area_diff_rel)

    alpha_sun_deg, beta_sun_deg = np.rad2deg(alpha_s_rad_list), np.rad2deg(beta_s_rad_list)
    if (plot_bool):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        my_cmap = plt.get_cmap('jet')
        ax.scatter(alpha_sun_deg, beta_sun_deg, area_diff_rel, c=area_diff_rel, cmap=my_cmap, alpha=0.5, label="Data")
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(f"Relative area difference, %")

        for i in range(6):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            z = (fourier_ellipse_coefficients[:, i + 2] - test_data[:, i + 2])
            id = z.argmax(axis=0)
            ax.scatter(alpha_sun_deg, beta_sun_deg, z, c=z, cmap=my_cmap, alpha=0.5, label="Data")
            ax.set_ylabel(f'{coefficients_labels[i]} coefficient error')

            plt.figure()
            plt.scatter(alpha_sun_deg, z, s=1)
            plt.xlabel('alpha_sun')
            plt.ylabel(f'{coefficients_labels[i]} coefficient error')

            plt.figure()
            plt.scatter(beta_sun_deg, z, s=1)
            plt.xlabel('beta_sun')
            plt.ylabel(f'{coefficients_labels[i]} coefficient error')

    print(f'maximum area difference: {max(area_diff_rel)}')
    print(f'minimum area difference: {min(area_diff_rel)}')
    print(f'mean area difference: {np.mean(area_diff_rel)}')
    print(f'STD area difference: {np.std(area_diff_rel)}')
    return [np.mean(area_diff_rel) - 2 * np.std(area_diff_rel), np.mean(area_diff_rel) + 2 * np.std(area_diff_rel)], the_id_relevance_list

def golden_section_wrapper(t, desired_two_sigma_interval_width=2):
    interval = truncated_fourier_ellipse(t)[0]
    interval_difference = interval[1] - interval[0]
    if (interval_difference > desired_two_sigma_interval_width):
        return 1
    else:
        return -1

if (COMPUTE_SCALING):
    scaling = golden(golden_section_wrapper, brack=(1e-7, 1), tol=1e-7)
    print(scaling)

# print the final function to be implemented in python
#_, the_id_relevance_list = truncated_fourier_ellipse(0)
the_id_relevance_list = [1700, 1700, 1700, 1700, 1700, 1700]
for i in range(6):
    filename = f'{AMS_directory}/Datasets/Ideal_model/vane_1/dominantFitTerms/{["A", "B", "C", "D", "E", "F"][i]}_shadow_{bool(sh_comp)}.txt'
    original_fourier_fit = np.genfromtxt(filename, delimiter=',', dtype=object)
    coefficients_array = "np.array(["
    expressions_array = "np.array(["
    for j in range(the_id_relevance_list[i]):
        coefficients_array += original_fourier_fit[j, 0].decode("utf-8") + ", "
        current_expression = original_fourier_fit[j, 2]
        expressions_array += current_expression.decode("utf-8") + ", "

    coefficients_array += "])"
    expressions_array += "])"
    print(i)
    print(expressions_array)
    print(coefficients_array)
    print('\n')


