import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
cmap = plt.get_cmap('jet')

def absolute_evolution_plot_LTT(x_arrays_list, y_arrays_list, xlabel, ylabel, savefig_label, close_fig_bool=True):
    plt.figure()
    for i in range(len(x_arrays_list)):
        plt.plot(x_arrays_list[i], y_arrays_list[i],
                 c="#E69F00",
                 alpha=0.5,
                 linewidth=1)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # complete hard-coded, do not reproduce at home
    if ('peri' in savefig_label or 'apo' in savefig_label):
        state_history_array_analytical = np.loadtxt(
            "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")
        dependent_variable_history_array_analytical = np.loadtxt(
            "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")

        # Extract dependent variables
        t_hours_analytical = (dependent_variable_history_array_analytical[:, 0] -
                              dependent_variable_history_array_analytical[0, 0]) / (3600 * 24)
        keplerian_state_analytical = dependent_variable_history_array_analytical[:, 1:7]
        pericenter_analytical = keplerian_state_analytical[:, 0] * (1 - keplerian_state_analytical[:, 1]) / 1000
        apocenter_analytical = keplerian_state_analytical[:, 0] * (1 + keplerian_state_analytical[:, 1]) / 1000

        if ('peri' in savefig_label):
            plt.plot(t_hours_analytical, pericenter_analytical, label='Simplified model', linestyle='--', color='k')
        else:
            plt.plot(t_hours_analytical, apocenter_analytical, label='Simplified model', linestyle='--', color='k')
        plt.plot([], [],  c="#E69F00",
                 label=f'Numerical propagations',
                 alpha=0.5,
                 linewidth=1)
        plt.legend()
    plt.savefig(savefig_label, bbox_inches='tight', dpi=300)
    if (close_fig_bool): plt.close()

def relative_evolution_plot_LTT(x_arrays_list, y_arrays_list,
                                xlabel, ylabel,
                                ref_x_array, ref_y_array,
                                colour_bar_values_list, colour_bar_label,
                                savefig_label, close_fig_bool=True):

    c_norm_omega_all = plt.Normalize(np.min(colour_bar_values_list), np.max(colour_bar_values_list))
    c_omega_all = cmap(c_norm_omega_all(colour_bar_values_list))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(x_arrays_list)):
        current_spline = CubicSpline(ref_x_array, ref_y_array)
        diff = y_arrays_list[i] - current_spline(x_arrays_list[i])

        plt.plot(x_arrays_list[i], diff,
                 c=c_omega_all[i],
                 label=f'{i}',
                 linewidth=1)

    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    cbar = fig.colorbar(cm.ScalarMappable(norm=c_norm_omega_all, cmap=cmap), ax=ax)
    cbar.set_label(colour_bar_label, rotation=270, labelpad=13)
    plt.savefig(savefig_label, bbox_inches='tight', dpi=300)
    if (close_fig_bool): plt.close()
