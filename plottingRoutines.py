import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
cmap = plt.get_cmap('seismic')

def absolute_evolution_plot_LTT(x_arrays_list, y_arrays_list, xlabel, ylabel, savefig_label, close_fig_bool=True):
    plt.figure()
    for i in range(len(x_arrays_list)):
        plt.plot(x_arrays_list[i], y_arrays_list[i],
                 c='royalblue',
                 label=f'{i}',
                 alpha=0.5,
                 linewidth=1)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.savefig(savefig_label, bbox_inches='tight', dpi=1200)
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
    plt.savefig(savefig_label, bbox_inches='tight', dpi=1200)
    if (close_fig_bool): plt.close()
