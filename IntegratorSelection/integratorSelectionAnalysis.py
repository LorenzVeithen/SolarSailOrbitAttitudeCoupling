import sys
from generalConstants import tudat_path
sys.path.insert(0, tudat_path)
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from tudatpy.util import compare_results, result2array
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix
import os
from pathlib import Path
from integratorSelectionSailModel import integrator_selection_data_directory
from generalConstants import Project_directory

selected_benchmark_step = 2**(-3)
lower_than_selected_time_step = 2**(-4)
markers_list = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p",
                "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", "o", "v", "^", "<", ">"]

PLOT_CHECKS = False
# load benchmark data
benchmark_state_history_array = np.loadtxt(
    integrator_selection_data_directory + f'/BenchmarkSelection/Cowell/state_history_benchmark_dt_{selected_benchmark_step}.dat')
benchmark_dependent_variable_array = np.loadtxt(
    integrator_selection_data_directory + f'/BenchmarkSelection/Cowell/dependent_variable_history_benchmark_dt_{selected_benchmark_step}.dat')
benchmark_state_history_dict, benchmark_dependent_variables_dict = {}, {}
for j, time in enumerate(benchmark_state_history_array[:, 0]):
    benchmark_state_history_dict[time] = benchmark_state_history_array[j, 1:]
    benchmark_dependent_variables_dict[time] = benchmark_dependent_variable_array[j, 1:]

# load smaller
benchmark_check_state_history_array = np.loadtxt(
    integrator_selection_data_directory + f'/BenchmarkSelection/Cowell/state_history_benchmark_dt_{lower_than_selected_time_step}.dat')
benchmark_check_state_history_dict = {}
for j, time in enumerate(benchmark_check_state_history_array[:, 0]):
    benchmark_check_state_history_dict[time] = benchmark_check_state_history_array[j, 1:]

# get benchmark error
benchmark_state_error = compare_results(benchmark_check_state_history_dict, benchmark_state_history_dict, benchmark_state_history_array[6:-6, 0])
benchmark_state_error_array = result2array(benchmark_state_error)
max_position_error_norm = max(np.sqrt(np.sum(benchmark_state_error_array[:, 1:4] ** 2, axis=1)))
max_velocity_error_norm = max(np.sqrt(np.sum(benchmark_state_error_array[:, 4:7] ** 2, axis=1)))
max_rotational_velocity_error_norm = max(np.rad2deg(np.sqrt(np.sum(benchmark_state_error_array[:, 11:14] ** 2, axis=1))))

# initialise dictionary to store information
integrator_dicts = {}

integrator_directory = integrator_selection_data_directory + "/IntegratorSelection"
p = Path(integrator_directory)
integrator_sub_directories = [x for x in p.iterdir() if x.is_dir()]
num_integrators = len(integrator_sub_directories)
for int_sub_dir in integrator_sub_directories:
    current_integrator_str = str(int_sub_dir).split('/')[-1]
    print(f'---- {current_integrator_str} ----')
    p = Path(int_sub_dir)
    tolerance_sub_directories = [x for x in p.iterdir() if x.is_dir()]
    time_array_list = []
    position_error_norm_list, velocity_error_norm_list, omega_error_norm_list = [], [], []
    max_position_error_list, max_velocity_error_list, max_omega_error_list, num_eval_list = [], [], [], []
    state_hist_array_list, dependent_variable_hist_array_list = [], []
    state_error_array_list, dependent_variable_error_array_list = [], []
    propagator_list, success_boolean_list = [], []

    tolerances_list = []
    tolerance_sub_directories = sorted(tolerance_sub_directories)
    for tol_sub_dir in tolerance_sub_directories:
        current_tol_str = str(tol_sub_dir).split('/')[-1]
        print(f'---- {current_tol_str} ----')
        # retrieve state history and dependent variables arrays
        current_state_history_array = np.loadtxt(str(tol_sub_dir) + f'/state_history.dat')
        current_dependent_variable_history_array = np.loadtxt(str(tol_sub_dir) + f'/dependent_variable_history.dat')

        # convert arrays to dictionaries to allow for easy comparison
        current_state_history_dict, current_dependent_variables_history_dict = {}, {}
        for j, time in enumerate(current_state_history_array[:, 0]):
            current_state_history_dict[time] = current_state_history_array[j, 1:]
            current_dependent_variables_history_dict[time] = current_dependent_variable_history_array[j, 1:]

        # cut time nodes at which errors in interpolation due to Runge's phenomenon are expected
        current_state_history_time_array = current_state_history_array[6:-6, 0]  # remove first six points and last six points due to interpolation error
        current_state_history_time_array = current_state_history_time_array[
            current_state_history_time_array > benchmark_state_history_array[6, 0]]
        current_state_history_time_array = current_state_history_time_array[
            current_state_history_time_array < benchmark_state_history_array[-6, 0]]

        # evaluate error with respect to benchmark
        state_history_error_array = result2array(compare_results(benchmark_state_history_dict, current_state_history_dict, current_state_history_time_array))
        dependent_variable_history_error_array = result2array(compare_results(benchmark_dependent_variables_dict, current_dependent_variables_history_dict, current_state_history_time_array))

        position_error_norm = np.sqrt(np.sum(state_history_error_array[:, 1:4] ** 2, axis=1))
        velocity_error_norm = np.sqrt(np.sum(state_history_error_array[:, 4:7] ** 2, axis=1))
        rotational_velocity_error_norm = np.rad2deg(np.sqrt(np.sum(state_history_error_array[:, 11:14] ** 2, axis=1)))

        time_array_list.append(current_state_history_time_array)
        state_hist_array_list.append(current_state_history_array)
        dependent_variable_hist_array_list.append(current_dependent_variable_history_array)

        state_error_array_list.append(state_history_error_array)
        dependent_variable_error_array_list.append(dependent_variable_history_error_array)

        position_error_norm_list.append(position_error_norm)
        velocity_error_norm_list.append(velocity_error_norm)
        omega_error_norm_list.append(rotational_velocity_error_norm)

        max_position_error_list.append(max(position_error_norm))
        max_velocity_error_list.append(max(velocity_error_norm))
        max_omega_error_list.append(max(rotational_velocity_error_norm))

        # read ancillary file
        with open(str(tol_sub_dir) + "/ancillary_simulation_info.txt", 'r') as f:
            for l, line in enumerate(f):
                info = line.split("\t")[-1]
                if (l == 0):
                    num_function_evaluations = int(info)
                elif (l == 1):
                    propagation_success_boolean = bool(info)
                elif (l == 2):
                    used_propagator = info
                elif (l == 3):
                    current_tol_str = info
                    used_tolerance = float(info)
        print(f'num_function_evaluations: {num_function_evaluations}')
        print(f'used_tolerance: {used_tolerance}')
        print(f'used_propagator: {used_propagator}')
        print(f'propagation_success_boolean: {propagation_success_boolean}')
        num_eval_list.append(num_function_evaluations)
        tolerances_list.append(used_tolerance)
        propagator_list.append(used_propagator)
        success_boolean_list.append(propagation_success_boolean)

    integrator_dicts[current_integrator_str + "_tol_list"] = tolerances_list
    integrator_dicts[current_integrator_str + "_num_eval_list"] = num_eval_list
    integrator_dicts[current_integrator_str + "_propagator_list"] = propagator_list
    integrator_dicts[current_integrator_str + "_success_boolean_list"] = success_boolean_list

    integrator_dicts[current_integrator_str + "_time_hist_list"] = time_array_list
    integrator_dicts[current_integrator_str + "_state_hist_list"] = state_hist_array_list
    integrator_dicts[current_integrator_str + "_dependent_variable_hist_list"] = dependent_variable_hist_array_list

    integrator_dicts[current_integrator_str + "_state_error_list"] = state_error_array_list
    integrator_dicts[current_integrator_str + "_dependent_variable_error_list"] = dependent_variable_error_array_list

    integrator_dicts[current_integrator_str + "_pos_error_hist_list"] = position_error_norm_list
    integrator_dicts[current_integrator_str + "_vel_error_hist_list"] = velocity_error_norm_list
    integrator_dicts[current_integrator_str + "_omega_error_hist_list"] = omega_error_norm_list

    integrator_dicts[current_integrator_str + "_max_pos_error_list"] = max_position_error_list
    integrator_dicts[current_integrator_str + "_max_vel_error_list"] = max_velocity_error_list
    integrator_dicts[current_integrator_str + "_max_omega_error_list"] = max_omega_error_list


integrator_list = [str(x).split("/")[-1] for x in integrator_sub_directories]

colors_integrators = pl.cm.jet(np.linspace(0, 1, num_integrators))  # initialise colors for each integrator

req_pos = 1
plt.figure()
for i, int in enumerate(integrator_list):
    plt.plot(integrator_dicts[int + "_num_eval_list"], integrator_dicts[int + "_max_pos_error_list"], label=int, marker=markers_list[i], color=colors_integrators[i])
plt.yscale('log')
plt.xscale('log')
plt.axhline(y=req_pos, color='r', linestyle='--', label='Max position error')
plt.axhline(y=max_position_error_norm, color='b', linestyle='--', label='Benchmark accuracy')
plt.grid(True, which="both")
plt.xlabel('Number of functions evaluations, N, [-]', fontsize=14)
plt.ylabel(r'Maximum position error norm, $\epsilon_r$, [m]', fontsize=14)
plt.legend(ncol=2)
plt.savefig(Project_directory + '/0_FinalPlots/Integrator_Selection/Integrator_Comparison_position.png',
            dpi=1200,
            bbox_inches='tight')

req_vel = 1e-3
plt.figure()
for i, int in enumerate(integrator_list):
    plt.plot(integrator_dicts[int + "_num_eval_list"], integrator_dicts[int + "_max_vel_error_list"], label=int, marker=markers_list[i], color=colors_integrators[i])
plt.yscale('log')
plt.xscale('log')
plt.axhline(y=req_vel, color='r', linestyle='--', label='Max velocity error')
plt.axhline(y=max_velocity_error_norm, color='b', linestyle='--', label='Benchmark accuracy')
plt.grid(True, which="both")
plt.xlabel('Number of functions evaluations, N, [-]', fontsize=14)
plt.ylabel(r'Maximum velocity error norm, $\epsilon_v$, [m/s]', fontsize=14)
plt.legend(ncol=2)
plt.savefig(Project_directory + '/0_FinalPlots/Integrator_Selection/Integrator_Comparison_velocity.png',
            dpi=1200,
            bbox_inches='tight')

req_omega = 5e-2
plt.figure()
for i, int in enumerate(integrator_list):
    plt.plot(integrator_dicts[int + "_num_eval_list"], integrator_dicts[int + "_max_omega_error_list"], label=int, marker=markers_list[i], color=colors_integrators[i])
plt.yscale('log')
plt.xscale('log')
plt.axhline(y=req_omega, color='r', linestyle='--', label=r'Max rotational velocity error')
plt.axhline(y=max_rotational_velocity_error_norm, color='b', linestyle='--', label='Benchmark accuracy')
plt.grid(True, which="both")
plt.xlabel('Number of functions evaluations, N, [-]', fontsize=14)
plt.ylabel('Maximum rotational velocity error norm,\n' +'$\epsilon_{\omega}$, [deg/s]', fontsize=14)
plt.legend(ncol=2)
plt.savefig(Project_directory + '/0_FinalPlots/Integrator_Selection/Integrator_Comparison_omega.png',
            dpi=1200,
            bbox_inches='tight')

if (PLOT_CHECKS):
    selected_integrator_and_tol = {"rkf_45": [1e-13, 1e-14],
                                   "rkdp_87": [1e-13, 1e-14],
                                   "rkf_78": [1e-13, 1e-14],
                                   "rkf_89": [1e-13, 1e-14]
                                    }
    for v in range(4):
        plt.figure()
        for k in selected_integrator_and_tol.keys():
            for t in selected_integrator_and_tol[k]:
                idx = integrator_dicts[k + "_tol_list"].index(t)
                current_state_array = integrator_dicts[k + "_state_hist_list"][idx]
                plt.plot((current_state_array[:, 0] - current_state_array[0, 0]) / 3600, current_state_array[:, 7 + v],
                         label=f"{k} tol={t}")
        plt.plot((benchmark_state_history_array[:, 0] - benchmark_state_history_array[0, 0]) / 3600,
                 benchmark_state_history_array[:, 7 + v], label=f"Benchmark")
        plt.legend()
        plt.xlabel("Time [hours]")
        plt.ylabel(f"Quaternion - {v}")
        plt.grid(True)

    for v in range(8):
        if (v < 4):
            y_label = f"alpha_1, vane {v}"
        else:
            y_label = f"alpha_2, vane {v-4}"
        plt.figure()
        for k in selected_integrator_and_tol.keys():
            for t in selected_integrator_and_tol[k]:
                idx = integrator_dicts[k + "_tol_list"].index(t)
                current_dependent_variable_array = integrator_dicts[k + "_dependent_variable_hist_list"][idx]
                plt.plot((current_dependent_variable_array[:, 0] - current_dependent_variable_array[0, 0]) / 3600,
                         current_dependent_variable_array[:, 21 + v], label=f"{k} tol={t}")
        plt.plot((benchmark_dependent_variable_array[:, 0] - benchmark_dependent_variable_array[0, 0]) / 3600,
                 benchmark_dependent_variable_array[:, 21 + v], label=f"Benchmark")
        plt.legend()
        plt.xlabel("Time [hours]")
        plt.ylabel(y_label)
        plt.grid(True)
