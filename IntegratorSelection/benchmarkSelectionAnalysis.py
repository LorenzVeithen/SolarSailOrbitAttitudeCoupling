import sys
sys.path.insert(0, r"/Users/lorenz_veithen/tudat-bundle/build/tudatpy")
import matplotlib.pyplot as plt
import numpy as np
from tudatpy.util import compare_results, result2array
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix
from integratorSelectionSailModel import integrator_selection_data_directory

PLOT_CHECKS = True
benchmark_time_steps = [2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 2**0, 2**(-1), 2**(-2), 2**(-3), 2**(-4), 2**(-5)]    # , , 2**(-6), 2**(-7)
#benchmark_time_steps = [2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**0] #, 2**0, 2**(-1), 2**(-2), 2**(-3)
state_history_arrays_list, dependent_variable_arrays_list = [], []
state_history_dicts_list, dependent_variable_dicts_list = [], []

for dt in benchmark_time_steps:
    print(f"loading {dt} s benchmark")

    state_history_array = np.loadtxt(integrator_selection_data_directory + f'/IntegratorSelection/BenchmarkSelection/Cowell/state_history_benchmark_dt_{dt}.dat')
    state_history_arrays_list.append(state_history_array)

    dependent_variable_array = np.loadtxt(integrator_selection_data_directory + f'/IntegratorSelection/BenchmarkSelection/Cowell/dependent_variable_history_benchmark_dt_{dt}.dat')
    dependent_variable_arrays_list.append(dependent_variable_array)

    current_state_history_dict = {}
    current_dependent_variables_dict = {}
    for j, time in enumerate(state_history_array[:, 0]):
        current_state_history_dict[time] = state_history_array[j, 1:]
        current_dependent_variables_dict[time] = dependent_variable_array[j, 1:]
    state_history_dicts_list.append(current_state_history_dict)
    dependent_variable_dicts_list.append(current_dependent_variables_dict)

time_list = []
position_error_norm_list, velocity_error_norm_list, omega_error_norm_list = [], [], []
max_position_error, max_velocity_error, max_omega_error = [], [], []
vane_angles_error_deg_list = []
detumbling_time_list = []
for i in range(len(state_history_arrays_list) - 1):
    print(f"processing {benchmark_time_steps[i]} s benchmark")
    total_time = state_history_arrays_list[i][-1, 0] - state_history_arrays_list[i][0, 0]

    current_state_history = state_history_arrays_list[i]
    current_state_history_time_array = current_state_history[:, 0]

    next_state_history = state_history_arrays_list[i + 1]
    next_state_history_at_current_time_steps = next_state_history[::2, :]

    # remove some data points to avoid issues due to Runge's phenomenon (6 on each side of the interpolated benchmark, for both)
    current_state_history_time_array = current_state_history_time_array[6:-6]   # coarser one in either case

    state_error = compare_results(state_history_dicts_list[i+1], state_history_dicts_list[i], current_state_history_time_array)
    state_error_array = result2array(state_error)

    dependent_array_error = compare_results(dependent_variable_dicts_list[i+1], dependent_variable_dicts_list[i], current_state_history_time_array)
    dependent_array_error_array = result2array(dependent_array_error)

    time_list.append(current_state_history_time_array)
    position_error_norm = np.sqrt(np.sum(state_error_array[:, 1:4]**2, axis=1))
    position_error_norm_list.append(position_error_norm)
    velocity_error_norm = np.sqrt(np.sum(state_error_array[:, 4:7] ** 2, axis=1))
    velocity_error_norm_list.append(velocity_error_norm)
    rotational_velocity_error_norm = np.rad2deg(np.sqrt(np.sum(state_error_array[:, 11:14] ** 2, axis=1)))
    omega_error_norm_list.append(rotational_velocity_error_norm)

    max_position_error.append(max(position_error_norm))
    max_velocity_error.append(max(velocity_error_norm))
    max_omega_error.append(max(rotational_velocity_error_norm))

    # Difference in the vane angles
    vane_angles_error_deg = np.rad2deg(dependent_array_error_array[:, 21:29])
    vane_angles_error_deg_list.append(vane_angles_error_deg)

for i in range(len(state_history_arrays_list)):
    current_state_history_time_array = state_history_arrays_list[i][:, 0]
    # detumbling time
    list_indices_zero_angles = np.where(dependent_variable_arrays_list[i][:, 21] == 0)[0]
    if (len(list_indices_zero_angles) !=0):
        detumbling_time_list.append((current_state_history_time_array[list_indices_zero_angles[0]]-current_state_history_time_array[0])/3600)
    else:
        detumbling_time_list.append(None)

print(detumbling_time_list)
plt.figure()
for i in range(len(benchmark_time_steps)-1):
    plt.semilogy((time_list[i]-time_list[i][0])/3600, position_error_norm_list[i], label=f"dt={benchmark_time_steps[i]} s")
plt.xlabel(r"Time, $t$ [hours]", fontsize=14)
plt.ylabel(r'Position error norm, $||\Delta r||$, [m]', fontsize=14)
plt.grid(True)
plt.legend()

plt.figure()
for i in range(len(benchmark_time_steps)-1):
    plt.semilogy((time_list[i]-time_list[i][0])/3600, velocity_error_norm_list[i], label=f"dt={benchmark_time_steps[i]} s")
plt.xlabel(r"Time, $t$ [hours]", fontsize=14)
plt.ylabel(r'Velocity error norm, $||\Delta V||$, [m/s]', fontsize=14)
plt.grid(True)
plt.legend()

plt.figure()
for i in range(len(benchmark_time_steps)-1):
    plt.semilogy((time_list[i]-time_list[i][0])/3600, omega_error_norm_list[i], label=f"dt={benchmark_time_steps[i]} s")
plt.xlabel(r"Time, $t$ [hours]", fontsize=14)
plt.ylabel('Rotational velocity error norm,\n' +'$||\Delta \omega||$, [deg/s]', fontsize=14)
plt.grid(True)
plt.legend()

plt.figure()
plt.loglog(benchmark_time_steps[:-1], max_position_error, marker="o")
plt.xlabel(r"Fixed time step size, $\Delta t$, [s]", fontsize=14)
plt.ylabel(r'Maximum position error norm, $\epsilon_r$, [m]', fontsize=14)
plt.grid(True, which='both')

plt.figure()
plt.loglog(benchmark_time_steps[:-1], max_velocity_error, marker="o")
plt.xlabel(r"Fixed time step size, $\Delta t$, [s]", fontsize=14)
plt.ylabel(r'Maximum velocity error norm, $\epsilon_v$, [m/s]', fontsize=14)
plt.grid(True, which='both')

plt.figure()
plt.loglog(benchmark_time_steps[:-1], max_omega_error, marker="o")
plt.xlabel(r"Fixed time step size, $\Delta t$, [s]", fontsize=14)
plt.ylabel('Maximum rotational velocity error norm,\n' +'$\epsilon_{\omega}$, [deg/s]', fontsize=14)
plt.grid(True, which='both')

if (PLOT_CHECKS):
    for v in range(4):
        plt.figure()
        for k in range(5, len(benchmark_time_steps)):
            plt.plot((state_history_arrays_list[k][:, 0]-state_history_arrays_list[k][0, 0])/3600, state_history_arrays_list[k][:, 7+v], label=f"dt={benchmark_time_steps[k]}")
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
        for k in range(5, len(benchmark_time_steps)):
            plt.plot((dependent_variable_arrays_list[k][:, 0]-dependent_variable_arrays_list[k][0, 0])/3600, dependent_variable_arrays_list[k][:, 21+v], label=f"dt={benchmark_time_steps[k]}")
        plt.legend()
        plt.xlabel("Time [hours]")
        plt.ylabel(y_label)
        plt.grid(True)

plt.show()



