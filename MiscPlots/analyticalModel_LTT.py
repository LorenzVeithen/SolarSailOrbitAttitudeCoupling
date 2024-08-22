import numpy as np
import matplotlib.pyplot as plt

state_history_array_analytical = np.loadtxt("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")
dependent_variable_history_array_analytical = np.loadtxt(
    "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_0.0_omega_y_0.0_omega_z_0.0_reduced_area_LTT.dat")

# Extract dependent variables
t_hours_analytical = (dependent_variable_history_array_analytical[:, 0] - dependent_variable_history_array_analytical[0, 0]) / (3600*24)
keplerian_state_analytical = dependent_variable_history_array_analytical[:, 1:7]
pericenter_analytical = keplerian_state_analytical[:, 0] * (1 - keplerian_state_analytical[:, 1])/1000
apocenter_analytical = keplerian_state_analytical[:, 0] * (1 + keplerian_state_analytical[:, 1])/1000

state_history_array_numerical = np.loadtxt("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_40.0_omega_y_-70.0_omega_z_0.0.dat")
dependent_variable_history_array_numerical = np.loadtxt(
    "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/LTT_Data/Sun_Pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/dependent_variable_history/dependent_variable_history_omega_x_40.0_omega_y_-70.0_omega_z_0.0.dat")

# Extract dependent variables
t_hours_numerical = (dependent_variable_history_array_numerical[:, 0] - dependent_variable_history_array_numerical[0, 0]) / (3600*24)
keplerian_state_numerical = dependent_variable_history_array_numerical[:, 1:7]
pericenter_numerical = keplerian_state_numerical[:, 0] * (1 - keplerian_state_numerical[:, 1])/1000
apocenter_numerical = keplerian_state_numerical[:, 0] * (1 + keplerian_state_numerical[:, 1])/1000


plt.figure()
plt.plot(t_hours_numerical, pericenter_numerical, label='Numerical', color="#E69F00")
plt.plot(t_hours_analytical, pericenter_analytical, label='Analytical', linestyle='--', color='k')
plt.xlabel(r"$t$ [days]")
plt.ylabel(r"$r_{p}$ [km]")
plt.grid(True)
plt.legend(prop={'size': 8})
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc/AnalyticalPericenter.png",
            bbox_inches='tight', dpi=1200)

plt.figure()
plt.plot(t_hours_numerical, apocenter_numerical, label='Numerical', color="#E69F00")
plt.plot(t_hours_analytical, apocenter_analytical, label='Analytical', linestyle='--', color='k')
plt.xlabel(r"$t$ [days]", fontsize=14)
plt.ylabel(r"$r_{a}$ [km]", fontsize=14)
plt.grid(True)
plt.legend(prop={'size': 8})
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc/AnalyticalApocenter.png",
            bbox_inches='tight', dpi=1200)
plt.show()
