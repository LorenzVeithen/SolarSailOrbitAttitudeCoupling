import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
line_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
plt.rc("axes", prop_cycle=line_cycler)

# Load data
normal_state_history_array = np.loadtxt("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_5.0_omega_y_5.0_omega_z_5.0_normal.dat")
normal_dependent_variable_history_array = np.loadtxt(
    "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_5.0_omega_y_5.0_omega_z_5.0_normal.dat")


direct_state_history_array = np.loadtxt("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/state_history_omega_x_5.0_omega_y_5.0_omega_z_5.0_direct.dat")
direct_dependent_variable_history_array = np.loadtxt(
    "/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_GeneratedData/PropagationData/dependent_variable_history_omega_x_5.0_omega_y_5.0_omega_z_5.0_direct.dat")

# Extract dependent variables
normal_t_dependent_variables_hours = (normal_dependent_variable_history_array[:, 0]-normal_dependent_variable_history_array[0, 0])/3600
normal_vanes_x_rotations = np.rad2deg(normal_dependent_variable_history_array[:, 21:25])
normal_vanes_y_rotations = np.rad2deg(normal_dependent_variable_history_array[:, 25:29])

direct_t_dependent_variables_hours = (direct_dependent_variable_history_array[:, 0]-direct_dependent_variable_history_array[0, 0])/3600
direct_vanes_x_rotations = np.rad2deg(direct_dependent_variable_history_array[:, 21:25])
direct_vanes_y_rotations = np.rad2deg(direct_dependent_variable_history_array[:, 25:29])

detumbling_normal = normal_t_dependent_variables_hours[np.where(np.sum(normal_vanes_x_rotations, axis=1)==0)[0][0]]
detumbling_direct = direct_t_dependent_variables_hours[np.where(np.sum(direct_vanes_x_rotations, axis=1)==0)[0][0]]

print(100 * (detumbling_direct-detumbling_normal)/detumbling_normal)

plt.figure()
plt.plot(direct_t_dependent_variables_hours, direct_vanes_x_rotations[:, 0], label='DIRECT')
plt.plot(normal_t_dependent_variables_hours, normal_vanes_x_rotations[:, 0], label='Hybrid')
plt.xlim((0, 10))
plt.legend()
plt.grid(True)
plt.xlabel(r'$t$ [hours]', fontsize=14)
plt.ylabel(r'$\theta_{v}$ vane 1 [deg]', fontsize=14)
plt.savefig("/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject/MSc_Thesis_Source_Python/0_FinalPlots/Misc/DIRECTvsHybridComparison.png",
            bbox_inches='tight', dpi=1200)
#plt.show()

