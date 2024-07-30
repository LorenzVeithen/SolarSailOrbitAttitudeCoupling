import numpy as np
import itertools
import matplotlib.pyplot as plt


def hole_area(dp, rho_p, rho_t, Vp_rel, ct, cp, t_t, d_p):
    C1 = 2.8
    p1 = -0.035
    p2 = 0.335
    p3 = 0.113
    p4 = 0.516
    C2 = 0.6
    Dh = dp * (C1 * (rho_p/rho_t)**p1
                * (np.linalg.norm(Vp_rel)/ct)**p2
                * (np.linalg.norm(Vp_rel)/cp)**p3
                * (t_t/d_p)**p4
                * np.cos(0.026)**(0.026)) + C2
    return 0


def initial_rotational_velocity(mp, Vp_rel, Ib_sc, rp, beta, omega_0=np.array([0, 0, 0])):
    """

    :param mp: float, Mass of the projectile in kg.
    :param Vp_rel: (3,) numpy array, Relative projectile velocity, in m/s. This is the impact velocity.
    :param Ib_sc: (3,3) numpy array, Spacecraft inertia tensor in kg kg m^2.
    :param rp: (3,) numpy array, Position of the impact on the sail.
    :param beta: float, Momentum enhancement factor to be used
    :return:
    """
    p_vec = mp * Vp_rel
    p_vec_enhanced = p_vec * beta
    torque_on_sail = np.cross(rp, p_vec_enhanced)
    delta_omega = np.linalg.inv(Ib_sc, torque_on_sail)
    return omega_0 + delta_omega

def projectile_direction_body_frame(theta, phi):
    # This is the same as the sunlight vector, but with theta and phi instead!
    return np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    -np.cos(theta)])

# First considering the initial rotational velocity
beta_list = [1, 2, 3, 4, 5]
position_on_boom_list = np.linspace(-7, 7, 10)
boom_direction_list = [np.array([1, 0, 0]), np.array([0, 1, 0])]    # First do the X-booms and then the Y-booms
impact_position_vector_list = []
for comb in list(itertools.product(position_on_boom_list, boom_direction_list)):
    impact_position_vector_list.append(comb[0] * comb[1])

# Necessary sail characteristics
sail_I = np.zeros((3, 3))
sail_I[0, 0] = 10.5
sail_I[1, 1] = 10.5
sail_I[2, 2] = 21
booms_length = 7    # m

# Orbital debris - assuming aluminium
rho_debris = 2700   # kg/m^3
projectile_diameter = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]) * 1e-3    # m  from 0.1 \micro m to 10 cm
projectile_mass_list = (4 / 3) * np.pi * ((projectile_diameter/2) ** 3) * rho_debris
projectile_velocity_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) * 1e3  # m/s
theta_list = [0]#np.linspace(-np.pi, np.pi, 10)
phi_list = [0]#np.linspace(-np.pi, np.pi, 10)


num_combinations = len(projectile_mass_list) * len(projectile_velocity_list) * len(beta_list) * len(impact_position_vector_list) * len(theta_list) * len(phi_list)
print(num_combinations)

# Prepare array of all combinations
HVI_data_arr = np.empty((num_combinations, 11), dtype=float)

counter = 0
for mp in projectile_mass_list:                                             # Mass of the projectile
    for velocity in projectile_velocity_list:                               # Projectile velocity
        for beta in beta_list:                                              # Momentum enhancement factor
            p_enhanced = (mp * velocity) * beta                             # Enhanced linear momentum of the projectile
            for impact_body_fixed_position in impact_position_vector_list:  # Position on the given boom
                for theta_id, theta in enumerate(theta_list):               # First direction angle
                    for phi_id, phi in enumerate(phi_list):                 # Second direction angle

                        # Store data
                        HVI_data_arr[counter, 0] = mp
                        HVI_data_arr[counter, 1] = velocity
                        HVI_data_arr[counter, 2] = beta
                        HVI_data_arr[counter, 3] = 0 if (impact_body_fixed_position[0] != 0) else 1     # 0 for X-axis and 1 for Y-axis
                        HVI_data_arr[counter, 4] = np.linalg.norm(impact_body_fixed_position)
                        HVI_data_arr[counter, 5] = theta
                        HVI_data_arr[counter, 6] = phi

                        # Compute attitude perturbation
                        transferred_momentum_vector = p_enhanced * projectile_direction_body_frame(theta, phi)
                        omega_increment = np.dot(np.linalg.inv(sail_I), (np.cross(impact_body_fixed_position, transferred_momentum_vector)))

                        # Store attitude perturbation
                        HVI_data_arr[counter, 7] = omega_increment[0]
                        HVI_data_arr[counter, 8] = omega_increment[1]
                        HVI_data_arr[counter, 9] = omega_increment[2]
                        HVI_data_arr[counter, 10] = np.linalg.norm(omega_increment)

                        counter += 1
                        if (counter % 1000):
                            print(100 * counter/num_combinations)


colors_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
# First plot - influence of the mass and relative velocity on the omega increment
plt.figure()
# Filter the dataset to the ones that are interesting
mp_HVI_data_arr = HVI_data_arr[np.where(HVI_data_arr[:, 3] == 0)[0], :]     # Focus on the 'x' axis
mp_HVI_data_arr = mp_HVI_data_arr[np.where(mp_HVI_data_arr[:, 4] == 7)[0], :]   # Focus on the ones at the tip of the booms
mp_HVI_data_arr = mp_HVI_data_arr[np.where(mp_HVI_data_arr[:, 5] == 0)[0], :]   # Keep theta=0 for this analysis
mp_HVI_data_arr = mp_HVI_data_arr[np.where(mp_HVI_data_arr[:, 6] == 0)[0], :]   # Keep phi=0 for this analysis
for mp_id, mp in enumerate(projectile_mass_list):
    current_HVI_data_arr_mp = mp_HVI_data_arr[np.where(mp_HVI_data_arr[:, 0] == mp)[0], :]
    current_HVI_data_arr_mp_min_beta = current_HVI_data_arr_mp[np.where(current_HVI_data_arr_mp[:, 2] == 1)[0], :]
    current_HVI_data_arr_mp_max_beta = current_HVI_data_arr_mp[np.where(current_HVI_data_arr_mp[:, 2] == 5)[0], :]
    plt.plot(current_HVI_data_arr_mp_min_beta[:, 1]/1000, np.rad2deg(current_HVI_data_arr_mp_min_beta[:, 10]),
             color=colors_list[mp_id])
    plt.plot(current_HVI_data_arr_mp_max_beta[:, 1]/1000, np.rad2deg(current_HVI_data_arr_mp_max_beta[:, 10]),
             color=colors_list[mp_id])
    plt.fill_between(current_HVI_data_arr_mp_max_beta[:, 1]/1000,
                     np.rad2deg(current_HVI_data_arr_mp_max_beta[:, 10]),
                     np.rad2deg(current_HVI_data_arr_mp_min_beta[:, 10]),
                     color=colors_list[mp_id],
                     alpha=0.5,
                     label=r'$d_{p}$ = ' + f"{'{:0.0e}'.format(projectile_diameter[mp_id])} m")


plt.grid(True)
plt.xlabel(r'$V_{p}$ [km/s]', fontsize=14)
plt.ylabel(r'$||\omega_{0}||$ [deg/s]', fontsize=14)
plt.axhline(y=0.1, color='k', linestyle='--', label='Significance level')
plt.axhline(y=15, color='k', linestyle=':', label='Capability level')
plt.yscale('log')
plt.legend(ncol=3, loc='lower left', prop={'size': 8})
plt.ylim(1e-18, np.rad2deg(max(HVI_data_arr[:, 10]))*2)

# Second figure: linear momentum and beta
plt.figure()
# Filter the dataset to the ones that are interesting
beta_HVI_data_arr = HVI_data_arr[np.where(HVI_data_arr[:, 3] == 0)[0], :]     # Focus on the 'x' axis
beta_HVI_data_arr = beta_HVI_data_arr[np.where(beta_HVI_data_arr[:, 4] == 7)[0], :]   # Focus on the ones at the tip of the booms
beta_HVI_data_arr = beta_HVI_data_arr[np.where(beta_HVI_data_arr[:, 5] == 0)[0], :]   # Keep theta=0 for this analysis
beta_HVI_data_arr = beta_HVI_data_arr[np.where(beta_HVI_data_arr[:, 6] == 0)[0], :]   # Keep phi=0 for this analysis
for beta_id, beta in enumerate(beta_list):
    current_HVI_data_arr_beta = beta_HVI_data_arr[np.where(beta_HVI_data_arr[:, 2] == beta)[0], :]
    p = current_HVI_data_arr_beta[:, 0] * current_HVI_data_arr_beta[:, 1]   # Linear momentum of the considered particles

    reduced_arr = np.empty((len(p), 2))
    reduced_arr[:, 0] = p
    reduced_arr[:, 1] = current_HVI_data_arr_beta[:, 10]
    reduced_arr = reduced_arr[reduced_arr[:, 0].argsort()]  # order by the linear momentum
    plt.plot(reduced_arr[:, 0], np.rad2deg(reduced_arr[:, 1]),
             color=colors_list[beta_id], label=rf'$\beta$ = {beta}')
plt.grid(True)
plt.xlabel(r'$p_{p}$ [kg m/s]', fontsize=14)
plt.ylabel(r'$||\omega_{0}||$ [deg/s]', fontsize=14)
plt.axhline(y=0.1, color='k', linestyle='--', label='Significance level')
plt.axhline(y=15, color='k', linestyle=':', label='Capability level')
plt.yscale('log')
plt.xscale('log')
plt.legend(ncol=1, prop={'size': 8})

# Third figure: dependence of the initial rotational velocity on the angles
f, ax = plt.subplots()
ax.set_xlabel(r'$\omega_{x}$ [deg/s]', fontsize=14)
ax.set_ylabel(r'$\omega_{y}$ [deg/s]', fontsize=14)
tpc = ax.tripcolor(current_last_comp_array[:, 0], current_last_comp_array[:, 1], current_last_comp_array[:, i + 2],
                   shading='flat', cmap=my_cmap)
cbar = f.colorbar(tpc)
cbar.set_label(labels_change_list[i], rotation=270, labelpad=13, fontsize=14)

plt.show()