import numpy as np

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
    # TODO: check that this is indeed correct
    return np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    -np.cos(theta)])

# First considering the initial rotational velocity
beta_list = [1, 2, 3, 4, 5]
position_on_boom_list = np.linspace(0, 7, 10)

# Orbital debris
rho_debris = 2700   # kg/m^3
projectile_diameter = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]) * 1e-3    # m  from 0.1 \micro m to 10 cm
projectile_mass = (4/3) * np.pi * projectile_diameter**3 * rho_debris
projectile_velocity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) * 1e3  # m/s
theta = np.linspace(0, 90, 10)
phi = np.linspace(0, 90, 10)


#TODO: add direction
#TODO: think about what is too much to consider and what is not enough, and a proper justification for these
all_combinations = []
for mp in projectile_mass:
    for beta in beta_list:
        for direction_id, direction_array in enumerate([np.array([1, 0, 0]), np.array([0, 1, 0])]):
            for position_on_boom in position_on_boom_list:
                impact_body_fixed_position = position_on_boom * direction_array

                momentum_transfered = beta * np.array([0, 0, 1]) * p
                omega_increment = np.linalg.inv(sail_I) * (np.cross(impact_body_fixed_position, momentum_transfered))
                print("------")
                print(p)
                print(np.rad2deg(np.linalg.norm(omega_increment)))
                all_combinations.append((omega_increment[0], omega_increment[1], omega_increment[2]))