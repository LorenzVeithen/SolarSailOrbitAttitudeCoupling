import numpy as np
import matplotlib.pyplot as plt

def hole_area(theta_p, dp, rho_p, rho_t, Vp_rel, ct, cp, t_t, d_p):
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
                * np.cos(theta_p)**(0.026)) + C2
    return 0


