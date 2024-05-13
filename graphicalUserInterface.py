import matplotlib.pyplot as plt
import numpy as np
from constants import *
from MiscFunctions import quiver_data_to_segments, set_axes_equal
import matplotlib.animation as animation
import matplotlib

fps = 40
time = 25
quiver_length = 0.3 * R_E
quiver_widths = 1
thr_previous_spacecraft_positions_fade_down = 1
thr_sun_rays = 1 * 24 * 3600

fig = plt.figure()
ax2 = fig.add_subplot(122, projection='3d')
ax1 = fig.add_subplot(121, projection='3d')

# Orbital side
t = np.linspace(0, time, time * fps)
omega = 2 * np.pi / 5
R = (R_E + 3000e3)
x = R * np.cos(omega * t)
y = R * np.sin(omega * t)
z = R * np.zeros(np.shape(t))

u_E, v_E = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_E = R_E * np.cos(u_E)*np.sin(v_E)
y_E = R_E * np.sin(u_E)*np.sin(v_E)
z_E = R_E * np.cos(v_E)
ax1.plot_wireframe(x_E, y_E, z_E, color="b", label="Earth")

previous_spacecraft_position = ax1.scatter([], [], [], c="r")
current_spacecraft_position = ax1.scatter(x[0], y[0], z[0], c="k", label="Spacecraft")
#ax.set(xlim=[min(x)*1.1, max(x)*1.1], ylim=[min(y)*1.1, max(y)*1.1], zlim=[-R*1.1, R*1.1])
ax1.legend()
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.set_zlabel("Z [m]")

# Sun rays
xgrid = np.linspace(min(x)*1.1, max(x)*1.1, 5)
ygrid = np.linspace(min(y)*1.1, max(y)*1.1, 5)
zgrid = np.linspace(-R*1.1, R*1.1, 5)
Xg, Yg, Zg = np.meshgrid(xgrid, ygrid, zgrid)

# Define constant vector components for the field
u = -np.ones_like(Xg)  # Constant x-component
v = np.zeros_like(Yg) # Constant y-component
w = np.zeros_like(Zg) # Constant z-component

# Plot the sun rays and the sail normal vector
sun_rays = ax1.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5, linewidth=quiver_widths, length=quiver_length)
current_sail_normal = ax1.quiver(x[0], y[0], z[0], 1, 0, 0, color="k")
set_axes_equal(ax1)
def updateOrbit(frame):
    # for each frame, update the data stored on each artist.
    global sun_rays, current_sail_normal
    if (frame / fps <= thr_previous_spacecraft_positions_fade_down): # Start-up phase
        xd = x[:frame-1]
        yd = y[:frame-1]
        zd = z[:frame-1]
    else:
        xd = x[frame - thr_previous_spacecraft_positions_fade_down * fps:frame - 1]
        yd = y[frame - thr_previous_spacecraft_positions_fade_down * fps:frame - 1]
        zd = z[frame - thr_previous_spacecraft_positions_fade_down * fps:frame - 1]
    # Update the first point
    data_first = np.stack([[x[frame]], [y[frame]], [z[frame]]]).T
    current_spacecraft_position._offsets3d = (data_first[:, 0], data_first[:, 1], data_first[:, 2])
    current_sail_normal.remove()
    current_sail_normal = ax1.quiver(x[frame], y[frame], z[frame], 1, 0, 0, color="r", normalize=True, linewidth=quiver_widths, length=quiver_length)

    # Update previous points
    if frame > 1:
        data_previous = np.stack([xd, yd, zd]).T
        previous_spacecraft_position._offsets3d = (data_previous[:, 0], data_previous[:, 1], data_previous[:, 2])
        #shading_down = np.linspace(1, 0, len(data_previous[:, 0]))
        #previous_spacecraft_position.set_alpha(shading_down)    # TODO: does not seem to be working perfectly yet



    ax1.set_title(f"time={round(frame / fps, 1)} s")
    if (frame > fps * thr_sun_rays):
        sun_rays.remove()
        sun_rays = ax1.quiver(Xg, Yg, Zg, u, v, w, normalize=True, color="gold", alpha=0.5, linewidth=quiver_widths, length=quiver_length)
    return (previous_spacecraft_position)

ani = animation.FuncAnimation(fig=fig, func=updateOrbit, frames=fps * time, interval=1000 / fps)




plt.show()