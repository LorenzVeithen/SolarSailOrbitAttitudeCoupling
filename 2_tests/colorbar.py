import matplotlib.pyplot as plt
import numpy as np

# Create some example data
data1 = np.random.rand(10, 10)
data2 = np.random.rand(10, 10)

# Create the first figure and axis
fig1, ax1 = plt.subplots()
cax1 = ax1.imshow(data1, cmap='viridis')
fig1.colorbar(cax1)  # Optional: Individual colorbars for verification

# Create the second figure and axis
fig2, ax2 = plt.subplots()
cax2 = ax2.imshow(data2, cmap='viridis')
fig2.colorbar(cax2)  # Optional: Individual colorbars for verification

# Determine the global min and max values for the colorbar
vmin = min(data1.min(), data2.min())
vmax = max(data1.max(), data2.max())

# Create the shared colorbar in a new figure
fig3, ax3 = plt.subplots(figsize=(1, 5))
norm = plt.Normalize(vmin, vmax)
fig3.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=ax3)

# Display all figures
plt.show()