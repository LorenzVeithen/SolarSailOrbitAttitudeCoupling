from MiscFunctions import compute_panel_geometrical_properties
import numpy as np

panel_coords = np.array([[0, 0, 0],
                        [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0]])

print(compute_panel_geometrical_properties(panel_coords))