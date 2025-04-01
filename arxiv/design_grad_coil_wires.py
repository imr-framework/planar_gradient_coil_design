# TODO:
# 1. Overlapping wires issue by adding a height of wire spacing - need to check this by simulating
# 2. Save the coordinates of the optimized coil
# 3. Test the multi-objective optimization
# 4. Run the mid-optimization 
# --------------------------------------------------------------
# Import necessary libraries

from amri_planar_gradient_coil import PlanarGradientCoil
from utilities import *
import magpylib as magpy
import numpy as np
from colorama import Fore, Style
import time 

# -----------------------------------------------------------------------------
# Generate target magnetization and linearity
grad_max = 27 * 1e-3 # T/m --> 0.1 B0 * 1e-2 
grad_dir = 'x'
dsv = 31 * 1e-3 # m
res = 4 * 1e-3 # m
symmetry = False
viewing = True
dsv_sensors, pos, Bz_target = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, 
                                                   res=res, viewing=viewing, symmetry=symmetry)
linearity_percentage = 5 # 5% linearity

# -----------------------------------------------------------------------------
# Generate the gradient coil
radius =  0.5 * 6 * 0.0254 # m
current = 100 # A
res_design = 4 * 1e-3 # m
mesh = 5 # 2 * int(radius / res_design) # number of points in the mesh 
thickness = 1.3 * 1e-3 # m
spacing = 2 * thickness # m
viewing  = True
heights = [-36 * 1e-3, 36 * 1e-3]  # m
symmetry = False
num_psi = mesh ** 2
magnetization = 24 * 1e2 # A/m - need to calibrate this value - 3.7 Am-1 for AWG 16
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil(grad_dir = grad_dir, radius=radius, heights = heights, magnetization=magnetization, mesh=mesh, 
                                         thickness=thickness, spacing=spacing, symmetry = symmetry)

# -----------------------------------------------------------------------------
# Load the coil with the optimized stream function
psi_init = get_stream_function(grad_dir='x', x =tenacity_grad_coil.x, 
                                    y=tenacity_grad_coil.y, viewing = False).T
num_levels = 20 # Need to make this more adaptive to the value of psi
loop_tolerance = 2e-3 #m
fname_psi_save = 'psi_mesh_' + str(mesh) + '.npy'
psi = np.load(fname_psi_save)
biplanar_coil_pattern_wires, _ = tenacity_grad_coil.get_wire_patterns(psi, levels = num_levels, stream_function = psi_init, loop_tolerance=loop_tolerance,
                                        x = tenacity_grad_coil.x, y = tenacity_grad_coil.y, heights = tenacity_grad_coil.heights, current = 1, viewing = True)

# -----------------------------------------------------------------------------
# Plot the optimized coil
tenacity_grad_coil.view(current_dir=True)
fname_csv_loops = 'optimized_coil_loops.csv'
loop_tolerance = loop_tolerance * 1e3 # mm
# tenacity_grad_coil.save_loops(fname_csv_file=fname_csv_loops, loop_tolerance = 1, viewing = False)