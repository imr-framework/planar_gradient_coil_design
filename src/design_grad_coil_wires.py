#-----------------------------------------------------------------------------
# Import libraries
from amri_planar_gradient_coil import PlanarGradientCoil
from utilities import *
import magpylib as magpy
import numpy as np
from colorama import Fore, Style
from optimize_gradient_design import planar_gradient_problem
import scipy.optimize as opt
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import time 
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# Filenmaes to save the results
mesh = 5
date_string = datetime.now().strftime("%Y-%m-%d")
fname_psi_save =  'psi_mesh_5_date_2025-04-25.npy' # Make sure this is the same as the desired filename
fname_csv_loops = 'optimized_coil_loops.csv'
fname_gcode = None # 'gradient_coil_xdir_multi_depths.nc'
fname_stl = None # 'gradient_coil_xdir_multi_depths.stl'
fname_excel = 'gradient_coil_xdir_multi_depths.xlsx'
wire_depth = 1.5 # mm
viewing = True
depth_array = np.linspace(0.01, 1.2, 120)
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

# radius =  0.5 * 6.1 * 0.0254 # m
# radius = 148 * 0.5 * 1e-3 # m
radius = 0.06
current = 100 # A
res_design = 4 * 1e-3 # m   
mesh = 5      # 2 * int(radius / res_design) # number of points in the mesh 
thickness = 1.3 * 1e-3 # m
spacing = 2 * thickness # m4dhhrhjfffcfffvvfc ccccv
viewing  = True
heights = [-36 * 1e-3, 36 * 1e-3]  # m  
num_psi = mesh ** 2
magnetization = 24 * 1e2 # 24 * 1e2 A/m - need to calibrate this value - 3.7 Am-1 for AWG 16
num_levels = 14   # Need to make this more adaptive to the value of psi
shape = 'square' # 'circle' or 'square'
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil(grad_dir = grad_dir, radius=radius, current = current, heights = heights, magnetization=magnetization, mesh=mesh, 
                                         thickness=thickness, spacing=spacing, symmetry = symmetry, levels=num_levels, shape = shape)

# -----------------------------------------------------------------------------
# Setup optimization problem including the preconditioner
psi_init = get_stream_function(grad_dir=grad_dir, x =tenacity_grad_coil.x, 
                                    y=tenacity_grad_coil.y, viewing = True, shape=shape).T

#------------------------------------------------------------------------------
# Choose the best solution and visualize the psi and resulting magnetic field 
with open(fname_psi_save, 'rb') as f:
    vars = pickle.load(f)
planar_coil_pattern, spiral_plate_lower, spiral_plate_upper, wire_smoothness = tenacity_grad_coil.get_wire_patterns(vars, levels = num_levels, stream_function = psi_init, 
                                        x = tenacity_grad_coil.x, y = tenacity_grad_coil.y, heights = tenacity_grad_coil.heights, current = 1, viewing = True)

# -----------------------------------------------------------------------------
# Plot the results - psi, contours, wire patterns, and resulting gradient field
# tenacity_grad_coil.view()
magpy.show(tenacity_grad_coil.biplanar_coil_pattern_wires, dsv_sensors)

# -----------------------------------------------------------------------------
# Load the CSV file and plot the gradient coil - validation step
x = spiral_plate_lower[:, 0]
y = spiral_plate_upper[:, 1]
# Plot the gradient coil
plt.figure(figsize=(10, 10))
plt.plot(x, y, 'r-', label='Gradient Coil')
plt.title('Gradient Coil - Spiral')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()
# -----------------------------------------------------------------------------
# write G-code to file
# -----------------------------------------------------------------------------
# G-code to be written to a file
if grad_dir == 'x':
    # this gradient has four components stored in a dictionary
    tol_zero = 2 #mm

    # 3. The positive gradient coil (x > 0)
    x = spiral_plate_lower[:, 0] 
    y = spiral_plate_lower[:, 1]
  
    x_pos = x[np.where(x > 0)] + tol_zero
    y_pos = y[np.where(x > 0)]
    
    x_pos = x_pos[10:] # remove the last two points
    y_pos = y_pos[10:] # remove the last two points
    
    print(np.min(x_pos), np.max(x_pos))

    # 4. The negative gradient coil (x < 0)
    x_neg = x[np.where(x < 0)] - tol_zero
    y_neg = y[np.where(x < 0)]
    
    x_neg = x_neg[10:] # remove the last two points
    y_neg = y_neg[10:] # remove the last two points
    
    plt.plot(x_pos, y_pos, 'r-', label='Positive Gradient Coil')
    plt.plot(x_neg, y_neg, 'b-', label='Negative Gradient Coil')
        
    # # 1. A circle of radius "radius" in the x-y plane
    # x1, y1 = circle(radius, num_points=360)  # convert to mm
    # x1 = x1 * 1000 # convert to mm
    # y1 = y1 * 1000 # convert to mm
    # plt.plot(x1, y1, 'k-', label='Circle')
    
    # 1. A square of side two times the radius in the x-y plane
    radius = (np.max(tenacity_grad_coil.x)*1e3 + 3) * 1e-3
    x1, y1 = square(radius, num_points=360)  # convert to mm
    x1 = x1 * 1000 # convert to mm
    y1 = y1 * 1000 # convert to mm
    plt.plot(x1, y1, 'k-', label='Square')

    # 2. A diameter across the circle at x = 0
    x2, y2 = diameter(radius, num_points=360, axis='x')  # convert to mm
    x2 = x2 * 1000 # convert to mm
    y2 = y2 * 1000 # convert to mm
    plt.plot(x2, y2, 'k-', label='Diameter')
    plt.title('Gradient Coil - Print view')
    
    
    # Write the  G-code to a file based on the dictionary
    grad_dict = {
        'positive_gradient_coil': (x_pos, y_pos),
        'negative_gradient_coil': (x_neg, y_neg),
        'diameter': (x2, y2),
        'square': (x1, y1),

    }
elif grad_dir == 'z':
    write_gcode(x, y, filename=fname_gcode, depths = depth_array)
if viewing:
    plt.show()
else:
    plt.savefig('gradient_coil.png')
    plt.close()
if fname_gcode is not None:
    write_gcode(filename=fname_gcode, grad=grad_dict, depths=depth_array)

if fname_stl is not None:
    # Write the grad_dict to an STL file
    for key, value in grad_dict.items():
        x, y = value
        z = np.ones_like(x) * wire_depth
        fname_stl_ind   = fname_stl.replace('.stl', f'_{key}.stl') 
        write_stl(fname_stl_ind, x, y, z)
        
if fname_excel is not None:
    # Write the grad_dict to an Excel file
    for key, value in grad_dict.items():
        x, y = value
        z = np.ones_like(x) * wire_depth
        fname_excel_ind   = fname_excel.replace('.xlsx', f'_{key}.xlsx') 
        write_excel(fname_excel_ind, x, y, z)
        