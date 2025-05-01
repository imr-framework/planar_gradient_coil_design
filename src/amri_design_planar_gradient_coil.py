#-----------------------------------------------------------------------------
# Import libraries
from amri_planar_gradient_coil import PlanarGradientCoil

from utilities import *
import magpylib as magpy
import numpy as np
from colorama import Fore, Style
from optimize_gradient_design import planar_gradient_problem
# import scipy.optimize as opt
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import time 
from datetime import datetime
import pickle

# -----------------------------------------------------------------------------
# Filenmaes to save the results
fname_psi_save = 'psi_mesh_5.npy'
fname_csv_loops = 'optimized_coil_loops.csv'


# -----------------------------------------------------------------------------
# Generate target magnetization and linearity
grad_max = 27 * 1e-3 # T/m --> 0.1 B0 * 1e-2 
grad_dir = 'x'
dsv = 41 * 1e-3 # m
res = 4 * 1e-3 # m
symmetry = False
viewing = True
dsv_sensors, pos, Bz_target = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, 
                                                   res=res, viewing=viewing, symmetry=symmetry)
linearity_percentage = 5 # 5% linearity

# -----------------------------------------------------------------------------
# Generate the gradient coil

radius =  0.5 * 6.1 * 0.0254 # m
current = 100 # A
res_design = 4 * 1e-3 # m   
mesh = 5     # 2 * int(radius / res_design) # number of points in the mesh 
thickness = 1.3 * 1e-3 # m
spacing = 2 * thickness # 
viewing  = True
heights = [-36 * 1e-3, 36 * 1e-3]  # m  
num_psi = mesh ** 2
magnetization = 24 * 1e2 # 24 * 1e2 A/m - need to calibrate this value - 3.7 Am-1 for AWG 16
num_levels = 14    # Need to make this more adaptive to the value of psi
shape = 'square' # 'circle' or 'square'
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil(grad_dir = grad_dir, radius=radius, current = current, heights = heights, magnetization=magnetization, mesh=mesh, 
                                         thickness=thickness, spacing=spacing, symmetry = symmetry, levels=num_levels, shape = shape)

# -----------------------------------------------------------------------------
# Setup optimization problem including the preconditioner
psi_init = get_stream_function(grad_dir=grad_dir, x =tenacity_grad_coil.x, 
                                    y=tenacity_grad_coil.y, viewing = True, shape=shape).T
# psi_0 = np.concatenate((psi_init.flatten(), psi_init.flatten()), axis=0)
# pop = Population.new("vars", psi_0)

num_objectives = 4 # number of objectives 1 or 5 for now
num_constraints = 1 # multi-objective optimization with multiple constraints
order = 2 # lp -> high fidelity for now
iterations = 50 # number of iterations
# opt_tool = 'pymoo-ga' # 'scipy-opt' or 'pymoo-ga'
tenacity_gradient_optimize = planar_gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos, 
                                                     target_field= Bz_target, psi = psi_init, n_constr=num_constraints, order=order,n_obj=num_objectives, 
                                                     linearity_percentage=linearity_percentage)
        

if tenacity_grad_coil.shape == 'square':
    pop_size = 4 * (tenacity_grad_coil.num_psi_weights * 2)
    # pop_size = 200

elif tenacity_grad_coil.shape == 'circle':
    pop_size = 4 * (tenacity_grad_coil.num_psi_weights * 2)

print(Fore.YELLOW + 'The population size is: ' + str(pop_size) + Style.RESET_ALL)
algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
tic = time.time()
res_psi = minimize(tenacity_gradient_optimize,
                algorithm, ('n_gen', iterations),
                verbose=True)
toc = time.time()
print(Fore.YELLOW + 'Wire pattern search ends ...')
print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
vars = res_psi.X

    
if vars is None:
    print(Fore.RED + 'No solution found')
    exit()
# -----------------------------------------------------------------------------
if type(vars) is np.ndarray: # has a list of solutions that we need to choose from
    # vars = vars[0] # for now choose the first solution
    print(Fore.YELLOW + 'Identifying the best solution out of multiple solutions: ' + str(vars.shape[0]) + Style.RESET_ALL)
    F_min = np.min(res_psi.F[0])
    vars_chosen = vars[0]
    for sol in range(vars.shape[0]):
        F = np.min(res_psi.F[sol])
        if F < F_min:
            F_min = F
            vars_chosen = vars[sol]

    vars = vars_chosen
print(Fore.YELLOW + 'The best solution has been chosen!' + Style.RESET_ALL)
# -----------------------------------------------------------------------------
# Display the resulting gradient coil induced magnetic field - need to handle more than one solution scenario
tenacity_grad_coil.load(vars, tenacity_grad_coil.num_psi_weights, psi_init, viewing = False)
Bz_grad_coil_mags = get_magnetic_field(tenacity_grad_coil.biplanar_coil_pattern, dsv_sensors, axis = 2)
# print(Fore.YELLOW + 'Visualizing the magnetic field ' + Style.RESET_ALL)
# display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad_coil_mags, title = 'Gradient Coil Field - magnets')

#------------------------------------------------------------------------------
# Choose the best solution and visualize the psi and resulting magnetic field 
print(Fore.YELLOW + 'Computing the gradient coils now ...' + Style.RESET_ALL)
planar_coil_pattern, spiral_plate_lower, spiral_plate_upper, wire_smoothness = tenacity_grad_coil.get_wire_patterns(vars, levels = num_levels, stream_function = psi_init, 
                                        x = tenacity_grad_coil.x, y = tenacity_grad_coil.y, heights = tenacity_grad_coil.heights, current = 1, viewing = False)
psi = np.array([vars[f"x{child:02}"] for child in range(0, num_psi)]) # all children should have same magnet positions to begin with
date_string = datetime.now().strftime("%Y-%m-%d")
fname_psi_save =  'psi_mesh_' + str(mesh) + '_date_' + date_string + '.npy'
with open(fname_psi_save, 'wb') as f:
    pickle.dump(vars, f)
# -----------------------------------------------------------------------------
# Plot the results - psi, contours, wire patterns, and resulting gradient field
# tenacity_grad_coil.view()
# magpy.show(tenacity_grad_coil.biplanar_coil_pattern_wires, dsv_sensors)
# Bz_grad_coil_wires = get_magnetic_field(tenacity_grad_coil.biplanar_coil_pattern_wires, dsv_sensors, axis = 2)
# display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad_coil_wires, title = 'Gradient Coil Field - wires')

# -----------------------------------------------------------------------------
# Characterize the gradient coil


# -----------------------------------------------------------------------------
# Save the gradient coil cooordinates - one half is enough - x,y 
coil_coords = np.array([spiral_plate_lower[:, 0], spiral_plate_lower[:, 1]]).T
np.savetxt(fname_csv_loops, coil_coords, delimiter=',', header='x,y', comments='saving only one half')


# -----------------------------------------------------------------------------
# Load the CSV file and plot the gradient coil - very time consuming, do not run this unless needed
# coil_coords = np.loadtxt(fname_csv_loops, delimiter=',', skiprows=1)
# x = coil_coords[:, 0]
# y = coil_coords[:, 1]
# # Plot the gradient coil
# plt.figure(figsize=(10, 10))
# plt.plot(x, y, 'r-', label='Gradient Coil')
# plt.title('Gradient Coil')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.axis('equal')
# plt.grid()
# plt.legend()
# plt.show()
# -----------------------------------------------------------------------------


