# TODO:
# 1. Overlapping wires issue by adding a height of wire spacing - need to check this by simulating
# 2. Save the coordinates of the optimized coil
# 3. Test the multi-objective optimization
# 4. Run the mid-optimization 
# --------------------------------------------------------------
# Import necessary libraries

from rectangular_gradient_coil import PlanarGradientCoil_rectangle
from utils import create_magpy_sensors
import magpylib as magpy
import numpy as np
from colorama import Fore, Style
from optimize_design_rectangle import gradient_problem
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import cvxpy as cp
import time 

# --------------------------------------------------------------
# Setup geometry for a planar gradient coil
grad_dir = 'x'
radius =  0.5 * 6 * 0.0254 # m
current = 100 # A
res_design = 2 * 1e-3 # m
mesh = 5 # 2 * int(radius / res_design) # number of points in the mesh 
target_field = 1 # get_field() # T
wire_thickness = 1.3 * 1e-3 # m
wire_spacing = 2 * wire_thickness # m
viewing  = True
heights = [-36 * 1e-3, 36 * 1e-3]  # m
symmetry = False
psi_weights = mesh ** 2
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil_rectangle(grad_dir = grad_dir, radius=radius, heights = heights, current=current, mesh=mesh, target_field=target_field, 
                            wire_thickness=wire_thickness, wire_spacing=wire_spacing, symmetry = symmetry)
                            
# --------------------------------------------------------------
# Set up the target magnetic field
grad_max = 27 * 1e-3 # T/m --> 0.1 B0 * 1e-2 
dsv = 31 * 1e-3 # m
res = 4 * 1e-3 # m
viewing = True
dsv_sensors, pos, Bz_target = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, res=res, viewing=viewing, symmetry=symmetry)
linearity_percentage = 5 # 5% linearity
#---------------------------------------------------------------
# Optimize coil design 
# Set up the optimization algorithm
num_objectives = 3 # number of objectives 1 or 5 for now
num_constraints = 1 # multi-objective optimization with multiple constraints
num_levels = 10 # No of contour levels to extract from the stream function per plate
order = 2 # lp -> high fidelity for now
iterations = 200 # number of iterations

if num_constraints == 0:
    num_regularizers = 5
    alpha = np.zeros(num_regularizers)
    alpha[0] = 1 # minimize max Bz difference
    alpha[1] = 1 # minimize random jumps in wires - improve smoothness of wires/contols levels indirectly
    alpha[2] = 1 # minimize random jumps in stream function - improve smoothness of the stream function
    alpha[3] = 1
    alpha[4] = 0 # minimize current
    alpha[5] = 0 # ensure smoothness or wire patterns and avoid overlaps
else:   
    alpha = np.zeros(1)


tenacity_grad_coil_optimize = gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos,
                                               target_field=Bz_target,
                                               order=order, alpha=alpha, beta=0.5, B_tol = 5, num_levels=num_levels, linearity_percentage=linearity_percentage,
                                               n_obj=num_objectives, n_constr=num_constraints)
opt_library = 'pymoo' # 'pymoo' or 'cvxpy'
if opt_library == 'pymoo':
    #---------------------------------------------------------------
    # Use pymoo to optimize the coil design
    pop_size = 4 * (tenacity_grad_coil_optimize.num_psi_weights + 1)
    print(Fore.YELLOW + 'The population size is: ' + str(pop_size) + Style.RESET_ALL)
    algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    tic = time.time()
    res_psi = minimize(tenacity_grad_coil_optimize,
                    algorithm, ('n_gen', iterations),
                    verbose=True)
    toc = time.time()
    print(Fore.YELLOW + 'Wire pattern search ends ...')
    print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
    psi = res_psi.X

#---------------------------------------------------------------
# Get the optimized gradient locations and visualize the coil and field
if psi is not None and len(psi) > 0:
    for i in range(len(psi)):
        tenacity_grad_coil.load(psi[i], tenacity_grad_coil_optimize.num_psi_weights, tenacity_grad_coil_optimize.num_levels, 
                        tenacity_grad_coil_optimize.pos, tenacity_grad_coil_optimize.sensors, viewing = False)
        tenacity_grad_coil.view(sensors = dsv_sensors, pos = pos, symmetry=True)
        non_linearity_pc = 100 * np.linalg.norm(tenacity_grad_coil.Bz_grad - Bz_target, ord=np.inf) / (np.linalg.norm(Bz_target, ord=np.inf))
        print(Fore.YELLOW + 'Non-linearity percentage for solution: ' + str(i) +':' + str(np.round(non_linearity_pc, decimals=2)) + Style.RESET_ALL)
else:
    print(Fore.RED + 'Optimization did not produce a valid result.' + Style.RESET_ALL)

#---------------------------------------------------------------
# Compute coil performance metrics

# --------------------------------------------------------------
# Out of the optimized gradient coil patterns, choose the one to save

#---------------------------------------------------------------
# Save the results to a csv file with coordinates for the positive and negative wires


# print(Fore.YELLOW + 'Saving the optimized wire pattern ...')
# fname = 'tenacity_grad_coil_' + tenacity_grad_coil.grad_dir + '.csv'
# tenacity_grad_coil.save(fname=fname)
# tenacity_grad_coil.save_loops(fname_csv_file=fname,loop_tolerance=5)
# tenacity_grad_coil.combine_loops_current_direction(loop_tolerance = 5, fname_csv_file = fname)
# print(Fore.YELLOW + 'Optimized wire pattern saved to: ' + fname + Style.RESET_ALL)




#---------------------------------------------------------------
# Filter the wire pattern to remove overlapping wires by adding a height of wire spacing and store the positive and negative wires in two files
# print(Fore.YELLOW + 'Filtering the wire pattern ...')
# tenacity_grad_coil.filter_wires_and_save(fname=fname)

#---------------------------------------------------------------
# Check the new filtered wires for Bz field achieved
# tenacity_grad_coil.load_from_csv(dirname = None, fname_pattern = ['negative_wires', 'positive_wires'])
# tenacity_grad_coil.view(sensors = dsv_sensors, pos = pos, symmetry=True)
# non_linearity_pc = 100 * np.linalg.norm(tenacity_grad_coil.Bz_grad - Bz_target, ord=np.inf) / (np.linalg.norm(Bz_target, ord=np.inf))
# print(Fore.YELLOW + 'Non-linearity percentage: ' + str(np.round(non_linearity_pc, decimals=2)) + Style.RESET_ALL)
# # Check non-linearity 

