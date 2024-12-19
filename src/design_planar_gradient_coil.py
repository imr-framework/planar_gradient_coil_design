# TODO:
# 1. Overlapping wires issue by adding a height of wire spacing - need to check this by simulating
# 2. Save the coordinates of the optimized coil
# 3. Test the multi-objective optimization
# 4. Run the mid-optimization 
# --------------------------------------------------------------
# Import necessary libraries

from planar_gradient_coil import PlanarGradientCoil
from utils import create_magpy_sensors
import magpylib as magpy
import numpy as np
from colorama import Fore, Style
from optimize_design import gradient_problem
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
current = 10 # A
mesh = 50 # number of points in the mesh 
target_field = 1 # get_field() # T
wire_thickness = 2.6 * 1e-3 # m
wire_spacing = 2 * wire_thickness # m
viewing  = True
heights = [-40 * 1e-3, 40 * 1e-3]  # m
symmetry = False
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil(grad_dir = grad_dir, radius=radius, heights = heights, current=current, mesh=mesh, target_field=target_field, 
                            wire_thickness=wire_thickness, wire_spacing=wire_spacing, symmetry = symmetry)
                            
# --------------------------------------------------------------
# Set up the target magnetic field

grad_max = 27 * 1e-3 # T/m --> 0.1 B0 * 1e-2 
dsv = 31 * 1e-3 # m
res = 2 * 1e-3 # m
viewing = True
dsv_sensors, pos, Bz_target = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, res=res, viewing=viewing, symmetry=symmetry)

#---------------------------------------------------------------
# Optimize coil design 
# Set up the optimization algorithm
num_objectives = 1 # number of objectives 1 or 5 for now
num_constraints = 0 # we will implement this later

if num_objectives == 1:
    num_regularizers = 5
    alpha = np.zeros(num_regularizers)
    alpha[0] = 1 # minimize point-wise Bz difference
    alpha[1] = 0.1 # minimize max Bz difference 
    alpha[2] = 0.1 # minimize  resistance
    alpha[3] = 0.1 # minimize current
    alpha[4] = 0.5 # ensure smoothness or wire patterns and avoid overlaps
order = 2 # lp -> high fidelity for now
iterations = 1 # number of iterations
tenacity_grad_coil_optimize = gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos,
                                               num_triangles_total=tenacity_grad_coil.num_triangles_total, target_field=Bz_target,
                                               order=order, alpha=alpha, beta=0.5, B_tol = 5, n_obj=num_objectives, n_constr=num_constraints)

opt_library = 'pymoo' # 'pymoo' or 'cvxpy'
if opt_library == 'pymoo':
    #---------------------------------------------------------------
    # Use pymoo to optimize the coil design
    pop_size = 2 * tenacity_grad_coil.nodes.shape[0]
    algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    tic = time.time()
    res_psi = minimize(tenacity_grad_coil_optimize,
                    algorithm, ('n_gen', iterations),
                    verbose=True)
    toc = time.time()
    print(Fore.YELLOW + 'Wire pattern search ends ...')
    print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
    psi = res_psi.X
    
elif opt_library == 'cvxpy':
    tenacity_grad_coil_optimize.evaluate_cvx()
    
#---------------------------------------------------------------
# Get the optimized gradient locations and visualize the coil and field

tenacity_grad_coil.load(psi, len(psi), pos, dsv_sensors, viewing = False)
tenacity_grad_coil.view(sensors = dsv_sensors, pos = pos, symmetry=True)

#---------------------------------------------------------------
# Compute coil performance metrics

#---------------------------------------------------------------
# Save the results to a csv file with coordinates for the positive and negative wires
print(Fore.YELLOW + 'Saving the optimized wire pattern ...')
fname = 'tenacity_grad_coil_' + tenacity_grad_coil.grad_dir + '.csv'
tenacity_grad_coil.save(fname=fname)

#---------------------------------------------------------------
# Filter the wire pattern to remove overlapping wires by adding a height of wire spacing and store the positive and negative wires in two files
print(Fore.YELLOW + 'Filtering the wire pattern ...')
tenacity_grad_coil.filter_wires_and_save(fname=fname)

#---------------------------------------------------------------
# Check the new filtered wires for Bz field achieved
# tenacity_grad_coil.load_from_csv(fnames = [fname_positive, fname_negative])


