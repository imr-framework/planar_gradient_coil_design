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
radius =0.5 * 6 * 0.0254 # m
current = 10 # A
mesh = 40 # number of points in the mesh 
target_field = 1 # get_field() # T
wire_thickness = 1.3 * 1e-3 # m
wire_spacing = 0.5 * wire_thickness # m
viewing  = True
heights = [-40 * 1e-3, 40 * 1e-3]  # m
symmetry = True
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil(grad_dir = grad_dir, radius=radius, heights = heights, current=current, mesh=mesh, target_field=target_field, 
                            wire_thickness=wire_thickness, wire_spacing=wire_spacing, symmetry = symmetry)
                            
# --------------------------------------------------------------
# Set up the target magnetic field

grad_max = 27 * 1e-3 * 1e-2 # T/m --> 0.1 B0
dsv = 31 * 1e-3 # m
res = 2 * 1e-3 # m
viewing = True
dsv_sensors, pos, Bz_target = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, res=res, viewing=viewing, symmetry=True)

#---------------------------------------------------------------
# Optimize coil design 
tenacity_grad_coil_optimize = gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos,
                                               num_triangles_total=tenacity_grad_coil.num_triangles_total, target_field=Bz_target)
opt_library = 'pymoo' # 'pymoo' or 'cvxpy'
if opt_library == 'pymoo':
    #---------------------------------------------------------------
    # Use pymoo to optimize the coil design
    pop_size = 2 * tenacity_grad_coil.triangles.shape[0]
    algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    tic = time.time()
    res_psi = minimize(tenacity_grad_coil_optimize,
                    algorithm, ('n_gen', 10),
                    verbose=True)
    toc = time.time()
    print(Fore.YELLOW + 'Wire pattern search ends ...')
    print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
    psi = res_psi.X
    
elif opt_library == 'cvxpy':
    tenacity_grad_coil_optimize.evaluate_cvx()
    
#---------------------------------------------------------------
# Get the optimized gradient locations and visualize the coil and field
# visualize_gradient_coil(biplanar_coil_pattern) - check for whole field 

tenacity_grad_coil.load(psi, len(psi), pos, dsv_sensors, viewing = True)
dsv_sensors_full, pos_full, _ = create_magpy_sensors(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, res=res, viewing=False, symmetry=False)
tenacity_grad_coil.load(psi, len(psi), pos_full, dsv_sensors_full, viewing = True)
print(Fore.GREEN + 'Optimized coil pattern loaded ...' + Style.RESET_ALL)

#---------------------------------------------------------------
# Compute coil performance metrics



# Plot the results

# Save the results to an STL file



