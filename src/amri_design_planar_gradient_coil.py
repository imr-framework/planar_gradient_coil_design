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
# Setup optimization problem including the preconditioner
psi_init = get_stream_function(grad_dir='x', x =tenacity_grad_coil.x, 
                                    y=tenacity_grad_coil.y, viewing = False).T
psi_0 = np.concatenate((psi_init.flatten(), psi_init.flatten()), axis=0)
# pop = Population.new("vars", psi_0)

num_objectives = 1 # number of objectives 1 or 5 for now
num_constraints = 1 # multi-objective optimization with multiple constraints
order = 2 # lp -> high fidelity for now
iterations = 120 # number of iterations
opt_tool = 'pymoo-ga' # 'scipy-opt' or 'pymoo-ga'
tenacity_gradient_optimize = planar_gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos, 
                                                     target_field= Bz_target, psi = psi_init, n_constr=num_constraints, order=order,n_obj=num_objectives)
    
# Evaluator().eval(tenacity_gradient_optimize, pop)
# -----------------------------------------------------------------------------
# Solve the optimization problem
if opt_tool == 'scipy-opt':
    tenacity_gradient_optimize = planar_gradient_problem(grad_coil=tenacity_grad_coil, sensors=dsv_sensors, pos=pos, target_field= Bz_target, psi = psi_0)
    result = opt.minimize(tenacity_gradient_optimize.objective, psi_0, method="L-BFGS-B", 
                        bounds=[(-1, 1)] * psi_0.shape[0], options={'disp': True, 'maxiter': 100})

elif opt_tool == 'pymoo-ga':
    pop_size = 4 * (tenacity_grad_coil.psi_weights * 2)
    
    print(Fore.YELLOW + 'The population size is: ' + str(pop_size) + Style.RESET_ALL)
    algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    # algorithm = NSGA2(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    tic = time.time()
    res_psi = minimize(tenacity_gradient_optimize,
                    algorithm, ('n_gen', iterations),
                    verbose=True)
    toc = time.time()
    print(Fore.YELLOW + 'Wire pattern search ends ...')
    print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
    vars = res_psi.X
    print(type(vars))
# -----------------------------------------------------------------------------
# Display the resulting gradient coil induced magnetic field - need to handle more than one solution scenario
tenacity_grad_coil.load(vars, tenacity_grad_coil.psi_weights, psi_init, viewing = True)
Bz_grad_coil_mags = get_magnetic_field(tenacity_grad_coil.biplanar_coil_pattern, dsv_sensors, axis = 2)
display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad_coil_mags, title = 'Gradient Coil Field - magnets')

#------------------------------------------------------------------------------
# Choose the best solution and visualize the psi and resulting magnetic field 
num_levels = 15 # Need to make this more adaptive to the value of psi
psi = np.array([vars[f"x{child:02}"] for child in range(0, 2 * num_psi)]) # all children should have same magnet positions to begin with
date_string = datetime.now().strftime("%Y-%m-%d")
fname_psi_save =  'psi_mesh_' + str(mesh) + '_date_' + date_string + '.npy'
np.save(fname_psi_save, psi)
binplanar_coil_pattern_wires, _ = tenacity_grad_coil.get_wire_patterns(psi, levels = num_levels, stream_function = psi_init, 
                                        x = tenacity_grad_coil.x, y = tenacity_grad_coil.y, heights = tenacity_grad_coil.heights, current = 1, viewing = True)
# -----------------------------------------------------------------------------
# Plot the results - psi, contours, wire patterns, and resulting gradient field
# tenacity_grad_coil.view()
Bz_grad_coil_wires = get_magnetic_field(tenacity_grad_coil.biplanar_coil_pattern_wires, dsv_sensors, axis = 2)
display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad_coil_wires, title = 'Gradient Coil Field - wires')

# -----------------------------------------------------------------------------
# Characterize the gradient coil


# -----------------------------------------------------------------------------
# Save the gradient coil



