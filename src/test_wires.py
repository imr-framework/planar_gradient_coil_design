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
current = 1 # A
res_design = 2 * 1e-3 # m
mesh = 5 # 2 * int(radius / res_design) # number of points in the mesh 
target_field = 1 # get_field() # T
wire_thickness = 1.3 * 1e-3 # m
wire_spacing = 2 * wire_thickness # m
viewing  = True
heights = [-40 * 1e-3, 40 * 1e-3]  # m
symmetry = False
psi_weights = mesh ** 2
# Make an instance of the planar gradient coil class
tenacity_grad_coil = PlanarGradientCoil_rectangle(grad_dir = grad_dir, radius=radius, heights = heights, current=current, mesh=mesh, target_field=target_field, 
                            wire_thickness=wire_thickness, wire_spacing=wire_spacing, symmetry = symmetry)
fname = 'tenacity_grad_coil_x.csv'
tenacity_grad_coil.save_loops(fname=fname)