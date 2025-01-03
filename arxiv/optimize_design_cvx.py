# Define the objective function for the optimization

# Define the constraints for the optimization

# Run the optimization



# target_B0_2_shim_locations
# The problem can be defined as:
# Find the combination of children in the magnet collections that when used provide the most homogeneous magnetic field with a given tolerance
# 
import magpylib as magpy
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from utils import *
import scipy.optimize as opt
import cvxpy as cp
from colorama import Fore, Style

class gradient_problem(ElementwiseProblem):
   
    def __init__(self, grad_coil, sensors, pos, target_field, order=2, num_levels = 10, 
                 alpha=[0.5], beta=0.5, B_tol = 5, linearity_percentage = 5, n_obj=1, n_constr=0, **kwargs):
        self.grad_coil = grad_coil
        self.sensors = sensors
        self.target_field = target_field
        self.target_field_norm, self.target_field_range = self.normalize_field(self.target_field, diameter = self.grad_coil.radius * 2)
        self.pos = pos
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.B_tol = B_tol
        self.num_psi_weights = self.grad_coil.psi_weights * 2 # for two plates
        self.num_levels = num_levels * 2 # for two plates
        self.x = np.random.rand(self.num_psi_weights)
        self.n_obj = n_obj
        self.num_constr = n_constr
        self.linearity_percentage = linearity_percentage
        
        
    def _evaluate(self, x, out, *args, **kwargs):
        self.biplanar_coil_pattern, self.coil_resistance, self.coil_current, max_ji, self.psi_smoothness, self.wire_smoothness = self.grad_coil.load(vars = x, opt = 'cvx', 
                                                                                    num_psi_weights = self.num_psi_weights, num_levels = self.num_levels, 
                                                                                    pos = self.pos, sensors = self.sensors, viewing = False)
        if (len(self.biplanar_coil_pattern.children[0])==0) and (len(self.biplanar_coil_pattern.children[1])==0): # for the case when no wires are present
           self.grad_coil_field = 0
        else:
            self.grad_coil_field= get_magnetic_field(self.biplanar_coil_pattern, self.sensors, axis = 2)
        
        # Minimize range of B and maximize mean
        if self.n_constr == 0:
            f = cost_fn(B_grad = self.grad_coil_field, B_target = self.target_field, 
                        coil_resistance=self.coil_resistance, coil_current=self.coil_current, psi_smoothness=self.psi_smoothness, wire_smoothness=self.wire_smoothness,
                        p=self.order, alpha=self.alpha, beta=self.beta, weight=1e3, case='target_field')
        
            out["F"] = [f]
            
        if self.n_constr > 0:
            f = cost_fn(B_grad = self.grad_coil_field, B_target = self.target_field, 
                        coil_resistance=self.coil_resistance, coil_current=self.coil_current, psi_smoothness=self.psi_smoothness, wire_smoothness=self.wire_smoothness,
                        p=self.order, alpha=self.alpha, beta=self.beta, weight=1e3, case='target_field')
        
            # print(Fore.YELLOW + 'Constraint violation value: ' + str(np.round(f[0]- self.linearity_percentage, decimals=2)) + Style.RESET_ALL)
            out["F"] = [f[1], f[2], f[3]]
            out["G"] = [f[0]- self.linearity_percentage]
        
        # print(Fore.YELLOW + 'Constraint violation value: ' + str(np.round(f[0]- self.linearity_percentage, decimals=2)) + Style.RESET_ALL)
        # out["G"] = [g2]
    
    def objective(self, x):
        self.biplanar_coil_pattern = self.grad_coil.load_cvx(vars = x, opt = 'cvx', num_psi_weights = self.num_psi_weights, num_levels = self.num_levels, 
                                                                                    pos = self.pos, sensors = self.sensors, viewing = False)
        if (len(self.biplanar_coil_pattern.children[0])==0) and (len(self.biplanar_coil_pattern.children[1])==0): # for the case when no wires are present
           self.grad_coil_field = 0
        else:
            self.grad_coil_field= get_magnetic_field(self.biplanar_coil_pattern, self.sensors, axis = 2)
        
        self.grad_coil_field_norm, self.grad_coil_field_range = self.normalize_field(self.grad_coil_field, diameter = self.grad_coil.radius * 2)
        
        f1 = 100 * np.linalg.norm(self.grad_coil_field_norm - self.target_field_norm, ord=np.inf) / (np.linalg.norm(self.target_field_norm, ord=np.inf)) # minimizing peak field
        f2 = np.sqrt((self.grad_coil_field_range - self.target_field_range)**2)
        f = f1 + f2 
        # f = cost_fn(B_grad = self.grad_coil_field, B_target = self.target_field, 
        #                 coil_resistance=self.coil_resistance, coil_current=self.coil_current, psi_smoothness=self.psi_smoothness, wire_smoothness=self.wire_smoothness,
        #                 p=self.order, alpha=self.alpha, beta=self.beta, weight=1e3, case='target_field')
        
        return f

    def normalize_field(self, B, diameter):
        
        B_norm = 100 * (B / np.linalg.norm(B, ord=np.inf))
        B_range = (np.max(B_norm) - np.min(B_norm)) / diameter
        
        return B_norm, B_range

    
            