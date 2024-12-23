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
from pymoo.core.variable import Real, Integer, Choice
import cvxpy as cp
from colorama import Fore, Style

class gradient_problem(ElementwiseProblem):
   
    def __init__(self, grad_coil, sensors, pos, target_field, order=2, 
                 alpha=[0.5], beta=0.5, B_tol = 5, num_levels = 10, linearity_percentage = 5, n_obj=1, n_constr=0, **kwargs):
        self.grad_coil = grad_coil
        self.sensors = sensors
        self.target_field = target_field
        self.pos = pos
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.B_tol = B_tol
        self.num_psi_weights = self.grad_coil.psi_weights * 2 # for two plates
        self.num_levels = num_levels * 2 # for two plates
    
        self.x = prepare_vars(num_psi = self.num_psi_weights, types = ['Real'], num_levels=self.num_levels,
                              options = [-1, 1, -1, 1])
        self.n_obj = n_obj
        self.num_constr = n_constr
        self.linearity_percentage = linearity_percentage
        
        super().__init__(vars=self.x, n_ieq_constr=self.num_constr, n_obj=self.n_obj, **kwargs)
       
        
    def _evaluate(self, x, out, *args, **kwargs):
        self.biplanar_coil_pattern, self.coil_resistance, self.coil_current, max_ji, self.psi_smoothness, self.wire_smoothness =self.grad_coil.load(vars = x,  
                                                                                    num_psi_weights = self.num_psi_weights, num_levels = self.num_levels, pos = self.pos, sensors = self.sensors, viewing = False)
        if (len(self.biplanar_coil_pattern)==0):
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
        
            
            out["F"] = [f[1], f[2]]
            out["G"] = [f[0]- self.linearity_percentage]
        
        # print(Fore.YELLOW + 'Constraint violation value: ' + str(np.round(f[0]- self.linearity_percentage, decimals=2)) + Style.RESET_ALL)
        # out["G"] = [g2]
    
    
     
        


    
            