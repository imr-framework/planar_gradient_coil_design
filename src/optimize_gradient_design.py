# -----------------------------------------------------------------------------
# Import the libraries
import magpylib as magpy
import scipy.optimize as opt
import numpy as np
from utilities import *
from colorama import Fore, Style
from pymoo.core.problem import ElementwiseProblem

# -----------------------------------------------------------------------------
# Initialize the class for the planar gradient coil optimization
class planar_gradient_problem(ElementwiseProblem):
   
    def __init__(self, grad_coil, sensors, pos, target_field, psi, order=2, 
                 alpha=[0.5], B_tol = 5, num_levels = 10, linearity_percentage = 5, 
                 n_obj=1, n_constr=0, symmetry= True, **kwargs):
        
        self.grad_coil = grad_coil
        self.sensors = sensors
        self.target_field = target_field
        self.pos = pos
        self.order = order
        self.alpha = alpha
        self.B_tol = B_tol
        self.num_psi_weights = self.grad_coil.num_psi_weights # for two plates
        self.num_levels = num_levels # for two plates
        self.psi_init = psi
        self.n_obj = n_obj
        self.num_constr = n_constr
        self.linearity_percentage = linearity_percentage
        self.symmetry = symmetry
        self.x = prepare_vars(num_psi = self.num_psi_weights, types = ['Real'], 
                              options = [-1, 1]) # [-1, 1] 
        super().__init__(vars=self.x, n_ieq_constr=self.num_constr, n_obj=self.n_obj, **kwargs)
       
        
# -----------------------------------------------------------------------------
# Define the objective function
    def objective(self, vars, viewing = False):
        self.biplanar_coil_pattern = self.grad_coil.load(vars, self.num_psi_weights, self.psi_init, viewing = viewing)
        if (len(self.biplanar_coil_pattern.children[0])==0) and (len(self.biplanar_coil_pattern.children[1])==0): # for the case when no wires are present
           self.grad_coil_field = 0
        else:
            self.grad_coil_field = get_magnetic_field(self.biplanar_coil_pattern, self.sensors, axis = 2)
        
        if viewing is True:
            display_scatter_3D(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], self.grad_coil_field, title = 'Gradient Coil Field')        
        
        
        f1 = 100 * np.linalg.norm(self.grad_coil_field - self.target_field, ord=np.inf) / (np.linalg.norm(self.target_field, ord=np.inf)) # minimizing peak field
        
        f = f1
        return f

    def _evaluate(self, vars, out, viewing = False, *args, **kwargs):
        self.biplanar_coil_pattern = self.grad_coil.load(vars, self.num_psi_weights, self.psi_init, viewing = viewing)
        
        
        
        if (len(self.biplanar_coil_pattern.children[0])==0) and (len(self.biplanar_coil_pattern.children[1])==0): # for the case when no wires are present
           self.grad_coil_field = 0
        else:
            self.grad_coil_field = get_magnetic_field(self.biplanar_coil_pattern, self.sensors, axis = 2)
        
        
        
        if viewing is True:
            display_scatter_3D(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], self.grad_coil_field, title = 'Gradient Coil Field')        
        
        f0 = 100 * np.linalg.norm(self.grad_coil_field - self.target_field, ord=np.inf) / (np.linalg.norm(self.target_field, ord=np.inf)) # minimizing peak field
        f1 = 100 * np.linalg.norm(self.grad_coil_field - self.target_field, ord=2) / (np.linalg.norm(self.target_field, ord=2)) # minimizing RMSE error
        if (f0 - self.linearity_percentage) <= 0:
            self.biplanar_coil_pattern_wires, _, _, _ = self.grad_coil.get_wire_patterns(vars, levels = self.grad_coil.levels, stream_function = self.psi_init, 
                                        x = self.grad_coil.x, y = self.grad_coil.y, heights = self.grad_coil.heights, current = 1, viewing = False)
        
            # _, Lorentz_force_mag = self.grad_coil.get_Lorentz_force(self.biplanar_coil_pattern_wires, 
            Lorentz_force_mag = 0                                                                 # self.biplanar_coil_pattern)
        else:
            Lorentz_force_mag = np.inf
        
        
        f2 = 1e4 * Lorentz_force_mag # 100 * get_psi_smoothness(vars, self.num_psi_weights, self.psi_init, self.grad_coil.mesh)
        
        # vars_2 = np.array([vars[f"x{child:02}"] for child in range(0, 2 * self.num_psi_weights)]) # all children should have same magnet positions to begin with
        f3 = 0 # 1000 * np.abs(np.sum(vars_2[:self.num_psi_weights])) # symmetry constraint
        f4 = 0 # 1000 * np.abs(np.sums(vars_2[self.num_psi_weights:])) # symmetry constraint
        # print(f0, f1, f2, f3, f4)
        out["F"] = [f1, f2, f3, f4]
        out["G"] = [f0 - self.linearity_percentage]

# -----------------------------------------------------------------------------
# Define the constraints



