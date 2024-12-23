import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from colorama import Fore, Style
import magpylib as magpy
from utils import *
import csv

class PlanarGradientCoil_rectangle:
    ''' This class describes the geometry and properties of a planar gradient coil. '''
    
    def __init__(self, grad_dir, radius,  mesh, target_field, heights, current =1,
                 wire_thickness=0.1, wire_spacing=0.1, resistivity=1.68e-8, symmetry=True):
        ''' Initialize the planar gradient coil. '''
        self.grad_dir = grad_dir
        self.radius = radius
        self.current = current
        self.wire_thickness = wire_thickness
        self.wire_spacing = wire_spacing
        self.mesh = mesh
        self.target_field = target_field
        self.resistivity = resistivity # Ohm-m
        self.current = current # A
        self.heights = heights
        self.upper_coil_plate_height = heights[0]
        self.lower_coil_plate_height = heights[1]
        self.symmetry = symmetry
        self.psi_weights = mesh ** 2
        self.get_xy_coords_stream()
            
    def get_xy_coords_stream(self):
        self.x = np.linspace(-self.radius, self.radius, self.mesh)
        self.y = np.linspace(-self.radius, self.radius, self.mesh)
        self.stream_function = get_stream_function(grad_dir='x', x =self.x, y=self.y, viewing = False)
        pass

    def load(self, vars, num_psi_weights, num_levels, pos, sensors, viewing = False):
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        psi_flatten = np.array([vars[f"x{child:02}"] for child in range(0, num_psi_weights)]) # all children should have same magnet positions to begin with
        # levels = np.array([vars[f"x{child:02}"] for child in range(num_psi_weights, num_psi_weights + num_levels)])
        
        psi_flatten_upper_plate = psi_flatten[:num_psi_weights//2]
        psi_flatten_lower_plate = psi_flatten[num_psi_weights//2:]
        
        # levels_upper_plate = levels[:num_levels//2]
        # levels_lower_plate = levels[num_levels//2:]
        levels_upper_plate = num_levels / 2
        levels_lower_plate = num_levels / 2
        
        psi_upper_plate = np.reshape(psi_flatten_upper_plate, (self.mesh, self.mesh))
        psi_lower_plate = np.reshape(psi_flatten_lower_plate, (self.mesh, self.mesh))
        
        upper_plate_wire_patterns, upper_plate_wire_smoothness = get_wire_patterns_contour_rect(psi = psi_upper_plate, levels = levels_upper_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.upper_coil_plate_height)
        lower_plate_wire_patterns, lower_plate_wire_smoothness = get_wire_patterns_contour_rect(psi = psi_lower_plate, levels = levels_lower_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.lower_coil_plate_height)
        
        biplanar_coil_pattern.add(upper_plate_wire_patterns)
        biplanar_coil_pattern.add(lower_plate_wire_patterns)
        
        
        # Smoothness parameters
        gradient_psi_ux, gradient_psi_uy = np.gradient(psi_upper_plate)
        gradient_psi_lx, gradient_psi_ly = np.gradient(psi_lower_plate)
        
        
        psi_smoothness = 100 * np.linalg.norm(np.sqrt(gradient_psi_ux**2 + gradient_psi_uy**2 + gradient_psi_lx**2 + gradient_psi_ly**2))
        wire_smoothness = 100 * (upper_plate_wire_smoothness + lower_plate_wire_smoothness)
        
        # TODO parameter for Litz wire implementation here
        coil_resistance = 0 # TODO: compute coil resistance
        coil_current = 1 # A TODO: compute coil current
        max_ji = 1 # TODO: compute max_ji
        
        
        if viewing is True:
            # biplanar_coil_pattern.show(backend='matplotlib')
            visualize_gradient_coil(biplanar_coil_pattern)
            Bz_grad = get_magnetic_field(biplanar_coil_pattern, sensors, axis = 2)
            display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad, title='Magnetic Field of the Planar Gradient Coil')
    
        return biplanar_coil_pattern, coil_resistance, coil_current, max_ji, psi_smoothness, wire_smoothness

    def view(self,  sensors, pos, symmetry=False):
        ''' Visualize the planar gradient coil. '''

        visualize_gradient_coil(self.biplanar_coil_pattern)
            
        Bz_grad = get_magnetic_field(self.biplanar_coil_pattern, sensors, axis = 2)
        display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad, title='Magnetic Field of the Planar Gradient Coil')
        pass
    
    