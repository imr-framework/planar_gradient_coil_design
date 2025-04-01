# --------------------------------------------------------
# Import libraries
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from colorama import Fore, Style
import magpylib as magpy
from utilities import *
import csv
import os
import warnings
from mpl_toolkits import mplot3d 
import numba
# --------------------------------------------------------
# Declare the class for the planar gradient coil

class PlanarGradientCoil:
    ''' This class describes the geometry and properties of a planar gradient coil. '''
# --------------------------------------------------------
# Initialize the planar gradient coil

    def __init__(self, grad_dir, radius,  mesh, heights, levels, current, magnetization = 8 * 1e5,
                 thickness=0.1, spacing=0.1, resistivity=1.68e-8, symmetry=True):
        ''' Initialize the planar gradient coil. '''
        self.grad_dir = grad_dir
        self.radius = radius
        self.thickness = thickness
        self.spacing = spacing
        self.mesh = mesh
        self.resistivity = resistivity # Ohm-m
        self.heights = heights
        self.upper_coil_plate_height = heights[0]
        self.lower_coil_plate_height = heights[1]
        self.symmetry = symmetry
        self.num_psi_weights = mesh ** 2
        self.magnet_side_length = 2 * radius / mesh
        self.magnetization_fact = magnetization 
        self.levels = levels
        self.biplanar_coil_pattern = None
        self.biplanar_coil_pattern_wires = None
        self.current = current
        self.get_xy_coords()
        
        
        # Check if the x and y coordinates are at least spacing apart
        if self.x[1] - self.x[0] < self.spacing:
            print(Fore.RED + 'The coordinates are not at least spacing apart' + Style.RESET_ALL)
            
        
    def get_xy_coords(self):
        a =  self.radius / np.sqrt(2)  # 4 * r**2  = 2 * (2a)**2;  a = r / sqrt(2) 
        self.x = np.linspace(-a, a, self.mesh)
        self.y = np.linspace(-a, a, self.mesh)
        if self.symmetry is True:
            if self.grad_dir == 'x':
                self.y = self.y[self.y >= 0]
            elif self.grad_dir == 'y':
                self.x = self.x[self.x >= 0]
            elif self.grad_dir == 'z':
                self.x = self.x[self.x >= 0]
                self.y = self.y[self.y >= 0]
        pass
      
    # --------------------------------------------------------
    # load the coil
    def load(self, vars, num_psi_weights, psi_init, viewing = False):
        biplanar_coil_pattern = magpy.Collection()
          
        vars = np.array([vars[f"x{child:02}"] for child in range(0, 2 * num_psi_weights)]) # all children should have same magnet positions to begin with
            
        psi_flatten_upper_plate = vars[:num_psi_weights]
        psi_flatten_lower_plate = vars[num_psi_weights:]
        
        psi_upper_plate = psi_flatten_upper_plate.reshape(self.mesh, self.mesh) * psi_init
        psi_lower_plate = psi_flatten_lower_plate.reshape(self.mesh, self.mesh) * psi_init
        
        upper_plate = self.load_plate(psi_upper_plate, self.x, self.y, self.upper_coil_plate_height)
        lower_plate = self.load_plate(psi_lower_plate, self.x, self.y, self.lower_coil_plate_height)
        
        biplanar_coil_pattern.add(upper_plate)
        biplanar_coil_pattern.add(lower_plate)

        if viewing is True:
            magpy.show(biplanar_coil_pattern, backend='matplotlib')
        
        self.biplanar_coil_pattern = biplanar_coil_pattern
            
        return biplanar_coil_pattern
    def load_plate(self, psi_plate, x, y, z):
        ''' Load the plate. '''

        plate = magpy.Collection()
        
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        positions = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, z)))
        magnetizations = self.magnetization_fact * psi_plate.ravel()
        valid_indices = np.abs(magnetizations) > 0.10 * self.magnetization_fact
        positions = positions[valid_indices]
        magnetizations = magnetizations[valid_indices]

        for pos, mag in zip(positions, magnetizations):
            style_color = 'r' if mag > 0 else 'b'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cube = magpy.magnet.Cuboid(
                        dimension=[self.magnet_side_length, self.magnet_side_length, self.magnet_side_length],
                        position=pos,
                        magnetization=[0, 0, mag],
                        style_color=style_color
                        )
                plate.add(cube)
                
                
        return plate
    
    def get_wire_patterns(self, vars, levels, stream_function, x, y, heights, 
                          current = 1, loop_tolerance = 2e-3, viewing = False):
    # check if x, y and z are in mm, if not convert to mm
    
        planar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        wire_smoothness = 0
        psi = np.array([vars[f"x{child:02}"] for child in range(0, 2 * self.num_psi_weights)]) 
        for z in heights:
            if z == heights[0]:
                psi_plate = psi[:self.num_psi_weights]
                spiral_plate_lower = []
            else:
                psi_plate = psi[self.num_psi_weights:]
                spiral_plate_upper = []
            loop_stack = []    
            contours = psi2contour(x, y, psi_plate, stream_function, levels = levels, viewing = viewing) # viewing = viewing
            spiral = connect_contours_to_spiral(contours, gap_pts=10)
            
            for collection, level  in zip(contours.collections, contours.levels):
                paths = collection.get_paths()
                # current_direction = np.sign(level)
                
                current_direction = level
                if z == heights[0]:
                    current_direction = -current_direction
                    
                for path in paths:
                    vertices = path.vertices 
                    loops_vertices = identify_loops(vertices, loop_tolerance = loop_tolerance)  # m at this stage
                    loop_stack.append(loops_vertices)
                    # all_vertices.extend(loops)
                    
                    for vertices in loops_vertices:
                        # Check if the loop intersects with any of the previous loops
                        # intersect = check_intersection(vertices, vertices_all)
                        # change the contour to spirals
                    
                            
                        loop_vertices_wire_widths = np.array([vertices[:, 0], vertices[:, 1], z * np.ones(vertices[:, 0].shape)]).T
                        gradient_x = np.gradient(loop_vertices_wire_widths[:, 0])
                        gradient_y = np.gradient(loop_vertices_wire_widths[:, 1])
                        wire_smoothness += np.linalg.norm(np.sqrt(gradient_x**2 + gradient_y**2))
                        
                        if current_direction > 0:
                            planar_coil_pattern.add(magpy.current.Polyline(current = current * current_direction, 
                                                                vertices = loop_vertices_wire_widths, style_color = 'red'))
                        else:
                            planar_coil_pattern.add(magpy.current.Polyline(current = current * current_direction, 
                                                                vertices = loop_vertices_wire_widths, style_color = 'blue'))
            # plt.plot(vertices_all[:, 0], vertices_all[:, 1], 'ro')
            # plt.show()
            
                            
            if z == heights[0]:
                spiral_plate_lower = np.array(spiral)
                if viewing is True:
                    planar_coil_pattern.show()
                    plt.plot(spiral_plate_lower[:, 0], spiral_plate_lower[:, 1], 'ro')
                    plt.show()
            else:
                spiral_plate_upper = np.array(spiral)
                if viewing is True:
                    planar_coil_pattern.show()
                    plt.plot(spiral_plate_upper[:, 0], spiral_plate_upper[:, 1], 'ro')
                    plt.show()
                        
        self.biplanar_coil_pattern_wires = planar_coil_pattern
        return planar_coil_pattern, spiral_plate_lower, spiral_plate_upper, wire_smoothness
       
# --------------------------------------------------------
# visualize the coil

    def view(self, current_dir = False):
        if self.biplanar_coil_pattern_wires is None:
            print(Fore.RED + 'No wire patterns found - aborting!' + Style.RESET_ALL)
        elif current_dir is False:
            magpy.show(self.biplanar_coil_pattern_wires, backend='matplotlib')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for wire_pattern in self.biplanar_coil_pattern_wires.children:
                current = wire_pattern.current
                vertices = wire_pattern.vertices
                if current > 0:
                    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'r-')
                else:
                    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-')
            plt.show()

# --------------------------------------------------------
# characterize the coil

# --------------------------------------------------------
# save the coil
    def save_loops(self, fname_csv_file:str=None, loop_tolerance = 1, viewing = False, dirname = './csv_files'):
        ''' Filter the wire patterns to remove overlapping wires. '''
        # Load the wire patterns' coordinates and current direction from file in fname
        # Load the wire patterns' coordinates and current direction from file in fname
        biplanar_coil_pattern = self.biplanar_coil_pattern_wires
        wire_num_negative= int(0)
        wire_num_positive=int(0)
        for wire_pattern in biplanar_coil_pattern.children:
                current = wire_pattern.current
                coordinates = wire_pattern.vertices
                
                if current > 0:
                    positive_wires = np.round(coordinates * 1e3, decimals=0)
                    positive_wires_collated = identify_loops(positive_wires, loop_tolerance = loop_tolerance)
                    wire_num_positive += 1
                    fname = os.path.join(dirname, 'positive_wires_'  + str(wire_num_positive) + '_' + fname_csv_file)   
                    self.write_loop_csv(positive_wires[: : 2, :], fname = fname)
                else:
                    negative_wires = np.round(coordinates * 1e3, decimals=0)
                    negative_wires_collated = identify_loops(negative_wires, loop_tolerance = loop_tolerance)
                    wire_num_negative += 1
                    fname = os.path.join(dirname, 'negative_wires_'  + str(wire_num_negative) + '_' + fname_csv_file)   
                    self.write_loop_csv(negative_wires[: : 2, :], fname = fname)
        
            
    def write_loop_csv(self, coordinates, fname:str=None):
        ''' Write the wire patterns to a CSV file. '''
        with open(fname, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in coordinates:
                writer.writerow(row)
           
        
        pass

    def get_Lorentz_force(self, biplanar_coil_pattern_wires, biplanar_magnets):
        ''' Get the Lorentz force on the wires. '''
        Lorentz_force_vector = np.zeros(3)
        Lorentz_force_mag = 0
        
        # @numba.njit
        def compute_forces(biplanar_magnets, vertices, current):
            b_field = magpy.getB(biplanar_magnets, vertices[:-1] + np.diff(vertices, axis=0) / 2.0)
            length_vectors = np.diff(vertices, axis=0)
            forces_on_segments = current * np.cross(length_vectors, b_field)
            return forces_on_segments

        for wire_pattern in biplanar_coil_pattern_wires.children:
            current = wire_pattern.current
            vertices = wire_pattern.vertices
            forces_on_segments = compute_forces(biplanar_magnets, vertices, current)
            Lorentz_force_vector += np.sum(forces_on_segments, axis=0)
            Lorentz_force_mag += np.sum(np.linalg.norm(forces_on_segments, axis=1))
            
        return Lorentz_force_vector, Lorentz_force_mag
    