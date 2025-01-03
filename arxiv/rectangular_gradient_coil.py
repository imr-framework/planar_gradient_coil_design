import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from colorama import Fore, Style
import magpylib as magpy
from utils import *
import csv
import os

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
        a =  self.radius / np.sqrt(2)  # 4 * r**2  = 2 * (2a)**2;  a = r / sqrt(2) 
        self.x = np.linspace(-a, a, self.mesh)
        self.y = np.linspace(-a, a, self.mesh)
        self.stream_function = get_stream_function(grad_dir='x', x =self.x, y=self.y, viewing = False)
        pass

    def load(self, vars, num_psi_weights, num_levels, pos, sensors, opt='ga', viewing = False):
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        
        if opt == 'ga':
            psi_flatten = np.array([vars[f"x{child:02}"] for child in range(0, num_psi_weights)]) # all children should have same magnet positions to begin with
            num_levels = np.array([vars[f"x{child:02}"] for child in range(num_psi_weights, num_psi_weights + 1)]) # only one number of levels for now - maybe two for each plate
            
            psi_flatten_upper_plate = psi_flatten[:num_psi_weights//2]
            psi_flatten_lower_plate = psi_flatten[num_psi_weights//2:]
            num_levels = int(num_levels)
            # print(Fore.YELLOW + 'Number of levels: ' + str(num_levels) + Style.RESET_ALL)
            levels_upper_plate = num_levels / 2
            levels_lower_plate = num_levels / 2 # multiplied by 2 because of two plates
            
        elif opt == 'cvx':
            psi_flatten_upper_plate = vars[:num_psi_weights//2]
            psi_flatten_lower_plate = vars[num_psi_weights//2:]
            levels_upper_plate = num_levels
            levels_lower_plate = num_levels
        
        # levels_upper_plate = levels[:num_levels//2]
        # levels_lower_plate = levels[num_levels//2:]
            levels_upper_plate = num_levels / 2
            levels_lower_plate = num_levels / 2
        
        psi_upper_plate = np.reshape(psi_flatten_upper_plate, (self.mesh, self.mesh))
        psi_lower_plate = np.reshape(psi_flatten_lower_plate, (self.mesh, self.mesh))
        
        upper_plate_wire_patterns, upper_plate_wire_smoothness = get_wire_patterns_contour_rect(psi = psi_upper_plate, levels = levels_upper_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.upper_coil_plate_height, current = self.current)
        lower_plate_wire_patterns, lower_plate_wire_smoothness = get_wire_patterns_contour_rect(psi = psi_lower_plate, levels = levels_lower_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.lower_coil_plate_height, current = self.current)
        
        biplanar_coil_pattern.add(upper_plate_wire_patterns)
        biplanar_coil_pattern.add(lower_plate_wire_patterns)
        self.biplanar_coil_pattern = biplanar_coil_pattern
        
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
    
        return self.biplanar_coil_pattern, coil_resistance, coil_current, max_ji, psi_smoothness, wire_smoothness

    def load_cvx(self, vars, num_psi_weights, num_levels, pos, sensors, opt='cvx', viewing = False):
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        
        psi_flatten_upper_plate = vars[:num_psi_weights//2]
        psi_flatten_lower_plate = vars[num_psi_weights//2:]

        levels_upper_plate = num_levels / 2
        levels_lower_plate = num_levels / 2
        
        psi_upper_plate = np.reshape(psi_flatten_upper_plate, (self.mesh, self.mesh))
        psi_lower_plate = np.reshape(psi_flatten_lower_plate, (self.mesh, self.mesh))
        
        upper_plate_wire_patterns, _ = get_wire_patterns_contour_rect(psi = psi_upper_plate, levels = levels_upper_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.upper_coil_plate_height, current = self.current)
        lower_plate_wire_patterns, _ = get_wire_patterns_contour_rect(psi = psi_lower_plate, levels = levels_lower_plate, stream_function = self.stream_function,
                                                                   x = self.x, y = self.y, z= self.lower_coil_plate_height, current = self.current)
        
        biplanar_coil_pattern.add(upper_plate_wire_patterns)
        biplanar_coil_pattern.add(lower_plate_wire_patterns)
        
        self.biplanar_coil_pattern = biplanar_coil_pattern
        return self.biplanar_coil_pattern
        
    def view(self,  sensors, pos, symmetry=False):
        ''' Visualize the planar gradient coil. '''

        visualize_gradient_coil(self.biplanar_coil_pattern)
            
        Bz_grad = get_magnetic_field(self.biplanar_coil_pattern, sensors, axis = 2)
        display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad, title='Magnetic Field of the Planar Gradient Coil')
        self.Bz_grad = Bz_grad
        pass
    
    def save(self, fname:str=None):
        ''' Save the planar gradient coil. '''
        visualize_gradient_coil(self.biplanar_coil_pattern, save = True, fname_save=fname)
        pass
    
    def save_loops(self, fname_csv_file:str=None, loop_tolerance = 1, viewing = False):
        ''' Filter the wire patterns to remove overlapping wires. '''
        # Load the wire patterns' coordinates and current direction from file in fname
        # Load the wire patterns' coordinates and current direction from file in fname
        biplanar_coil_pattern = self.biplanar_coil_pattern
        wire_num_negative= int(0)
        wire_num_positive=int(0)
        for plate in biplanar_coil_pattern.children:
            for wire_pattern in plate.children:
                current = wire_pattern.current
                coordinates = wire_pattern.vertices
                
                if current > 0:
                    positive_wires = np.round(coordinates * 1e3, decimals=0)
                    positive_wires_collated = identify_loops(positive_wires, loop_tolerance = loop_tolerance)
                    for wire in range(0, len(positive_wires_collated)):
                        positive_wires_each = positive_wires_collated[wire]
                        wire_num_positive += 1
                        positive_wires_each = positive_wires_each[: : 2, :]
                        fname = 'positive_wires_'  + str(wire_num_positive) + '_' + fname_csv_file
                        self.write_loop_csv(positive_wires_each[:-1, :], fname = fname)
                        if viewing is True:
                            plt.plot(positive_wires_each[:, 0], positive_wires_each[:, 1], 'ro', label='Positive Wires - before filtering')
                            plt.show()
                else:
                    negative_wires = np.round(coordinates * 1e3, decimals=0)
                    negative_wires_collated = identify_loops(negative_wires, loop_tolerance = loop_tolerance)
                    for wire in range(0, len(negative_wires_collated)):
                        negative_wires_each = negative_wires_collated[wire]
                        wire_num_negative += 1
                        negative_wires_each = negative_wires_each[: : 2, :]
                        fname = 'negative_wires_'  + str(wire_num_negative) + '_' + fname_csv_file
                        self.write_loop_csv(negative_wires_each[: -1, :], fname = fname)
                        if viewing is True:
                            plt.plot(negative_wires_each[:, 0], negative_wires_each[:, 1], 'bo', label='Negative Wires - before filtering')
                            plt.show()
                    
        plt.show()
        
            
    def write_loop_csv(self, coordinates, fname:str=None):
        ''' Write the wire patterns to a CSV file. '''
        with open(fname, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in coordinates:
                writer.writerow(row)
           
        
        pass
    
    def combine_loops_current_direction(self, loop_tolerance = 5, fname_csv_file:str=None):
        ''' Combine the wire patterns and current direction. '''
        
        biplanar_coil_pattern = self.biplanar_coil_pattern
        plate_num = 0
        
        for plate in biplanar_coil_pattern.children:
            plate_num += 1
            positive_wires_collated = []
            negative_wires_collated = []
            
            for wire_pattern in plate.children:
                current = wire_pattern.current
                coordinates = wire_pattern.vertices
                
                # Collect the wire patterns
                if current > 0:
                    positive_wires = np.round(coordinates * 1e3, decimals=0)
                    positive_wires_loops = identify_loops(positive_wires, loop_tolerance = loop_tolerance)
                    for loop in positive_wires_loops:
                        positive_wires_collated.append(loop)
                else:
                    negative_wires = np.round(coordinates * 1e3, decimals=0)
                    negative_wires_loops = identify_loops(negative_wires, loop_tolerance = loop_tolerance)
                    for loop in negative_wires_loops:
                        negative_wires_collated.append(loop)
                    
            if len(positive_wires_collated) == 0:
                print(Fore.RED + 'No positive wires found for plate ' + str(plate_num) + Style.RESET_ALL)
            else:
                positive_loops_combined = combine_loops(positive_wires_collated)
                fname = 'positive_loops_combined_' + str(plate_num) + '_' + fname_csv_file
                self.write_loop_csv(positive_loops_combined, fname = fname)
            
            if len(negative_wires_collated) == 0:
                print(Fore.RED + 'No negative wires found for plate ' + str(plate_num) + Style.RESET_ALL)
            else:
                negative_loops_combined = combine_loops(negative_wires_collated)
                fname = 'negative_loops_combined_' + str(plate_num) + '_'+ fname_csv_file
                self.write_loop_csv(negative_loops_combined, fname = fname)
        pass
    
    
    def load_from_csv(self, dirname, fname_pattern=None):
        ''' Load the wire patterns from a CSV file. '''
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        if dirname is None:
            print(Fore.YELLOW + 'Using current directory' + Style.RESET_ALL)
            dirname = os.getcwd()
            files = os.listdir(dirname)
        else:
            files = os.listdir(dirname)
        num_file =0 
        for file in files:
            if fname_pattern[0] in file:
                num_file += 1
                print(Fore.YELLOW + 'Loading negative wire patterns from ' + file + Style.RESET_ALL)
                negative_wires_coordinates = np.loadtxt(os.path.join(dirname, file), delimiter=',')     
                negative_wires_coordinates = negative_wires_coordinates / 1e3  # convert back to meters
                biplanar_coil_pattern.add(magpy.current.Polyline(current=-1, vertices=negative_wires_coordinates))     
            elif fname_pattern[1] in file:
                num_file += 1
                print(Fore.YELLOW + 'Loading positive wire patterns from ' + file + Style.RESET_ALL)
                positive_wires_coordinates = np.loadtxt(os.path.join(dirname, file), delimiter=',')
                positive_wires_coordinates = positive_wires_coordinates / 1e3
                biplanar_coil_pattern.add(magpy.current.Polyline(current=1, vertices=positive_wires_coordinates))
        self.biplanar_coil_pattern = biplanar_coil_pattern
        print(Fore.YELLOW + 'Loaded ' + str(num_file) + ' files' + Style.RESET_ALL)
        return
        
        
