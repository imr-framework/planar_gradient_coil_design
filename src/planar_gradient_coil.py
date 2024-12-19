import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from colorama import Fore, Style
import magpylib as magpy
from utils import get_surface_current_density, compute_resistance, make_wire_patterns_contours, get_magnetic_field, display_scatter_3D, visualize_gradient_coil


class PlanarGradientCoil:
    ''' This class describes the geometry and properties of a planar gradient coil. '''
    
    def __init__(self, grad_dir, radius,  mesh, target_field, heights, current =10,
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
        self.triangle_thickness = wire_thickness
        self.current = current # A
        self.heights = heights
        self.upper_coil_plate_height = heights[0]
        self.lower_coil_plate_height = heights[1]
        self.symmetry = symmetry
        self.make_coil_plates()
        
        
    def get_triangles(self, viewing=False):
        ''' Get triangles from the mesh of a particular circular region. '''
        
        # Create a circular region mesh
        theta = np.linspace(0, 2 * np.pi, self.mesh)
        r = np.linspace(0, self.radius, self.mesh)
        r, theta = np.meshgrid(r, theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
  
        # Flatten the arrays
        x = x.flatten()
        y = y.flatten()

        if self.symmetry is True:
            x, y = self.do_symmetry(x, y)

        # Create the Delaunay triangulation
        triang = tri.Triangulation(x, y)

        # Extract nodes of the triangulation
        nodes = np.vstack((triang.x, triang.y)).T
        print(Fore.GREEN + 'Number of nodes: ', nodes.shape[0], Style.RESET_ALL)
        

        # Extract triangles
        triangles = triang.triangles
        print(Fore.GREEN + 'Number of triangles: ', triangles.shape[0], Style.RESET_ALL)
        
        debug = False
        if debug is True:
            print(Fore.GREEN + 'Triangles: ', triangles[0, :], Style.RESET_ALL)
            print(Fore.GREEN + 'Nodes: ', nodes[triangles[0, :]], Style.RESET_ALL)
            psi = np.random.random(triangles.shape[0])
            ji = get_surface_current_density(triangles[0, :], nodes, psi, p=2)
        
            
        # Display the triangles
        if viewing is True:
            plt.triplot(triang, 'go-', lw=1.0)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title('Triangulation of Circular Region')
            plt.show()
        self.triangles = triangles
        self.nodes = nodes
        self.num_nodes_total = nodes.shape[0]
        self.num_triangles_total = triangles.shape[0]
        
        return triangles, nodes
    
    def do_symmetry(self, x, y):
        ''' Exploit symmetry of the planar gradient coil to reduce the coil nodes. '''
        
        if self.grad_dir == 'x':
            mask = y >= 0
            x = x[mask]
            y = y[mask]
            
        elif self.grad_dir == 'y':
            mask = x >= 0
            x = x[mask]
            y = y[mask]
            
        return x, y
        
        
    def make_coil_plates(self):
        ''' Make the coil plates for the planar gradient coil. '''
        triangles, nodes = self.get_triangles()

        self.upper_coil_plate_triangles = triangles
        self.lower_coil_plate_triangles = triangles
        self.num_triangles_total = triangles.shape[0] * 2 # upper and lower plates
        
        self.upper_coil_plate_nodes = nodes
        self.lower_coil_plate_nodes = nodes
        
        self.num_nodes_total = nodes.shape[0] * 2 # upper and lower plates
        
        self.upper_coil_plate_nodes_3D = np.concatenate((nodes, self.heights[0] * np.ones((nodes.shape[0], 1))), axis=1)
        self.lower_coil_plate_nodes_3D = np.concatenate((nodes, self.heights[1] * np.ones((nodes.shape[0], 1))), axis=1)
        
        pass
        
    
    def make_wire_patterns(self):
        ''' Make wire patterns for the triangular mesh. '''
        self.upper_coil_wire_pattern = make_wire_patterns(self.upper_coil_plate_triangles, self.upper_coil_plate_nodes_3D,
                                                                  self.upper_coil_plate_height, current = self.current, w=2, g=1, viewing=True)
        self.lower_coil_wire_pattern = make_wire_patterns(self.lower_coil_plate_triangles, self.lower_coil_plate_nodes_3D,
                                                                  self.lower_coil_plate_height, current = self.current, w=2, g=1, viewing=True)
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r') 
        biplanar_coil_pattern.add(self.upper_coil_wire_pattern)
        biplanar_coil_pattern.add(self.lower_coil_wire_pattern)
        
        self.biplanar_coil_pattern = biplanar_coil_pattern
          
        pass
    
    def load(self, psi, num_nodes, pos, sensors, viewing=False):
        ''' Load the planar gradient coil. '''
        biplanar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
        self.psi = psi
        psi = np.array([psi[f"x{child:02}"] for child in range(0, num_nodes)]) # all children should have same magnet positions to begin with
        # compute current densities for all triangles
        N_each_plate = int(len(psi) * 0.5)
        upper_plate_triangles, upper_plate_ji, upper_plate_areas= get_surface_current_density(self.upper_coil_plate_triangles, self.upper_coil_plate_nodes_3D, psi[0:N_each_plate], p=2)
        lower_plate_triangles, lower_plate_ji, lower_plate_area = get_surface_current_density(self.lower_coil_plate_triangles, self.lower_coil_plate_nodes_3D, psi[N_each_plate:], p=2)
        
    
        # make wire patterns accordingly including the magnet collection
        # self.upper_coil_wire_pattern = make_wire_patterns(upper_plate_triangles,upper_plate_ji, self.upper_coil_plate_nodes_3D,
                                                                #   height = self.upper_coil_plate_height, current =self.current, w=self.wire_thickness, g=self.wire_spacing, psi=psi, viewing=False)
        
        self.upper_coil_wire_pattern = make_wire_patterns_contours(upper_plate_triangles, self.upper_coil_plate_nodes_3D, psi=psi[0:N_each_plate], 
                                                                   current=self.current, viewing=False)
        self.lower_coil_wire_pattern = make_wire_patterns_contours(lower_plate_triangles, self.lower_coil_plate_nodes_3D, psi=psi[N_each_plate:],
                                                                   current = self.current, viewing=False)
        
        max_ji_upper = np.max(upper_plate_ji)
        # self.lower_coil_wire_pattern = make_wire_patterns(lower_plate_triangles,lower_plate_ji, self.lower_coil_plate_nodes_3D,
                                                                #   height = self.lower_coil_plate_height, current = self.current, w=self.wire_thickness, g=self.wire_spacing, psi = psi, viewing=False)
        max_ji_lower = np.max(lower_plate_ji)
        max_ji = np.max([max_ji_upper, max_ji_lower])
        
        
        # Compute terms required for optimization
        #---------------------------------------------------------------
        # Compute the resistance of the coil plates
        upper_plate_resistance = self.compute_resistance(upper_plate_ji, upper_plate_areas)
        lower_plate_resistance = self.compute_resistance(lower_plate_ji, lower_plate_area)
        
        # compute the resistance of the coil
        self.coil_resistance = upper_plate_resistance + lower_plate_resistance
        
        #---------------------------------------------------------------
        # Compute the current in the coil from all the triangles
        upper_plate_coil_current = self.compute_current_triangles(upper_plate_ji, upper_plate_areas)
        lower_plate_coil_current = self.compute_current_triangles(lower_plate_ji, lower_plate_area)
        self.coil_current = upper_plate_coil_current + lower_plate_coil_current
         
        biplanar_coil_pattern.add(self.upper_coil_wire_pattern)
        biplanar_coil_pattern.add(self.lower_coil_wire_pattern)
        self.biplanar_coil_pattern = biplanar_coil_pattern
        self.psi_array = psi
        if viewing is True:
            # biplanar_coil_pattern.show(backend='matplotlib')
            visualize_gradient_coil(biplanar_coil_pattern)
            Bz_grad = get_magnetic_field(biplanar_coil_pattern, sensors, axis = 2)
            display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad, title='Magnetic Field of the Planar Gradient Coil')
        return self.biplanar_coil_pattern, self.coil_resistance, self.coil_current, max_ji
    
    def view(self,  sensors, pos, symmetry=False):
        ''' Visualize the planar gradient coil. '''
        if symmetry is False:
            coil_pattern = self.biplanar_coil_pattern_all
        else:
            coil_pattern = self.biplanar_coil_pattern
            
        visualize_gradient_coil(coil_pattern)
            
        Bz_grad = get_magnetic_field(coil_pattern, sensors, axis = 2)
        display_scatter_3D(pos[:, 0], pos[:, 1], pos[:, 2], Bz_grad, title='Magnetic Field of the Planar Gradient Coil')
        pass
    
    def compute_magnetic_field(self): # for Lorentz force later
        ''' Compute the magnetic field of the planar gradient coil. '''
        
        pass
    
    def compute_error(self):
        ''' Compute the error between the target and planar gradient magnetic field. '''
        pass
    
    def compute_energy(self):
        ''' Compute the energy of the planar gradient coil. '''
        pass
    
    def compute_inductance(self):
        ''' Compute the inductance of the planar gradient coil. '''
        pass
    
    def compute_resistance(self, triangles_ji, triangles_area):
        ''' Compute the resistance of the planar gradient coil. '''
        self.triangle_resistance = []
        triangles_ji = np.array(triangles_ji)
        triangles_area = np.array(triangles_area)
        
        for triangle in range(triangles_ji.shape[0]):
            resistance_triangle = compute_resistance(triangles_ji[triangle], triangles_area[triangle],  p=2, fact_include = False)
            self.triangle_resistance.append(resistance_triangle)
            
        coil_resistance = np.sum(self.triangle_resistance) *  self.resistivity / (self.triangle_thickness * self.current**2)
        return coil_resistance
    
    def compute_power_dissipation(self):
        ''' Compute the power dissipation of the planar gradient coil. '''
        pass
    
    def compute_current_triangles(self, triangles_ji, triangles_area,p=2):
        ''' Compute the current in the planar gradient coil. '''
        self.triangle_current = []
        triangles_ji = np.array(triangles_ji)
        triangles_area = np.array(triangles_area)
        
        for triangle in range(triangles_ji.shape[0]):
            current_triangle = np.linalg.norm(triangles_ji[triangle], ord=p, axis=0) * triangles_area[triangle]
            self.triangle_current.append(current_triangle)
            
        coil_current = np.sum(self.triangle_current)
        return coil_current
    
    def undo_symmetry(self):
        ''' Undo the symmetry reduction of the planar gradient coil. '''
        self.symmetry = False
        self.biplanar_coil_pattern_all = self.biplanar_coil_pattern
        biplanar_coil_pattern = self.biplanar_coil_pattern
        if (len(biplanar_coil_pattern)) > 1: # expects two plates
            for plate in range(len(biplanar_coil_pattern)): 
                plate_new = biplanar_coil_pattern.children[plate].copy()
                for wire in range(len(biplanar_coil_pattern.children[plate].children)):
                    wire_pattern = biplanar_coil_pattern.children[plate].children[wire]
                    wire_pattern.vertices = np.array(wire_pattern.vertices)
                    
                    if self.grad_dir == 'x':
                        wire_pattern.vertices[:, 1] = - wire_pattern.vertices[:, 1]
                    elif self.grad_dir == 'y':
                        wire_pattern.vertices[:, 0] = - wire_pattern.vertices[:, 0]
                        
                    vertices = np.array(wire_pattern.vertices)
                    plate_new.add(magpy.current.Polyline(current=wire_pattern.current, vertices=vertices))
                
                self.biplanar_coil_pattern_all.add(plate_new) 
               
        pass
    
    def save(self, fname:str=None):
        ''' Save the planar gradient coil. '''
        visualize_gradient_coil(self.biplanar_coil_pattern, save = True, fname_save=fname)
        pass