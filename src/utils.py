'''This file contains the utility functions for the planar gradient coil design problem.'''
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from pymoo.core.variable import Real, Integer, Choice, Binary
from colorama import Fore, Style

# Design the target magnetic field - Bz
def set_target_field(grad_dir:str ='x', grad_max:float = 27, dsv:float = 30, res:float = 2, symmetry = True, viewing = False):
    ''' Design the target magnetic field. '''
    pts = int(np.ceil(dsv / res))
    Bz_max = grad_max * dsv * 0.5
    r = np.linspace(-0.5 * dsv, 0.5 * dsv, pts)
    
    X, Y, Z = np.meshgrid(r, r, r)
    Bz = np.zeros((pts, pts, pts))
    
    if grad_dir == 'x':
        Bz = Bz_max * X / (0.5 * dsv)
        if symmetry is True:
            mask1 = Y < 0 # consider only the positive of the other two dimensions
            mask2 = Z < 0 
    elif grad_dir == 'y':
        Bz = Bz_max * Y / (0.5 * dsv)
        if symmetry is True:
            mask1 = X < 0 # consider only the positive of the other two dimensions
            mask2 = Z < 0 
    elif grad_dir == 'z':
        Bz = Bz_max * Z / (0.5 * dsv)
        if symmetry is True:
            mask1 = X < 0 # consider only the positive of the other two dimensions
            mask2 = Y < 0 
    
    # Remove the coordinates in the sphere that are outside the -dsv/2 to dsv/2 range
    mask = np.sqrt(X**2 + Y**2 + Z**2) > 0.5 * dsv
    Bz[mask] = 0
    
    if symmetry is True:
        Bz[mask1] = 0
        Bz[mask2] = 0
        
    X = X[Bz!=0]
    Y = Y[Bz!=0]
    Z = Z[Bz!=0]
    Bz = Bz[Bz!=0]
    
    # Convert the magnetic field and co-ordinates to a 1D arrays
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    Bz = Bz.flatten()

     
    if viewing is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X, Y, Z, c=Bz, cmap='jet')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.title('3D Spherical Volume Bz')
        plt.colorbar(img)
        plt.show()
    
    return X, Y, Z, Bz



# Get the surface current density of a particular triangular mesh
def get_surface_current_density_triangle(triangle, nodes, psi, p=2, debug = False):
    ''' Get the surface current density of the triangular mesh. '''
    if debug is True:   
        print(Fore.GREEN + 'psi: ', psi, Style.RESET_ALL)
    P = nodes[triangle[0]]
    Q = nodes[triangle[1]]
    R = nodes[triangle[2]]
    
    # Compute the edge lengths
    PQ = np.linalg.norm(P - Q,ord=p, axis=0)
    QR = np.linalg.norm(Q - R,ord=p, axis=0)
    RP = np.linalg.norm(R - P,ord=p, axis=0)
    
    ei = np.max([PQ, QR, RP], axis=0) # magntiude of the longest edge
    
    # Get the direction of ei
    if ei == PQ:
        ei = P - Q
    elif ei == QR:
        ei = Q - R
    else:
        ei = R - P
        
    
    # Calculate the semi-perimeter
    s = (PQ + QR + RP) / 2
    
    # Calculate the area using Heron's formula
    area = np.sqrt(s * (s - PQ) * (s - QR) * (s - RP))

    # Compute the surface current density
    surface_current_density = ei * (psi) / (2 * area)
    return surface_current_density, area


# Make wire patterns for the particular triangular mesh
def make_wire_patterns(triangles, triangles_ji, nodes, height,  w=2e-3, g=1e-3, 
                       viewing=False, current =1, psi= 1, debug = False):
    ''' Make wire patterns for the triangular mesh with strips of rectangles. 
    Two uses: 
    1. input to magpy lib to make the magnet collection
    2. input to subsequent STL file
    '''
    # Initialize a magpy collection 
    planar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
    triangles = np.array(triangles)

    
    for i in range(triangles.shape[0]):
        triangle = triangles[i, :]
        triangle_ji = triangles_ji[i]
        if np.sum(triangle_ji) != 0:
            P = nodes[triangle[0]]
            Q = nodes[triangle[1]]
            R = nodes[triangle[2]]
        
        current_direction = np.sign(psi[i])
        if debug is True:
            print(Fore.GREEN + f'Current direction: {current_direction * current}' + Style.RESET_ALL)
        
        # Calculate the vectors PQ and PR
        PQ = Q - P
        PR = R - P
        
        # Calculate the number of strips that can fit within the triangle
        num_strips = int(np.linalg.norm(PQ) // (w + g))
        
        wire_patterns = []
        
        for i in range(num_strips):
            # Calculate the starting point of the strip
            start_point = P + i * (w + g) * (PQ / np.linalg.norm(PQ))
            
            # Calculate the end point of the strip
            end_point = start_point + w * (PQ / np.linalg.norm(PQ))
            
            # Create the rectangle strip
            strip = [start_point, end_point, end_point + PR, start_point + PR]
            wire_patterns.append(strip)
            
            start_point_magpy = [start_point[0] + (0.5 * w), start_point[1] + (0.5 * w), height]
            end_point_magpy = [end_point[0] + (0.5 * w), end_point[1] + (0.5 * w), height]
            
            
            # Add this to the magpy lib collection for both plates
            planar_coil_pattern.add(magpy.current.Polyline(current=current * current_direction, 
                                                           vertices = [start_point_magpy, end_point_magpy]))
    if viewing is True:
        planar_coil_pattern.show(backend='matplotlib', style='wire', style_color='k')
    return planar_coil_pattern

def visualize_gradient_coil(biplanar_coil_pattern):
    ax = plt.figure().add_subplot(111, projection='3d')
    if (len(biplanar_coil_pattern)) > 1: # expects two plates
        for plate in range(len(biplanar_coil_pattern)): 
            for wire in range(len(biplanar_coil_pattern.children[plate].children)):
                wire = biplanar_coil_pattern.children[plate].children[wire]
                if wire.current > 0:
                    color = 'r'
                else:
                    color = 'b'
                
                vertices = np.array(wire.vertices)
                ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color)
        plt.show()
        
            # mag_cuboid_trace = mag_collection_template.children[child1].get_trace()
            # vertices = np.array([mag_cuboid_trace['x'], mag_cuboid_trace['y'], mag_cuboid_trace['z']]).T + cube_magnet.position.T  #cube_magnet.orientation.apply(cube_magnet.position.T)
    





# Compute the resistance of the triangular mesh
def compute_resistance(j_triangle, area_triangle,  p=2, fact_include = False,
                       resistivity=1.68e-8, current = 10, thickness = 1.6):
    
    ''' Compute the resistance of the triangular mesh. '''
    resistance = np.linalg.norm(j_triangle, ord=np.inf, axis=0) * area_triangle * np.linalg.norm(j_triangle, ord=p, axis=0)
    if fact_include is True:
        resistance = resistance *  resistivity / (thickness * current**2)
  
    return resistance


# Compute the power dissipation of the planar gradient coil


# Visualize wire patterns for the planar gradient coil
def plot_wire_patterns(wire_patterns):
    ''' Visualize wire patterns for the planar gradient coil. '''
    
    if wire_patterns is not None:
        fig, ax = plt.subplots()
        
        for wire_pattern in wire_patterns:
            for i in range(len(wire_pattern)):
                ax.plot([wire_pattern[i][0], wire_pattern[(i + 1) % len(wire_pattern)][0]], 
                        [wire_pattern[i][1], wire_pattern[(i + 1) % len(wire_pattern)][1]], 'k-')
        
        ax.set_aspect('equal')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Wire Patterns of Planar Gradient Coil')
        plt.show()
    pass

def get_magnetic_field(magnets, sensors, axis = None):
    B = sensors.getB(magnets)
    if axis is None:
        B_eff = np.linalg.norm(B, axis=2) # interested only in Bz for now; concomitant fields later
    else:
        B_eff = np.squeeze(B[:, axis]) 
    return B_eff

def get_surface_current_density(triangles, nodes, psi,p=2):
    ''' Get the surface current density of the triangular mesh. '''
    J = 0
    triangle_ji = []
    triangle_area = []
    triangles_used = []
    if triangles is not None:
        for i in range(triangles.shape[0]):
        # for triangle, psi_i in triangles, psi:
            ji, area = get_surface_current_density_triangle(triangles[i, :], nodes, psi[i], p=p)
            if np.sum(ji) != 0:
                triangle_ji.append(ji)
                triangle_area.append(area)
                triangles_used.append(triangles[i, :])
    return triangles_used, triangle_ji, triangle_area

def cost_fn(B_grad, B_target, coil_resistance, coil_current, case = 'vector_BEM',
            p=2,  alpha = 0.1, beta = 0.1, weight = 1000):
    ''' Compute the cost function for the optimization problem. '''
    gammabar = 42.58e6
    if case == 'target_field':
        f1 = np.linalg.norm(B_grad - B_target, ord=p) * weight
        f2 = np.linalg.norm(B_grad - B_target, ord=np.inf) * weight
        f = f1 + f2
        # f = (100 * np.linalg.norm(B_grad - B_target, ord=np.inf))/ (np.linalg.norm(B_target, ord=np.inf)) 
        
    elif case == 'vector_BEM':
        f1 = alpha * coil_resistance
        f2 = beta *  (1 - alpha) * coil_current 
        f3 = (100 * np.linalg.norm(B_grad - B_target, ord=np.inf))/ (gammabar * np.linalg.norm(B_target, ord=np.inf)) 
        f = f1 + f2 + f3
        
    elif case == 'resistance_BEM':  
        f  =  alpha * coil_resistance
    
    elif case == 'J_BEM': 
        f = beta *  (1 - alpha) * coil_current 
    
    return f

def compute_constraints(B_grad, B_target, B_tol, current, wire_thickness, J_max, gammabar = 42.58e6):
         B_term = (100 * np.linalg.norm(B_grad - B_target, ord=np.inf))/ (gammabar * np.linalg.norm(B_target, ord=np.inf))  
         g1 = B_term - B_tol
         
         J_tol = current / wire_thickness 
         g2 = J_max - J_tol
         
         #---------------------------------------------------------------
         # Need to include Lorentz force constraints here
          
         return g1, g2
              

def prepare_vars(num_triangles, types = ['Choice'], options = [[0]]):
    vars = dict()
    for var in range(num_triangles):
        vars[f"x{var:02}"] = Choice(options=options[0])
    return vars


def display_scatter_3D(x, y, z, B, center:bool=False, title:str=None, clim_plot = None, vmin = 0.265, vmax = 0.271):
    
    if center is True:
        x = (x - 0.5 *  np.max(x)) 
        y = (y - 0.5 * np.max(y)) 
        z = (z - 0.5 * np.max(z)) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # img = ax.scatter(x * 1e3, y * 1e3, z * 1e3, c=B, cmap='jet', vmin = vmin, vmax = vmax) #coolwarm
    
    img = ax.scatter(x * 1e3, y * 1e3, z * 1e3, c=B, cmap='jet') #coolwarm
    
    plt.title(title)
    plt.colorbar(img)
    # plt.xticks([-16, 0, 16])
    # plt.yticks([-16, 0, 16])
    # ax.set_zticks([-16, 0, 16])
    # plt.axis('off')
    plt.show()
    
def create_magpy_sensors(grad_dir, grad_max, dsv, res, viewing, symmetry):
    x, y, z, Bz_target = set_target_field(grad_dir=grad_dir, grad_max=grad_max, dsv=dsv, res=res, viewing=viewing, symmetry=symmetry)

    #---------------------------------------------------------------
    # Set up the sensors in magpy lib
    pos = np.zeros((x.shape[0], 3))
    pos[:, 0] = x # length
    pos[:, 1] = y # depth
    pos[:, 2] = z # height

    dsv_sensors = magpy.Collection(style_label='sensors')
    sensor1 = magpy.Sensor(position=pos,style_size=2)
    dsv_sensors.add(sensor1)
    print(Fore.GREEN + 'Done creating position sensors' + Style.RESET_ALL)
    
    return dsv_sensors, pos, Bz_target
    