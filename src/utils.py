'''This file contains the utility functions for the planar gradient coil design problem.'''
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from pymoo.core.variable import Real, Integer, Choice, Binary
from colorama import Fore, Style
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
from scipy.interpolate import interp2d 


# Design the target magnetic field - Bz
def set_target_field(grad_dir:str ='x', grad_max:float = 27, dsv:float = 30, res:float = 2, 
                     symmetry = True, viewing = False, normalize = True):
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
    elif grad_dir == 'y':
        Bz = Bz_max * Y / (0.5 * dsv)
        if symmetry is True:
            mask1 = X < 0 # consider only the positive of the other two dimensions
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
        if grad_dir == 'z':
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
    
    if normalize is True:
        Bz = 100 * Bz / np.max(Bz)  # This should put the range from -100 to 100

     
    if viewing is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X, Y, Z, c=Bz, cmap='jet')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.title('Target Bz')
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
    
    psi_P = psi[triangle[0]]
    psi_Q = psi[triangle[1]]
    psi_R = psi[triangle[2]]
    
    # Compute the edge lengths
    PQ = np.linalg.norm(P - Q,ord=p, axis=0)
    QR = np.linalg.norm(Q - R,ord=p, axis=0)
    RP = np.linalg.norm(R - P,ord=p, axis=0)
    
    ei = np.max([PQ, QR, RP], axis=0) # magntiude of the longest edge
    
    # Get the direction of ei
    if ei == PQ:
        ei = P - Q
        psi = psi_R - psi_P
    elif ei == QR:
        ei = Q - R
        psi = psi_P - psi_Q
    else:
        ei = R - P
        psi = psi_Q - psi_R
        
    
    # Calculate the semi-perimeter
    s = (PQ + QR + RP) / 2
    
    # Calculate the area using Heron's formula
    area = np.sqrt(s * (s - PQ) * (s - QR) * (s - RP))

    # Compute the surface current density
    surface_current_density = ei * (psi) / (2 * area)
    return surface_current_density, area

def make_wire_patterns_contours(nodes, psi, levels, current, grad_dir = 'x',
                                wire_width = 1.4e-3, wire_gap = 0.7e-3, 
                                viewing = False):
    planar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')

    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    wire_patterns = []
    contours = psi2contour(x, y, psi, levels_values = levels, grad_dir = grad_dir, viewing = viewing )
    wire_smoothness = 0
    
    for collection, level  in zip(contours.collections, contours.levels):
        paths = collection.get_paths()
        # print(Fore.GREEN + f'Level: {level}' + Style.RESET_ALL)
        current_direction = np.sign(level)
        for path in paths:
            vertices = path.vertices
        
            # if  is_closed(vertices) is False:
            #     vertices = np.vstack((vertices, vertices[0, :]))
            loop_vertices_wire_widths = np.array([vertices[:, 0], vertices[:, 1], z[0] * np.ones(vertices[:, 0].shape)]).T
        
            # store all loops in a variable
            # wire_patterns.append(loop_vertices_wire_widths)
            
            # increase wire thickness based on level
            # loop_vertices_wire_widths = gen_wire_vertices(loop_vertices_wire_widths, level, wire_width, wire_gap)
            
            # Check for all loops in wire_patterns if the current loop is overalapping with any coordinates of previous loops, if so exclude it
            overlap = False
            # loop1 = list(loop_vertices_wire_widths[:, :2])
            # for previous_pattern in wire_patterns[:-1]:
            #     loop2 = list(previous_pattern[:, :2])
            #     if check_intersection(loop1, loop2):
            #         overlap = True
            #         break

            if overlap is False:
                gradient_x = np.gradient(loop_vertices_wire_widths[:, 0])
                gradient_y = np.gradient(loop_vertices_wire_widths[:, 1])
                
                wire_smoothness += np.linalg.norm(np.sqrt(gradient_x**2 + gradient_y**2))
                planar_coil_pattern.add(magpy.current.Polyline(current=current * current_direction, 
                                                        vertices = loop_vertices_wire_widths))  
        # else:
        #     print(Fore.RED + 'Path is not closed' + Style.RESET_ALL)
    
    return planar_coil_pattern, wire_smoothness




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

def visualize_gradient_coil(biplanar_coil_pattern, save = False, fname_save='coil_design.csv'):
    ''' Visualize the wire patterns of the planar gradient coil. '''
    ax = plt.figure().add_subplot(111, projection='3d')
    if save is True:
        vertices_write = []
    if (len(biplanar_coil_pattern)) > 1: # expects two plates

            
        for plate in range(len(biplanar_coil_pattern)): 
            for wire in range(len(biplanar_coil_pattern.children[plate].children)):
                wire_pattern = biplanar_coil_pattern.children[plate].children[wire]
                if wire_pattern.current > 0:
                    color = 'r'
                else:
                    color = 'b'
                
                vertices = np.array(wire_pattern.vertices)
                if save is True:
                    if wire_pattern.current > 0:
                        vertices_write.append(np.hstack((np.ones((vertices.shape[0], 1)), 1e3 * vertices)))
                    else:
                        vertices_write.append(np.hstack((-1 * np.ones((vertices.shape[0], 1)),1e3 * vertices)))
                    
                        
                        
                        
                ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color)
        if save is True:
            np.savetxt(fname_save, np.vstack(vertices_write), delimiter=',')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()
        
          
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

def get_magnetic_field(magnets, sensors, axis = None, normalize = True):
    B = sensors.getB(magnets)
    if axis is None:
        B_eff = np.linalg.norm(B, axis=2) # interested only in Bz for now; concomitant fields later
    else:
        B_eff = np.squeeze(B[:, axis]) 
        
    if normalize is True:
        B_eff = 100 * B_eff / np.max(B_eff) # This should put the range from -100 to 100
    
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
            ji, area = get_surface_current_density_triangle(triangles[i, :], nodes, psi, p=p)
            if np.sum(ji) != 0:
                triangle_ji.append(ji)
                triangle_area.append(area)
                triangles_used.append(triangles[i, :])
    return triangles_used, triangle_ji, triangle_area

def cost_fn(B_grad, B_target,  psi_smoothness, wire_smoothness,
            case = 'target_field', coil_resistance=0, coil_current=0,
            p=2,  alpha = [0.1], beta = 0.1, weight = 1):
    ''' Compute the cost function for the optimization problem. '''

    if case == 'target_field':
        # print(Fore.GREEN + ' max B_grad: ', np.max(B_grad), Style.RESET_ALL)
        f0 = 100 * np.linalg.norm(B_grad - B_target, ord=np.inf) / (np.linalg.norm(B_target, ord=np.inf)) # minimizing peak field
        f1 = wire_smoothness # 100 * np.linalg.norm(np.divide((B_grad - B_target), B_target), ord=p) * weight # target field method
        # avoiding overlaps in the wire patterns by enforcing a smooth transition
        # N_plate = int(psi.shape[0] * 0.5)
        f2 = np.abs(psi_smoothness)
        f3 = coil_resistance # minimizing resistance
        f4 = coil_current # minimizing peak current
        
        if alpha.shape[0] > 1:
            f = (alpha[0] * f0) + (alpha[1] * f1) + (alpha[2] * f2) + (alpha[3] * f3) 
            + (alpha[4] * f4)

            # print(Fore.YELLOW + 'f0: ', alpha[0] * f0, 'f1: ', alpha[1] * f1, 'f2: ', alpha[2] * f2, 'f3: ', alpha[3] * f3, 'f4: ', alpha[4] * f4, Style.RESET_ALL)
        else:
            f = [f0, f1, f2, f3, f4]
            # print(f0, f1, f2)
            
    elif case == 'vector_BEM':
        f1 = alpha * coil_resistance
        f2 = beta *  (1 - alpha) * coil_current 
        f3 = (100 * np.linalg.norm(B_grad - B_target, ord=np.inf))/ (gammabar * np.linalg.norm(B_target, ord=np.inf)) 
        f = f1 + f2 + f3
        
    elif case == 'resistance_BEM':  
        f  =  alpha * coil_resistance
    
    elif case == 'J_BEM': 
        f = beta *  (1 - alpha) * coil_current 
    
    f = np.array(f)
    f[np.isnan(f)] = np.inf
    return f

def compute_constraints(B_grad, B_target, B_tol, current, wire_thickness, J_max, gammabar = 42.58e6):
         B_term = (100 * np.linalg.norm(B_grad - B_target, ord=np.inf))/ (gammabar * np.linalg.norm(B_target, ord=np.inf))  
         g1 = B_term - B_tol
         
         J_tol = current / wire_thickness 
         g2 = J_max - J_tol
         
         #---------------------------------------------------------------
         # Need to include Lorentz force constraints here
          
         return g1, g2
              

def prepare_vars(num_psi, types = ['Real'],  num_levels =10, options = [[0]]):
    vars = dict()
    if types[0] == 'Real':
        for var in range(num_psi):
            vars[f"x{var:02}"] = Real(bounds=(options[0], options[1]))
        # for var in range(num_psi, num_psi + num_levels):
        #     vars[f"x{var:02}"] = Real(bounds=(options[2], options[3]))
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
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
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

def psi2contour(x, y, psi, grad_dir = 'x', levels_values = np.linspace(-1, 1, 10), viewing = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  

    # Need to shape the stream function to expedite convergence
    if grad_dir == 'x':
        weights = (np.sin(np.pi * (x)/np.max(np.abs(x))))
        # psi = (psi) * np.sign(x) * weights
        psi = (psi) *  weights
    elif grad_dir == 'y':
        weights = (np.sin(np.pi * (y)/np.max(np.abs(y))))
        psi = (psi) *  weights
        
    xi = np.linspace(min(x), max(x), 100) # Hardocded but good enough for now
    yi = np.linspace(min(y), max(y), 100)
    psi_2D = griddata((x, y), psi, (xi[None,:], yi[:,None]), method='cubic')
    X, Y = np.meshgrid(xi, yi)
    # levels_values_sorted = np.unique(np.round(np.sort(levels_values), decimals=1))
    levels_values_sorted = np.sort(levels_values)
    contours = ax.contour(X, Y, psi_2D, levels = levels_values_sorted , cmap='jet')
    
    
    plt.close()
    # viewing = True
    if viewing is True:
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.plot_surface(X, Y, psi_2D, cmap='jet')
       plt.show()
       
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')  
       ax.contour(X, Y, psi_2D, levels = np.sort(levels_values),cmap='jet')
       plt.show()
    
    return  contours


def gen_wire_vertices(vertices, level, wire_width, wire_gap):
    ''' Generate the vertices of the wire patterns. '''
    vertices_wire_widths = []
    for vertex in vertices:
        x, y, z = vertex
        
        # Thicken the line by a factor
        factor = np.abs(level) * wire_width / 2
        x_thick = x + factor * np.cos(level)
        y_thick = y + factor * np.sin(level)
        
        # Check if the new line thickness exceeds wire_width
        if np.linalg.norm([x_thick - x, y_thick - y]) > wire_width:
            x_thick += wire_gap * np.cos(level)
            y_thick += wire_gap * np.sin(level)
        
        vertices_wire_widths.append([x_thick, y_thick, z])

    return np.array(vertices_wire_widths)
    



def check_intersection(loop1, loop2):
    """
    Checks if two loops intersect.

    Args:
        loop1 (list of tuples): List of (x, y) coordinate tuples representing the first loop.
        loop2 (list of tuples): List of (x, y) coordinate tuples representing the second loop.

    Returns:
        bool: True if the loops intersect, False otherwise.
    """

    for i in range(len(loop1)):
        for j in range(len(loop2)):
            p1, p2 = loop1[i], loop1[(i + 1) % len(loop1)]
            q1, q2 = loop2[j], loop2[(j + 1) % len(loop2)]

            if do_intersect(p1, p2, q1, q2):
                return True

    return False

def do_intersect(p1, p2, q1, q2):
    """
    Checks if two line segments intersect.

    Args:
        p1 (tuple): (x, y) coordinates of the first point of the first segment.
        p2 (tuple): (x, y) coordinates of the second point of the first segment.
        q1 (tuple): (x, y) coordinates of the first point of the second segment.
        q2 (tuple): (x, y) coordinates of the second point of the second segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    # Special cases (collinear)
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

def on_segment(p, q, r):
    """
    Checks if point q lies on segment pr.

    Args:
        p (tuple): (x, y) coordinates of the first point.
        q (tuple): (x, y) coordinates of the second point.
        r (tuple): (x, y) coordinates of the third point.

    Returns:
        bool: True if q lies on segment pr, False otherwise.
    """

    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
def get_psi_smoothness(nodes, psi):
    ''' Compute the smoothness of the stream function. '''
    
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    r = np.sqrt(x**2 + y**2)
    
    d_psi_r = np.gradient(psi, r)
    d_psi_r = np.nan_to_num(d_psi_r, nan=0, posinf=0, neginf=0)
    
    # psi_smoothness = np.linalg.norm(d_psi_dx, ord=1) + np.linalg.norm(d_psi_dy, ord=1)  #TV like norm
    psi_smoothness = np.linalg.norm(d_psi_r, ord=1) #TV like norm
    
    return psi_smoothness


def fit_ellipse_contour(vertices, oversampling = 2):
    ''' Fit an ellipse to the given vertices and extract N vertices of the fitted ellipse. '''
    N = vertices.shape[0] * oversampling
    

    # Convert vertices to a format suitable for OpenCV
    vertices = np.array(vertices, dtype=np.float32)
    vertices = vertices.reshape(-1, 1, 2)

    # Fit an ellipse to the vertices
    ellipse = cv2.fitEllipse(vertices)

    # Extract parameters of the fitted ellipse
    center, axes, angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # semi-major and semi-minor axes

    # Generate N vertices of the fitted ellipse
    theta = np.linspace(0, 2 * np.pi, N)
    ellipse_vertices = np.zeros((N, 2))

    for i in range(N):
        x = a * np.cos(theta[i])
        y = b * np.sin(theta[i])

        # Rotate the point by the ellipse angle
        x_rot = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
        y_rot = x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))

        # Translate the point to the ellipse center
        ellipse_vertices[i, 0] = x_rot + center[0]
        ellipse_vertices[i, 1] = y_rot + center[1]
      

    return ellipse_vertices
  


def is_polygon(vertices):
    try:
        Polygon(vertices)
        return True
    except ValueError:
        return False

def is_closed(vertices):
    dist =  np.sum(vertices[0] - vertices[-1])
    if dist < 1e-3:
        return True
    else:
        return False
    
def get_stream_function(grad_dir ='x', x =None, y=None, viewing = False):
    if grad_dir == 'x':
        X, Y = np.meshgrid(x, y)
        stream_function = (np.sin(np.pi * (X)/np.max(np.abs(X))))
       
    elif grad_dir == 'y':
        X, Y = np.meshgrid(x, y)
        stream_function = (np.sin(np.pi * (Y)/np.max(np.abs(Y))))
       
       
    if viewing is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, stream_function, cmap='jet')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.show()
    return stream_function

def get_wire_patterns_contour_rect(psi, levels, stream_function, x, y, z, current = 1):
    planar_coil_pattern = magpy.Collection(style_label='coil', style_color='r')
    wire_smoothness = 0
    contours = psi2contour_rect(x, y, psi, stream_function, levels = levels, viewing = False)

    for collection, level  in zip(contours.collections, contours.levels):
        paths = collection.get_paths()
        current_direction = np.sign(level)
        for path in paths:
            vertices = path.vertices
            loop_vertices_wire_widths = np.array([vertices[:, 0], vertices[:, 1], z * np.ones(vertices[:, 0].shape)]).T
            gradient_x = np.gradient(loop_vertices_wire_widths[:, 0])
            gradient_y = np.gradient(loop_vertices_wire_widths[:, 1])
                
            wire_smoothness += np.linalg.norm(np.sqrt(gradient_x**2 + gradient_y**2))
            planar_coil_pattern.add(magpy.current.Polyline(current=current * current_direction, 
                                                        vertices = loop_vertices_wire_widths)) 
    return planar_coil_pattern, wire_smoothness


def psi2contour_rect(x, y, psi, stream_function, levels, viewing = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  

    # Need to shape the stream function to expedite convergence
    psi_weighted = psi * stream_function.T # TODO - figure out 
    X, Y = np.meshgrid(x, y)
    
    if x.shape[0] < 100:
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        f = RegularGridInterpolator((x, y), psi_weighted, method='cubic')
        X, Y = np.meshgrid(xi, yi)
        psi_weighted_rsz = f((X, Y))
    else:
        psi_weighted_rsz = psi_weighted
        
    
    levels_values = np.linspace(-1, 1, int(levels)) # if we want to only determine psi and not both

    
    contours = ax.contour(X, Y, psi_weighted_rsz, levels = levels_values , cmap='jet')
    plt.close()
    # viewing = True
    
    if viewing is True:
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.plot_surface(X, Y, psi_weighted_rsz, cmap='jet')
       plt.xlabel('X (mm)')
       plt.ylabel('Y (mm)')
       plt.show()
       
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')  
       ax.contour(X, Y, psi_weighted_rsz, levels = np.sort(levels_values),cmap='jet')
       plt.xlabel('X (mm)')
       plt.ylabel('Y (mm)')
       plt.show()
    
    return  contours