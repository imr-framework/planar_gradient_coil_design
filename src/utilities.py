# ----------------------
# Import the libraries
import numpy as np
import magpylib as magpy
from colorama import Fore, Style
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.core.variable import Real
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString
import numpy as np
from stl import mesh
import pandas as pd
# ----------------------
# Create the sensors for magpy

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

def set_target_field(grad_dir:str ='x', grad_max:float = 27, dsv:float = 30, res:float = 2, 
                     symmetry = True, viewing = False, normalize = False):
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
# Define the 2D Gaussian function
def gaussian_2d(x, y, amplitude, center_x, center_y, sigma_x, sigma_y):
    exponent = -((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2))
    return amplitude * np.exp(exponent)
# ----------------------
# generate the initial psi 
def get_stream_function(grad_dir ='x', x =None, y=None, viewing = False, symmetry = False, shape='circle'):
    amplitude_pos = 1
    amplitude_neg = -1
    threshold_pos = 0.5
    threshold_neg = -0.5


    if grad_dir == 'x' and symmetry is False:
        # Generate the Gaussian surfaces
        center_x_pos = 0.5 *  np.max(x)
        center_y_pos = 0
        center_x_neg = -0.5 *  np.max(x)
        center_y_neg = 0
        if shape == 'circle':
            X, Y = circular_meshgrid(np.max(x), x.shape[0])
            sigma_x =   0.35 * np.max(x) # 0.5
            sigma_y = 1 * np.max(y) # 0.5
        elif shape == 'square':   
            X, Y = np.meshgrid(x, y)
            sigma_x = 0.5 * np.max(x) # 0.5
            sigma_y = 0.5 * np.max(y) # 0.5
            
        Z_pos = gaussian_2d(X, Y, amplitude_pos, center_x_pos, center_y_pos, sigma_x, sigma_y)
        Z_neg = gaussian_2d(X, Y, amplitude_neg, center_x_neg, center_y_neg, sigma_x, sigma_y)

        Z_pos = Z_pos / np.max(np.abs(Z_pos)) # otherwise psi_optimum will never scale to max when weighted
        Z_neg = Z_neg / np.max(np.abs(Z_neg)) # otherwise psi_optimum will never scale to max when weighted
       
        Z_pos[Z_pos > threshold_pos] = 1
        Z_neg[Z_neg < threshold_neg] = -1
        if shape == 'circle':
            # Z_pos = Z_pos / np.max(np.abs(Z_pos)) # otherwise psi_optimum will never scale to max when weighted
            # Z_neg = Z_neg / np.max(np.abs(Z_neg)) # otherwise psi_optimum will never scale to max when weighted
            stream_function = Z_pos + Z_neg
            
            # stream_function[0, :] = 0
            # stream_function[-1, :] = 0
            # stream_function[:, 0] = 0
            # stream_function[:, -1] = 0   
            
        elif shape == 'square':
            stream_function = Z_pos + Z_neg  

            # stream_function[0, :] = 0
            # stream_function[-1, :] = 0
            # stream_function[:, 0] = 0
            # stream_function[:, -1] = 0   

    elif grad_dir == 'x' and symmetry is True:  
        x_new = np.linspace(0, np.max(x), int(0.5 * x.shape[0]))
        X, Y = np.meshgrid(x_new, y)
        # Generate the Gaussian surfaces
        center_x_pos = 0.5 *  np.max(x)
        center_y_pos = 0
        sigma_x = 0.4 * np.max(x) 
        sigma_y = 50 * np.max(y) # 0.5
        
        stream_function = gaussian_2d(X, Y, amplitude_pos, center_x_pos, center_y_pos, sigma_x, sigma_y)
        # stream_function = stream_function / np.max(stream_function) # otherwise psi_optimum will never scale to max when weighted
        stream_function[0, :] = 0
        stream_function[-1, :] = 0
        stream_function[:, 0] = 0
        stream_function[:, -1] = 0
        
        
    elif grad_dir == 'y': # This is a useless case as x-grad rotated by 90 degrees is the same as y-grad
      print('This is a useless case as x-grad rotated by 90 degrees is the same as y-grad; please use x-gradient')
      return None
    
    elif grad_dir == 'z': 
        X, Y = np.meshgrid(x, y)
        # Generate the Gaussian surfaces
        center_x = 0
        center_y = 0
        sigma_x = 0.5 * np.max(x)
        sigma_y = 0.5 * np.max(y)
        stream_function = gaussian_2d(X, Y, amplitude_pos, center_x, center_y, sigma_x, sigma_y)
    
        stream_function[0, :] = 0
        stream_function[-1, :] = 0
        stream_function[:, 0] = 0
        stream_function[:, -1] = 0
    
     
    # stream_function = np.pad(stream_function, pad_width=1, mode='constant', constant_values=0)[1:-1, 1:-1]  # ensuring that the function always ends at 0 
    if viewing is True:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, stream_function, cmap='jet')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Stream function  - initial weights')

        plt.show()
        
    return stream_function

# ----------------------
# Compute resulting magnetization 

def get_magnetic_field(magnets, sensors, axis = None, normalize = False):
    B = sensors.getB(magnets)
    if axis is None:
        B_eff = np.linalg.norm(B, axis=2) # interested only in Bz for now; concomitant fields later
    else:
        B_eff = np.squeeze(B[:, axis]) 
        

    
    if normalize is True:
        B_eff = 100 * B_eff / np.max(np.abs(B_eff)) # This should put the range from -100 to 100
    
    return B_eff




# ----------------------
# 3D scatter plot of magnetic field
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
    # ax.set_zlabel('Z (mm)')
    
    # plt.xticks([-16, 0, 16])
    # plt.yticks([-16, 0, 16])
    # ax.set_zticks([-16, 0, 16])
    # plt.axis('off')
    plt.show()


#----------------------
# Prepare the variables for optimization

def prepare_vars(num_psi, types = ['Real'],  options = [[0]]):
    vars = dict()
    for num_var in range(len(types)):
        if types[num_var] == 'Real':
            for var in range(num_psi):
                vars[f"x{var:02}"] = Real(bounds=(options[0], options[1]))
                    
    return vars
# ----------------------
# convert the psi to contours
def psi2contour(x, y, psi, stream_function, levels, viewing = True, symmetry=False, filter = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    
    psi = np.reshape(psi, (x.shape[0], y.shape[0]))
    # Need to shape the stream function to expedite convergence
    psi_weighted = psi * stream_function # TODO - figure out 
    
    
    X, Y = np.meshgrid(x, y)
    
    if x.shape[0] < 100:
        xi = np.linspace(min(x), max(x), 512)
        yi = np.linspace(min(y), max(y), 512)
        f = RegularGridInterpolator((x, y), psi_weighted, method='cubic')
        X, Y = np.meshgrid(xi, yi)
        psi_weighted_rsz = f((X, Y))
    else:
        psi_weighted_rsz = psi_weighted
        
    ramp_length = 64 #112
    
    x_begin_ramp = np.linspace(0, psi_weighted_rsz[ramp_length, :], ramp_length) 
    x_end_ramp = np.linspace(psi_weighted_rsz[-ramp_length, :], 0, ramp_length)
    y_begin_ramp = np.linspace(0, psi_weighted_rsz[:, ramp_length], ramp_length)
    y_end_ramp = np.linspace(psi_weighted_rsz[:, -ramp_length], 0, ramp_length)
    
    
    psi_weighted_rsz[:ramp_length, :] = x_begin_ramp
    psi_weighted_rsz[-ramp_length:, :] = x_end_ramp
    psi_weighted_rsz[:, :ramp_length] = y_begin_ramp.T
    psi_weighted_rsz[:, -ramp_length:] = y_end_ramp.T
    
    
    
    
    psi_weighted_rsz[:2, :] = 0
    psi_weighted_rsz[-2:, :] = 0
    psi_weighted_rsz[:, :2] = 0
    psi_weighted_rsz[:, -2:] = 0
    
    # low pass filter psi_weighted_rsz so that there are no sharp transitions near the edges
    if filter is True:
        kernel = np.ones((3, 3)) / 9
        psi_weighted_rsz = np.convolve(psi_weighted_rsz.flatten(), kernel.flatten(), mode='same').reshape(psi_weighted_rsz.shape)
     
    # f = RegularGridInterpolator((xi, yi), psi_weighted_rsz, method='cubic')
    # X, Y = np.meshgrid(xi, yi)
    # psi_weighted_rsz = f((X, Y))
    

    min_level = np.min(psi_weighted_rsz)
    max_level = np.max(psi_weighted_rsz)
    # min_level = -1
    # max_level = 1
    
    levels_values = np.linspace(min_level, max_level, int(levels)) # if we want to only determine psi and not both
    # changed this to make sure we have even distribution of levels
    # levels_values = np.linspace(-1, 1, int(levels)) # if we want to only determine psi and not both
    
    contours = ax.contour(X, Y, psi_weighted_rsz, levels = levels_values , cmap='jet')
    
                
    plt.close(fig)

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
# ----------------------
# Convert the psi to wire patterns



def identify_loops(vertices, loop_tolerance = 5):
    vertices_collated = []
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    loop_starts = np.where(dist > loop_tolerance)[0] + 1
    
    if len(loop_starts) == 0:
        vertices_collated.append(vertices)
    else:
        loop_starts = np.insert(loop_starts, 0, 0)
        loop_starts = np.append(loop_starts, len(x))
        for i in range(len(loop_starts) - 1):
            vertices_collated.append(vertices[loop_starts[i]:loop_starts[i + 1], :])
    
    return vertices_collated
    
def check_intersection(curve1, curve2):
    ''' Check if two loops intersect. '''
    curve1 = LineString(curve1)
    curve2 = LineString(curve2)
    return curve1.crosses(curve2) 


def get_psi_smoothness(vars, num_psi_weights, psi_init, mesh):
    
    vars = np.array([vars[f"x{child:02}"] for child in range(0, 2 * num_psi_weights)]) # all children should have same magnet positions to begin with
            
    psi_flatten_upper_plate = vars[:num_psi_weights]
    psi_flatten_lower_plate = vars[num_psi_weights:]
    
    psi_upper_plate = psi_flatten_upper_plate.reshape(mesh, mesh) * psi_init
    psi_lower_plate = psi_flatten_lower_plate.reshape(mesh, mesh) * psi_init
        
    grad_psi_upper_x, grad_psi_upper_y = np.gradient(psi_upper_plate)
    grad_psi_lower_x, grad_psi_lower_y = np.gradient(psi_lower_plate)

    psi_smoothness = np.sum(np.sqrt(grad_psi_upper_x**2 + grad_psi_upper_y**2)) + np.sum(np.sqrt(grad_psi_lower_x**2 + grad_psi_lower_y**2))
    
    return psi_smoothness

def get_psi_volumes(vars, num_psi_weights, psi_init, mesh, debug = True):
    
    vars = np.array([vars[f"x{child:02}"] for child in range(0, 2 * num_psi_weights)]) # all children should have same magnet positions to begin with
            
    psi_flatten_upper_plate = vars[:num_psi_weights]
    psi_flatten_lower_plate = vars[num_psi_weights:]
    
    psi_upper_plate = psi_flatten_upper_plate.reshape(mesh, mesh) * psi_init
    psi_lower_plate = psi_flatten_lower_plate.reshape(mesh, mesh) * psi_init
    if debug is True:
        print('')
    # The total volume should be zero for the positive and negative volumes should cancel out
    psi_volumes_upper = np.trapz(np.trapz(psi_upper_plate, axis=0), axis=0)
    psi_volumes_lower = np.trapz(np.trapz(psi_lower_plate, axis=0), axis=0)
    
    psi_volumes = psi_volumes_upper + psi_volumes_lower
    psi_volumes = 0
    
    return psi_volumes

def connect_contours_to_spiral(contours, gap_pts=3):
    spiral = []
    for contour in contours.collections:
        for path in contour.get_paths():
            vertices = path.vertices * 1e3 # convert to mm
            spiral.extend(vertices[:-gap_pts])
                
            # spiral2 = np.array(spiral) 
            # plt.plot(spiral2[:, 0], spiral2[:, 1], 'ro-')
            # plt.show()
            
    
    return np.array(spiral)

def impose_symmetry(psi, x=None, y = None, viewing = False):
    ''' Impose symmetry on the psi function by adding the mirror. '''
    psi_sym = np.zeros([psi.shape[0] * 2, psi.shape[1]], dtype=psi.dtype)
    psi_sym[psi.shape[0]:, :psi.shape[1]] = psi
    psi_sym[:psi.shape[0], :psi.shape[1]] = -psi #[::-1, :]
    # psi_sym = psi_sym.T

    if viewing is True:
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, psi_sym, cmap='jet')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.show()
    return psi_sym
# ----------------------
    
# Example usage:
# contours = psi2contour(x, y, psi, stream_function, levels, viewing=True)
# spiral = connect_contours_to_spiral(contours)
# ----------------------------
# Function to write x and y coordinates to G-code
# x and y are in m 

def write_gcode(filename="output.nc", grad:dict = None, depths:np.ndarray = [0.01]):
    

    # check if the filename ends with .nc
    if not filename.endswith(".nc"):
        raise ValueError("Filename must end with .nc")
    
    # convert the x and y arrays to mm and int
    # x = np.array(x) * 1e3
    # y = np.array(y) * 1e3
    # x = x.astype(int)
    # y = y.astype(int)
    
    
    
    
    with open(filename, "w") as file:
        file.write("; G-code generated from X and Y values\n")
        file.write("G21 ; Set units to millimeters\n")
        file.write("G90 ; Absolute positioning\n")
        file.write("M3 S1000 ; Start spindle at 1000 RPM\n")
        # item = -1
        for depth in range(depths.shape[0]):
            for key, (x, y) in grad.items():
                file.write(f"; {key} coordinates\n")
                # check if the x and y arrays are of the same length
                if len(x) != len(y):
                    raise ValueError("x and y arrays must have the same length")
                # item += 1
                file.write(f"G0 Z5 ; Move to safe height\n")
                file.write(f"G0 X0 Y0 ; Move to start position\n")
                file.write(f"G0 X{x[0]:.3f} Y{y[0]:.3f}\n") 
                file.write(f"G0 Z-{depths[depth]:.3f} ; Move down to cutting depth\n")
                file.write(f"G1 F1000 ; Set feed rate\n")
                # write the coordinates to the file
                
                # check if the x and y arrays are of the same length
                if len(x) != len(y):
                    raise ValueError("x and y arrays must have the same length")
            
                for i in range(len(x)):
                    file.write(f"G1 X{x[i]:.3f} Y{y[i]:.3f}\n")

                file.write(f"G0 Z5 ; Move to safe height\n")
                file.write(f"G0 X0 Y0 ; Move to start position\n")
        file.write("M30 ; End of program\n")
    print(f"G-code written to {filename}")
    
    
def circle(radius, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y

def square(radius, num_points=100):
    """
    Returns the coordinates of a square with a given radius.

    Parameters:
        radius (float): Radius of the square.
        num_points (int): Number of points along each side of the square.

    Returns:
        tuple: Two arrays representing the x and y coordinates of the square.
    """
    x = np.concatenate((np.linspace(-radius, radius, num_points), np.full(num_points, radius), 
                        np.linspace(radius, -radius, num_points), np.full(num_points, -radius)))
    y = np.concatenate((np.full(num_points, -radius), np.linspace(-radius, radius, num_points),
                        np.full(num_points, radius), np.linspace(radius, -radius, num_points)))
                        
    return x, y

def diameter(radius, num_points=360, axis='x'):
        """
        Returns the coordinates of the diameter of a circle at x=0 or y=0.

        Parameters:
            radius (float): Radius of the circle.
            num_points (int): Number of points along the diameter.
            axis (str): Axis along which the diameter is defined ('x' or 'y').

        Returns:
            tuple: Two arrays representing the x and y coordinates of the diameter.
        """
        if axis == 'x':
            x = np.zeros(num_points)
            y = np.linspace(-radius, radius, num_points)
        elif axis == 'y':
            y = np.zeros(num_points)
            x = np.linspace(-radius, radius, num_points)
        else:
            raise ValueError("Axis must be 'x' or 'y'.")
        return x, y
    
    
def  balance_force_contours(contours):
    """
    Balances the forces on the contours by ensuring that the sum of the forces in each direction is zero.

    Parameters:
        contours (list): List of contours, where each contour is a list of points.

    Returns:
        list: List of balanced contours.
    """

    fig, ax = plt.subplots()
    balanced_contours = ax.contour([], [], [])
    for collection, level  in zip(contours.collections, contours.levels):
                paths = collection.get_paths()
                # current_direction = np.sign(level)
                if level >=0:
                   for path in paths:
                       vertices = path.vertices
                       balanced_contours.append({'vertices': vertices, 'level': level})
                       vertices_mirror = (vertices * np.array([-1, 1]))
                       balanced_contours.append({'vertices': vertices_mirror, 'level': -level})

                    
    return balanced_contours
# -----------------------------------------------------------------------------
def spiral_symmetry(spiral):
    x = spiral[:, 0]
    y = spiral[:, 1]
    
    x_pos = x[np.where(x >= 0)]
    y_pos = y[np.where(x >= 0)]
    
    
    # Create a new array for the mirrored points
    mirrored_x = -x_pos
    mirrored_y = y_pos
    # Combine the original and mirrored points
    spiral_mirror = np.vstack((x_pos, y_pos)).T
    spiral_mirror = np.vstack((spiral_mirror, np.vstack((mirrored_x, mirrored_y)).T))
    
    return spiral_mirror



def circular_meshgrid(radius, num_points):
  """
  Generates a circular meshgrid.

  Args:
    radius: The radius of the circle.
    num_points: The number of points along the radius and angle.

  Returns:
    A tuple (X, Y) of 2D arrays representing the x and y coordinates of the meshgrid.
  """
  r = np.linspace(0, radius, num_points)
  theta = np.linspace(0, 2 * np.pi, num_points)
  R, Theta = np.meshgrid(r, theta)

  X = R * np.cos(Theta)
  Y = R * np.sin(Theta)
  return X, Y



# X and Y now contain the coordinates of the circular meshgrid
# You can use these for plotting or other calculations

def write_stl(filename, x, y, z):
    """
    Writes an STL file using the provided x, y, and z coordinates.

    Parameters:
        filename (str): Name of the STL file to write.
        x (array-like): X coordinates of the vertices.
        y (array-like): Y coordinates of the vertices.
        z (array-like): Z coordinates of the vertices.
    """
    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("x, y, and z arrays must have the same length")

    # Create the vertices array
    vertices = np.column_stack((x, y, z))

    # Create the faces array (triangulation)
    faces = []
    for i in range(len(vertices) - 2):
        faces.append([0, i + 1, i + 2])
    faces = np.array(faces)

    # Create the mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j], :]

    # Write the STL file
    stl_mesh.save(filename)
    print(f"STL file written to {filename}")
    

def write_excel(fname, x, y, z):    
    """
    Write the x, y, z coordinates and magnetic field values to an Excel file.

    Parameters:
        fname (str): Name of the Excel file to write.
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        z (array-like): Z coordinates.
        Bz (array-like): Magnetic field values in the z direction.
        Bx (array-like): Magnetic field values in the x direction.
        By (array-like): Magnetic field values in the y direction.
    """


    # Create a DataFrame from the data
    data = {
        'X': x,
        'Y': y,
        'Z': z,
    }
    df = pd.DataFrame(data)
    # Write the DataFrame to an Excel file
    df.to_excel(fname, index=False)
    print(f"Excel file written to {fname}")