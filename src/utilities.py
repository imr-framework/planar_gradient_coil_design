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

# ----------------------
# generate the initial psi 
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
def psi2contour(x, y, psi, stream_function, levels, viewing = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    psi = np.reshape(psi, (x.shape[0], y.shape[0]))
    # Need to shape the stream function to expedite convergence
    psi_weighted = psi * stream_function # TODO - figure out 
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