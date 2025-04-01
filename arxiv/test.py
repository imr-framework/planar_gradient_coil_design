import numpy as np

def lorentz_force_loop(current, loop_segments, magnetic_field):
    """
    Calculates the total Lorentz force on a loop of wire.

    Args:
        current (float): The current flowing in the loop (in Amperes).
        loop_segments (np.ndarray): An array of line segments
            representing the loop, where each segment is a 
            [start_point, end_point] (each point is a 3D vector).
        magnetic_field (np.ndarray or function): The magnetic field 
            (in Tesla). It can be a constant vector or a function 
            that takes a position vector and returns the 
            magnetic field at that point.

    Returns:
        np.ndarray: The total Lorentz force vector (3D).
    """
    total_force = np.zeros(3)
    for segment in loop_segments:
      start_point, end_point = segment
      length_vector = end_point - start_point
      midpoint = (start_point + end_point) / 2.0
      
      if callable(magnetic_field):
          b_field = magnetic_field(midpoint)
      else:
          b_field = magnetic_field
      
      force_on_segment = current * np.cross(length_vector, b_field)
      total_force += force_on_segment
    return total_force