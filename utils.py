import numpy as np
import os

def clear_screen():
    # TODO: add doc heres
    os.system('cls' if os.name=='nt' else 'clear')


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def angular_error(v1, v2):
    """ 
        Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(rad_angle)


def print_stats(errors):
    # TODO: add doc here
    print("mean: " + str(np.mean(errors)) 
        + " median: " + str(np.median(errors))
        + " max: " + str(np.max(errors)))