import numpy as np
import io_data

DEPTH_INTR_MTX =     np.array([[475.62,     0.0,        311.125],
                                [0.0,        475.62,     245.965],
                                [0.0,        0.0,        1.0]])
COLOR_INTR_MTX = np.array([[617.173,    0.0,       315.453],
                           [0.0,        617.173,    242.259],
                           [0.0,        0.0,        1.0]])
COLOR_EXTR_MTX = np.array([[1.0, 0.0, 0.0, 24.7],
                           [0.0, 1.0, 0.0, -0.0471401],
                           [0.0, 0.0, 1.0, 3.72045],
                           [0.0, 0.0, 0.0, 1.0]])
PROJECT_MTX =    np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

def get_joint_in_color_space(joint):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    joint_pixel = np.dot(DEPTH_INTR_MTX, joint)
    joint_pixel = joint_pixel / joint[2]
    v = int(joint_pixel[0])
    u = int(joint_pixel[1])
    return u, v