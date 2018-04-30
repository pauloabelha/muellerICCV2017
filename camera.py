import numpy as np
import io_data

DEPTH_INTR_MTX =     np.array([[475.62,     0.0,        311.125],
                                [0.0,        475.62,     245.965],
                                [0.0,        0.0,        1.0]])
DEPTH_INTR_MTX_INV =     np.array([[0.00210252, 0., -0.65414617],
                                [0., 0.00210252, -0.51714604],
                                [0., 0., 1.]])
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


def joint_color2depth(u, v, proj_const):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    joint_uv = np.array([v, u, 1.0]).reshape((3,))
    joint_uv *= proj_const
    joint_depth = np.dot(DEPTH_INTR_MTX_INV, joint_uv)
    return joint_depth

def joint_depth2color(joint_depth):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    joint_depth_z = joint_depth[2]
    joint_pixel = np.dot(DEPTH_INTR_MTX, joint_depth)
    joint_pixel /= joint_depth_z
    v = int(joint_pixel[0])
    u = int(joint_pixel[1])
    return u, v, joint_depth_z