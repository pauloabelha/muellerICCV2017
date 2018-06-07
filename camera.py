import numpy as np

def joint_color2depth_new(u, v, proj_const, dataset_handler):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    joint_uv = np.array([u, v, proj_const]).reshape((3,))
    #joint_uv *= proj_const
    mtx = np.dot(dataset_handler.COLOR_INTR_MTX, np.eye(3))
    joint_depth = np.dot(mtx, joint_uv)
    return joint_depth

def joint_color2depth(u, v, proj_const, depth_intr_matrix):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    joint_uv = np.array([u, v, 1.0]).reshape((3,))
    joint_uv *= proj_const
    joint_depth = np.dot(depth_intr_matrix, joint_uv)
    return joint_depth


def joint_depth2color(joint_depth, depth_intr_matrix, handroot=None):
    ''' Gets the joint in the color image space

    :param joint: join in depth camera space (mm)
    :param depth_intr_mtx: depth camera intrinsic params
    :return: u and v of joint in pixel space
    '''
    if handroot is None:
        handroot = np.zeros((1, 3))
    joint_depth = joint_depth + handroot
    if joint_depth.shape[0] == 1:
        joint_depth = joint_depth[0]
    joint_depth_z = joint_depth[2]
    joint_pixel = np.dot(depth_intr_matrix, joint_depth)
    if joint_depth_z == 0:
        joint_pixel = [0, 0, joint_depth_z]
    else:
        joint_pixel /= joint_depth_z
    u = int(joint_pixel[0])
    v = int(joint_pixel[1])
    return u, v, joint_depth_z


def joints_depth2color(joints_depth, depth_intr_matrix, handroot=None, img_res=None, orig_res=None):
    if handroot is None:
        handroot = np.zeros((1, 3))
    joints_colorspace = np.zeros((joints_depth.shape[0], 3))
    for i in range(joints_colorspace.shape[0]):
        joints_colorspace[i, 0], joints_colorspace[i, 1], joints_colorspace[i, 2] \
            = joint_depth2color(joints_depth[i, :], depth_intr_matrix, handroot=handroot)
    if not img_res is None:
        for dim in range(2):
            joints_colorspace[:, dim] *= (img_res[dim] / orig_res[dim])
    return joints_colorspace
