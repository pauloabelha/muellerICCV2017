import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import operator

class Joint(object):
    def __init__(self, bone=None, axes=None):
        self.bone = bone
        self.axes = axes


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def get_rot_mtx(axis, theta):
    if theta == 0.0:
        return np.eye(3)
    rot_mtx = np.eye(3)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    if axis == 0:
        rot_mtx[1, 1] = cos_theta
        rot_mtx[1, 2] = -sin_theta
        rot_mtx[2, 1] = sin_theta
        rot_mtx[2, 2] = cos_theta
    elif axis == 1:
        rot_mtx[0, 0] = cos_theta
        rot_mtx[0, 2] = sin_theta
        rot_mtx[2, 0] = -sin_theta
        rot_mtx[2, 2] = cos_theta
    elif axis == 2:
        rot_mtx[0, 0] = cos_theta
        rot_mtx[0, 1] = -sin_theta
        rot_mtx[1, 0] = sin_theta
        rot_mtx[1, 1] = cos_theta
    else:
        return None
    return rot_mtx

def skeleton_joints():
    x_axis = [1.0, 0.0, 0.0]
    y_axis = [0.0, 1.0, 0.0]
    z_axis = [0.0, 0.0, 1.0]
    thumb_mcp_axis = [0.1875, 0.1875, 0.625]
    thumb_dip_axis = [0.5, 0.5, 0.0]
    joints = []
    # handroot
    joints[0] = Joint(0, [x_axis, y_axis, z_axis])
    # thumb
    joints[1] = Joint(1, [thumb_mcp_axis, thumb_dip_axis])
    joints[2] = Joint(2, [thumb_dip_axis])
    joints[3] = Joint(3, [thumb_dip_axis])
    # index
    joints[4] = Joint(4, )


def skeleton_bone_lengths():
    '''

    :return skeleton model fixed bone lengths in mm
    '''
    bone_lengths = np.zeros((20, 1))
    # thumb
    bone_lengths[0] = 40
    bone_lengths[1] = 50
    bone_lengths[2] = 40
    bone_lengths[3] = 30
    # index
    bone_lengths[4] = 110
    bone_lengths[5] = 40
    bone_lengths[6] = 20
    bone_lengths[7] = 20
    # middle
    bone_lengths[8] = 110
    bone_lengths[9] = 40
    bone_lengths[10] = 20
    bone_lengths[11] = 20
    # ring
    bone_lengths[12] = 100
    bone_lengths[13] = 40
    bone_lengths[14] = 20
    bone_lengths[15] = 20
    # little
    bone_lengths[16] = 90
    bone_lengths[17] = 30
    bone_lengths[18] = 20
    bone_lengths[19] = 20
    return bone_lengths

def skeleton_bone_angles():
    '''

    :return: canonical global angles of resting skeleton
    '''
    bone_angles = [0] * 20
    # thumb
    bone_angles[0] = 5.495
    # index\
    bone_angles[4] = 5.8875
    # middle
    bone_angles[8] = 0.0
    # ring
    bone_angles[12] = 0.3925
    # little
    bone_angles[16] = 0.785
    return bone_angles

def plot_bone_line(bones):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgbyk')
    ax.plot([0, bones[0][0]],
            [0, bones[0][1]],
            [0, bones[0][2]])
    i = 1
    while i < len(bones):
        ax.plot([bones[i-1][0], bones[i][0]],
                [bones[i-1][1], bones[i][1]],
                [bones[i-1][2], bones[i][2]])
        i += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-150, 150])
    ax.set_ylim3d([-150, 150])
    ax.set_zlim3d([-150, 150])
    plt.show()

def plot_bone_lines(bone_lines):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgby')
    for i in range(len(bone_lines)):
        ax.plot([0, bone_lines[i][0][0]],
                [0, bone_lines[i][0][1]],
                [0, bone_lines[i][0][2]])
        j = 1
        while j < len(bone_lines[i]):
            ax.plot([bone_lines[i][j - 1][0], bone_lines[i][j][0]],
                    [bone_lines[i][j - 1][1], bone_lines[i][j][1]],
                    [bone_lines[i][j - 1][2], bone_lines[i][j][2]])
            j += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-150, 150])
    ax.set_ylim3d([-150, 150])
    ax.set_zlim3d([-150, 150])
    plt.show()

def update_point_and_children(joints_Theta, bone_lengths, bone_angles, bone_ix, axes_theta_ix, children_points, eps=1e-6):
    main_point = [bone_lengths[bone_ix][0], 0, 0]
    for i in range(len(children_points)):
        children_points[i] += main_point
    rotations = [None, None, None]
    for ax_ix in range(3):
        theta_ix = axes_theta_ix[ax_ix]
        if theta_ix is None:
            continue
        rotations[ax_ix] = get_rot_mtx(ax_ix, joints_Theta[theta_ix])
    if bone_angles[bone_ix] > eps:
        rotations[1] = np.dot(rotations[1], get_rot_mtx(1, bone_angles[bone_ix]))
    for rotation in rotations:
        if rotation is None:
            continue
        main_point = np.dot(rotation, main_point)
        for i in range(len(children_points)):
            children_points[i] = np.dot(rotation, children_points[i])
    return main_point, children_points

def get_bone_line_args(finger_ix):
    bone_ixs = [3, 2, 1, 0]
    bone_ixs = [x + (finger_ix * 4) for x in bone_ixs]
    axes_theta = []
    for i in range(4):
        axis_theta = [None, None, None]
        if i < 3:
            axis_theta[1] = (6 + (finger_ix * 4)) - i
        if i == 2:
            axis_theta[2] = axis_theta[1] - 1
        if i == 3:
            axis_theta = [2, 1, 0]
        axes_theta.append(axis_theta)
    return bone_ixs, axes_theta

def skeleton_bone_lines(joints_Theta, eps=1e-6):
    '''

    :param joints_Theta:
    :return: skeleton relative pose from 13 joint angles and global handroot 3D position (16 parameters in total)
    '''
    bone_lengths = skeleton_bone_lengths()
    bone_angles = skeleton_bone_angles()
    fingers_bone_lines = []
    for finger_ix in [0, 1, 2, 3, 4]:
        bone_ixs, axes_thetas = get_bone_line_args(finger_ix)
        finger_points = []
        for i in range(4):
            bone_ix = bone_ixs[i]
            axes_theta = axes_thetas[i]
            finger_main_pt, finger_children = update_point_and_children(
                joints_Theta, bone_lengths, bone_angles, bone_ix, axes_theta, finger_points)
            finger_points = [finger_main_pt] + finger_points
        fingers_bone_lines.append(finger_points)
    plot_bone_lines(fingers_bone_lines)
    return skeleton_bone_lines


joints_Theta = [0] * 26
joints_Theta[5] = 0.7
joints_Theta[6] = 0.7

skeleton_bone_lines = skeleton_bone_lines(joints_Theta)


