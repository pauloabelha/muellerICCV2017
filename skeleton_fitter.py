import autograd.numpy as np  # Thinly-wrapped numpy
from autograd.builtins import list
from autograd import grad
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

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
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def get_rot_mtx(axis, theta):
    if theta == 0.0:
        return np.eye(3)
    rot_mtx = np.eye(3)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
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

def rotate_diff_x(vec, ix_start, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = vec[0]
    y = cos_theta * vec[ix_start+1] - sin_theta * vec[ix_start+2]
    z = sin_theta * vec[ix_start+1] + cos_theta * vec[ix_start+2]
    return x, y, z

def rotate_diff_y(vec, ix_start, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = cos_theta * vec[ix_start] + sin_theta * vec[ix_start+2]
    y = vec[1]
    z = -sin_theta * vec[ix_start] + cos_theta * vec[ix_start+2]
    return x, y, z

def rotate_diff_z(vec, ix_start, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = cos_theta * vec[ix_start] - sin_theta * vec[ix_start+1]
    y = sin_theta * vec[ix_start] + cos_theta * vec[ix_start+1]
    z = vec[ix_start+2]
    return x, y, z

def rotate_diff_axis(axis, vec, ix_start, theta):
    if axis == 0:
        x, y, z = rotate_diff_x(vec, ix_start, theta)
    elif axis == 1:
        x, y, z = rotate_diff_y(vec, ix_start, theta)
    elif axis == 2:
        x, y, z = rotate_diff_z(vec, ix_start, theta)
    else:
        return None
    return np.array([x, y, z])

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
    bone_lengths[0] = 52
    bone_lengths[1] = 43
    bone_lengths[2] = 35
    bone_lengths[3] = 32
    # index
    bone_lengths[4] = 86
    bone_lengths[5] = 42
    bone_lengths[6] = 34
    bone_lengths[7] = 29
    # middle
    bone_lengths[8] = 78
    bone_lengths[9] = 48
    bone_lengths[10] = 34
    bone_lengths[11] = 28
    # ring
    bone_lengths[12] = 77
    bone_lengths[13] = 50
    bone_lengths[14] = 32
    bone_lengths[15] = 29
    # little1
    bone_lengths[16] = 77
    bone_lengths[17] = 29
    bone_lengths[18] = 21
    bone_lengths[19] = 23
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

def plot_bone_lines(bone_lines, handroot=None, fig=None, show=True, lim=200):
    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgby')
    if handroot is None:
        handroot = np.zeros((1, 3))
    for i in range(len(bone_lines)):
        ax.plot([handroot[0], bone_lines[i][0][0]],
                [handroot[1], bone_lines[i][0][1]],
                [handroot[2], bone_lines[i][0][2]])
        j = 1
        while j < len(bone_lines[i]):
            ax.plot([bone_lines[i][j - 1][0], bone_lines[i][j][0]],
                    [bone_lines[i][j - 1][1], bone_lines[i][j][1]],
                    [bone_lines[i][j - 1][2], bone_lines[i][j][2]])
            j += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-lim, lim])
    if show:
        plt.show()
    return fig

def plot_hand_matrix(hand_matrix, fig=None, show=True, lim=200):
    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgby')
    handroot = np.zeros((1, 3))
    for i in range(5):
        mcp_ix = (i*4)
        ax.plot([handroot[0, 0], hand_matrix[mcp_ix, 0]],
                [handroot[0, 1], hand_matrix[mcp_ix, 1]],
                [handroot[0, 2], hand_matrix[mcp_ix, 2]])
        for j in range(3):
            ax.plot([hand_matrix[mcp_ix+j, 0], hand_matrix[mcp_ix+j+1, 0]],
                    [hand_matrix[mcp_ix+j, 1], hand_matrix[mcp_ix+j+1, 1]],
                    [hand_matrix[mcp_ix+j, 2], hand_matrix[mcp_ix+j+1, 2]])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-lim, lim])
    if show:
        plt.show()
    return fig

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
    # add hand root rotation
    if bone_ix % 4 == 0:
        for i in range(3):
            if joints_Theta[i] > eps:
                rotations = [get_rot_mtx(i, joints_Theta[i])] + rotations
    for rotation in rotations:
        if rotation is None:
            continue
        main_point = np.dot(rotation, main_point)
        for i in range(len(children_points)):
            children_points[i] = np.dot(rotation, children_points[i])
    return main_point, children_points

def update_point_and_children2(joints_Theta, bone_lengths, bone_angles, bone_ix, axes_theta_ix, children_points, eps=1e-6):
    main_point = [bone_lengths[bone_ix][0], 0, 0]
    for i in range(len(children_points)):
        children_points[i] += main_point
    rotations = [None, None, None]
    for ax_ix in range(3):
        theta_ix = axes_theta_ix[ax_ix]
        if theta_ix is None:
            continue
        if abs(joints_Theta[theta_ix]) > eps:
            rotations[ax_ix] = get_rot_mtx(ax_ix, joints_Theta[theta_ix])
    if bone_angles[bone_ix] > eps:
        if rotations[1] is None:
            rotations[1] = get_rot_mtx(1, bone_angles[bone_ix])
        else:
            rotations[1] = np.dot(rotations[1], get_rot_mtx(1, bone_angles[bone_ix]))
    # add hand root rotation
    if bone_ix % 4 == 0:
        for i in range(3):
            if joints_Theta[i] > eps:
                rotations = [get_rot_mtx(i, joints_Theta[i])] + rotations

    main_point2 = main_point
    children_points2 = children_points
    if bone_ix % 4 == 0:
        for i in range(3):
            ix = 2-i
            if joints_Theta[ix] > eps:
                rot_mtx = get_rot_mtx(ix, joints_Theta[ix])
                #print('--------------------------------------------------------')
                #print(rot_mtx)
                main_point2 = rotate_diff_axis(ix, main_point2, 0, joints_Theta[ix])
                for j in range(len(children_points2)):
                    children_points2[j] = rotate_diff_axis(ix, children_points2[j], 0, joints_Theta[ix])
    for ax_ix in range(3):
        theta_ix = axes_theta_ix[ax_ix]
        if theta_ix is None:
            continue
        if ax_ix == 1: # bone rotation
            if bone_angles[bone_ix] > eps:
                rot_mtx = get_rot_mtx(1,  bone_angles[bone_ix])
                #print('--------------------------------------------------------')
                #print(rot_mtx)
                main_point2 = rotate_diff_axis(1, main_point2, 0, bone_angles[bone_ix])
                for j in range(len(children_points2)):
                    children_points2[j] = rotate_diff_axis(1, children_points2[j], 0, bone_angles[bone_ix])
        aa = joints_Theta[theta_ix]
        rot_mtx = get_rot_mtx(ax_ix, joints_Theta[theta_ix])
        #print('--------------------------------------------------------')
        #print(rot_mtx)
        main_point2 = rotate_diff_axis(ax_ix, main_point2, 0, joints_Theta[theta_ix])
        for j in range(len(children_points2)):
            children_points2[j] = rotate_diff_axis(ax_ix, children_points2[j], 0, joints_Theta[theta_ix])


    for rotation in rotations:
        if rotation is None:
            continue
        main_point = np.dot(rotation, main_point)
        for i in range(len(children_points)):
            children_points[i] = np.dot(rotation, children_points[i])

    print(np.linalg.norm(main_point - main_point2))
    #for rotation in rotations:
    #    print('-----------------------------------------------------')
    #    print(rotation)
    return main_point2, children_points2

def get_bone_line_args(finger_ix):
    bone_ixs = [3, 2, 1, 0]
    bone_ixs = [x + (finger_ix * 4) for x in bone_ixs]
    axes_theta = list([])
    for i in range(4):
        axis_theta = [None, None, None]
        if i < 3:
            if finger_ix == 0:
                ax_ix = 1
            else:
                ax_ix = 2
            axis_theta[ax_ix] = (6 + (finger_ix * 4)) - i
        if i == 2:
            axis_theta[1] = (6 + (finger_ix * 4)) - i
            axis_theta[2] = (6 + (finger_ix * 4)) - i - 1
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
    bone_lines = list([])
    for finger_ix in [0, 1, 2, 3, 4]:
        bone_ixs, axes_thetas = get_bone_line_args(finger_ix)
        finger_points = []
        for i in range(4):
            bone_ix = bone_ixs[i]
            axes_theta = axes_thetas[i]
            finger_main_pt, finger_children = update_point_and_children(
                joints_Theta, bone_lengths, bone_angles, bone_ix, axes_theta, finger_points)
            finger_points = [finger_main_pt] + finger_points
        finger_points = np.array(finger_points)
        #finger_points = finger_points + joints_Theta[-3:] # do not consider global position
        bone_lines.append(finger_points)
    return bone_lines

def fingers_bone_lines_to_matrix(fingers_bone_lines, handroot):
    hand_matrix = np.array(fingers_bone_lines).reshape((20, 3))
    return hand_matrix


def joints_theta_ok():
    joints_Theta = [0] * 26
    joints_Theta[4] = 5.75
    joints_Theta[7] = 1.57
    joints_Theta[9] = 1.57
    joints_Theta[10] = 0.7
    joints_Theta[11] = 1.57
    joints_Theta[12] = 0.7
    joints_Theta[13] = 1.57
    joints_Theta[14] = 0.7
    joints_Theta[15] = 1.57
    joints_Theta[16] = 0.7
    joints_Theta[17] = 1.57
    joints_Theta[18] = 0.7
    joints_Theta[19] = 1.57
    joints_Theta[21] = 1.57

    joints_Theta[23] = 100
    return joints_Theta

def joints_Theta_to_hand_matrix(joints_Theta):
    fingers_bone_lines = skeleton_bone_lines(joints_Theta)
    hand_matrix = fingers_bone_lines_to_matrix(fingers_bone_lines, joints_Theta[-3:])
    return hand_matrix

def animate_skeleton(pausing=0.001):
    fig = None
    for h in range(3):
        joints_Theta = [0.0] * 26
        ix = 2 -h
        joints_Theta[ix] = 1.57
        for i in range(23):
            for j in range(10):
                if i % 4 == 0:
                    continue
                joints_Theta[i+3] = 0.1 * j
                fingers_bone_lines = skeleton_bone_lines(joints_Theta)
                fig = plot_bone_lines(fingers_bone_lines, handroot=joints_Theta[-3:], fig=fig, show=False)
                plt.pause(pausing)
                plt.clf()
    plt.show()

def E_pos3D(joints_Theta, joints_pred):
    hand_matrix = joints_Theta_to_hand_matrix(joints_Theta)
    dist = np.abs((hand_matrix - joints_pred)).sum()
    return dist

def get_example_target_joints():
    joints_vec = np.array([
                 [ 3.81632347e+01,  1.14704266e+01, -3.37704353e+01],
                 [ 6.10587921e+01,  2.33903408e+01, -6.82850800e+01],
                 [ 8.05751648e+01,  4.75567703e+01, -8.45160522e+01],
                 [ 9.82698898e+01,  7.10361176e+01, -9.79136353e+01],
                 [ 8.31332245e+01,  1.65777664e+01, -1.59413128e+01],
                 [ 1.18601105e+02,  3.94201927e+01, -2.28066750e+01],
                 [ 1.35169754e+02,  6.50885391e+01, -3.82870293e+01],
                 [ 1.42985275e+02,  8.90216675e+01, -5.33211937e+01],
                 [ 7.25996475e+01,  2.82628822e+01,  5.69833565e+00],
                 [ 1.12488670e+02,  5.47043686e+01, -1.17148340e+00],
                 [ 1.34326385e+02,  7.71675949e+01, -1.57214651e+01],
                 [ 1.37153976e+02,  9.35943451e+01, -3.92123222e+01],
                 [ 6.31909981e+01,  3.92918282e+01,  2.07988148e+01],
                 [ 9.71118088e+01,  7.10300827e+01,  1.67733021e-02],
                 [ 1.13407402e+02,  9.21796188e+01, -1.90750809e+01],
                 [ 1.07807945e+02,  1.04189819e+02, -4.57797546e+01],
                 [ 5.04926300e+01,  4.86349411e+01,  3.22667580e+01],
                 [ 6.29547806e+01,  7.22900848e+01,  1.90970001e+01],
                 [ 7.12234039e+01,  8.81342850e+01,  7.43657589e+00],
                 [ 8.01767883e+01,  1.06043503e+02, -4.03247738e+00]])
    return joints_vec

def bone_lengths_from_hand_matrix(hand_matrix):
    handroot = np.zeros((1, 3))
    bone_ix = 0
    bone_lengths = [0.] * 20
    for i in range(5):
        mcp_ix = (i*4)
        bone_lengths[bone_ix] = np.linalg.norm(handroot[0, :] - hand_matrix[mcp_ix, :])
        bone_ix += 1
        for j in range(3):
            bone_lengths[bone_ix] = np.linalg.norm(hand_matrix[mcp_ix+j, :] - hand_matrix[mcp_ix+j+1, :])
            bone_ix += 1
    return bone_lengths

joint_pred = get_example_target_joints()

joints_Theta = [0.] * 26
for i in range(26):
    if i < 3:
        continue
    if i % 4 == 0:
        continue
    joints_Theta[i] = 1.
fingers_bone_lines = skeleton_bone_lines(joints_Theta)
#plot_hand_matrix(joint_pred)
print(joints_Theta)
plot_bone_lines(fingers_bone_lines, handroot=joints_Theta[-3:])
#animate_skeleton()

bone_lengths = bone_lengths_from_hand_matrix(joint_pred)

grad_E_pos3D = grad(E_pos3D, 0)
lr = 0.01
i = 0
loss = 0.
prev_loss = 0.
theta = np.array([1.] * 26)
num_iter = 10000
for i in range(num_iter):
    grad_calc = grad_E_pos3D(theta, joint_pred)
    theta -= lr * grad_calc
    loss = E_pos3D(theta, joint_pred)
    diff_loss = np.abs((loss - prev_loss))
    if i > 0 and i % 10 == 0:
        print('Iter {} : Loss {} : Loss Diff {}'.format(i, loss, diff_loss))
    if diff_loss < 1e-4:
        print('Found very small loss diff: {}'.format(diff_loss))
        break
    prev_loss = loss

print('Num iter: {}'.format(i))
print('Final loss: {}'.format(loss))
print('Theta:\n{}'.format(theta))
fingers_bone_lines = skeleton_bone_lines(theta)

plot_hand_matrix(joint_pred)
plot_bone_lines(fingers_bone_lines, handroot=theta[-3:])


