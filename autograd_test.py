import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

def plot_bone_lines(bone_lines, handroot=None, fig=None, show=True, lim=200):
    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgby')
    if handroot is None:
        handroot = [0., 0., 0.]
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

def rotate_diff_axis(axis, vec, ix_start, theta, eps=1e-6):
    if axis == 0:
        x, y, z = rotate_diff_x(vec, ix_start, theta)
    elif axis == 1:
        x, y, z = rotate_diff_y(vec, ix_start, theta)
    elif axis == 2:
        x, y, z = rotate_diff_z(vec, ix_start, theta)
    else:
        return None
    return np.array([x, y, z])


def finger1(theta, finger_len=10.):
    ret = []
    # theta is 4 x 1 (root position and angle in y rot)
    finger = np.array([finger_len, 0., 0.]).reshape((3, 1))
    finger = rotate_diff_axis(1, finger, 0, theta[0])
    ret.append(finger)
    finger = rotate_diff_axis(2, finger, 0, theta[1])
    ret.append(finger)
    return ret

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

def thumb(theta):
    bone_lengths = skeleton_bone_lengths()
    bone_angles = skeleton_bone_angles()
    bone_ixs, axes_thetas = get_bone_line_args(0)
    bones = []
    for bone_ix in range(len(axes_thetas)):
        bone_vec = [bone_lengths[bone_ix][0], 0., 0.]
        for ax_ix in range(3):
            if axes_thetas[bone_ix][ax_ix] is None:
                continue
            angle_rot = theta[axes_thetas[bone_ix][ax_ix]]
            bone_vec = rotate_diff_axis(ax_ix, bone_vec, 0, angle_rot)
        bone_vec = rotate_diff_axis(1, bone_vec, 0, bone_angles[bone_ix])
        bones = [bone_vec] + bones
    return bones

def loss_finger(theta, finger_pred):
    finger = thumb(theta)
    dist = 0.
    for i in range(len(finger)):
        dist = dist + np.abs((finger[i] - finger_pred)).sum()
    return dist

theta = np.array([0.] * 26)
print('Theta pred:\n{}'.format(theta))

finger_pred = thumb(theta)
print('Finger pred:\n{}'.format(finger_pred))
plot_bone_lines([finger_pred])

grad_loss = grad(loss_finger, 0)

theta = np.array([0.] * 26)
prev_loss = 0.
lr = 0.001
for i in range(200):
    grad_calc = grad_loss(theta, finger_pred)
    theta -= lr * grad_calc
    loss = loss_finger(theta, finger_pred)
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
final_thumb = thumb(theta)
print('Finger:\n{}'.format(final_thumb))

plot_bone_lines([final_thumb])

a = 0