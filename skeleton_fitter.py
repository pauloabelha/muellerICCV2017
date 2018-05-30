from autograd import grad
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd.builtins import list
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

# hand 'canonical' pose is:
#   Hand root (wrist) in origin
#   Middle finger accross positive X axis
#   Palm facing positive Y axis import autograd.numpy as np  # Thinly-wrapped numpy(Y axis is the normal to the palm)
# bones start as a vector [bone_length, 0., 0.] before being rotated

def plot_bone_lines(bone_lines, fig=None, show=True, lim=200):
    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_color_cycle('rgby')
    for i in range(len(bone_lines)):
        ax.plot([0., bone_lines[i][0][0]],
                [0., bone_lines[i][0][1]],
                [0., bone_lines[i][0][2]])
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

def get_bones_lengths():
    '''

    :return skeleton model fixed bone lengths in mm
    '''
    bone_lengths = list([[]] * 5)
    # finger
    bone_lengths[0] = [52., 43., 35., 32.]
    # index
    bone_lengths[1] = [86., 42., 34., 29.]
    # middle
    bone_lengths[2] = [78., 48., 34., 28.]
    # ring
    bone_lengths[3] = [77., 50., 32., 29.]
    # little1
    bone_lengths[4] = [77., 29., 21., 23.]
    return bone_lengths

def get_fingers_angles_canonical():
    '''

    :return: bone angles of hand canonical pose
    '''
    finger_angles = list([[0., 0., 0.]] * 5)
    # finger
    finger_angles[0] = [5.8875, 5.495, 0.75]
    # index
    finger_angles[1] = [1.57, 5.8875, 0.]
    # middle
    finger_angles[2] = [1.57, 0., 0.]
    # ring
    finger_angles[3] = [1.57, 0.3925, 0.]
    # little
    finger_angles[4] = [1.57, 0.785, 0.]
    return finger_angles

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
    if abs(theta) <= eps:
        return vec
    if axis == 0:
        x, y, z = rotate_diff_x(vec, ix_start, theta)
    elif axis == 1:
        x, y, z = rotate_diff_y(vec, ix_start, theta)
    elif axis == 2:
        x, y, z = rotate_diff_z(vec, ix_start, theta)
    else:
        return None
    return list([x, y, z])

def get_finger_bone_seq_no_rot(finger_ix, bones_lengths):
    finger_bone_angles_ixs = [0] * 4
    for i in range(4):
        finger_bone_angles_ixs[i] = i + 3
    finger_bone_seq = list([[0., 0., 0.]] * 4)
    curr_seq_len = 0.
    for i in range(4):
        finger_bone_seq[i] = [curr_seq_len + bones_lengths[finger_ix][i], 0., 0.]
        curr_seq_len += bones_lengths[finger_ix][0]
    return finger_bone_seq

def get_finger_canonical_bone_seq(finger_ix, bones_lengths, fingers_angles):
    finger_bone_seq = get_finger_bone_seq_no_rot(finger_ix, bones_lengths)
    for i in range(4):
        for ax in range(3):
            angle = fingers_angles[finger_ix][ax]
            finger_bone_seq[i] = rotate_diff_axis(ax, finger_bone_seq[i], 0, angle)
    return finger_bone_seq

def get_finger_theta_ixs_and_axes(finger_ix):
    axes_rot = [2, 1, 1, 1]
    theta_ixs = [0.] * 4
    for i in range(4):
        theta_ixs[i] = ((finger_ix+1) * 4) -1 + i
    return theta_ixs, axes_rot

def get_finger_bone_seq(finger_ix, Theta, bones_lengths, fingers_angles):
    # get local bone sequence positions, without rotation
    finger_bone_seq = list([[0., 0., 0.]] * 4)
    for i in range(4):
        finger_bone_seq[i] = [bones_lengths[finger_ix][i], 0., 0.]
    # get finger-dependent indexes of Theta and axes of rotation
    theta_ixs, axes_rot = get_finger_theta_ixs_and_axes(finger_ix)
    # rotate each finger bone with Theta
    for i in range(4):
        ix_rev = 3 - i
        angle = Theta[theta_ixs[ix_rev]]
        ax_rot = axes_rot[ix_rev]
        finger_bone_seq[ix_rev] = rotate_diff_axis(ax_rot, finger_bone_seq[ix_rev], 0, angle)
        # update "children" bones
        j = ix_rev
        while j < 3:
            finger_bone_seq[j+1] = rotate_diff_axis(ax_rot, finger_bone_seq[j+1], 0, angle)
            j += 1
    # put all finger bones in absolute position to hand root
    for i in range(3):
        finger_bone_seq[i+1] = [finger_bone_seq[i+1][0] + finger_bone_seq[i][0],
                                finger_bone_seq[i+1][1] + finger_bone_seq[i][1],
                                finger_bone_seq[i+1][2] + finger_bone_seq[i][2]]
    # rotate each finger with its finger canonical angle for each axis
    for i in range(4):
        for ax in range(3):
            angle = fingers_angles[finger_ix][ax]
            finger_bone_seq[i] = rotate_diff_axis(ax, finger_bone_seq[i], 0, angle)
    # rotate each finger according to the hand root rotation
    for i in range(4):
        for j in range(3):
            finger_bone_seq[i] = rotate_diff_axis(j, finger_bone_seq[i], 0, Theta[j])
    return finger_bone_seq

def get_hand_seq_canonical(bones_lengths, fingers_angles):
    hand_seq = list([[]] * 5)
    for finger_ix in range(5):
        hand_seq[finger_ix] = get_finger_canonical_bone_seq(finger_ix, bones_lengths, fingers_angles)
    return hand_seq

def get_hand_seq(Theta, bones_lengths, fingers_angles):
    hand_seq = list([[]] * 5)
    for finger_ix in range(5):
        hand_seq[finger_ix] = get_finger_bone_seq(finger_ix, Theta, bones_lengths, fingers_angles)
    return hand_seq

def hand_seq_to_matrix(hand_seq):
    hand_matrix = np.array(hand_seq).reshape((20, 3))
    return hand_matrix

def Theta_to_hand_matrix(Theta, bones_lengths, fingers_angles):
    #Theta = np.minimum(Theta, 6.28)
    #Theta = np.maximum(Theta, 0.)
    hand_seq = get_hand_seq(Theta, bones_lengths, fingers_angles)
    hand_matrix = hand_seq_to_matrix(hand_seq)
    return hand_matrix

def animate_skeleton(pausing=0.001):
    fig = None
    Theta = [0.] * 23
    for i in range(len(Theta)):
        for j in range(10):
            Theta[i] = 0.05 * j
            hand_seq = get_hand_seq(Theta, bones_lengths, fingers_angles)
            fig = plot_bone_lines(hand_seq, fig=fig, show=False)
            plt.pause(pausing)
            plt.clf()
    plt.show()

def get_example_target_matrix():
    target_matrix = np.array([
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
    return target_matrix

def get_example_target_matrix2():
    target_matrix = np.array([
                [5.04926300e+01, 4.86349411e+01, 3.22667580e+01],
                [6.29547806e+01, 7.22900848e+01, 1.90970001e+01],
                [7.12234039e+01, 8.81342850e+01, 7.43657589e+00],
                [8.01767883e+01, 1.06043503e+02, -4.03247738e+00],
                [6.31909981e+01, 3.92918282e+01, 2.07988148e+01],
                [9.71118088e+01, 7.10300827e+01, 1.67733021e-02],
                [1.13407402e+02, 9.21796188e+01, -1.90750809e+01],
                [1.07807945e+02, 1.04189819e+02, -4.57797546e+01],
                [7.25996475e+01, 2.82628822e+01, 5.69833565e+00],
                [1.12488670e+02, 5.47043686e+01, -1.17148340e+00],
                [1.34326385e+02, 7.71675949e+01, -1.57214651e+01],
                [1.37153976e+02, 9.35943451e+01, -3.92123222e+01],
                [8.31332245e+01, 1.65777664e+01, -1.59413128e+01],
                [1.18601105e+02, 3.94201927e+01, -2.28066750e+01],
                [1.35169754e+02, 6.50885391e+01, -3.82870293e+01],
                [1.42985275e+02, 8.90216675e+01, -5.33211937e+01],
                [3.81632347e+01, 1.14704266e+01, -3.37704353e+01],
                [6.10587921e+01, 2.33903408e+01, -6.82850800e+01],
                [8.05751648e+01, 4.75567703e+01, -8.45160522e+01],
                [9.82698898e+01, 7.10361176e+01, -9.79136353e+01]
                 ])
    return target_matrix

def E_lim(Theta):
    loss_lim = 0.
    for i in range(23):
        if 0 <= Theta[i] and Theta[i] <= 6.28:
            loss_angle = 0.
        else:
            loss_angle = np.abs(np.abs(Theta[i] - 6.28))
        loss_lim += loss_angle
    return loss_lim

def E_pos3D(Theta, target_matrix, bones_lengths, fingers_angles):
    hand_matrix = Theta_to_hand_matrix(Theta, bones_lengths, fingers_angles)
    dist = np.abs((hand_matrix - target_matrix)).sum()
    return dist

def Epsilon_Loss(Theta, target_matrix, bones_lengths, fingers_angles):
    loss_pos = E_pos3D(Theta, target_matrix, bones_lengths, fingers_angles)
    loss_lim = E_lim(Theta)
    loss_eps = loss_pos + loss_lim
    return loss_eps

def fit_skeleton(loss_func, target_matrix, bones_lengths, fingers_angles, initial_theta=None, num_iter=1000, log_interval=10, lr=0.01):
    grad_fun = grad(loss_func, 0)
    i = 0
    loss = 0.
    if initial_theta is None:
        theta = np.array([1.] * 26)
    else:
        theta = np.array(initial_theta)
    for i in range(num_iter):
        grad_calc = grad_fun(theta, target_matrix, bones_lengths, fingers_angles)
        theta -= lr * grad_calc
        if i % log_interval == 0:
            loss = loss_func(theta, target_matrix, bones_lengths, fingers_angles)
            print('Iter {} : Loss {}'.format(i, loss))
        if i % (10 * log_interval) == 0:
            print('Theta:\t{}'.format(theta))
    print('Num iter: {}'.format(i))
    print('Final loss: {}'.format(loss))
    print('Theta:\n{}'.format(theta))
    return theta

bones_lengths = get_bones_lengths()
fingers_angles = get_fingers_angles_canonical()
hand_seq = get_hand_seq_canonical(bones_lengths, fingers_angles)

Theta = np.array([1.] * 23)
Theta[0] = 5.57
# thumb dip and tip
for i in range(len(Theta)):
    ix = i - 1
    if ix > 0 and ix % 4 == 0:
        Theta[i] = np.pi / 4
print(Theta)
hand_seq = get_hand_seq(Theta, bones_lengths, fingers_angles)
#plot_bone_lines(hand_seq)

hand_matrix = Theta_to_hand_matrix(Theta, bones_lengths, fingers_angles)
print(hand_matrix)

target_matrix = get_example_target_matrix()
print(target_matrix)
#plot_hand_matrix(target_matrix)

loss = E_pos3D(Theta, target_matrix, bones_lengths, fingers_angles)
print(loss)

Theta_fit = fit_skeleton(E_pos3D, target_matrix, bones_lengths, fingers_angles,
                         initial_theta=Theta, num_iter=200, log_interval=10, lr=3e-5)
hand_seq_fit = get_hand_seq(Theta_fit, bones_lengths, fingers_angles)

hand_matrix = Theta_to_hand_matrix(Theta, bones_lengths, fingers_angles)
print(hand_matrix)

plot_hand_matrix(target_matrix)
plot_bone_lines(hand_seq_fit)




