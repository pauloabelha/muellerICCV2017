import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

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
    if axis == 1:
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

def loss_finger(theta, finger_pred):
    finger = finger1(theta)
    dist = 0.
    for i in range(len(finger)):
        dist = dist + np.abs((finger[i] - finger_pred)).sum()
    return dist

theta = np.zeros((5, 1))
theta[0] = np.pi
theta[1] = np.pi / 2
theta[2] = 117.
print('Theta pred:\n{}'.format(theta))

finger_pred = finger1(theta)
print('Finger pred:\n{}'.format(finger_pred))

grad_loss = grad(loss_finger, 0)

theta = np.zeros((4, 1))
prev_loss = 0.
lr = 0.001
for i in range(1000):
    grad_calc = grad_loss(theta, finger_pred)
    theta -= lr * grad_calc
    loss = loss_finger(theta, finger_pred)
    diff_loss = np.abs((loss - prev_loss))
    if i > 0 and i % 100 == 0:
        print('Iter {} : Loss {:.2f} : Loss Diff {:.5f}'.format(i, loss, diff_loss))
    if diff_loss < 1e-4:
        print('Found very small loss diff: {:.5f}'.format(diff_loss))
        break
    prev_loss = loss

print('Num iter: {}'.format(i))
print('Final loss: {}'.format(loss))
print('Theta:\n{}'.format(theta))
print('Finger:\n{}'.format(finger1(theta)))