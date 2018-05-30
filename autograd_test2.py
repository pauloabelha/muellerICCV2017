import autograd.numpy as np
from autograd import grad

def rotate_diff_z(vec, ix_start, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = cos_theta * vec[ix_start] - sin_theta * vec[ix_start+1]
    y = sin_theta * vec[ix_start] + cos_theta * vec[ix_start+1]
    z = vec[ix_start+2]
    vec = np.array([x, y, z])
    return vec

def rotate_point(Theta):
    point = np.cos(Theta)
    return point

def loss_fun(Theta, target_point):
    point = rotate_point(Theta)
    loss = np.abs(point - target_point)
    return loss


target_point = np.array([0.75])
Theta = np.array([0.1])
print('Rot point:\t\t{}'.format(rotate_point(Theta)))
print('Target point:\t{}'.format(target_point))
print('Loss:\t\t\t{}'.format(loss_fun(Theta, target_point)))

num_iter = 1000
lr = 0.01
grad_fun = grad(loss_fun, 0)
i = 0
loss = 0.
for i in range(num_iter):
    grad_calc = grad_fun(Theta, target_point)
    Theta -= lr * grad_calc
    if i % 10 == 0:
        loss = loss_fun(Theta, target_point)
        print('Iter {} : Loss {}'.format(i, loss))

print('Num iter: {}'.format(i))
print('Final loss: {}'.format(loss))
print('Theta:\n{}'.format(Theta))
print('Rot point:\n{}'.format(rotate_point(Theta)))