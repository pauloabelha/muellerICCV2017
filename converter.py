import numpy as np


def numpy_to_plottable_rgb(numpy_img):
    channel_axis = 0
    for i in numpy_img.shape:
        if i == 3 or i == 4:
            break
        channel_axis += 1
    if channel_axis == 0:
        img = numpy_img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
    elif channel_axis == 1:
        img = numpy_img.swapaxes(1, 2)
    elif channel_axis == 2:
        img = numpy_img
    else:
        return None
    return img[:, :, 0:3].astype(int)

def batch_numpy_to_plottable_rgb(batch_numpy_img, batch_axis=0):
    if batch_axis == 0:
        imgs_shape = (batch_numpy_img.shape[batch_axis], batch_numpy_img.shape[0],
                         batch_numpy_img.shape[1], batch_numpy_img.shape[2])
    elif batch_axis == 1:
        imgs_shape = (batch_numpy_img.shape[0], batch_numpy_img.shape[batch_axis],
                      batch_numpy_img.shape[1], batch_numpy_img.shape[2])
    elif batch_axis == 2:
        imgs_shape = (batch_numpy_img.shape[0], batch_numpy_img.shape[1],
                      batch_numpy_img.shape[batch_axis], batch_numpy_img.shape[2])
    elif batch_axis == 3:
        imgs_shape = (batch_numpy_img.shape[0], batch_numpy_img.shape[1],
                      batch_numpy_img.shape[2], batch_numpy_img.shape[batch_axis])
    else:
        return None
    imgs = np.zeros(imgs_shape)
    for batch_idx in range(batch_numpy_img.shape[batch_axis]):
        if batch_axis == 0:
            img = numpy_to_plottable_rgb(batch_numpy_img[batch_idx, :, :, :])
        elif batch_axis == 1:
            img = numpy_to_plottable_rgb(batch_numpy_img[:, batch_idx, :, :])
        elif batch_axis == 2:
            img = numpy_to_plottable_rgb(batch_numpy_img[:, :, batch_idx, :])
        elif batch_axis == 3:
            img = numpy_to_plottable_rgb(batch_numpy_img[:, :, :, batch_idx])
        else:
            return None
        imgs[batch_idx] = img
    return imgs

def heatmaps_to_joints_colorspace(heatmaps):
    num_joints = heatmaps.shape[0]
    joints_colorspace = np.zeros((num_joints, 2))
    for joint_ix in range(num_joints):
        heatmap = heatmaps[joint_ix, :, :]
        joints_colorspace[joint_ix, :] = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return joints_colorspace