import numpy as np
import torch
from torch.autograd import Variable
import camera


def numpy_swap_cols(np_array):
    np_array[:,[0, 1]] = np_array[:,[1, 0]]
    return np_array

def numpy_swap_cols(np_array, col_from, col_to):
    np_array[:, [col_from, col_to]] = np_array[:, [col_to, col_from]]
    return np_array

def numpy_to_plottable_rgb(numpy_img):
    img = numpy_img
    if len(numpy_img.shape) == 3:
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
        img = img[:, :, 0:3]
    img = img.swapaxes(0, 1)
    return img.astype(int)

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


def normalize_output(output):
    output_positive = output + abs(np.min(output, axis=(0, 1)))
    norm_output = output_positive / np.sum(output_positive, axis=(0, 1))
    return norm_output


def convert_torch_targetheatmap_to_canonical(target_heatmap, res=(640, 480)):
    assert target_heatmap.shape[0] == res[0]
    assert target_heatmap.shape[1] == res[1]
    return target_heatmap


def convert_torch_dataoutput_to_canonical(data, res=(640, 480)):
    if len(data.shape) < 3:
        image = data
    else:
        image = data[0, :, :]
    # put channels at the end
    image = np.swapaxes(image, 0, 1)
    assert image.shape[0] == res[0]
    assert image.shape[1] == res[1]
    return image


def convert_torch_dataimage_to_canonical(data, res=(640, 480)):
    image = data[0:3, :, :]
    # put channels at the end
    image = image.astype(np.uint8)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    assert image.shape[0] == res[0]
    assert image.shape[1] == res[1]
    assert image.shape[2] == 3
    return image

def convert_labels_2D_new_res(color_space_label, orig_img_res, heatmap_res):
    new_ix_res1 = int(color_space_label[0] /
                      (orig_img_res[0] / heatmap_res[0]))
    new_ix_res2 = int(color_space_label[1] /
                      (orig_img_res[1] / heatmap_res[1]))
    return np.array([new_ix_res1, new_ix_res2])

def color_space_label_to_heatmap(color_space_label, heatmap_res, orig_img_res=(640, 480)):
    '''
    Convert a (u,v) color-space label into a heatmap
    In this case, the heat map has only one value set to 1
    That is, the value (u,v)
    :param color_space_label: a pair (u,v) of color space joint position
    :param image_res: a pair (U, V) with the values for image resolution
    :return: numpy array of dimensions image_res with one position set to 1
    '''
    SMALL_PROB = 0.0
    heatmap = np.zeros(heatmap_res) + SMALL_PROB
    new_label_res = convert_labels_2D_new_res(color_space_label, orig_img_res, heatmap_res)
    heatmap[new_label_res[0], new_label_res[1]] = 1 - (SMALL_PROB * heatmap.size)
    return heatmap


def data_to_batch(data):
    batch = np.zeros((1, data.shape[0], data.shape[1], data.shape[2]))
    batch[0, :, :, :] = data
    batch = Variable(torch.from_numpy(batch).float())
    return batch


def jornet_local_to_global_joints(jornet_joints, handroot):
    jornet_joints_global = np.zeros((21, 3))
    jornet_joints_global[0, :] = handroot
    jornet_joints = jornet_joints.reshape((20, 3))
    jornet_joints_global[1:, :] = jornet_joints + handroot
    return jornet_joints_global


def joints_globaldepth_to_colorspace(jornet_joints_global, dataset_handler, img_res=(320, 240), orig_res=(640, 480)):
    joints_colorspace = np.zeros((21, 3))
    for i in range(21):
        u, v, z = camera.joint_depth2color(jornet_joints_global[i, :], dataset_handler.DEPTH_INTR_MTX)
        joints_colorspace[i, 0] = u
        joints_colorspace[i, 1] = v
        joints_colorspace[i, 2] = z
    joints_colorspace[:, 0] *= img_res[0] / orig_res[0]
    joints_colorspace[:, 1] *= img_res[1] / orig_res[1]
    return joints_colorspace


