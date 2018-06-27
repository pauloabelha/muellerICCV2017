from scipy import misc
import numpy as np
import converter as conv


def change_res_image(image, new_res):
    image = misc.imresize(image, new_res)
    return image


def read_RGB_image(image_filepath, new_res=None):
    image = misc.imread(image_filepath)
    image = image.swapaxes(0, 1)
    if new_res:
        image = change_res_image(image, new_res)
    return image

def get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res):
    res_transf_u = (heatmap_res[0] / (crop_coords[2] - crop_coords[0]))
    res_transf_v = (heatmap_res[1] / (crop_coords[3] - crop_coords[1]))
    labels_ix = 0
    labels_heatmaps = np.zeros((len(joint_ixs), heatmap_res[0], heatmap_res[1]))
    labels_colorspace_mapped = np.copy(labels_colorspace)
    for joint_ix in joint_ixs:
        label_crop_local_u = labels_colorspace[joint_ix, 0] - crop_coords[0]
        label_crop_local_v = labels_colorspace[joint_ix, 1] - crop_coords[1]
        label_u = int(label_crop_local_u * res_transf_u)
        label_v = int(label_crop_local_v * res_transf_v)
        labels_colorspace_mapped[joint_ix, 0] = label_u
        labels_colorspace_mapped[joint_ix, 1] = label_v
        label = conv.color_space_label_to_heatmap(labels_colorspace_mapped[joint_ix, :], heatmap_res,
                                                     orig_img_res=heatmap_res)
        label = label.astype(float)
        labels_heatmaps[labels_ix, :, :] = label
        labels_ix += 1
    return labels_heatmaps, labels_colorspace_mapped

def get_crop_coords(joints_uv, image_rgbd):
    min_u = min(joints_uv[:, 0]) - 10
    min_v = min(joints_uv[:, 1]) - 10
    max_u = max(joints_uv[:, 0]) + 10
    max_v = max(joints_uv[:, 1]) + 10
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    # get coords
    coords = [u0, v0, u1, v1]
    return coords

def crop_hand_rgbd(joints_uv, image_rgbd, crop_res):
    crop_coords = get_crop_coords(joints_uv, image_rgbd)
    # crop hand
    crop = image_rgbd[:, crop_coords[0]:crop_coords[2], crop_coords[1]:crop_coords[3]]
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop_rgb = change_res_image(crop[:, :, 0:3], crop_res)
    crop_depth = change_res_image(crop[:, :, 3], crop_res)
    # normalize depth
    crop_depth = np.divide(crop_depth, np.max(crop_depth))
    crop_depth = crop_depth.reshape(crop_depth.shape[0], crop_depth.shape[1], 1)
    crop_rgbd = np.append(crop_rgb, crop_depth, axis=2)
    crop_rgbd = crop_rgbd.swapaxes(1, 2)
    crop_rgbd = crop_rgbd.swapaxes(0, 1)
    return crop_rgbd, crop_coords

def crop_image_get_labels(data, labels_colorspace, joint_ixs=range(21), crop_res=(128, 128)):
    data, crop_coords = crop_hand_rgbd(labels_colorspace, data, crop_res=crop_res)
    labels_heatmaps, labels_colorspace =\
        get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res=crop_res)
    return data, crop_coords, labels_heatmaps, labels_colorspace