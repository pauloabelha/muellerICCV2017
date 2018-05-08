import sys

import io_image
import synthhands_handler
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Ignoring matplotlib import error")
    pass
import torch
from torch.autograd import Variable



# implement a depth-dependent image crop

def imcrop(crop_res, heatmap_root_joint, max_heatmap, depth_at_max, image):
    magic_constant = 100
    # calculate a side length ivnersely proportional to depth
    side_length = int(min(heatmap_root_joint.shape[0], magic_constant / depth_at_max))
    # get valid bounding box
    crop_min_x = max(0, max_heatmap[0] - side_length)
    crop_min_y = max(0, max_heatmap[1] - side_length)
    crop_max_x = min(heatmap_root_joint.shape[0], max_heatmap[0] + side_length)
    crop_max_y = min(heatmap_root_joint.shape[1], max_heatmap[1] + side_length)
    # crop
    crop = image[:, crop_min_x:crop_max_x, crop_min_y:crop_max_y]
    # normalize crop
    crop[3, :, :] /= depth_at_max
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop = io_image.change_res_image(crop[:, :, 0:3], crop_res)
    return crop

def imcrop2(joints_uv, image_rgbd, crop_res):
    min_u = min(joints_uv[:, 0]) - 10
    min_v = min(joints_uv[:, 1]) - 10
    max_u = max(joints_uv[:, 0]) + 10
    max_v = max(joints_uv[:, 1]) + 10
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    crop = image_rgbd[:, u0:u1, v0:v1]
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop = io_image.change_res_image(crop[:, :, 0:3], crop_res)
    plt.imshow(crop.astype(int))
    plt.show()
    return crop



#image_to_show = io_data.convert_torch_dataimage_to_canonical(image)
#image_to_show = image_to_show.swapaxes(0, 1)
#plt.imshow(image_to_show)

def get_joints_uv(target_heatmaps):
    num_joints = target_heatmaps.shape[0]
    joints_uv = np.zeros((num_joints, 2))
    for joint_ix in range(num_joints):
        target_heatmap = target_heatmaps[joint_ix, :, :]
        max_heatmap = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        joints_uv[joint_ix, 0] = max_heatmap[0]
        joints_uv[joint_ix, 1] = max_heatmap[1]
    return joints_uv


def plot_torch_image(torch_data):
    aa = torch_data.data.numpy().swapaxes(0, 2)
    plt.imshow(aa[:, :, 0:3].astype(int))
    plt.show()

def crop_batch_input_images(img_batch, target_heatmaps, crop_res):
    num_imgs = img_batch.data.shape[0]
    cropped_img_batch = np.zeros((img_batch.shape[0], img_batch.shape[1], crop_res[0], crop_res[0]))
    for i in range(num_imgs):
        joints_uv = get_joints_uv(target_heatmaps.data.numpy()[i, :, :, :])
        print(joints_uv)
        plot_torch_image(img_batch[i, :, :, :])
        cropped_img = imcrop2(joints_uv, img_batch.data.numpy()[i, :, :, :], crop_res)
        cropped_img = cropped_img.swapaxes(0, 2)
        cropped_img = cropped_img.swapaxes(1, 2)
        cropped_img_batch[i, :, :, :] = cropped_img
    return Variable(torch.from_numpy(cropped_img_batch).float())

'''
root_folder = '/home/paulo/rds_muri/paulo/synthhands/SynthHands_Release/'
train_loader = io_data.get_SynthHands_trainloader(root_folder=root_folder, joint_ixs=range(20),
                                                  batch_size=1, verbose=True)
                                                  
for batch_idx, (data, target) in enumerate(train_loader):
    target_heatmaps, target_joints = target
    joints_uv = get_joints_uv(target_heatmaps)
    image = data.data.numpy()
    crop = imcrop2(joints_uv, image[0, :, :, :])
    crop_to_show = crop[:, :, 0:3]
    crop_to_show = crop_to_show.astype(np.uint8)
    plt.imshow(crop_to_show)
    plt.show()
'''
