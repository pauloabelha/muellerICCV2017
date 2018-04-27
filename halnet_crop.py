import sys
import io_data
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Ignoring matplotlib import error")
    pass

# implement a depth-dependent image crop



def imcrop(heatmap_root_joint, max_heatmap, depth_at_max, image):
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
    crop = io_data.change_res_image(crop[:, :, 0:3], (128, 128))
    return crop

def imcrop2(joints_uv, image_rgbd):
    min_u = min(joints_uv[:, 0]) - 10
    min_v = min(joints_uv[:, 1]) - 10
    max_u = max(joints_uv[:, 0]) + 10
    max_v = max(joints_uv[:, 1]) + 10
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    crop = image_rgbd[:, u0:u1, v0:v1]
    crop[3, :, :] /= 1
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop = io_data.change_res_image(crop[:, :, 0:3], (128, 128))
    return crop

root_folder = '/home/paulo/rds_muri/paulo/synthhands/SynthHands_Release/'
train_loader = io_data.get_SynthHands_trainloader(root_folder=root_folder, joint_ixs=range(20),
                                                  batch_size=1, verbose=True)

#image_to_show = io_data.convert_torch_dataimage_to_canonical(image)
#image_to_show = image_to_show.swapaxes(0, 1)
#plt.imshow(image_to_show)

def get_joints_uv(target_heatmaps):
    num_joints = target_heatmaps.shape[1]
    joints_uv = np.zeros((num_joints, 2))
    for joint_ix in range(num_joints):
        target_heatmap = target_heatmaps.data.numpy()[0, joint_ix, :, :]
        max_heatmap = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        joints_uv[joint_ix, 0] = max_heatmap[0]
        joints_uv[joint_ix, 1] = max_heatmap[1]
    return joints_uv


for batch_idx, (data, target) in enumerate(train_loader):
    target_heatmaps, target_joints = target
    joints_uv = get_joints_uv(target_heatmaps)
    image = data.data.numpy()
    crop = imcrop2(joints_uv, image[0, :, :, :])
    crop_to_show = crop[:, :, 0:3]
    crop_to_show = crop_to_show.astype(np.uint8)
    plt.imshow(crop_to_show)
    plt.show()
