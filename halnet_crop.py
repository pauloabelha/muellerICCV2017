import numpy as np

# implement a depth-dependent image crop

#heatmap_max = np.unravel_index(np.argmax(heatmap_root_joint), heatmap_root_joint.shape)

def imcrop(heatmap_root_joint, max_heatmap, depth_at_max):
    # calculate a side length ivnersely proportional to depth
    side_length = max(heatmap_root_joint.shape, 1 / depth_at_max)
    # get valid bounding box
    crop_min_x = max(0, max_heatmap[0] - side_length)
    crop_min_y = max(0, max_heatmap[1] - side_length)
    crop_max_x = min(heatmap_root_joint.shape[0], max_heatmap[0] + side_length)
    crop_max_y = min(heatmap_root_joint.shape[1], max_heatmap[1] + side_length)
    # crop
    crop = heatmap_root_joint[crop_min_x:crop_max_x, crop_min_y:crop_max_y]
    # normalize crop
    crop = crop / depth_at_max
    return crop
