import numpy as np
import probs

def cross_entropy_loss_p_logq(torchvar_p, torchvar_logq, eps=1e-9):
    batch_size = torchvar_p.data.shape[0]
    return (-((torchvar_p + eps) * torchvar_logq + eps).sum(dim=1).sum(dim=1)).sum() / batch_size

def calculate_loss_multiple(output, target, iter_size,
                   weight_loss_intermed1, weight_loss_intermed2,
                   weight_loss_intermed3, weight_loss_main, eps=1e-9):
    #loss_intermed1 = F.l1_loss(output[0][:, 0, :, :], target)
    #loss_intermed2 = F.l1_loss(output[1][:, 0, :, :], target)
    #loss_intermed3 = F.l1_loss(output[2][:, 0, :, :], target)
    #loss_main = F.l1_loss(output[-1][:, 0, :, :], target)
    # loss = (weight_loss_intermed1 * loss_intermed1) +\
    #       (weight_loss_intermed2 * loss_intermed2) + \
    #       (weight_loss_intermed3 * loss_intermed3) + \
    #       (weight_loss_main * loss_main)
    loss_intermed1 = cross_entropy_loss_p_logq(output[0][:, 0, :, :], target)
    loss_intermed2 = cross_entropy_loss_p_logq(output[1][:, 0, :, :], target)
    loss_intermed3 = cross_entropy_loss_p_logq(output[2][:, 0, :, :], target)
    loss_main = cross_entropy_loss_p_logq(output[-1][:, 0, :, :], target)
    loss = (weight_loss_intermed1 * loss_intermed1) +\
           (weight_loss_intermed2 * loss_intermed2) + \
           (weight_loss_intermed3 * loss_intermed3) + \
           (weight_loss_main * loss_main)
    loss = loss / iter_size
    #print('-------------------------------')
    #print(loss_intermed1.data[0])
    #print(loss_intermed2.data[0])
    #print(loss_intermed3.data[0])
    #print(loss_main.data[0])
    #print('-------------------------------')
    return loss

def euclidean_loss(output_joints, target_joints):
    return (output_joints - target_joints).sum().abs()

def calculate_loss_main_with_joints(output, target_heatmaps, target_joints, iter_size):
    loss_main = 0
    output_main = output[0]
    # calculate main loss for "heatmaps"
    for joint_output_ix in range(output_main.shape[1]):
        loss_joint = cross_entropy_loss_p_logq(
            output_main[:, joint_output_ix, :, :], target_heatmaps[:, joint_output_ix, :, :])
        loss_main += loss_joint
    loss_main = loss_main / iter_size
    # calculate loss for joints
    loss_joints = euclidean_loss(output[1], target_joints)
    loss = loss_main + loss_joints
    return loss

def calculate_loss_main(output, target, iter_size):
    loss_main = 0
    for joint_output_ix in range(output.shape[1]):
        loss_joint = cross_entropy_loss_p_logq(
            output[:, joint_output_ix, :, :], target[:, joint_output_ix, :, :])
        loss_main += loss_joint
    loss_main = loss_main / iter_size
    return loss_main

def accumulate_pixel_dist_loss_main(pixel_dist_loss, output, target, BATCH_SIZE):
    size_batch = target.data.shape[0]
    iter_size = int(BATCH_SIZE / size_batch)
    avg_dist_loss = 0
    for i in range(size_batch):
        output_heatmap = output.data.cpu().numpy()[i][0, :, :]
        max_output = np.unravel_index(np.argmax(output_heatmap), output_heatmap.shape)
        target_heatmap = target.data.cpu().numpy()[i, :, :]
        max_target = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        dist_loss = np.sqrt(np.power((max_output[0] - max_target[0]), 2) +
                            np.power((max_output[1] - max_target[1]), 2))
        avg_dist_loss += dist_loss / size_batch
    pixel_dist_loss += avg_dist_loss / iter_size
    pixel_dist_loss = round(pixel_dist_loss, 1)
    return pixel_dist_loss

def calculate_pixel_loss_max(output_heatmap, target_heatmap):
    max_output = np.unravel_index(np.argmax(output_heatmap), output_heatmap.shape)
    max_target = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
    dist_loss = np.sqrt(np.power((max_output[0] - max_target[0]), 2) +
                        np.power((max_output[1] - max_target[1]), 2))
    return dist_loss

def calculate_pixel_loss_sample(output_heatmap, target_heatmap):
    output_sample, _ = probs.sample_from_2D_output(output_heatmap, is_log_prob=True)
    max_target = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
    dist_loss = np.sqrt(np.power((output_sample[0] - max_target[0]), 2) +
                        np.power((output_sample[1] - max_target[1]), 2))
    return dist_loss

def accumulate_pixel_dist_loss_multiple(pixel_dist_losses, output, target, BATCH_SIZE,
                                        dist_func=calculate_pixel_loss_max):
    size_batch = target.data.shape[0]
    num_channels = target.data.shape[1]
    iter_size = int(BATCH_SIZE / size_batch)
    for channel_ix in range(num_channels):
        avg_dist_loss = 0
        for batch_ix in range(size_batch):
            output_heatmap = output.data.cpu().numpy()[batch_ix, channel_ix, :, :]
            target_heatmap = target.data.cpu().numpy()[batch_ix, channel_ix, :, :]
            dist_loss = dist_func(output_heatmap, target_heatmap)
            avg_dist_loss += dist_loss / size_batch
        pixel_dist_losses[channel_ix] += avg_dist_loss / iter_size
        pixel_dist_losses[channel_ix] = round(pixel_dist_losses[channel_ix], 1)
    return pixel_dist_losses

def accumulate_pixel_dist_loss_from_sample_multiple(pixel_dist_losses, output, target, BATCH_SIZE):
    return accumulate_pixel_dist_loss_multiple(pixel_dist_losses, output, target, BATCH_SIZE,
                                        dist_func=calculate_pixel_loss_sample)

def accumulate_pixel_dist_loss_from_sample(pixel_dist_loss, output, target, BATCH_SIZE):
    size_batch = target.data.shape[0]
    iter_size = int(BATCH_SIZE / size_batch)
    avg_dist_loss = 0
    output_numpy = output.data.cpu().numpy()
    for i in range(size_batch):
        output_sample, _ = probs.sample_from_2D_output(output_numpy[i][0, :, :], is_log_prob=True)
        target_tensor = target.data.cpu().numpy()[i, :, :]
        max_target = np.unravel_index(np.argmax(target_tensor), target_tensor.shape)
        dist_loss = np.sqrt(np.power((output_sample[0] - max_target[0]), 2) +
                            np.power((output_sample[1] - max_target[1]), 2))
        avg_dist_loss += dist_loss / size_batch
    pixel_dist_loss += avg_dist_loss / iter_size
    pixel_dist_loss_sample = round(pixel_dist_loss, 1)
    return pixel_dist_loss_sample