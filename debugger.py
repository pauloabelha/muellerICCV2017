import numpy as np
import math
import converter
import visualize
import probs
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("WARNING: Ignoring matplotlib import error")
    pass

def print_verbose(str, verbose, n_tabs=0, erase_line=False):
    prefix = '\t' * n_tabs
    msg = prefix + str
    if verbose:
        if erase_line:
            print(msg, end='')
        else:
            print(prefix + str)
    return msg

def show_target_and_output_to_image_info(data, target_heatmaps, output):
    batch_idxs = [0, 1]
    n_joints = target_heatmaps.data.shape[1]
    rows = math.ceil(math.sqrt(n_joints))
    cols = rows
    for BATCH_IDX in batch_idxs:
        fig = plt.figure(figsize=(rows, cols))
        for joint_ix in range(target_heatmaps.data.shape[1]):
            output_heatmap_example1 = output.data.cpu().numpy()[BATCH_IDX, joint_ix, :, :]
            heatmap = output_heatmap_example1
            image = data.data.cpu().numpy()[BATCH_IDX]
            img_title = 'Joint ' + str(joint_ix)
            heatmap = converter.convert_torch_targetheatmap_to_canonical(heatmap)
            heatmap = heatmap.swapaxes(0, 1)
            image = converter.convert_torch_dataimage_to_canonical(image)
            image = image.swapaxes(0, 1)
            fig.add_subplot(rows, cols, joint_ix + 1)
            plt.imshow(image)
            fig.add_subplot(rows, cols, joint_ix + 1)
            plt.imshow(255 * heatmap, alpha=0.6, cmap='viridis', interpolation='nearest')
            plt.title("Joint " + str(joint_ix))
    plt.title("Joint heatmaps")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

def show_target_and_prob_output_to_image_info(data, target, output, debug_visually=True):
    BATCH_IDX = 0
    print("Showing info for first datum of batch and for every joint:")
    for joint_ix in range(target.data.shape[1]):
        print("-------------------------------------------------------------------------------------------")
        target_heatmap = target.data.cpu().numpy()[BATCH_IDX, joint_ix, :, :]
        max_target = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        print("Max of target: " + str(max_target))
        max_value_target = np.max(target_heatmap)
        print("Max value of target: " + str(max_value_target))
        output_heatmap_example1 = output.data.cpu().numpy()[BATCH_IDX, joint_ix, :, :]
        max_output = np.unravel_index(np.argmax(output_heatmap_example1), output_heatmap_example1.shape)
        print("Max of output: " + str(max_output))
        max_value_output = np.max(output_heatmap_example1)
        print("Max value of output (prob): " + str(np.exp(max_value_output)))
        output_sample_flat_ix = np.random.choice(range(len(output_heatmap_example1.flatten())),
                                                 1, p=np.exp(output_heatmap_example1).flatten())
        prob_mass_window = probs.prob_mass_n_pixels_radius(output_heatmap_example1,
                                                           u_p=max_output[0],
                                                           v_p=max_output[1])
        print("Probability mass in a 5x5 pixel window around maximum: " + str(prob_mass_window))
        prob_mass_window = probs.prob_mass_n_pixels_radius(output_heatmap_example1,
                                                           u_p=max_output[0],
                                                           v_p=max_output[1],
                                                           n_pixels=10)
        print("Probability mass in a 10x10 pixel window around maximum: " + str(prob_mass_window))
        output_sample_uv = np.unravel_index(output_sample_flat_ix, output_heatmap_example1.shape)
        print("Sample of output: (" + str(output_sample_uv[0][0]) + ", " + str(output_sample_uv[1][0]) + ")")
        max_value_output = np.max(output_heatmap_example1)
        print("Sample value of output (prob): " + str(np.exp(max_value_output)))
        prob_mass_window = probs.prob_mass_n_pixels_radius(output_heatmap_example1,
                                                           u_p=output_sample_uv[0][0],
                                                           v_p=output_sample_uv[1][0])
        print("Probability mass in a 5x5 pixel window around sample: " + str(prob_mass_window))
        if debug_visually:
            visualize.show_halnet_output_as_heatmap(output_heatmap_example1,
                                                    data.data.cpu().numpy()[BATCH_IDX],
                                                    img_title='Joint ' + str(joint_ix))