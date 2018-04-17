import numpy as np
import visualize
import probs

def print_verbose(str, verbose, n_tabs=0, erase_line=False):
    prefix = '\t' * n_tabs
    if verbose:
        if erase_line:
            print(prefix + str, end='')
        else:
            print(prefix + str)

def show_target_and_output_to_image_info(data, target, output, debug_visually=True):
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
        print("Max value of output (prob): " + str(max_value_output))
        if debug_visually:
            visualize.show_halnet_output_as_heatmap(output_heatmap_example1,
                                                    data.data.cpu().numpy()[BATCH_IDX],
                                                    img_title='Joint ' + str(joint_ix))

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