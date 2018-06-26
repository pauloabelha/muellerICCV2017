# exampel calls
# -i female_object/seq01/cam01/01/00000000 -r /home/paulo/SynthHands_Release/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_for_valid_38000.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_70000.pth.tar
# -i Fruits/color_on_depth/image_00000 -r /home/paulo/EgoDexter/data/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_30000.pth.tar

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io_image
import trainer
import egodexter_handler
import argparse
import converter as conv
import HALNet, JORNet
import time
import visualize

IMG_RES = (320, 240)

def print_time(str_, time_diff):
    print(str_ + str(round(time_diff*1000)) + ' ms')

def plot_joints(joints_colorspace, show_legend=True, linewidth=4):
    num_joints = joints_colorspace.shape[0]
    joints_colorspace = conv.numpy_swap_cols(joints_colorspace, 0, 1)
    plt.plot(joints_colorspace[0, 1], joints_colorspace[0, 0], 'ro', color='C0')
    plt.plot(joints_colorspace[0:2, 1], joints_colorspace[0:2, 0], 'ro-', color='C0', linewidth=linewidth)
    joints_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    legends = []
    if show_legend:
        palm_leg = mpatches.Patch(color='C0', label='Palm')
        legends.append(palm_leg)
    for i in range(4):
        plt.plot([joints_colorspace[0, 1], joints_colorspace[(i * 4) + 5, 1]],
                 [joints_colorspace[0, 0], joints_colorspace[(i * 4) + 5, 0]], 'ro-', color='C0', linewidth=linewidth)
    for i in range(num_joints - 1):
        if (i + 1) % 4 == 0:
            continue
        color = 'C' + str(int(np.ceil((i + 1) / 4)))
        plt.plot(joints_colorspace[i + 1:i + 3, 1], joints_colorspace[i + 1:i + 3, 0], 'ro-', color=color, linewidth=linewidth)
        if show_legend and i % 4 == 0:
            joint_name = joints_names[int(np.floor((i+1)/4))]
            legends.append(mpatches.Patch(color=color, label=joint_name))
    if show_legend:
        plt.legend(handles=legends)
    return joints_colorspace

def get_image_name(image_basename, ix, dataset_name):
    str_to_add = str(ix)
    if ix == 0:
        ix = 1
    algs = int(np.log10(ix))
    image_basename = image_basename[:-(algs+1)] + str_to_add
    return image_basename

def get_image_as_data(dataset_folder, input_img_namebase, dataset_name, img_res):
    data = []
    if dataset_name == 'SynthHands_Release':
        data = synthhands_handler._get_data(dataset_folder, input_img_namebase, img_res)
    elif dataset_name == 'EgoDexter':
        data = egodexter_handler.get_data(dataset_folder, input_img_namebase, img_res=img_res)
    return data

def load_examples_to_memory(start_ix, end_ix, dataset):
    data = []
    img_paths = []
    i = start_ix
    while i < end_ix:
        img_paths.append(dataset.filenamebases[i])
        img_data, img_labels = dataset[i]
        img_labels_2D, img_labels_heatmaps, img_labels_3D = img_labels
        data.append((img_data, (img_labels_2D, img_labels_heatmaps, img_labels_3D)))
        i += 1
    return data, img_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
    parser.add_argument('-r', dest='dataset_folder', default='', type=str, required=True,
                        help='Dataset folder')
    parser.add_argument('--halnet', dest='halnet_filepath', type=str, required=True,
                        help='Filepath to trained HALNet checkpoint')
    parser.add_argument('--jornet', dest='jornet_filepath', type=str, required=True,
                        help='Filepath to trained HALNet checkpoint')
    parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=False,
                        help='Whether to use cuda for training')
    parser.add_argument('-o', dest='output_filepath', default='',
                        help='Output file for logging')
    parser.add_argument('-s', dest='start_ix', default='', type=int, required=True,
                        help='Dataset starting example ix')
    parser.add_argument('-e', dest='end_ix', default='', type=int, required=True,
                        help='Dataset end example ix')
    return parser.parse_args()

def print_divisor(num=100):
    print('-' * num)

args = parse_args()
dataset_name = args.dataset_folder.split('/')[-2]

egodexter = egodexter_handler.EgoDexterDataset(root_folder=args.dataset_folder, type_='full', heatmap_res=IMG_RES)
num_examples = args.end_ix - args.start_ix

print_divisor()
print('Arguments')
print('\tDataset root: {}'.format(args.dataset_folder))
print('\tHALNet filepath: {}'.format(args.halnet_filepath))
print('\tJORNet filepath: {}'.format(args.jornet_filepath))
print('\tUsing CUDA: {}'.format(args.use_cuda))
print_divisor()
print('Dataset: ')
print('\tSize of dataset: {}'.format(len(egodexter)))
print('\tNumber of examples to process: {}'.format(num_examples))
start = time.time()
dataset_data, img_paths = load_examples_to_memory(args.start_ix, args.end_ix, egodexter)
print_time('\tLoaded examples to memory: ', time.time() - start)
print_divisor()

print('Neural networks: ')
# load nets
start = time.time()
halnet, _, _, _ = trainer.load_checkpoint(filename=args.halnet_filepath,
                                          model_class=HALNet.HALNet,
                                          use_cuda=args.use_cuda)
print_time('\tHALNet loaded: ', time.time() - start)

start = time.time()
jornet, _, _, _ = trainer.load_checkpoint(filename=args.jornet_filepath,
                                          model_class=JORNet.JORNet,
                                          use_cuda=args.use_cuda)
print_time('\tJORNet loaded: ', time.time() - start)
print_divisor()

NUM_JOINTS = 5
dataset_name = 'EgoDexter'

def get_label_index(dataset_name, idx):
    if dataset_name == 'EgoDexter':
        idx = (idx+1)*4
    return idx

losses_halnet_per_joints = np.zeros((num_examples, NUM_JOINTS))
losses_halnet_joints_tot = []
losses_jornet_per_joints = np.zeros((num_examples, NUM_JOINTS))
losses_jornet_joints_tot = []
losses_jornet_per_joints_depth = np.zeros((num_examples, NUM_JOINTS))
losses_jornet_depth_tot = []
num_valid_loss_iter = 0
tot_loss = 0
example_ix = 0
while example_ix < (args.end_ix - args.start_ix):

    print('Processing example #{}: {}'.format(example_ix, img_paths[example_ix]))
    dataset_datum = dataset_data[example_ix]
    img_data, img_labels = dataset_datum
    img_numpy = img_data.data.numpy()
    img_labels_2D, img_labels_heatmaps, img_labels_3D = img_labels
    img_labels_2D = img_labels_2D.data.numpy()

    print('\tHALNet:')
    start = time.time()
    halnet_input = conv.data_to_batch(img_data)
    print_time('\t\tHALNet image convertion: ', time.time() - start)

    start = time.time()
    output_halnet = halnet(halnet_input)
    print_time('\t\tHALNet pass: ', time.time() - start)

    halnet_main_out = output_halnet[3][0].data.numpy()
    halnet_handroot = np.array(np.unravel_index(np.argmax(halnet_main_out[0]), IMG_RES))
    print('\t\t\tHALNet hand root:\t{}'.format(halnet_handroot))

    # get halnet joints in colorspace from heatmaps
    halnet_joints_colorspace = conv.heatmaps_to_joints_colorspace(halnet_main_out)

    print('\tHALNet joint pixel loss:')
    num_valid_loss = 0
    loss_halnet_joints = 0
    halnet_out_fingertips = np.zeros((5, 2))
    for i in range(NUM_JOINTS):
        idx = get_label_index(dataset_name, i)
        if np.sum(img_labels_2D[i, :]) > 0:
            curr_loss = np.linalg.norm(img_labels_2D[i, :] - halnet_joints_colorspace[idx, :])
            loss_halnet_joints += curr_loss
            num_valid_loss += 1
            losses_halnet_per_joints[example_ix, i] = curr_loss
            print('\t\t\tJoint {}: {}\t{} : {}'.format(i, img_labels_2D[i, :], halnet_joints_colorspace[idx, :], curr_loss))
        else:
            print('\t\t\tJoint {}: {}\t{}'.format(i, 'No label', halnet_joints_colorspace[idx, :]))
        halnet_out_fingertips[i, :] = halnet_joints_colorspace[idx, :]
    if num_valid_loss > 0:
        losses_halnet_joints_tot.append(loss_halnet_joints / num_valid_loss)
        print('\t\tAverage loss for HALNet joints: {}'.format(losses_halnet_joints_tot[-1]))
    else:
        print('\t\tNo Loss Calculation due to lack of labels')
    print_divisor()

    #fig = visualize.plot_image(img_numpy)
    #visualize.plot_fingertips(img_labels_2D, fig=fig)
    #visualize.plot_joints(halnet_joints_colorspace, fig=fig)
    #visualize.plot_fingertips(halnet_out_fingertips, handroot=halnet_joints_colorspace[0, :], fig=fig)
    #visualize.show()

    print('\tJORNet:')
    start = time.time()
    data_crop, crop_coords, _, _ = io_image.crop_image_get_labels(img_numpy, halnet_joints_colorspace)
    batch_jornet = conv.data_to_batch(data_crop)
    print_time('\t\tJORNet image conversion: ', time.time() - start)

    #visualize.plot_image(data_crop)
    #visualize.show()

    start = time.time()
    output_jornet = jornet(batch_jornet)
    print_time('\t\tJORNet pass: ', time.time() - start)

    _, img_labels_2D_cropped = io_image.get_labels_cropped_heatmaps(
        img_labels_2D, joint_ixs=range(NUM_JOINTS), crop_coords=crop_coords, heatmap_res=(128, 128))

    print('\tImage labels (2D cropped to (128, 128)):')
    for i in range(NUM_JOINTS):
        depth_const = img_numpy[3, int(img_labels_2D_cropped[i, 0]), int(img_labels_2D_cropped[i, 1])]
        print('\t\tFinger tip {}: {}'.format(i, img_labels_2D_cropped[i, :]))
    print_divisor()

    #fig = visualize.plot_image(data_crop)
    #visualize.plot_fingertips(img_labels_2D_cropped, fig=fig)
    #visualize.show()

    print('\tJORNet joint pixel loss:')
    output_jornet_heatmaps_main = output_jornet[3][0].data.numpy()
    jornet_joints_colorspace = conv.heatmaps_to_joints_colorspace(output_jornet_heatmaps_main)
    num_valid_loss = 0
    loss_jornet_joints = 0
    for i in range(NUM_JOINTS):
        curr_loss = np.linalg.norm(img_labels_2D_cropped[i, :] - jornet_joints_colorspace[i, :])
        loss_jornet_joints += curr_loss
        losses_jornet_per_joints[example_ix, i] = curr_loss
        print('\t\t\tJoint {}: {}\t{} : {}'.format(i, img_labels_2D_cropped[i, :], jornet_joints_colorspace[i, :], curr_loss))
    losses_jornet_joints_tot.append(loss_jornet_joints / NUM_JOINTS)
    print('\t\tAverage loss for HALNet joints: {}'.format(losses_jornet_joints_tot[-1]))
    print_divisor()

    #fig = visualize.plot_image(data_crop)
    #visualize.plot_fingertips(img_labels_2D_cropped, fig=fig)
    #visualize.plot_fingertips(jornet_joints_colorspace, handroot=0, fig=fig)
    #visualize.show()

    output_jornet_joints_main = output_jornet[7][0].data.cpu().numpy().reshape((20, 3))
    #handroot = camera.joint_color2depth(halnet_joints_colorspace[0, 0],
    #                                    halnet_joints_colorspace[0, 1],
    #                                    200,
    #                                    synthhands_handler.DEPTH_INTR_MTX_INV)
    handroot = np.zeros((1, 3))
    jornet_joints_global = conv.jornet_local_to_global_joints(output_jornet_joints_main, handroot)


    #fig, ax = visualize.plot_3D_joints(jornet_joints_global)
    #visualize.plot_3D_joints(img_labels_3D, fig=fig, ax=ax)
    #visualize.show()

    print('\tJORNet joint depth loss:')
    num_valid_loss = 0
    loss_jornet_depth = 0
    for i in range(NUM_JOINTS):
        curr_loss = np.linalg.norm(img_labels_3D[i, :] - jornet_joints_global[i, :])
        losses_jornet_per_joints_depth[example_ix, i] = curr_loss
        loss_jornet_depth += curr_loss
        print('\t\t\tFinger tip {}: {}\t{} : {}'.format(i, img_labels_3D[i, :],
                                                        jornet_joints_global[i, :],
                                                        curr_loss))
    loss_jornet_depth /= NUM_JOINTS
    losses_jornet_depth_tot.append(loss_jornet_depth)
    print('\t\tAverage loss for JORNet depth: {}'.format(loss_jornet_depth))
    print_divisor()

    print('\tLosses:')
    print('\tHALNet')
    print('\t\tAverage joint loss: {}'.format(np.mean(losses_halnet_joints_tot)))
    print('\t\tStddev joint loss: {}'.format(np.std(losses_halnet_joints_tot)))
    print('\t\tPer joint loss (average ; std):')
    halnet_means_per_joint = [0.] * (NUM_JOINTS + 1)
    halnet_err_per_joint = [0.] * (NUM_JOINTS + 1)
    for joint_ix in range(NUM_JOINTS):
        halnet_means_per_joint[joint_ix] = float(np.mean(losses_halnet_per_joints[:, joint_ix]))
        halnet_err_per_joint[joint_ix] = float(np.std(losses_halnet_per_joints[:, joint_ix])) / 2.
        print('\t\t\tJoint\t{}:\t{}\t;\t{}'.
              format(joint_ix,
                     halnet_means_per_joint[joint_ix],
                     np.std(losses_halnet_per_joints[:, joint_ix])))
    halnet_means_per_joint[-1] = np.mean(losses_halnet_joints_tot)
    halnet_err_per_joint[-1] = np.std(losses_halnet_joints_tot)
    print('\tJORNet')
    print('\t\tPer joint loss (average ; std):')
    jornet_means_per_joint = [0.] * (NUM_JOINTS + 1)
    jornet_err_per_joint = [0.] * (NUM_JOINTS + 1)
    for joint_ix in range(NUM_JOINTS):
        jornet_means_per_joint[joint_ix] = float(np.mean(losses_jornet_per_joints[:, joint_ix]))
        jornet_err_per_joint[joint_ix] = float(np.std(losses_jornet_per_joints[:, joint_ix])) / 2.
        print('\t\t\tJoint\t{}:\t{}\t;\t{}'.
              format(joint_ix,
                     jornet_means_per_joint[joint_ix],
                     np.std(losses_jornet_per_joints[:, joint_ix])))
    jornet_means_per_joint[-1] = np.mean(losses_jornet_joints_tot)
    jornet_err_per_joint[-1] = np.std(losses_jornet_joints_tot)

    print('\t\tPer joint loss depth (average ; std):')
    jornet_means_per_joint_depth = [0.] * (NUM_JOINTS + 1)
    jornet_err_per_joint_depth = [0.] * (NUM_JOINTS + 1)
    for joint_ix in range(NUM_JOINTS):
        jornet_means_per_joint_depth[joint_ix] = float(np.mean(losses_jornet_per_joints_depth[:, joint_ix]))
        jornet_err_per_joint_depth[joint_ix] = float(np.std(losses_jornet_per_joints_depth[:, joint_ix])) / 2.
        print('\t\t\tJoint\t{}:\t{}\t;\t{}'.
              format(joint_ix,
                     jornet_means_per_joint_depth[joint_ix],
                     np.std(losses_jornet_per_joints_depth[:, joint_ix])))
    jornet_means_per_joint_depth[-1] = np.mean(losses_jornet_depth_tot)
    jornet_err_per_joint_depth[-1] = np.std(losses_jornet_depth_tot)

    print('\t\tAverage depth loss: {}'.format(np.mean(losses_jornet_depth_tot)))
    print('\t\tStddev depth loss: {}'.format(np.std(losses_jornet_depth_tot)))
    print_divisor()

    example_ix += 1

num_ranges = 30
max_range = 30
frames_per_mm_range = np.zeros((num_ranges, 1))
losses_jornet_joints_tot_array = np.array(losses_jornet_joints_tot)
for i in range(num_ranges):
    ixs = np.array(losses_jornet_joints_tot_array < i * (max_range/num_ranges))
    frames_per_mm_range[i] = (np.sum(ixs) / len(losses_jornet_joints_tot)) * 100
    print(frames_per_mm_range[i])
print_divisor()

visualize.plot_line(frames_per_mm_range, xlabel='Error threshold (pixels)', ylabel='Percentage of frames',
                    fontsize=30, tickwidth=10, linewidth=10)
visualize.show()

num_mm_ranges = 60
max_range = 60
frames_per_mm_range = np.zeros((num_mm_ranges, 1))
losses_jornet_depth_tot_array = np.array(losses_jornet_depth_tot)
for i in range(num_mm_ranges):
    ixs = np.array(losses_jornet_depth_tot_array < i * (max_range/num_mm_ranges))
    frames_per_mm_range[i] = (np.sum(ixs) / len(losses_jornet_depth_tot)) * 100
    print(frames_per_mm_range[i])
print_divisor()

visualize.plot_line(frames_per_mm_range, xlabel='Error threshold (mm)', ylabel='Percentage of frames',
                    fontsize=30, tickwidth=10, linewidth=10)
visualize.show()

visualize.plot_per_joint_bar_chart(halnet_means_per_joint, halnet_err_per_joint, fingertips_only=True, added_avg_value=True,
                                   horizontal=True, xlabel='Joint dist loss (pixels)',
                                   ylabel='Joint name', title='{} : HALNet: Loss per Joint (pixels)'.format(dataset_name))
visualize.show()

visualize.plot_per_joint_bar_chart(jornet_means_per_joint, jornet_err_per_joint, fingertips_only=True, added_avg_value=True,
                                   horizontal=True, xlabel='Joint dist loss (pixels)',
                                   ylabel='Joint name', title='{} : JORNet: Loss per Joint (pixel)'.format(dataset_name))
visualize.show()

visualize.plot_per_joint_bar_chart(jornet_means_per_joint_depth, jornet_err_per_joint_depth, fingertips_only=True, added_avg_value=True,
                                   horizontal=True, xlabel='Joint dist loss (mm)',
                                   ylabel='Joint name', title='{} : JORNet: Loss per Joint (depth)'.format(dataset_name))
visualize.show()
