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
import camera
import visualize


MAX_NUM_EXAMPLES = 10
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

def load_examples_to_memory(num_labels, dataset_folder, img_res):
    data = []
    img_paths = []
    egodexter = egodexter_handler.EgoDexterDataset('full', dataset_folder, img_res)
    for i in range(num_labels):
        img_paths.append(egodexter.filenamebases[i])
        img_data, img_labels = egodexter[i]
        data.append((img_data, img_labels))
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
    return parser.parse_args()

def print_divisor(num=100):
    print('-' * num)

args = parse_args()
dataset_name = args.dataset_folder.split('/')[-2]

egodexter = egodexter_handler.EgoDexterDataset('full', args.dataset_folder, IMG_RES)
num_examples = min(MAX_NUM_EXAMPLES, len(egodexter))

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
dataset_data, img_paths = load_examples_to_memory(num_examples, args.dataset_folder, img_res=IMG_RES)
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

losses_halnet_fingertips = []
losses_jornet_fingertips = []
losses_jornet_depth = []
num_valid_loss_iter = 0
tot_loss = 0



for example_ix in range(num_examples):
    start_beg = time.time()

    print('Processing example #{}: {}'.format(example_ix, img_paths[example_ix]))
    dataset_datum = dataset_data[example_ix]
    img_data, img_labels = dataset_datum
    img_numpy = img_data.data.numpy()
    img_labels_2D, img_labels_heatmaps, img_labels_3D = img_labels
    img_labels_2D = img_labels_2D.data.numpy()


    #print('\tImage labels (2D ; 3D):')
    #for i in range(5):
    #    if np.sum(img_labels_2D[i, :]) > 0:
    #        depth_const = img_numpy[3, int(img_labels_2D[i, 0]), int(img_labels_2D[i, 1])]
    #        print('\t\tFinger tip {}: {} {}; {}'.format(i, img_labels_2D[i, :], depth_const, img_labels_3D[i, :]))
    #        print('\t\tFinger tip {}: {}'.format(i, img_labels_2D[i, :]))
    #    else:
    #        print('\t\tFinger tip {}: {}'.format(i, img_labels_2D[i, :]))
    #print_divisor()

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

    print('\tHALNet heatmap pixel loss:')
    num_valid_loss = 0
    loss_halnet_fingertips = 0
    for i in range(5):
        fingertip_ix = (i+1)*4
        if np.sum(img_labels_2D[i, :]) > 0:
            curr_loss = np.linalg.norm(img_labels_2D[i, :] - halnet_joints_colorspace[fingertip_ix, :])
            loss_halnet_fingertips += curr_loss
            print('\t\t\tFinger tip {}: {}\t{} : {}'.format(i, img_labels_2D[i, :], halnet_joints_colorspace[fingertip_ix, :], curr_loss))
            num_valid_loss += 1
        else:
            print('\t\t\tFinger tip {}: {}\t{}'.format(i, 'No label', halnet_joints_colorspace[fingertip_ix, :]))
    if num_valid_loss > 0:
        loss_halnet_fingertips /= num_valid_loss
        losses_halnet_fingertips.append(loss_halnet_fingertips)
        print('\t\tAverage loss for HALNet fingertips: {}'.format(loss_halnet_fingertips))
    else:
        print('\t\tAverage loss for  HALNet fingertips: {}'.format('No loss'))
    print_divisor()

    #fig = visualize.plot_image(img_numpy)
    #visualize.plot_joints(halnet_joints_colorspace, fig=fig)
    #visualize.show()

    print('\tJORNet:')

    start = time.time()
    data_crop, crop_coords, _, _ = io_image.crop_image_get_labels(img_numpy, halnet_joints_colorspace)
    batch_jornet = conv.data_to_batch(data_crop)
    print_time('\t\tJORNet image conversion: ', time.time() - start)

    start = time.time()
    output_jornet = jornet(batch_jornet)
    print_time('\t\tJORNet pass: ', time.time() - start)

    _, img_labels_2D_cropped = io_image.get_labels_cropped_heatmaps(
        img_labels_2D, joint_ixs=range(5),
        crop_coords=crop_coords, heatmap_res=(128, 128))

    print('\tImage labels (2D cropped to (128, 128)):')
    for i in range(5):
        depth_const = img_numpy[3, int(img_labels_2D_cropped[i, 0]), int(img_labels_2D_cropped[i, 1])]
        if np.sum(img_labels_2D[i, :]) > 0:
            print('\t\tFinger tip {}: {}'.format(i, img_labels_2D_cropped[i, :]))
        else:
            print('\t\tFinger tip {}: No label'.format(i))
    print_divisor()

    print('\tJORNet heatmap pixel loss:')
    output_jornet_heatmaps_main = output_jornet[3][0].data.numpy()
    jornet_joints_colorspace = conv.heatmaps_to_joints_colorspace(output_jornet_heatmaps_main)
    num_valid_loss = 0
    loss_jornet_fingertips = 0
    for i in range(5):
        fingertip_ix = (i + 1) * 4
        if np.sum(img_labels_2D[i, :]) > 0:
            curr_loss = np.linalg.norm(img_labels_2D_cropped[i, :] - jornet_joints_colorspace[fingertip_ix, :])
            loss_jornet_fingertips += curr_loss
            print('\t\t\tFinger tip {}: {}\t{} : {}'.format(i, img_labels_2D_cropped[i, :], jornet_joints_colorspace[fingertip_ix, :],
                                                 curr_loss))
            num_valid_loss += 1
        else:
            print('\t\t\tFinger tip {}: {}\t{}'.format(i, 'No label', jornet_joints_colorspace[fingertip_ix, :]))
    if num_valid_loss > 0:
        loss_jornet_fingertips /= num_valid_loss
        losses_jornet_fingertips.append(loss_jornet_fingertips)
        print('\t\tAverage loss for JORNet fingertips: {}'.format(loss_jornet_fingertips))
    else:
        print('\t\tAverage loss for  JORNet fingertips: {}'.format('No loss'))
    print_divisor()

    #fig = visualize.plot_image(data_crop)
    #visualize.plot_joints(jornet_joints_colorspace, fig=fig)
    #visualize.show()

    output_jornet_joints_main = output_jornet[7][0].data.cpu().numpy().reshape((20, 3))
    handroot = camera.joint_color2depth(halnet_joints_colorspace[0, 0],
                                        halnet_joints_colorspace[0, 1],
                                        300,
                                        egodexter_handler.DEPTH_INTR_MTX_INV)
    jornet_joints_global = conv.jornet_local_to_global_joints(output_jornet_joints_main, handroot)

    print('\tJORNet joint depth loss:')
    num_valid_loss = 0
    loss_jornet_depth = 0
    for i in range(5):
        fingertip_ix = (i + 1) * 4
        if np.sum(img_labels_3D[i, :]) > 0:
            curr_loss = np.linalg.norm(img_labels_3D[i, :] - jornet_joints_global[fingertip_ix, :])
            loss_jornet_depth += curr_loss
            print('\t\t\tFinger tip {}: {}\t{} : {}'.format(i, img_labels_3D[i, :],
                                                            jornet_joints_global[fingertip_ix, :],
                                                            curr_loss))
            num_valid_loss += 1
        else:
            print('\t\t\tFinger tip {}: {}\t{}'.format(i, 'No label', jornet_joints_global[fingertip_ix, :]))
    if num_valid_loss > 0:
        loss_jornet_depth /= num_valid_loss
        losses_jornet_depth.append(loss_jornet_depth)
        print('\t\tAverage loss for JORNet fingertips: {}'.format(loss_jornet_depth))
    else:
        print('\t\tAverage loss for  JORNet fingertips: {}'.format('No loss'))
    print_divisor()

    print('\tLosses:')
    print('\tHALNet')
    print('\t\tAverage fingertip loss: {}'.format(np.mean(losses_halnet_fingertips)))
    print('\t\tStddev fingertip loss: {}'.format(np.std(losses_halnet_fingertips)))
    print('\tJORNet')
    print('\t\tAverage fingertip loss: {}'.format(np.mean(losses_jornet_fingertips)))
    print('\t\tStddev fingertip loss: {}'.format(np.std(losses_jornet_fingertips)))
    print('\t\tAverage depth loss: {}'.format(np.mean(losses_jornet_depth)))
    print('\t\tStddev depth loss: {}'.format(np.std(losses_jornet_depth)))
    print_divisor()