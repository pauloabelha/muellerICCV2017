# exampel calls
# -i female_object/seq01/cam01/01/00000000 -r /home/paulo/SynthHands_Release/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_for_valid_38000.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_70000.pth.tar
# -i Fruits/color_on_depth/image_00000 -r /home/paulo/EgoDexter/data/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_30000.pth.tar

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import trainer
import synthhands_handler
import egodexter_handler
import argparse
import converter as conv
import HALNet, JORNet
import time
import camera
import visualize

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-i', dest='input_img_namebase', default='', type=str, required=False,
                    help='Inpimport HALNet, JORNetut image file name base (e.g. female_noobject/seq01/cam01/01/00000000')
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
args = parser.parse_args()

dataset_name = args.dataset_folder.split('/')[-2]

def print_time(str_, time_diff):
    print(str_ + str(round(time_diff*1000)) + ' ms')

# load nets
start = time.time()
halnet, _, _, _ = trainer.load_checkpoint(filename=args.halnet_filepath,
                                          model_class=HALNet.HALNet,
                                          use_cuda=args.use_cuda)
print_time('HALNet loading: ', time.time() - start)

start = time.time()
jornet, _, _, _ = trainer.load_checkpoint(filename=args.jornet_filepath,
                                          model_class=JORNet.JORNet,
                                          use_cuda=args.use_cuda)
print_time('JORNet loading: ', time.time() - start)

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

def load_images_to_memory(num_images, dataset_folder, dataset_name, img_res):
    images = []
    for i in range(num_images):
        input_img_namebase = get_image_name(args.input_img_namebase, i, dataset_name)
        data = get_image_as_data(dataset_folder, input_img_namebase, dataset_name, img_res)
        images.append(data)
    return images

start = time.time()
images = load_images_to_memory(100, args.dataset_folder, dataset_name, (320, 240))
print_time('Loading images to memory: ', time.time() - start)

joints_colorspace = np.zeros((21, 2))

for i in range(100):
    start_beg = time.time()
    print('--------------------------------------------------------------------------')
    print(args.input_img_namebase)

    start = time.time()
    input_img_namebase = get_image_name(args.input_img_namebase, i, dataset_name)
    print_time('Image reading: ', time.time() - start)

    data = images[i]
    img_numpy = data.data.numpy()

    start = time.time()
    output_halnet = halnet(conv.data_to_batch(data))
    print_time('HALNet pass: ', time.time() - start)

    start = time.time()
    halnet_main_out = output_halnet[3][0].data.numpy()
    handroot_colorspace = np.unravel_index(np.argmax(halnet_main_out[0]), halnet_main_out[0].shape)
    handroot = camera.joint_color2depth(handroot_colorspace[0], handroot_colorspace[1],
                                        300,
                                        egodexter_handler.DEPTH_INTR_MTX_INV)
    print('Handroot (colorspace):\t{}'.format(handroot_colorspace))
    print('Handroot (colorspace), z:\t{}'.format(img_numpy[3, handroot_colorspace[0], handroot_colorspace[1]]))
    print('Handroot (depthspace):\t{}'.format(handroot))
    labels_colorspace = conv.heatmaps_to_joints_colorspace(halnet_main_out)

    data_crop, _, _, _ = synthhands_handler.crop_image_get_labels(img_numpy, labels_colorspace, range(21))
    batch_jornet = conv.data_to_batch(data_crop)
    print_time('JORNet image conversion: ', time.time() - start)

    start = time.time()
    output_jornet = jornet(batch_jornet)
    print_time('JORNet pass: ', time.time() - start)

    start = time.time()
    jornet_joints_mainout = output_jornet[7][0].data.cpu().numpy()

    jornet_joints_global = conv.jornet_local_to_global_joints(jornet_joints_mainout, handroot)
    joints_colorspace = conv.joints_globaldepth_to_colorspace(jornet_joints_global, handroot, img_res=(320, 240))
    mov_0 = labels_colorspace[0, 0] - joints_colorspace[0, 0]
    mov_1 = labels_colorspace[0, 1] - joints_colorspace[0, 1]
    for i in range(joints_colorspace.shape[0] ):
        joints_colorspace[i, 0] += mov_0
        joints_colorspace[i, 1] += mov_1
    print_time('Plot image preparation: ', time.time() - start)

    plt.imshow(conv.numpy_to_plottable_rgb(img_numpy))
    plot_joints(labels_colorspace, show_legend=False)
    total_elapsed_time = round(time.time() - start_beg, 2)
    plt.title(input_img_namebase + ' : ' + str(total_elapsed_time) + ' ms')
    plt.pause(0.001)
    plt.clf()
    print('--------------------------------------------------------------------------')

plt.show()