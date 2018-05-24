# exampel calls
# -i female_object/seq01/cam01/01/00000000 -r /home/paulo/SynthHands_Release/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_30000.pth.tar
# -i Fruits/color_on_depth/image_00000 -r /home/paulo/EgoDexter/data/ --halnet /home/paulo/muellericcv2017/trainednets/trained_HALNet_1493752625_.pth.tar --jornet /home/paulo/muellericcv2017/trainednets/trained_JORNet_1662451312_for_valid_30000.pth.tar

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import trainer
import synthhands_handler
import argparse
import converter as conv
import HALNet, JORNet

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

# load nets
print('Loading HALNet from: ' + args.halnet_filepath)
halnet, _, _, _ = trainer.load_checkpoint(filename=args.halnet_filepath,
                                          model_class=HALNet.HALNet,
                                          use_cuda=args.use_cuda)
print('Loading JORNet from: ' + args.jornet_filepath)
jornet, _, _, _ = trainer.load_checkpoint(filename=args.jornet_filepath,
                                          model_class=JORNet.JORNet,
                                          use_cuda=args.use_cuda)

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

def get_image_name(image_basename, ix):
    if ix == 0:
        ix = 1
    algs = int(np.log10(ix))
    str_to_add = str(ix)
    print(ix)
    print(algs)
    print(image_basename)
    print(image_basename[0:-(algs+1)])
    image_basename = image_basename[:-(algs+1)] + str_to_add
    print(image_basename)
    print('--------------------------------------------------------')
    return image_basename

joints_colorspace = np.zeros((21, 2))
for i in range(100):
    input_img_namebase = get_image_name(args.input_img_namebase, i)

    data = synthhands_handler._get_data(args.dataset_folder, input_img_namebase, (320, 240))
    labels_jointspace, _, _ = synthhands_handler.\
        get_labels_depth_and_color(args.dataset_folder, input_img_namebase)
    handroot = labels_jointspace[0, 0:3]

    img_numpy = data.data.numpy()
    output_halnet = halnet(conv.data_to_batch(data))
    halnet_main_out = output_halnet[3][0].data.numpy()
    labels_colorspace = conv.heatmaps_to_joints_colorspace(halnet_main_out)

    data_crop, _, _, _ = synthhands_handler.crop_image_get_labels(img_numpy, labels_colorspace, range(21))
    batch_jornet = conv.data_to_batch(data_crop)
    output_jornet = jornet(batch_jornet)
    jornet_joints_mainout = output_jornet[7][0].data.cpu().numpy()
    # plot depth
    jornet_joints_mainout *= 1.1
    jornet_joints_global = conv.jornet_local_to_global_joints(jornet_joints_mainout, handroot)
    joints_colorspace = conv.joints_globaldepth_to_colorspace(jornet_joints_global, handroot, img_res=(320, 240))
    plt.imshow(conv.numpy_to_plottable_rgb(img_numpy))
    plot_joints(joints_colorspace, show_legend=False)
    plt.title(input_img_namebase)
    plt.pause(0.01)
    plt.clf()

plt.show()