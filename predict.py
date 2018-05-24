import torch
import argparse
import HALNet, JORNet
import synthhands_handler
import visualize
import trainer
import converter as conv
import camera

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-i', dest='input_img_namebase', default='', type=str, required=False,
                    help='Input image file name base (e.g. female_noobject/seq01/cam01/01/00000000')
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

if args.use_cuda:
    torch.set_default_tensor_type('torch.cuda.Floaimg_filenamebasetTensor')

def predict_from_dataset(args, halnet, jornet):
    valid_loader = synthhands_handler.get_SynthHands_validloader(root_folder=args.dataset_folder,
                                                                 joint_ixs=range(21),
                                                                 heatmap_res=(320, 240),
                                                                 batch_size=pred_vars['max_mem_batch'],
                                                                 verbose=True)


def plot_halnet_joints_from_heatmaps(halnet_main_out, img_numpy, filenamebase):
    fig = visualize.create_fig()
    visualize.plot_joints_from_heatmaps(halnet_main_out, fig=fig, data=img_numpy)
    visualize.title('HALNet (joints from heatmaps): ' + filenamebase)
    visualize.show()

def plot_halnet_heatmap(halnet_mainout, img_numpy, heatmap_ix, filenamebase):
    visualize.plot_image_and_heatmap(halnet_mainout[heatmap_ix], data=img_numpy)
    joint_name = synthhands_handler.get_joint_name_from_ix(heatmap_ix)
    visualize.title('HALNet (heatmap for ' + joint_name + '): ' + filenamebase)
    visualize.show()

def plot_halnet_joints_from_heatmaps_crop(halnet_main_out, img_numpy, filenamebase, plot=True):
    labels_colorspace = conv.heatmaps_to_joints_colorspace(halnet_main_out)
    data_crop, crop_coords, labels_heatmaps, labels_colorspace = \
        synthhands_handler.crop_image_get_labels(img_numpy, labels_colorspace, range(21))
    if plot:
        fig = visualize.create_fig()
        visualize.plot_image(data_crop, title=filenamebase, fig=fig)
        visualize.plot_joints_from_colorspace(labels_colorspace, title=filenamebase, fig=fig, data=data_crop)
        visualize.title('HALNet (joints from heatmaps - cropped): ' + filenamebase)
        visualize.show()
    return data_crop

def plot_jornet_joints_global_depth(joints_global_depth, filenamebase,
                                    gt_joints=None, color_jornet_joints='C6'):
    if gt_joints is None:
        visualize.plot_3D_joints(joints_global_depth)
    else:
        fig, ax = visualize.plot_3D_joints(gt_joints)
        visualize.plot_3D_joints(joints_global_depth, fig=fig, ax=ax, color=color_jornet_joints)
    visualize.title('JORNet (GT multi-coloured; JORNet single color): ' + filenamebase)
    visualize.show()
    return joints_global_depth

def get_jornet_colorspace(joints_global_depth, handroot):
    joints_color_orig_res = camera.joints_depth2color(
        joints_global_depth,
        depth_intr_matrix=synthhands_handler.DEPTH_INTR_MTX,
        handroot=handroot)
    return joints_color_orig_res


def predict_from_image(args, halnet, jornet, img_res=(320, 240), orig_res=(640, 480)):
    # get data and labels
    data = synthhands_handler._get_data(args.dataset_folder, args.input_img_namebase, img_res)
    labels_jointspace, labels_colorspace, labels_joint_depth_z =\
        synthhands_handler.get_labels_depth_and_color(args.dataset_folder, args.input_img_namebase)
    # plot HALnet predictions
    output_halnet = halnet(conv.data_to_batch(data))
    halnet_main_out = output_halnet[3][0].data.numpy()
    img_numpy = data.data.numpy()
    #plot_halnet_heatmap(halnet_main_out, img_numpy, 8, args.input_img_namebase)
    #plot_halnet_joints_from_heatmaps(halnet_main_out, img_numpy, args.input_img_namebase)
    data_crop = plot_halnet_joints_from_heatmaps_crop(halnet_main_out, img_numpy, args.input_img_namebase, plot=False)
    # get JORNet outputs
    handroot = labels_jointspace[0, 0:3]
    batch_jornet = convert_data_to_batch(data_crop)
    output_jornet = jornet(batch_jornet)
    jornet_joints_mainout = output_jornet[7][0].data.cpu().numpy()
    # plot depth
    jornet_joints_mainout *= 1.1
    jornet_joints_global = get_jornet_global_depth(jornet_joints_mainout, handroot)
    plot_jornet_joints_global_depth(jornet_joints_global, args.input_img_namebase, gt_joints=labels_jointspace)
    joints_colorspace = joints_globaldepth_to_colorspace(jornet_joints_global, handroot, img_res=(640, 480))
    plot_jornet_colorspace(joints_colorspace, args.input_img_namebase)
    return output_halnet, output_jornet, jornet_joints_global


# load nets
print('Loading HALNet from: ' + args.halnet_filepath)
halnet, _, _, _ = trainer.load_checkpoint(filename=args.halnet_filepath,
                                          model_class=HALNet.HALNet,
                                          use_cuda=args.use_cuda)
print('Loading JORNet from: ' + args.jornet_filepath)
jornet, _, _, _ = trainer.load_checkpoint(filename=args.jornet_filepath,
                                          model_class=JORNet.JORNet,
                                          use_cuda=args.use_cuda)


if args.input_img_namebase == '':
    predict_from_dataset(args, halnet, jornet)
elif args.dataset_folder == '':
    raise('You need to define either a dataset folder (-r) or an image file name base (-i)')
else:
    for i in range(10):
        args.input_img_namebase = args.input_img_namebase[0:-1] + str(i)
        predict_from_image(args, halnet, jornet)



