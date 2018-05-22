import torch
from torch.autograd import Variable
import argparse
import HALNet, JORNet
import synthhands_handler
import numpy as np
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

def convert_data_to_batch(data):
    batch = np.zeros((1, data.shape[0], data.shape[1], data.shape[2]))
    batch[0, :, :, :] = data
    batch = Variable(torch.from_numpy(batch).float())
    return batch

def predict_from_image(args, halnet, jornet, img_res=(320, 240)):
    data = synthhands_handler._get_data(args.dataset_folder, args.input_img_namebase, img_res)
    labels_jointspace, labels_colorspace, labels_joint_depth_z =\
        synthhands_handler.get_labels_depth_and_color(args.dataset_folder, args.input_img_namebase)
    labels_jointvec, handroot = synthhands_handler.get_labels_jointvec(labels_jointspace, range(21), rel_root=False)
    handroot = labels_jointvec[0:3]

    batch_halnet = convert_data_to_batch(data)
    output_halnet = halnet(batch_halnet)

    filenamebase = args.input_img_namebase
    fig = visualize.create_fig()
    visualize.plot_joints_from_heatmaps(output_halnet[3][0].data.numpy(), fig=fig,
                                        title=filenamebase, data=batch_halnet[0].data.numpy())
    visualize.show()

    visualize.plot_image_and_heatmap(output_halnet[3][0][8].data.numpy(),
                                     data=data[0].data.numpy(),
                                     title=filenamebase + ' : index finger tip')
    visualize.show()

    labels_colorspace = conv.heatmaps_to_joints_colorspace(output_halnet[3][0].data.numpy())
    data_crop, crop_coords, labels_heatmaps, labels_colorspace = \
        synthhands_handler.crop_image_get_labels(batch_halnet[0].data.numpy(), labels_colorspace, range(21))
    visualize.plot_image(data_crop, title=filenamebase, fig=fig)
    visualize.plot_joints_from_colorspace(labels_colorspace, title=filenamebase, fig=fig, data=data_crop)
    visualize.show()

    batch_jornet = convert_data_to_batch(data_crop)
    output_jornet = jornet(batch_jornet)

    output_jornet_batch_numpy = output_jornet[7][0].data.cpu().numpy()
    visualize.plot_3D_joints(output_jornet_batch_numpy)
    visualize.show()

    temp = np.zeros((21, 3))
    output_batch_numpy_abs = output_jornet_batch_numpy.reshape((20, 3))
    temp[1:, :] = output_batch_numpy_abs
    output_batch_numpy_abs = temp
    output_joints_colorspace = camera.joints_depth2color(
        output_batch_numpy_abs,
        depth_intr_matrix=synthhands_handler.DEPTH_INTR_MTX,
        handroot=handroot)
    visualize.plot_3D_joints(output_joints_colorspace)
    visualize.show()

    fig = visualize.plot_image(batch_halnet[0].data.numpy(), title=filenamebase)
    visualize.plot_joints_from_colorspace(output_joints_colorspace, fig=fig)
    visualize.show()

    aa = 0


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
    predict_from_image(args, halnet, jornet)



