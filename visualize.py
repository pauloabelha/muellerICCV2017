import converter
import io_image

try:
    import cv2
except ImportError:
    print("WARNING: Ignoring opencv import error")
    pass
try:
    from torchviz import make_dot
except ImportError:
    print("WARNING: Ignoring torchviz import error")
    pass
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("WARNING: Ignoring matplotlib import error")
    pass
import numpy as np
import synthhands_handler
import camera
from torch.autograd import Variable
import torch
import pylab
import converter as conv
import matplotlib.patches as mpatches
import math
import converter as conv
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!


    #data_img_RGB = conv.numpy_to_plottable_rgb(data)
        #fig = visualize.plot_img_RGB(data_img_RGB, title=filenamebase)
        #visualize.plot_joints(joints_colorspace=labels_colorspace, num_joints=len(joint_ixs), fig=fig)
        #visualize.savefig('/home/paulo/' + filenamebase.replace('/', '_') + '_' + 'orig')
        #visualize.show()
        #data, crop_coords, labels_heatmaps, labels_colorspace =\
        #    crop_image_get_labels(data, labels_colorspace, joint_ixs)
        #data_img_RGB = conv.numpy_to_plottable_rgb(data)
        #fig = visualize.plot_img_RGB(data_img_RGB, title=filenamebase)
        #visualize.plot_3D_joints(joints_vec=labels_jointvec)
        #visualize.plot_joints(joints_colorspace=labels_colorspace, fig=fig)
        #visualize.show()


def save_graph_pytorch_model(model, model_input_shape, folder='', modelname='model', plot=False):
    x = Variable(torch.randn(model_input_shape), requires_grad=True)
    y = model(x)
    dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    dot.render(folder + modelname + '.gv', view=plot)


def show_nparray_with_matplotlib(np_array, img_title='Image'):
    plt.imshow(np_array)
    plt.title(img_title)
    plt.show()

def _add_small_square(image, u, v, color=[0, 0, 100], square_size=10):
    '''

    :param u: u in pixel space
    :param v: v in pixel space
    :return:
    '''
    half_square_size = int(square_size/2)
    for i in range(square_size):
        for j in range(square_size):
            new_u_ix = u - half_square_size + i
            if new_u_ix < 0 or new_u_ix >= image.shape[0]:
                continue
            new_v_ix = v - half_square_size + j
            if new_v_ix < 0 or new_v_ix >= image.shape[1]:
                continue
            image[new_u_ix, new_v_ix, 0] = color[0]
            image[new_u_ix, new_v_ix, 1] = color[1]
            image[new_u_ix, new_v_ix, 2] = color[2]
            #print(image[u - half_square_size + i, v - half_square_size + j, :])
    return image

def add_squares_for_joint_in_color_space(image, joint, color=[0, 0, 100]):
    u, v = joint
    image = _add_small_square(image, u, v, color)
    return image

def _add_squares_for_joints(image, joints, depth_intr_matrix):
    '''

    :param image: image to which add joint squares
    :param joints: joints in depth camera space
    :param depth_intr_mtx: depth camera intrinsic params
    :return: image with added square for each joint
    '''
    joints_color_space = np.zeros((joints.shape[0], 2))
    for joint_ix in range(joints.shape[0]):
        joint = joints[joint_ix, :]
        u, v = camera.joint_depth2color(joint, depth_intr_matrix)
        image = _add_small_square(image, u, v)
        joints_color_space[joint_ix, 0] = u
        joints_color_space[joint_ix, 1] = v
    return image, joints_color_space

def show_me_example(example_ix_str, depth_intr_matrix):
    '''

        :return: image of first example in dataset (also plot it)
        '''

    image = io_image._read_RGB_image(
        "/home/paulo/synthhands/example_data/01/00000" +
        example_ix_str + "_color_on_depth.png")

    joint_label = synthhands_handler._read_label(
        "/home/paulo/synthhands/example_data/01/00000" +
        example_ix_str + "_joint_pos.txt")

    joints = np.array(joint_label).astype(float)
    image, joints_color_space\
        = _add_squares_for_joints(image, joints, depth_intr_matrix)

    cv2.imshow('Example image with joints as blue squares', image)
    print("Press 0 to close image...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image, joints_color_space

def show_me_an_example(depth_intr_matrix):
    '''

    :return: image of first example in dataset (also plot it)
    '''
    return show_me_example('000', depth_intr_matrix)

def show_dataset_example_with_joints(dataset, example_ix=0):
    filenamebases = dataset.filenamebases
    img_title = "File namebase: " + dataset.color_on_depth_images_dict[
        filenamebases[example_ix]]
    print("\t" + str(example_ix+1) + " - " + img_title)
    # deal with image
    example_data, example_label = dataset[example_ix]
    final_image = converter.convert_torch_dataimage_to_canonical(example_data)
    # deal with label
    for i in range(20):
        joint_uv = dataset.get_colorspace_joint_of_example_ix(example_ix, i)
        #print("\tJoint " + str(i) + " (u,v): (" + str(joint_uv[0])
        #      + ", " + str(joint_uv[1]) + ")")
        final_image = \
            add_squares_for_joint_in_color_space(
                final_image, joint_uv, color=[i*10, 100-i*5, 100+i*5])
    img_title = "File namebase: " + dataset.color_on_depth_images_dict[
        filenamebases[example_ix]]
    show_nparray_with_matplotlib(final_image, img_title=img_title)

def show_data_as_image(example_data):
    data_image = converter.convert_torch_dataimage_to_canonical(example_data)
    plt.imshow(data_image)
    plt.show()

def show_halnet_data_as_image(dataset, example_ix=0):
    example_data, example_label = dataset[example_ix]
    show_data_as_image(example_data)

def show_halnet_output_as_heatmap(heatmap, image=None, img_title=''):
    heatmap = converter.convert_torch_targetheatmap_to_canonical(heatmap)
    heatmap = heatmap.swapaxes(0, 1)
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    if not image is None:
        image = converter.convert_torch_dataimage_to_canonical(image)
        image = image.swapaxes(0, 1)
        plt.imshow(image)
        plt.imshow(255 * heatmap, alpha=0.6, cmap='hot')
    plt.title(img_title)
    plt.show()

def plot_img_RGB(img_RGB, fig=None, title=''):
    if fig is None:
        fig = plt.figure()
    plt.imshow(img_RGB)
    plt.title(title)
    return fig

def plot_joints(joints_colorspace, fig=None, show_legend=True, linewidth=4):
    if fig is None:
        fig = plt.figure()
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
            joint_name = joints_names[math.floor((i+1)/4)]
            legends.append(mpatches.Patch(color=color, label=joint_name))
    if show_legend:
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=legends)
        plt.legend(handles=legends)
    return fig

def plot_joints_from_heatmaps(heatmaps, data=None, title='', fig=None, linewidth=2):
    if fig is None:
        fig = plt.figure()
    joints_colorspace = conv.heatmaps_to_joints_colorspace(heatmaps)
    fig = plot_joints(joints_colorspace, fig=fig, linewidth=linewidth)
    if not data is None:
        data_img_RGB = conv.numpy_to_plottable_rgb(data)
        fig = plot_img_RGB(data_img_RGB, fig=fig, title=title)
    return fig

def plot_joints_from_colorspace(joints_colorspace, data=None, title='', fig=None, linewidth=2):
    if fig is None:
        fig = plt.figure()
    fig = plot_joints(joints_colorspace, fig=fig, linewidth=linewidth)
    if not data is None:
        data_img_RGB = conv.numpy_to_plottable_rgb(data)
        fig = plot_img_RGB(data_img_RGB, fig=fig, title=title)
    return fig

def plot_3D_joints(joints_vec):
    fig = plt.figure()
    ax = Axes3D(fig)
    joints_vec = joints_vec.reshape((21, 3))
    for i in range(5):
        idx = (i * 4) + 1
        ax.plot([joints_vec[0, 1], joints_vec[idx, 1]],
                [joints_vec[0, 0], joints_vec[idx, 0]],
                [joints_vec[0, 2], joints_vec[idx, 2]],
                label='',
                color='C0')
    for j in range(5):
        idx = (j * 4) + 1
        for i in range(3):
            ax.plot([joints_vec[idx, 1], joints_vec[idx + 1, 1]],
                    [joints_vec[idx, 0], joints_vec[idx + 1, 0]],
                    [joints_vec[idx, 2], joints_vec[idx + 1, 2]],
                    label='',
                    color='C' + str(j+1))
            idx += 1
    print(joints_vec)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_zlim(0, 500)
    ax.view_init(azim=270, elev=250)
    plt.show()
    aa =0

def plot_image(data, title='', fig=None):
    if fig is None:
        fig = plt.figure()
    data_img_RGB = conv.numpy_to_plottable_rgb(data)
    plt.imshow(data_img_RGB)
    if not title == '':
        plt.title(title)
    return fig

def plot_image_and_heatmap(heatmap, data, title=''):
    plot_image(data, title=title)
    heatmap = np.exp(heatmap)
    heatmap = heatmap.swapaxes(0, 1)
    plt.imshow(255 * heatmap, alpha=0.6, cmap='hot')

def show():
    plt.show()

def savefig(filepath):
    pylab.savefig(filepath)

def create_fig():
    return plt.figure()
