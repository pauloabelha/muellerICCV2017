import converter
import io_image
import matplotlib

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

def plot_fingertips(fingertips_colorspace, handroot=None, fig=None, linewidth=10):
    if fig is None:
        fig = plt.figure()
    joints_names = ['Thumb TIP', 'Index TIP', 'Middle TIP', 'Ring TIP', 'Little TIP']
    legends = []
    for i in range(5):
        color = 'C' + str(i+1)
        plt.scatter(fingertips_colorspace[i, 0], fingertips_colorspace[i, 1], color=color, linewidths=linewidth)
        legends.append(mpatches.Patch(color=color, label=joints_names[i]))
    if not handroot is None:
        plt.scatter(handroot[0], handroot[1], color='C0', linewidths=linewidth)
        legends.append(mpatches.Patch(color='C0', label='Hand root'))
    plt.legend(handles=legends)
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
            joint_name = joints_names[int(math.floor((i+1)/4))]
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

def plot_3D_joints(joints_vec, title='', fig=None, ax=None, color=None):
    if fig is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    if joints_vec.shape[0] == 60:
        joints_vec = joints_vec.reshape((20, 3))
        joints_vec = np.vstack([np.zeros((1, 3)), joints_vec])
    else:
        joints_vec = joints_vec.reshape((21, 3))
    for i in range(5):
        idx = (i * 4) + 1
        if color is None:
            curr_color = 'C0'
        else:
            curr_color = color
        ax.plot([joints_vec[0, 1], joints_vec[idx, 1]],
                [joints_vec[0, 0], joints_vec[idx, 0]],
                [joints_vec[0, 2], joints_vec[idx, 2]],
                label='',
                color=curr_color)
    for j in range(5):
        idx = (j * 4) + 1
        for i in range(3):
            if color is None:
                curr_color = 'C' + str(j+1)
            else:
                curr_color = color
            ax.plot([joints_vec[idx, 1], joints_vec[idx + 1, 1]],
                    [joints_vec[idx, 0], joints_vec[idx + 1, 0]],
                    [joints_vec[idx, 2], joints_vec[idx + 1, 2]],
                    label='',
                    color=curr_color)
            idx += 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlim(0, 640)
    #ax.set_ylim(0, 480)
    #ax.set_zlim(0, 500)
    ax.view_init(azim=0, elev=180)
    ax.set_title(title)
    return fig, ax

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

def plot_line(values, fontsize=22, linewidth=3, tickwidth=3, xlabel='', ylabel='', title=''):
    plt.plot(values, linewidth=linewidth)
    ax = plt.gca()
    matplotlib.rcParams.update({'font.size': fontsize})
    ax.tick_params(width=tickwidth)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(title)

def title(title):
    plt.title(title)

def show():
    plt.show()

def savefig(filepath):
    pylab.savefig(filepath)

def create_fig():
    return plt.figure()


def plot_jornet_colorspace(joints_color_orig_res, filenamebase, dataset_folder, input_img_namebase, img_res=(640, 480)):
    data = synthhands_handler._get_data(dataset_folder, filenamebase, img_res)
    fig = plot_image(data[0:3, :, :].data.numpy(), title=input_img_namebase)
    plot_joints_from_colorspace(joints_color_orig_res, fig=fig)
    return joints_color_orig_res

def plot_bar_chart(bar_values, names_tuple, bar_err=None, horizontal=False, xlabel='', ylabel='', title=''):
    pos = np.arange(len(names_tuple))
    if horizontal:
        if bar_err is None:
            plt.barh(pos, bar_values, align='center', alpha=0.5)
        else:
            plt.barh(pos, bar_values, xerr=bar_err, align='center', alpha=0.5)
        plt.yticks(pos, names_tuple)
    else:
        if bar_err is None:
            plt.bar(pos, bar_values, align='center', alpha=0.5)
        else:
            plt.bar(pos, bar_values, yerr=bar_err, align='center', alpha=0.5)
        plt.xticks(pos, names_tuple)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def get_joint_names():
    joint_names = [''] * 21
    joint_names[0] = 'Hand root'
    joint_names[1] = 'Thumb MCP'
    joint_names[2] = 'Thumb DIP'
    joint_names[3] = 'Thumb PIP'
    joint_names[4] = 'Thumb TIP'
    joint_names[5] = 'Index MCP'
    joint_names[6] = 'Index DIP'
    joint_names[7] = 'Index PIP'
    joint_names[8] = 'Index TIP'
    joint_names[9] = 'Middle MCP'
    joint_names[10] = 'Middle DIP'
    joint_names[11] = 'Middle PIP'
    joint_names[12] = 'Middle TIP'
    joint_names[13] = 'Ring MCP'
    joint_names[14] = 'Ring DIP'
    joint_names[15] = 'Ring PIP'
    joint_names[16] = 'Ring TIP'
    joint_names[17] = 'Little MCP'
    joint_names[18] = 'Little DIP'
    joint_names[19] = 'Little PIP'
    joint_names[20] = 'Little TIP'
    return joint_names

def get_fingertip_names():
    joint_names = [''] * 5
    joint_names[0] = 'Thumb TIP'
    joint_names[1] = 'Index TIP'
    joint_names[2] = 'Middle TIP'
    joint_names[3] = 'Ring TIP'
    joint_names[4] = 'Little TIP'
    return joint_names


def plot_per_joint_bar_chart(joint_values, joint_std=None, fingertips_only=False, added_avg_value=False, horizontal=False, xlabel='', ylabel='', title=''):
    if fingertips_only:
        joint_names = get_fingertip_names()
    else:
        joint_names = get_joint_names()
    if added_avg_value:
        joint_names.append('Average')
    plot_bar_chart(bar_values=joint_values, names_tuple=joint_names,  bar_err=joint_std,
                   horizontal=horizontal, xlabel=xlabel, ylabel=ylabel, title=title)