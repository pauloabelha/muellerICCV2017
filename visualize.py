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
import io_data
import camera
from torch.autograd import Variable
import torch

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

def _add_squares_for_joints(image, joints):
    '''

    :param image: image to which add joint squares
    :param joints: joints in depth camera space
    :param depth_intr_mtx: depth camera intrinsic params
    :return: image with added square for each joint
    '''
    joints_color_space = np.zeros((joints.shape[0], 2))
    for joint_ix in range(joints.shape[0]):
        joint = joints[joint_ix, :]
        u, v = camera.get_joint_in_color_space(joint)
        image = _add_small_square(image, u, v)
        joints_color_space[joint_ix, 0] = u
        joints_color_space[joint_ix, 1] = v
    return image, joints_color_space

def show_me_example(example_ix_str):
    '''

        :return: image of first example in dataset (also plot it)
        '''

    image = io_data._read_RGB_image(
        "/home/paulo/synthhands/example_data/01/00000" +
        example_ix_str + "_color_on_depth.png")

    joint_label = io_data._read_label(
        "/home/paulo/synthhands/example_data/01/00000" +
        example_ix_str + "_joint_pos.txt")

    joints = np.array(joint_label).astype(float)
    image, joints_color_space\
        = _add_squares_for_joints(image, joints)

    cv2.imshow('Example image with joints as blue squares', image)
    print("Press 0 to close image...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image, joints_color_space

def show_me_an_example():
    '''

    :return: image of first example in dataset (also plot it)
    '''
    return show_me_example('000')

def show_dataset_example_with_joints(dataset, example_ix=0):
    filenamebase_keys = dataset.filenamebase_keys
    img_title = "File namebase: " + dataset.color_on_depth_images_dict[
        filenamebase_keys[example_ix]]
    print("\t" + str(example_ix+1) + " - " + img_title)
    # deal with image
    example_data, example_label = dataset[example_ix]
    final_image = io_data.convert_torch_dataimage_to_canonical(example_data)
    # deal with label
    for i in range(20):
        joint_uv = dataset.get_colorspace_joint_of_example_ix(example_ix, i)
        #print("\tJoint " + str(i) + " (u,v): (" + str(joint_uv[0])
        #      + ", " + str(joint_uv[1]) + ")")
        final_image = \
            add_squares_for_joint_in_color_space(
                final_image, joint_uv, color=[i*10, 100-i*5, 100+i*5])
    img_title = "File namebase: " + dataset.color_on_depth_images_dict[
                                               filenamebase_keys[example_ix]]
    show_nparray_with_matplotlib(final_image, img_title=img_title)

def show_data_as_image(example_data):
    data_image = io_data.convert_torch_dataimage_to_canonical(example_data)
    plt.imshow(data_image)
    plt.show()

def show_halnet_data_as_image(dataset, example_ix=0):
    example_data, example_label = dataset[example_ix]
    show_data_as_image(example_data)

def show_halnet_output_as_heatmap(heatmap, image=None, img_title=''):
    heatmap = io_data.convert_torch_targetheatmap_to_canonical(heatmap)
    heatmap = heatmap.swapaxes(0, 1)
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    if not image is None:
        image = io_data.convert_torch_dataimage_to_canonical(image)
        image = image.swapaxes(0, 1)
        plt.imshow(image)
        plt.imshow(255 * heatmap, alpha=0.6, cmap='hot')
    plt.title(img_title)
    plt.show()
