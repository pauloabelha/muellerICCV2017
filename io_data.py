import os
import numpy as np
import camera
import pickle
import torch
from torch.utils.data.dataset import Dataset
import shutil
import torch.optim as optim
from scipy import misc
try:
    import cv2
except ImportError:
    print("WARNING: Ignoring opencv import error")
    pass
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import pyplot as plt
import converter as conv
import visualize


# resolution in rows x cols
LABEL_RESOLUTION_HALNET = (320, 240)
IMAGE_RES_ORIG = (480, 640)
IMAGE_RES_HALNET = (320, 240)
NUM_JOINTS = 21

def load_checkpoint(filename, model_class, num_iter=0, log_interval=0,
                    log_interval_valid=0, batch_size=0, max_mem_batch=0):
    # load file
    try:
        torch_file = torch.load(filename)
        override_to_not_use_cuda = False
    except RuntimeError:
        override_to_not_use_cuda = True
        torch_file = torch.load(filename, map_location=lambda storage, loc: storage)
    # load model
    model_state_dict = torch_file['model_state_dict']
    try:
        joint_ixs = torch_file['train_vars']['joint_ixs']
    except:
        joint_ixs = torch_file['joint_ixs']

    try:
        train_vars = torch_file['train_vars']
        control_vars = torch_file['control_vars']
    except:
        # for older model dicts
        train_vars = {}
        train_vars['losses'] = torch_file['losses']
        train_vars['pixel_losses'] = torch_file['dist_losses']
        train_vars['pixel_losses_sample'] = torch_file['dist_losses_sample']
        train_vars['best_loss'] = torch_file['best_loss']
        train_vars['best_pixel_loss'] = torch_file['best_dist_loss']
        train_vars['best_pixel_loss_sample'] = torch_file['best_dist_loss_sample']
        train_vars['best_model_dict'] = {}
        train_vars['joint_ixs'] = joint_ixs
        control_vars = {}
        control_vars['start_epoch'] = torch_file['epoch']
        control_vars['start_iter'] = torch_file['curr_iter']
        control_vars['num_iter'] = num_iter
        control_vars['best_model_dict'] = {}
        control_vars['log_interval'] = log_interval
        control_vars['log_interval_valid'] = log_interval_valid
        control_vars['batch_size'] = batch_size
        control_vars['max_mem_batch'] = max_mem_batch
        control_vars['iter_size'] = int(batch_size / max_mem_batch)
        control_vars['tot_toc'] = 0
    params_dict = {}
    params_dict['joint_ixs'] = train_vars['joint_ixs']
    params_dict['use_cuda'] = train_vars['use_cuda']
    params_dict['cross_entropy'] = train_vars['cross_entropy']

    if override_to_not_use_cuda:
        params_dict['use_cuda'] = False

    model = model_class(params_dict)
    model.load_state_dict(model_state_dict)
    # load optimizer
    optimizer_state_dict = torch_file['optimizer_state_dict']
    optimizer = optim.Adadelta(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)


    return model, optimizer, train_vars, control_vars

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_dataset_split(root_folder):
    return pickle.load(open(root_folder + "dataset_split_files.p", "rb"))

def save_dataset_split(dataset_root_folder, perc_train=0.7, perc_valid=0.15, perc_test=0.15):

    print("Recursively traversing all files in root folder: " + dataset_root_folder)
    orig_num_tabs = len(dataset_root_folder.split('/'))
    len_root_folder = len(dataset_root_folder)
    num_files_to_process = 0
    for root, dirs, files in os.walk(dataset_root_folder, topdown=True):
        for filename in sorted(files):
            if filename[-18:-4] == 'color_on_depth':
                num_files_to_process += 1
        tabs = '  ' * (len(root.split('/')) - orig_num_tabs)
        print('Counting files (' + str(num_files_to_process) + ')' +  tabs + root)
    print("Number of files to process: " + str(num_files_to_process))
    filenamebases = [0] * num_files_to_process
    ix = 0
    for root, dirs, files in os.walk(dataset_root_folder, topdown=True):
        for filename in sorted(files):
            if filename[-18:-4] == 'color_on_depth':
                filenamebases[ix] = os.path.join(root, filename[0:8])[len_root_folder:]
                ix += 1
        tabs = '  ' * (len(root.split('/')) - orig_num_tabs)
        print(str(ix) + '/' + str(num_files_to_process) + ' files processed : ' + tabs + root)
    print("Done traversing files")
    print("Randomising file names...")
    ixs_randomize = np.random.choice(len(filenamebases), len(filenamebases), replace=False)
    filenamebases = np.array(filenamebases)
    filenamebases_randomized = filenamebases[ixs_randomize]
    print("Splitting into training, validation and test sets...")
    num_train = int(np.floor(len(filenamebases) * perc_train))
    num_valid = int(np.floor(len(filenamebases) * perc_valid))
    filenamebases_train = filenamebases_randomized[0: num_train]
    filenamebases_valid = filenamebases_randomized[num_train: num_train + num_valid]
    filenamebases_test = filenamebases_randomized[num_train + num_valid:]
    print("Dataset split")
    print("Percentages of split: training " + str(perc_train*100) + "%, " +
          "validation " + str(perc_valid*100) + "% and " +
          "test " + str(perc_test*100) + "%")
    print("Number of files of split: training " + str(len(filenamebases_train)) + ", " +
          "validation " + str(len(filenamebases_valid)) + " and " +
          "test " + str(len(filenamebases_test)))
    print("Saving split into pickle file: " + 'dataset_split_files.p')
    data = {
            'dataset_root_folder': dataset_root_folder,
            'perc_train': perc_train,
            'perc_valid': perc_valid,
            'perc_test': perc_test,
            'filenamebases': filenamebases,
            'ixs_randomize': ixs_randomize,
            'filenamebases_train': filenamebases_train,
            'filenamebases_valid': filenamebases_valid,
            'filenamebases_test': filenamebases_test
            }
    with open('dataset_split_files.p', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_output(output):
    output_positive = output + abs(np.min(output, axis=(0, 1)))
    norm_output = output_positive / np.sum(output_positive, axis=(0, 1))
    return norm_output

def convert_torch_targetheatmap_to_canonical(target_heatmap):
    #target_heatmap = np.swapaxes(target_heatmap, 0, 1)
    assert target_heatmap.shape[0] == LABEL_RESOLUTION_HALNET[0]
    assert target_heatmap.shape[1] == LABEL_RESOLUTION_HALNET[1]
    return target_heatmap

def convert_torch_dataoutput_to_canonical(data):
    if len(data.shape) < 3:
        image = data
    else:
        image = data[0, :, :]
    # put channels at the end
    image = np.swapaxes(image, 0, 1)
    assert image.shape[0] == LABEL_RESOLUTION_HALNET[0]
    assert image.shape[1] == LABEL_RESOLUTION_HALNET[1]
    return image

def convert_torch_dataimage_to_canonical(data, keep_depth=False):
    image = data[0:3, :, :]
    # put channels at the end
    image = image.astype(np.uint8)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    assert image.shape[0] == IMAGE_RES_HALNET[0]
    assert image.shape[1] == IMAGE_RES_HALNET[1]
    assert image.shape[2] == 3
    return image

def convert_torch_dataimage_to_canonical(data):
    image = data[0:3, :, :]
    # put channels at the end
    image = image.astype(np.uint8)
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    assert image.shape[0] == IMAGE_RES_HALNET[0]
    assert image.shape[1] == IMAGE_RES_HALNET[1]
    assert image.shape[2] == 3
    return image

def convert_color_space_label_to_heatmap(color_space_label, heatmap_res, orig_img_res=IMAGE_RES_ORIG):
    '''
    Convert a (u,v) color-space label into a heatmap
    In this case, the heat map has only one value set to 1
    That is, the value (u,v)
    :param color_space_label: a pair (u,v) of color space joint position
    :param image_res: a pair (U, V) with the values for image resolution
    :return: numpy array of dimensions image_res with one position set to 1
    '''
    SMALL_PROB = 0
    heatmap = np.zeros(heatmap_res) + SMALL_PROB
    new_ix_res1 = int(color_space_label[0] /
                      (orig_img_res[0] / heatmap_res[0]))
    new_ix_res2 = int(color_space_label[1] /
                      (orig_img_res[1] / heatmap_res[1]))
    heatmap[new_ix_res1, new_ix_res2] = 1 - (SMALL_PROB * heatmap.size)
    return heatmap

def _get_data(root_folder, filenamebase, new_res, as_torch=True, depth_suffix='_depth.png', color_on_depth_suffix='_color_on_depth.png'):
    # load color
    color_on_depth_image_filename = root_folder + filenamebase + color_on_depth_suffix
    color_on_depth_image = _read_RGB_image(color_on_depth_image_filename, new_res=new_res)
    # load depth
    depth_image_filename = root_folder + filenamebase + depth_suffix
    depth_image = _read_RGB_image(depth_image_filename, new_res=new_res, depth=True)
    depth_image = np.array(depth_image)
    depth_image = np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
    # get data
    RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
    RGBD_image = RGBD_image.swapaxes(1, 2).swapaxes(0, 1)
    #data = RGBD_image.astype(np.float)
    data = RGBD_image
    if as_torch:
        data = torch.from_numpy(RGBD_image).float()
    return data

def get_labels_depth_and_color(root_folder, filenamebase, label_suffix='_joint_pos.txt'):
    label_filename = root_folder + filenamebase + label_suffix
    labels_jointspace = _read_label(label_filename)
    labels_colorspace = np.zeros((labels_jointspace.shape[0], 2))
    labels_joint_depth_z = np.zeros((labels_jointspace.shape[0], 1))
    for i in range(labels_jointspace.shape[0]):
        labels_colorspace[i, 0], labels_colorspace[i, 1],  labels_joint_depth_z[i] \
            = camera.joint_depth2color(labels_jointspace[i])
    return labels_jointspace, labels_colorspace

def get_labels_jointvec(labels_jointspace, joint_ixs, rel_root=False):
    labels_ix = 0
    labels_jointvec = np.zeros((len(joint_ixs) * 3,))
    hand_root = np.copy(labels_jointspace[0, :])
    for joint_ix in joint_ixs:
        # get joint pos relative to hand root (paper's p^L)
        if rel_root:
            labels_jointspace[joint_ix, :] -= hand_root
        labels_jointvec[labels_ix * 3:(labels_ix * 3) + 3] = labels_jointspace[joint_ix, :]
        labels_ix += 1
    return labels_jointvec

def get_labels_heatmaps_and_jointvec(labels_jointspace, labels_colorspace, joint_ixs, heatmap_res):
    labels_heatmaps = np.zeros((len(joint_ixs), heatmap_res[0], heatmap_res[1]))
    labels_ix = 0
    labels_jointvec = np.zeros((len(joint_ixs) * 3,))
    for joint_ix in joint_ixs:
        label = convert_color_space_label_to_heatmap(labels_colorspace[joint_ix, :], heatmap_res)
        label = label.astype(float)
        labels_heatmaps[labels_ix, :, :] = label
        # joint labels
        labels_jointvec[labels_ix * 3:(labels_ix * 3) + 3] = labels_jointspace[joint_ix, :]
        labels_ix += 1
    return labels_heatmaps, labels_jointvec

def _get_labels(root_folder, filenamebase, heatmap_res, joint_ixs, label_suffix='_joint_pos.txt'):
    labels_jointspace, labels_colorspace = \
        get_labels_depth_and_color(root_folder, filenamebase, label_suffix=label_suffix)
    labels_heatmaps, labels_jointvec = \
        get_labels_heatmaps_and_jointvec(labels_jointspace, labels_colorspace, joint_ixs, heatmap_res)
    labels_jointvec = torch.from_numpy(labels_jointvec).float()
    labels_heatmaps = torch.from_numpy(labels_heatmaps).float()
    return labels_heatmaps, labels_jointvec, labels_colorspace

def crop_hand_rgbd(joints_uv, image_rgbd, crop_res):
    min_u = min(joints_uv[:, 0]) - 10
    min_v = min(joints_uv[:, 1]) - 10
    max_u = max(joints_uv[:, 0]) + 10
    max_v = max(joints_uv[:, 1]) + 10
    u0 = int(max(min_u, 0))
    v0 = int(max(min_v, 0))
    u1 = int(min(max_u, image_rgbd.shape[1]))
    v1 = int(min(max_v, image_rgbd.shape[2]))
    # get coords
    coords = [u0, v0, u1, v1]
    # crop hand
    crop = image_rgbd[:, u0:u1, v0:v1]
    crop = crop.swapaxes(0, 1)
    crop = crop.swapaxes(1, 2)
    crop_rgb = change_res_image(crop[:, :, 0:3], crop_res)
    crop_depth = change_res_image(crop[:, :, 3], crop_res)
    # normalize depth
    crop_depth = np.divide(crop_depth, np.max(crop_depth))
    crop_depth = crop_depth.reshape(crop_depth.shape[0], crop_depth.shape[1], 1)
    crop_rgbd = np.append(crop_rgb, crop_depth, axis=2)
    crop_rgbd = crop_rgbd.swapaxes(1, 2)
    crop_rgbd = crop_rgbd.swapaxes(0, 1)
    return crop_rgbd, coords

def get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res):
    res_transf_u = (heatmap_res[0] / (crop_coords[2] - crop_coords[0]))
    res_transf_v = (heatmap_res[1] / (crop_coords[3] - crop_coords[1]))
    labels_ix = 0
    labels_heatmaps = np.zeros((len(joint_ixs), heatmap_res[0], heatmap_res[1]))
    labels_colorspace_mapped = np.copy(labels_colorspace)
    for joint_ix in joint_ixs:
        label_crop_local_u = labels_colorspace[joint_ix, 0] - crop_coords[0]
        label_crop_local_v = labels_colorspace[joint_ix, 1] - crop_coords[1]
        label_u = int(label_crop_local_u * res_transf_u)
        label_v = int(label_crop_local_v * res_transf_v)
        labels_colorspace_mapped[joint_ix, 0] = label_u
        labels_colorspace_mapped[joint_ix, 1] = label_v
        label = convert_color_space_label_to_heatmap(labels_colorspace_mapped[joint_ix, :], heatmap_res,
                                                     orig_img_res=heatmap_res)
        label = label.astype(float)
        labels_heatmaps[labels_ix, :, :] = label
        labels_ix += 1
    return labels_heatmaps, labels_colorspace_mapped

def crop_image_get_labels(data, labels_colorspace, joint_ixs, crop_res=(128, 128)):
    data, crop_coords = crop_hand_rgbd(labels_colorspace, data, crop_res=crop_res)
    labels_heatmaps, labels_colorspace =\
        get_labels_cropped_heatmaps(labels_colorspace, joint_ixs, crop_coords, heatmap_res=crop_res)
    return data, crop_coords, labels_heatmaps, labels_colorspace

def _get_data_labels(root_folder, idx, filenamebases, heatmap_res, joint_ixs, flag_crop_hand=False):
    filenamebase = filenamebases[idx]
    heatmap_res = (heatmap_res[1], heatmap_res[0])
    if flag_crop_hand:
        data = _get_data(root_folder, filenamebase, as_torch=False, new_res=None)
        labels_jointspace, labels_colorspace = get_labels_depth_and_color(root_folder, filenamebase)
        labels_jointvec = get_labels_jointvec(labels_jointspace, joint_ixs, rel_root=True)
        labels_jointvec = torch.from_numpy(labels_jointvec).float()
        #data_img_RGB = conv.numpy_to_plottable_rgb(data)
        #fig = visualize.plot_img_RGB(data_img_RGB, title=filenamebase)
        #visualize.plot_joints(joints_colorspace=labels_colorspace, num_joints=len(joint_ixs), fig=fig)
        #visualize.savefig('/home/paulo/' + filenamebase.replace('/', '_') + '_' + 'orig')
        #visualize.show()
        data, crop_coords, labels_heatmaps, labels_colorspace =\
            crop_image_get_labels(data, labels_colorspace, joint_ixs)
        data_img_RGB = conv.numpy_to_plottable_rgb(data)
        fig = visualize.plot_img_RGB(data_img_RGB, title=filenamebase)
        visualize.plot_joints(joints_colorspace=labels_colorspace, fig=fig)
        visualize.show()
        data = torch.from_numpy(data).float()
        labels_heatmaps = torch.from_numpy(labels_heatmaps).float()
    else:
        data = _get_data(root_folder, filenamebase, heatmap_res)
        labels_heatmaps, labels_jointvec, _ = _get_labels(root_folder, filenamebase, heatmap_res, joint_ixs)
    labels = labels_heatmaps, labels_jointvec
    return data, labels

class SynthHandsDataset(Dataset):
    type = ''
    root_dir = ''
    filenamebases = []
    joint_ixs = []
    length = 0
    dataset_folder = ''
    heatmap_res = None
    crop_hand = False

    def __init__(self, root_folder, joint_ixs, type, heatmap_res, crop_hand):
        self.type = type
        self.joint_ixs = joint_ixs
        dataset_split_files = load_dataset_split(root_folder=root_folder)
        self.filenamebases = dataset_split_files['filenamebases_' + self.type]
        self.length = len(self.filenamebases)
        self.dataset_folder = root_folder
        self.heatmap_res = heatmap_res
        self.crop_hand = crop_hand

    def __getitem__(self, idx):
        return _get_data_labels(self.dataset_folder, idx, self.filenamebases,
                                self.heatmap_res, self.joint_ixs, flag_crop_hand=self.crop_hand)

    def get_filenamebase(self, idx):
        return self.filenamebases[idx]

    def get_raw_joints_of_example_ix(self, example_ix):
        return _read_label(self.filenamebases[example_ix])

    def get_colorspace_joint_of_example_ix(self, example_ix, joint_ix):
        prop_res_u = IMAGE_RES_HALNET[0] / IMAGE_RES_ORIG[0]
        prop_res_v = IMAGE_RES_HALNET[1] / IMAGE_RES_ORIG[1]
        label = _read_label(self.filenamebases[example_ix])
        u, v = camera.joint_depth2color(label[joint_ix])
        u = int(u * prop_res_u)
        v = int(v * prop_res_v)
        return u, v

    def __len__(self):
        return self.length

class SynthHandsTrainDataset(SynthHandsDataset):
     type = 'train'

class SynthHandsValidDataset(SynthHandsDataset):
    type = 'valid'

class SynthHandsTestDataset(SynthHandsDataset):
    type = 'test'

def _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, crop_hand, verbose, type, batch_size=1):
    if verbose:
        print("Loading synthhands " + type + " dataset...")
    if type == 'train':
        dataset = SynthHandsDataset(root_folder, joint_ixs, type, heatmap_res, crop_hand)
    elif type == 'valid':
        dataset = SynthHandsDataset(root_folder, joint_ixs, type, heatmap_res, crop_hand)
    elif type == 'test':
        dataset = SynthHandsDataset(root_folder, joint_ixs, type, heatmap_res, crop_hand)
    else:
        raise BaseException("Type " + type + " does not exist. Valid types are (train, valid, test)")
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)
    if verbose:
        data_example, label_example = dataset[0]
        labels_heatmaps, label_joints = label_example
        print("Synthhands " + type + " dataset loaded with " + str(len(dataset)) + " examples")
        print("\tExample shape: " + str(data_example.shape))
        print("\tLabel heatmap shape: " + str(labels_heatmaps.shape))
        print("\tLabel joint vector shape (N_JOINTS * 3): " + str(label_joints.shape))
    return dataset_loader

def get_SynthHands_trainloader(root_folder, joint_ixs, heatmap_res, crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, crop_hand, verbose, 'train', batch_size)

def get_SynthHands_validloader(root_folder, joint_ixs, heatmap_res, crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, crop_hand, verbose, 'valid', batch_size)

def get_SynthHands_testloader(root_folder, joint_ixs, heatmap_res, crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, crop_hand, verbose, 'test', batch_size)


def get_train_ixs(size, perc_train):
    '''
    Get an array of indices for training set
    :param size: size of training set
    :param perc_test: percentage to be given to training set
    :return:
    '''
    num_train_examples = size
    n_train_samples = int(np.floor(num_train_examples * perc_train))
    test_ixs = np.in1d(range(num_train_examples),
                       np.random.choice(num_train_examples,
                                        n_train_samples, replace=False))
    return test_ixs


def split_train_test_data(X, Y, perc_train):
    '''
    Randomly split tranining data into training and test
    :param X: Training data
    :param Y: Labels
    :param perc_test: Percentage to be given to test data
    :return: X_train, Y_train, X_test, Y_test
    '''
    train_ixs = get_train_ixs(X.shape[0], perc_train)
    X_train = X[train_ixs]
    Y_train = Y[train_ixs]
    X_test = X[~train_ixs]
    Y_test = Y[~train_ixs]
    return X_train, Y_train, X_test, Y_test, train_ixs

def _read_label(label_filepath):
    '''

    :param label_filepath: path to a joint positions groundtruth file
    :return: NUM_JOINTS X 3 numpy array, where NUM_JOINTS is number of joints
    '''
    with open(label_filepath, 'r') as f:
        first_line = f.readline()
    first_line_nums = first_line.split(',')
    return np.reshape(first_line_nums, (NUM_JOINTS, 3)).astype(float)

def _read_RGB_image_opencv(image_filepath, depth):
    if depth:
        image = cv2.imread(image_filepath, 0)
    else:
        image = cv2.imread(image_filepath)
    # COLOR_BGR2RGB requried when working with OpenCV
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _get_img_read_func_for_module(module_name):
    if module_name == 'matplotlib':
        return mpimg.imread
    elif module_name == 'scipy':
        return misc.imread
    elif module_name == 'opencv':
        return _read_RGB_image_opencv

def _resize_image_pillow(image, new_res):
    imagePIL = Image.fromarray(image.astype('uint8'), 'RGB')
    imagePIL.thumbnail(new_res, Image.ANTIALIAS)
    return np.array(image)

def _get_downsample_image_func(module_name):
    if module_name == 'matplotlib':
        return _resize_image_pillow
    elif module_name == 'scipy':
        return misc.imresize
    elif module_name == 'opencv':
        return cv2.resize

def change_res_image(image, new_res, module_name='scipy'):
    image = _get_downsample_image_func(module_name)(image, new_res)
    return image
def _read_RGB_image(image_filepath, new_res=None, depth=False, module_name='scipy'):
    '''
    Reads RGB image from filepath
    Can downsample image after reading (default is not downsampling)
    :param image_filepath: path to image file
    :return: opencv image object
    '''
    if module_name == 'opencv':
        image = _read_RGB_image_opencv(image_filepath, depth)
    else:
        image = _get_img_read_func_for_module(module_name)(image_filepath)
    if new_res:
        image = change_res_image(image, new_res)
    return image

def show_image(image):
    '''
    Shows an image and waits for the 0 key to be pressed
    After 0 key is pressed, kill all windows
    :param image: opencv image object
    :return:
    '''
    cv2.imshow('Press 0 to kill window...', cv2.IMREAD_UNCHANGED)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
