import numpy as np
import camera
import pickle
import torch
from torch.utils.data.dataset import Dataset
import converter as conv
from dataset_handler import load_dataset_split
from io_image import read_RGB_image, crop_image_get_labels
from scipy.spatial.distance import pdist, squareform

SPLIT_PREFIX_LENGTH = 8

DEPTH_INTR_MTX =     np.array([[475.62,     0.0,        311.125],
                                [0.0,        475.62,     245.965],
                                [0.0,        0.0,        1.0]])
DEPTH_INTR_MTX_INV =     np.array([[0.00210252, 0., -0.65414617],
                                [0., 0.00210252, -0.51714604],
                                [0., 0., 1.]])
COLOR_INTR_MTX = np.array([[617.173,    0.0,       315.453],
                           [0.0,        617.173,    242.259],
                           [0.0,        0.0,        1.0]])
COLOR_EXTR_MTX = np.array([[1.0, 0.0, 0.0, 24.7],
                           [0.0, 1.0, 0.0, -0.0471401],
                           [0.0, 0.0, 1.0, 3.72045],
                           [0.0, 0.0, 0.0, 1.0]])
PROJECT_MTX =    np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

DATASET_SPLIT_FILENAME = 'dataset_split_synthhands.p'

def get_finger_name_from_joint_ix(joint_ix):
    finger_name = ''
    if joint_ix == 0:
        finger_name = 'Hand root'
    elif joint_ix <= 4:
        finger_name = 'Thumb'
    elif joint_ix <= 8:
        finger_name = 'Index'
    elif joint_ix <= 12:
        finger_name = 'Middle'
    elif joint_ix <= 16:
        finger_name = 'Ring'
    elif joint_ix <= 20:
        finger_name = 'Little'
    return finger_name

def get_joint_acronym_from_joint_ix(joint_ix):
    if joint_ix == 0:
        return ''
    acr_names = ['MCP', 'PIP', 'DIP', 'TIP']
    acr_ix = int(np.ceil((joint_ix + 1) / 4))
    return acr_names[acr_ix]

def get_joint_name_from_ix(joint_ix):
    joint_name = get_finger_name_from_joint_ix(joint_ix)
    joint_name += ' ' + get_joint_acronym_from_joint_ix(joint_ix)
    return joint_name

def _get_joint_prior(dataset_folder,  prior_file_name):
    joint_prior_dict = pickle.load(open(dataset_folder + prior_file_name, "rb"))
    joint_prior = joint_prior_dict['pair_dist_prob']
    joint_prior /= joint_prior.sum()
    joint_prior = torch.from_numpy(joint_prior).float()
    return joint_prior

def _get_joints_dist_posterior(target_joints):
    joint_posterior = pair_dist_prob = np.zeros((210, 300))
    D = squareform(pdist(target_joints.reshape((21, 3))))
    ix_pair = 0
    for i in range(D.shape[0]):
        j = i + 1
        while j < D.shape[1]:
            # print('(' + str(i) + ', ' + str(j) + '): ' + str(D[i, j]))
            dist = D[i, j]
            pair_dist_prob[ix_pair, int(dist)] = 1
            j += 1
            ix_pair += 1
    joint_posterior = torch.from_numpy(joint_posterior).float()
    return joint_posterior

def _get_data(root_folder, filenamebase, new_res, as_torch=True, depth_suffix='_depth.png', color_on_depth_suffix='_color_on_depth.png'):
    # load color
    color_on_depth_image_filename = root_folder + filenamebase + color_on_depth_suffix
    color_on_depth_image = read_RGB_image(color_on_depth_image_filename, new_res=new_res)
    # load depth
    depth_image_filename = root_folder + filenamebase + depth_suffix
    depth_image = read_RGB_image(depth_image_filename, new_res=new_res)
    depth_image = np.array(depth_image)
    depth_image = np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
    # get data
    RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
    RGBD_image = RGBD_image.swapaxes(1, 2).swapaxes(0, 1)
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
            = camera.joint_depth2color(labels_jointspace[i], DEPTH_INTR_MTX)
    return labels_jointspace, labels_colorspace, labels_joint_depth_z

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
    return labels_jointvec, hand_root

def get_labels_heatmaps_and_jointvec(labels_jointspace, labels_colorspace, joint_ixs, heatmap_res):
    labels_heatmaps = np.zeros((len(joint_ixs), heatmap_res[0], heatmap_res[1]))
    labels_ix = 0
    labels_jointvec = np.zeros((len(joint_ixs) * 3,))
    for joint_ix in joint_ixs:
        label = conv.color_space_label_to_heatmap(labels_colorspace[joint_ix, :], heatmap_res)
        label = label.astype(float)
        labels_heatmaps[labels_ix, :, :] = label
        # joint labels
        labels_jointvec[labels_ix * 3:(labels_ix * 3) + 3] = labels_jointspace[joint_ix, :]
        labels_ix += 1
    return labels_heatmaps, labels_jointvec

def _get_labels(root_folder, filenamebase, heatmap_res, joint_ixs, label_suffix='_joint_pos.txt', orig_img_res=(640, 480)):
    labels_jointspace, labels_colorspace, labels_joint_depth_z = \
        get_labels_depth_and_color(root_folder, filenamebase, label_suffix=label_suffix)
    labels_heatmaps, labels_jointvec = \
        get_labels_heatmaps_and_jointvec(labels_jointspace, labels_colorspace, joint_ixs, heatmap_res)
    labels_colorspace[:, 0] = labels_colorspace[:, 0] * (heatmap_res[0] / orig_img_res[0])
    labels_colorspace[:, 1] = labels_colorspace[:, 1] * (heatmap_res[1] / orig_img_res[1])
    labels_jointvec = torch.from_numpy(labels_jointvec).float()
    labels_heatmaps = torch.from_numpy(labels_heatmaps).float()
    return labels_heatmaps, labels_jointvec, labels_colorspace, labels_joint_depth_z






def _read_label(label_filepath, num_joints=21):
    '''

    :param label_filepath: path to a joint positions groundtruth file
    :return: num_joints X 3 numpy array, where num_joints is number of joints
    '''
    with open(label_filepath, 'r') as f:
        first_line = f.readline()
    first_line_nums = first_line.split(',')
    reshaped_joints = np.reshape(first_line_nums, (num_joints, 3)).astype(float)
    return reshaped_joints

def _get_data_labels(root_folder, idx, filenamebases, heatmap_res, joint_ixs, flag_crop_hand=False):
    filenamebase = filenamebases[idx]
    if flag_crop_hand:
        data = _get_data(root_folder, filenamebase, as_torch=False, new_res=None)
        labels_jointspace, labels_colorspace, labels_joint_depth_z = get_labels_depth_and_color(root_folder, filenamebase)
        labels_jointvec, handroot = get_labels_jointvec(labels_jointspace, joint_ixs, rel_root=True)
        data, crop_coords, labels_heatmaps, labels_colorspace =\
            crop_image_get_labels(data, labels_colorspace, joint_ixs)
        data = torch.from_numpy(data).float()
        labels_heatmaps = torch.from_numpy(labels_heatmaps).float()
        labels_jointvec = torch.from_numpy(labels_jointvec).float()
    else:
        data = _get_data(root_folder, filenamebase, heatmap_res)
        labels_heatmaps, labels_jointvec, labels_colorspace, _ = _get_labels(root_folder, filenamebase, heatmap_res, joint_ixs)
        handroot = labels_jointvec[0:3]
    labels = labels_colorspace, labels_jointvec, labels_heatmaps, handroot
    return data, labels

class SynthHandsDataset(Dataset):
    type = ''
    root_dir = ''
    filenamebases = []
    joint_ixs = []
    length = 0
    num_splits = 0
    dataset_folder = ''
    heatmap_res = None
    crop_hand = False

    def __init__(self, root_folder, type_, joint_ixs=range(21), heatmap_res=(320, 240),
                 split_ix=0, crop_hand=False, splitfilename='dataset_split_files.p'):
        self.type = type_
        self.joint_ixs = joint_ixs
        self.num_splits = 0
        dataset_split_files = load_dataset_split(root_folder=root_folder, splitfilename=splitfilename)
        if self.type == 'full':
            self.filenamebases = dataset_split_files['filenamebases']
        elif self.type == 'split':
            self.filenamebases = dataset_split_files['filename_bases_list'][split_ix]
            self.num_splits = len(dataset_split_files['filename_bases_list'])
        else:
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

    def get_colorspace_joint_of_example_ix(self, example_ix, joint_ix, halnet_res=(320, 240), orig_res=(640, 480)):
        prop_res_u = halnet_res[0] / orig_res[0]
        prop_res_v = halnet_res[1] / orig_res[1]
        label = _read_label(self.filenamebases[example_ix])
        u, v = camera.joint_depth2color(label[joint_ix], DEPTH_INTR_MTX)
        u = int(u * prop_res_u)
        v = int(v * prop_res_v)
        return u, v

    def __len__(self):
        return self.length

class SynthHandsDataset_prior(SynthHandsDataset):
    type = ''
    root_dir = ''
    filenamebases = []
    joint_ixs = []
    length = 0
    dataset_folder = ''
    heatmap_res = None
    crop_hand = False
    prior_file_name = 'joint_prior.p'

    def __init__(self, root_folder, joint_ixs, type, heatmap_res, crop_hand):
        super(SynthHandsDataset_prior, self).__init__(root_folder, joint_ixs, type, heatmap_res, crop_hand)
        #self.joint_prior = _get_joint_prior(self.dataset_folder, self.prior_file_name)

    def __getitem__(self, idx):
        data, labels = _get_data_labels(self.dataset_folder, idx, self.filenamebases,
                                self.heatmap_res, self.joint_ixs, flag_crop_hand=self.crop_hand)
        labels_list = list(labels)
        target_joints = labels[1].numpy()
        joint_posterior = _get_joints_dist_posterior(target_joints)
        labels_list.append(joint_posterior)
        labels = tuple(labels_list)

        return data, labels

class SynthHandsTrainDataset(SynthHandsDataset):
     type = 'train'

class SynthHandsValidDataset(SynthHandsDataset):
    type = 'valid'

class SynthHandsTestDataset(SynthHandsDataset):
    type = 'test'

class SynthHandsFullDataset(SynthHandsDataset):
    type = 'full'

def _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, dataset_type, crop_hand, verbose, type, batch_size=1):
    list_of_types = ['prior', 'train', 'test', 'valid', 'full']
    if verbose:
        print("Loading synthhands " + type + " dataset...")
    dataset_class = SynthHandsDataset
    if not type in list_of_types:
        raise BaseException('Type ' + type + ' does not exist. Valid types are: ' + str(list_of_types))
    dataset = dataset_class(root_folder, type, joint_ixs, heatmap_res, crop_hand)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)
    if verbose:
        data_example, label_example = dataset[0]
        if dataset_type == 'prior':
            labels_heatmaps, label_joints, label_handroot, label_prior = label_example
        else:
            labels_heatmaps, label_joints, label_handroot = label_example
        print("Synthhands " + type + " dataset loaded with " + str(len(dataset)) + " examples")
        print("\tExample shape: " + str(data_example.shape))
        print("\tLabel heatmap shape: " + str(labels_heatmaps.shape))
        print("\tLabel joint vector shape (N_JOINTS * 3): " + str(label_joints.shape))
        if dataset_type == 'prior':
            print("\tLabel prior shape (pair * dist): " + str(label_prior.shape))
    return dataset_loader

def get_SynthHands_trainloader(root_folder, joint_ixs=range(21), heatmap_res=(320, 240), dataset_type='normal', crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, dataset_type, crop_hand, verbose, 'train', batch_size)

def get_SynthHands_validloader(root_folder, joint_ixs=range(21), heatmap_res=(320, 240), dataset_type='normal', crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, dataset_type, crop_hand, verbose, 'valid', batch_size)

def get_SynthHands_testloader(root_folder, joint_ixs=range(21), heatmap_res=(320, 240), dataset_type='normal', crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, dataset_type, crop_hand, verbose, 'test', batch_size)

def get_SynthHands_fullloader(root_folder, joint_ixs=range(21), heatmap_res=(320, 240), dataset_type='normal', crop_hand=False, batch_size=1, verbose=False):
    return _get_SynthHands_loader(root_folder, joint_ixs, heatmap_res, dataset_type, crop_hand, verbose, 'full', batch_size)
