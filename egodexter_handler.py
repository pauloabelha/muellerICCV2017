import numpy as np
from torch.utils.data.dataset import Dataset
import synthhands_handler
from dataset_handler import load_dataset_split
import camera
import torch
from converter import convert_labels_2D_new_res, color_space_label_to_heatmap
from io_image import read_RGB_image

SPLIT_PREFIX_LENGTH = 11

DEPTH_INTR_MTX =     np.array([[475.62,         0.0,            311.125],
                               [0.0,            475.62,         245.965],
                               [0.0,            0.0,             1.0]])

DEPTH_INTR_MTX_INV = np.array([[0.00210252,     0.0,            -0.65414617],
                               [0.0,            0.00210252,     -0.51714604],
                               [0.0,            0.0,            1.0]])

COLOR_INTR_MTX =    np.array([[617.173,         0.0,            315.453],
                              [0.0,             617.173,        242.259],
                              [0.0,             0.0,            1.0]])

COLOR_EXTR_MTX =    np.array([[1.0,             0.00090442,     -0.0074,        20.2365],
                              [-0.00071933,     0.9997,         0.0248,         1.2846],
                              [0.007,           -0.0248,        0.9997,         5.7360]])

PROJECT_MTX =       np.array([[1.0,         0.0,            0.0,        0.0],
                              [0.0,         1.0,            0.0,        0.0],
                              [0.0,         0.0,            1.0,        0.0]])

DATASET_SPLIT_FILENAME = 'dataset_split_egodexter.p'


def get_data(root_folder, filenamebase, color_on_depth_suffix='_color_on_depth.png', depth_suffix='_depth.png', img_res=(320, 240), as_torch=True):
    # load color
    color_on_depth_image_filepath = root_folder + filenamebase + color_on_depth_suffix
    color_on_depth_image = read_RGB_image(color_on_depth_image_filepath, new_res=img_res)
    # load depth
    filenamebase_split = filenamebase.split('/')
    depth_filenamebase = '/'.join(filenamebase_split[0:2]) + '/depth/' + filenamebase_split[-1]
    depth_image_filepath = root_folder + depth_filenamebase + depth_suffix
    depth_image = read_RGB_image(depth_image_filepath, new_res=img_res)
    depth_image = np.array(depth_image)
    depth_image = np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
    # get data
    RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
    RGBD_image = RGBD_image.swapaxes(1, 2).swapaxes(0, 1)
    img_data = RGBD_image
    if as_torch:
        img_data = torch.from_numpy(RGBD_image).float()
    return img_data


class EgoDexterDataset(Dataset):
    root_folder = ''
    data_folders = ['Desk/', 'Fruits/', 'Kitchen/', 'Rotunda/']
    type = ''
    orig_img_res = (640, 480)
    img_res = (640, 480)
    root_dir = ''
    filenamebases = []
    file_ixs = []
    length = 0
    dataset_folder = ''
    heatmap_res = None
    files_annotations = {}
    files_annotations_3D = {}
    img_labels = {}
    img_labels_3D = {}

    def __init__(self, type_, root_folder, heatmap_res, split_ix=0, joint_ixs=range(21), splitfilename='egodexter_split_10.p'):
        self.type = type_
        self.root_folder = root_folder
        self.img_res = heatmap_res
        self.joint_ixs = range(21)
        dataset_split_files = load_dataset_split(root_folder=root_folder, splitfilename=splitfilename)
        if self.type == 'full':
            self.filenamebases = dataset_split_files['filenamebases']
            self.file_ixs = dataset_split_files['ixs_randomize']
        elif self.type == 'split':
            self.filenamebases = dataset_split_files['filename_bases_list'][split_ix]
            self.num_splits = len(dataset_split_files['filename_bases_list'])
        else:
            self.filenamebases = dataset_split_files['filenamebases_' + self.type]
            self.file_ixs = dataset_split_files['file_ixs_' + self.type]
        self.length = len(self.filenamebases)
        self.dataset_folder = root_folder
        self.img_res = heatmap_res

        self._fill_files_annotations()
        self._fill_files_annotations_3D()
        self._fill_img_labels()

        for idx in range(10):
            self.__getitem__(idx)


    def __getitem__(self, idx):
        return self.get_image_and_labels(idx)

    def __len__(self):
        return self.length

    def get_image_and_labels(self, idx, as_torch=True):
        img_labels_2D, img_labels_heatmaps, img_labels_3D = self.get_labels(idx)
        img_data = self.get_image(idx, as_torch=as_torch)
        for i in range(5):
            if img_data[3, img_labels_2D[i, 0], img_labels_2D[i, 1]] == 0:
                img_labels_2D[i, :] = [-1, -1]
        if as_torch:
            img_labels_2D = torch.from_numpy(img_labels_2D).float()
            img_labels_heatmaps = torch.from_numpy(img_labels_heatmaps).float()
        return (img_data, (img_labels_2D, img_labels_heatmaps, img_labels_3D))

    def get_image(self, idx, as_torch=True, color_on_depth_suffix='_color_on_depth.png', depth_suffix='_depth.png'):
        filenamebase = self.filenamebases[idx]
        # load color
        color_on_depth_image_filepath = self.root_folder + filenamebase + color_on_depth_suffix
        color_on_depth_image = read_RGB_image(color_on_depth_image_filepath, new_res=self.img_res)
        # load depth
        filenamebase_split = filenamebase.split('/')
        depth_filenamebase = '/'.join(filenamebase_split[0:1]) + '/depth/' + filenamebase_split[-1]
        depth_image_filepath = self.root_folder + depth_filenamebase + depth_suffix
        depth_image = read_RGB_image(depth_image_filepath, new_res=self.img_res)
        depth_image = np.array(depth_image)
        depth_image = np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
        # get data
        RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
        RGBD_image = RGBD_image.swapaxes(1, 2).swapaxes(0, 1)
        img_data = RGBD_image
        if as_torch:
            img_data = torch.from_numpy(RGBD_image).float()



        return img_data

    def get_labels(self, idx):
        img_labels_2D = self.img_labels[self.filenamebases[idx]].astype(int)
        img_labels_3D = self.img_labels_3D[self.filenamebases[idx]].astype(int)
        for label_ix in range(img_labels_2D.shape[0]):
            img_labels_2D[label_ix, :] = convert_labels_2D_new_res(img_labels_2D[label_ix, :],
                                                                   self.orig_img_res, self.img_res)
        img_labels_heatmaps = self.get_labels_heatmaps(img_labels_2D)
        return img_labels_2D, img_labels_heatmaps, img_labels_3D

    def get_labels_heatmaps(self, img_labels_2D):
        labels_heatmaps = np.zeros((len(self.joint_ixs), self.img_res[0], self.img_res[1]))
        for label_ix in range(img_labels_2D.shape[0]):
            label_heatmap =\
                color_space_label_to_heatmap(img_labels_2D[label_ix, :], self.img_res)
            label_heatmap = label_heatmap.astype(float)
            labels_heatmaps[label_ix, :, :] = label_heatmap
        return labels_heatmaps

    def _fill_img_labels(self):
        self.img_labels = {}
        for filenamebase in self.filenamebases:
            filenamebase_split = filenamebase.split('/')
            img_data_folder = filenamebase_split[0] + '/'
            img_number = int(filenamebase_split[-1][-5:])
            self.img_labels[filenamebase] = self.files_annotations[img_data_folder][img_number, :]
            self.img_labels_3D[filenamebase] = self.files_annotations_3D[img_data_folder][img_number, :]


    def _fill_files_annotations(self):
        for data_folder in self.data_folders:
            filepath = self.root_folder + data_folder + 'annotation.txt'
            with open(filepath, 'rb') as f:
                n_lines = 0
                for line in f:
                    n_lines += 1
                f.seek(0)
                values = np.zeros((n_lines, 5, 2))
                line_ix = 0
                for line in f:
                    line_split = line.decode("utf-8").split(';')[:-1]
                    pair_ix = 0
                    for pair_str in line_split:
                        pair_split = pair_str.split(',')
                        values[line_ix, pair_ix, 0] = int(pair_split[0])
                        values[line_ix, pair_ix, 1] = int(pair_split[1])
                        pair_ix += 1
                    line_ix+= 1
            self.files_annotations[data_folder] = values

    def _fill_files_annotations_3D(self):
        for data_folder in self.data_folders:
            filepath = self.root_folder + data_folder + 'annotation.txt_3D.txt'
            with open(filepath, 'rb') as f:
                n_lines = 0
                for line in f:
                    n_lines += 1
                f.seek(0)
                values = np.zeros((n_lines, 5, 3))
                line_ix = 0
                for line in f:
                    line_split = line.decode("utf-8").split(';')[:-1]
                    pair_ix = 0
                    for pair_str in line_split:
                        pair_split = pair_str.split(',')
                        values[line_ix, pair_ix, 0] = float(pair_split[0])
                        values[line_ix, pair_ix, 1] = float(pair_split[1])
                        values[line_ix, pair_ix, 2] = float(pair_split[2])
                        pair_ix += 1
                    line_ix += 1
            self.files_annotations_3D[data_folder] = values

    def get_filenamebase(self, idx):
        return self.filenamebases[idx]

    def get_file_ix(self, idx):
        return self.file_ixs[idx]

    def get_raw_joints_of_example_ix(self, example_ix):
        return _read_label(self.filenamebases[example_ix])

    def get_colorspace_joint_of_example_ix(self, example_ix, joint_ix, halnet_res=(320, 240), orig_res=(640, 480)):
        prop_res_u = halnet_res[0] / orig_res[0]
        prop_res_v = halnet_res[1] / orig_res[1]
        lafile_ixs_randomizedbel = _read_label(self.filenamebases[example_ix])
        u, v = camera.joint_depth2color(label[joint_ix], DEPTH_INTR_MTX)
        u = int(u * prop_res_u)
        v = int(v * prop_res_v)
        return u, v

def get_loader(type, root_folder, img_res=(320, 240), batch_size=16, verbose=False):
    list_of_types = ['train', 'test', 'valid', 'full']
    if verbose:
        print("Loading synthhands " + type + " dataset...")
    if not type in list_of_types:
        raise BaseException('Type ' + type + ' does not exist. Valid types are: ' + str(list_of_types))
    dataset = EgoDexterDataset(type, root_folder, img_res)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)
    return dataset_loader