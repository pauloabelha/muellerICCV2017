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
except:
    pass
import matplotlib.image as mpimg
from PIL import Image

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/'
ORIG_DATASET_ROOT_FOLDER = ROOT_FOLDER + '../SynthHands_Release/'
HALNET_DATA_FILENAME = "HALNet_data.p"
#DATASET_PATH = "/home/paulo/synthhands/example_data/"
# resolution in rows x cols
LABEL_RESOLUTION_HALNET = (240, 320)
IMAGE_RES_ORIG = (480, 640)
IMAGE_RES_HALNET = (240, 320)
NUM_LOSSES = 4
LOSSES_RES = [(320, 240), (160, 120), (160, 120), (40, 30)]
NUM_JOINTS = 21

def load_checkpoint(filename, model_class, num_iter=0, log_interval=0,
                    log_interval_valid=0, batch_size=0, max_mem_batch=0):
    # load file
    torch_file = torch.load(filename)
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


def load_dataset_split():
    return pickle.load(open(ROOT_FOLDER + "dataset_split_files.p", "rb"))

def save_dataset_split(orig_dataset_root_folder=ORIG_DATASET_ROOT_FOLDER,
                       perc_train=0.9, perc_valid=0.05, perc_test=0.05):
    color_on_depth_images_dict = {}
    depth_images_dict = {}
    labels_dict = {}
    filenamebase_keys = []
    print("Recursively traversing all files in root folder: " + ORIG_DATASET_ROOT_FOLDER)
    for root, dirs, files in os.walk(orig_dataset_root_folder, topdown=True):
        for filename in sorted(files):
            if filename[-18:-4] == 'color_on_depth':
                filebaseprefix = os.path.join(root, filename[0:8])
                color_on_depth_images_dict[filebaseprefix] = os.path.join(root, filename)
                filenamebase_keys.append(filebaseprefix)
            elif filename[-9:-4] == 'depth':
                filebaseprefix = os.path.join(root, filename[0:8])
                depth_images_dict[filebaseprefix] = os.path.join(root, filename)
            elif filename[-13:-4] == 'joint_pos':
                filebaseprefix = os.path.join(root, filename[0:8])
                labels_dict[filebaseprefix] = os.path.join(root, filename)
    print("Done traversing files")
    print("Randomising file names...")
    ixs_randomize = np.random.choice(len(filenamebase_keys), len(filenamebase_keys), replace=False)
    filenamebase_keys = np.array(filenamebase_keys)
    filenamebase_keys_randomized = filenamebase_keys[ixs_randomize]
    print("Splitting into training, validation and test sets...")
    num_train = int(np.floor(len(filenamebase_keys) * perc_train))
    num_valid = int(np.floor(len(filenamebase_keys) * perc_valid))
    num_test = int(np.floor(len(filenamebase_keys) * perc_test))
    filenamebase_keys_train = filenamebase_keys_randomized[0: num_train]
    filenamebase_keys_valid = filenamebase_keys_randomized[num_train: num_train + num_valid]
    filenamebase_keys_test = filenamebase_keys_randomized[num_train + num_valid:]
    print("Dataset split")
    print("Percentages of split: training " + str(perc_train*100) + "%, " +
          "validation " + str(perc_valid*100) + "% and " +
          "test " + str(perc_test*100) + "%")
    print("Number of files of split: training " + str(len(filenamebase_keys_train)) + ", " +
          "validation " + str(len(filenamebase_keys_valid)) + " and " +
          "test " + str(len(filenamebase_keys_test)))
    print("Saving split into pickle file: " + 'dataset_split_files.p')
    data = {
            'orig_dataset_root_folder': orig_dataset_root_folder,
            'perc_train': perc_train,
            'perc_valid': perc_valid,
            'perc_test': perc_test,
            'filenamebase_keys': filenamebase_keys,
            'color_on_depth_images_dict': color_on_depth_images_dict,
            'depth_images_dict': depth_images_dict,
            'labels_dict': labels_dict,
            'ixs_randomize': ixs_randomize,
            'filenamebase_keys_train': filenamebase_keys_train,
            'filenamebase_keys_valid': filenamebase_keys_valid,
            'filenamebase_keys_test': filenamebase_keys_test
            }
    with open('dataset_split_files.p', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_output(output):
    output_positive = output + abs(np.min(output, axis=(0, 1)))
    #print(np.min(output_positive, axis=(0, 1)))
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

def convert_color_space_label_to_heatmap(color_space_label, heatmap_res):
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
                      (IMAGE_RES_ORIG[0] / heatmap_res[0]))
    new_ix_res2 = int(color_space_label[1] /
                      (IMAGE_RES_ORIG[1] / heatmap_res[1]))
    heatmap[new_ix_res1, new_ix_res2] = 1 - (SMALL_PROB * heatmap.size)
    return heatmap

def _load_dicts_color_depth_labels(root_dir):
    color_on_depth_images_dict = {}
    depth_images_dict = {}
    labels_dict = {}
    filenamebase_keys = []

    for root, dirs, files in os.walk(root_dir, topdown=True):
        for filename in files:
            file_ext = os.path.splitext(filename)[1][1:]
            curr_file_namebase = filename.split('_')[0]
            if filename[-18:-4] == 'color_on_depth':
                color_on_depth_images_dict[curr_file_namebase] = \
                    os.path.join(root, filename)
                filenamebase_keys.append(curr_file_namebase)
            elif filename[-9:-4] == 'depth':
                depth_images_dict[curr_file_namebase] = \
                    os.path.join(root, filename)
            elif file_ext == 'txt':
                labels_dict[curr_file_namebase] = \
                    os.path.join(root, filename)
    return color_on_depth_images_dict, depth_images_dict, \
           labels_dict, filenamebase_keys

def _get_data(filenamebase_key, color_on_depth_images_dict, depth_images_dict):
    # load color
    color_on_depth_image_filename = color_on_depth_images_dict[filenamebase_key]
    color_on_depth_image = \
        _read_RGB_image(color_on_depth_image_filename, new_res=IMAGE_RES_HALNET)

    # load depth
    depth_image_filename = depth_images_dict[filenamebase_key]
    depth_image = \
        _read_RGB_image(depth_image_filename, new_res=IMAGE_RES_HALNET)
    depth_image = np.reshape(depth_image,
                             (depth_image.shape[0], depth_image.shape[1], 1))

    # get data
    RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
    data = np.swapaxes(RGBD_image, 0, 2).astype(np.float)
    data = torch.from_numpy(data).float()
    return data

def _get_labels(filenamebase_key, labels_dict, joint_ixs):
    # get label
    label_depth_space = _read_label(labels_dict[filenamebase_key])
    label_color_space = np.zeros((label_depth_space.shape[0], 2))
    for i in range(label_depth_space.shape[0]):
        label_color_space[i, 0], label_color_space[i, 1] \
            = camera.get_joint_in_color_space(label_depth_space[i])
    labels = np.zeros((len(joint_ixs), LABEL_RESOLUTION_HALNET[0], LABEL_RESOLUTION_HALNET[1]))
    labels_ix = 0
    labels_joints = np.zeros((len(joint_ixs)*3, ))
    for joint_ix in joint_ixs:
        label = convert_color_space_label_to_heatmap(label_color_space[joint_ix, :], LABEL_RESOLUTION_HALNET)
        label = label.astype(float)
        labels[labels_ix, :, :] = label
        # joint labels
        labels_joints[labels_ix*3:(labels_ix*3)+3] = label_depth_space[joint_ix, :]
        labels_ix += 1
    labels_joints = torch.from_numpy(labels_joints).float()
    labels = torch.from_numpy(labels).float()
    return (labels, labels_joints)

def _get_data_labels(idx, labels_dict, color_on_depth_images_dict,
                      depth_images_dict, filenamebase_keys, joint_ixs):
    filenamebase_key = filenamebase_keys[idx]
    data = _get_data(filenamebase_key, color_on_depth_images_dict, depth_images_dict)
    labels = _get_labels(filenamebase_key, labels_dict, joint_ixs)
    return data, labels

class SynthHandsDataset(Dataset):
    type = ''
    root_dir = ''
    labels_dict = {}
    color_on_depth_images_dict = {}
    depth_images_dict = {}
    filenamebase_keys = []
    joint_ixs = []

    def __init__(self, joint_ixs, type):
        self.type = type
        self.joint_ixs = joint_ixs
        dataset_split_files = load_dataset_split()
        self.filenamebase_keys = dataset_split_files['filenamebase_keys_' + self.type]
        self.color_on_depth_images_dict = \
            { your_key: dataset_split_files['color_on_depth_images_dict'][your_key]
              for your_key in self.filenamebase_keys}
        self.depth_images_dict = \
            {your_key: dataset_split_files['depth_images_dict'][your_key]
             for your_key in self.filenamebase_keys}
        self.labels_dict = \
            {your_key: dataset_split_files['labels_dict'][your_key]
             for your_key in self.filenamebase_keys}

    def __getitem__(self, idx):
        return _get_data_labels(idx, self.labels_dict,
                                self.color_on_depth_images_dict,
                                self.depth_images_dict,
                                self.filenamebase_keys,
                                self.joint_ixs)

    def get_raw_joints_of_example_ix(self, example_ix):
        filenamebase = self.filenamebase_keys[example_ix]
        return _read_label(self.labels_dict[filenamebase])

    def get_colorspace_joint_of_example_ix(self, example_ix, joint_ix):
        prop_res_u = IMAGE_RES_HALNET[0] / IMAGE_RES_ORIG[0]
        prop_res_v = IMAGE_RES_HALNET[1] / IMAGE_RES_ORIG[1]
        filenamebase = self.filenamebase_keys[example_ix]
        label = _read_label(self.labels_dict[filenamebase])
        u, v = camera.get_joint_in_color_space(label[joint_ix])
        u = int(u * prop_res_u)
        v = int(v * prop_res_v)
        return u, v

    def __len__(self):
        return len(self.filenamebase_keys)

class SynthHandsTrainDataset(SynthHandsDataset):
     type = 'train'

class SynthHandsValidDataset(SynthHandsDataset):
    type = 'valid'

class SynthHandsTestDataset(SynthHandsDataset):
    type = 'test'

def _get_SynthHands_loader(joint_ixs, verbose, type, batch_size=1):
    if verbose:
        print("Loading synthhands " + type + " dataset...")
    if type == 'train':
        dataset = SynthHandsDataset(joint_ixs, type)
    elif type == 'valid':
        dataset = SynthHandsDataset(joint_ixs, type)
    elif type == 'test':
        dataset = SynthHandsDataset(joint_ixs, type)
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

def get_SynthHands_trainloader(joint_ixs, batch_size=1, verbose=False):
    return _get_SynthHands_loader(joint_ixs, verbose, 'train', batch_size)

def get_SynthHands_validloader(joint_ixs, batch_size=1, verbose=False):
    return _get_SynthHands_loader(joint_ixs, verbose, 'valid', batch_size)

def get_SynthHands_testloader(joint_ixs, batch_size=1, verbose=False):
    return _get_SynthHands_loader(joint_ixs, verbose, 'test', batch_size)


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





def load_HALNet_data(perc_train=0.8):
    '''

    :param dataset_filepath: path to the dataset base root,
        where all label files are (e.g. ~/synthhands_release/')
    :param perc_train: percentage of training samples
    :return: X_train, Y_train, X_test, Y_test
    '''
    try:
        print("Loading HALNet data from file: " + HALNET_DATA_FILENAME)
        data = pickle.load(open(HALNET_DATA_FILENAME, "rb"))
        X_train, Y_train, X_test, Y_test, files_namebase, train_ixs = data
    except:
        print("Could not load data file " +
              HALNET_DATA_FILENAME +
              ". Creating new one from synthhand dataset...")
        X, Y, files_namebase = load_images_and_labels()
        train_ixs = get_train_ixs(Y.shape[0], perc_train)
        files_namebase = np.append(files_namebase[~train_ixs],
                                   files_namebase[train_ixs])
        X_train = X[train_ixs]
        X_test = X[~train_ixs]
        Y_train = []
        Y_test = []
        for ix_loss in range(NUM_LOSSES):
            Y_heatmap = np.zeros((Y.shape[0],
                                  LOSSES_RES[ix_loss][0],
                                  LOSSES_RES[ix_loss][1]))
            for ix_sample in range(Y.shape[0]):
                Y_heatmap[ix_sample] =\
                    convert_color_space_label_to_heatmap(
                        Y[ix_sample, 0, :], LOSSES_RES[ix_loss])
            Y_heatmap_train = Y_heatmap[train_ixs]
            Y_heatmap_test = Y_heatmap[~train_ixs]
            Y_train.append(Y_heatmap_train)
            Y_test.append(Y_heatmap_test)
        data = [X_train, Y_train, X_test, Y_test, files_namebase, train_ixs]
        pickle.dump(data, open(HALNET_DATA_FILENAME, "wb"))
    print("Data loaded")
    print("Reformatting data...")
    X_train = np.swapaxes(X_train, 1, 3)
    #X_train = np.swapaxes(X_train, 2, 3)
    X_train = torch.from_numpy(X_train.astype(np.float))
    for i in range(len(Y_train)):
        Y_train[i] = torch.from_numpy(Y_train[i].astype(np.float64))
    X_test = torch.from_numpy(X_test.astype(np.float))
    X_test = np.swapaxes(X_test, 1, 3)
    #X_test = np.swapaxes(X_test, 2, 3)
    for i in range(len(Y_test)):
        Y_test[i] = torch.from_numpy(Y_test[i].astype(np.float))
    print("Data ready")
    return X_train, Y_train, X_test, Y_test, files_namebase, train_ixs


def load_images_and_labels():
    '''

        :param dataset_filepath: path to the dataset base root,
            where all label files are (e.g. ~/synthhands_release/')
        :return: N X NUM_JOINTS X 3 numpy array,
            where N is number of labels and NUM_JOINTS is number of joints
        '''
    files_namebase = []
    labels = []
    color_on_depth_images_dict = {}
    depth_images_dict = {}
    for root, dirs, files in os.walk(DATASET_PATH, topdown=True):
        for filename in files:
            curr_file_namebase = filename.split('_')[0]
            file_ext = os.path.splitext(filename)[1][1:]
            if filename[-18:-4] == 'color_on_depth':
                color_on_depth_images_dict[curr_file_namebase] =\
                    _read_RGB_image(os.path.join(root, filename),
                                    new_res=IMAGE_RES_HALNET)
            elif filename[-9:-4] == 'depth':
                depth_images_dict[curr_file_namebase] = \
                    _read_RGB_image(os.path.join(root, filename),
                                    new_res=IMAGE_RES_HALNET)
            if file_ext == 'txt':
                label_depth_space = _read_label(os.path.join(root, filename))
                label_color_space = np.zeros((label_depth_space.shape[0], 2))
                for i in range(label_depth_space.shape[0]):
                    label_color_space[i, 0], label_color_space[i, 1]\
                        = camera.get_joint_in_color_space(label_depth_space[i])
                labels.append(label_color_space)
                files_namebase.append(curr_file_namebase)

    RGBD_images = []
    for filenamebase_key in color_on_depth_images_dict.keys():
        color_on_depth_image = color_on_depth_images_dict[filenamebase_key]
        depth_image = depth_images_dict[filenamebase_key]
        depth_image = np.reshape(depth_image,
                                 (depth_image.shape[0], depth_image.shape[1], 1))
        RGBD_image = np.concatenate((color_on_depth_image, depth_image), axis=-1)
        RGBD_images.append(RGBD_image)
    RGBD_images_np_array = np.stack(RGBD_images, axis=0)
    IMAGE_RES1, IMAGE_RES2 = IMAGE_RES_HALNET
    assert RGBD_images_np_array.shape[1] == IMAGE_RES1
    assert RGBD_images_np_array.shape[2] == IMAGE_RES2
    assert RGBD_images_np_array.shape[3] == 4
    label_np_array = np.stack(labels, axis=0)
    assert label_np_array.shape[1] == NUM_JOINTS
    assert label_np_array.shape[2] == 2
    files_namebase_np_array = np.array(files_namebase)
    return RGBD_images_np_array, label_np_array, files_namebase_np_array




def _read_label(label_filepath):
    '''

    :param label_filepath: path to a joint positions groundtruth file
    :return: NUM_JOINTS X 3 numpy array, where NUM_JOINTS is number of joints
    '''
    with open(label_filepath, 'r') as f:
        first_line = f.readline()
    first_line_nums = first_line.split(',')
    return np.reshape(first_line_nums, (NUM_JOINTS, 3)).astype(float)

def _read_RGB_image_opencv(image_filepath):
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

def _read_RGB_image(image_filepath, new_res=None, module_name='matplotlib'):
    '''
    Reads RGB image from filepath
    Can downsample image after reading (default is not downsampling)
    :param image_filepath: path to image file
    :return: opencv image object
    '''
    image = _get_img_read_func_for_module(module_name)(image_filepath)
    if new_res:
        if module_name == 'opencv':
            new_res = (new_res[1], new_res[0])
        image = _get_downsample_image_func(module_name)(image, new_res)
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
