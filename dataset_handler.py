import os
import pickle
import numpy as np


def dataset_save_split(dataset_root_folder, splitfilename, filenamebases, save_folder, perc_train, perc_valid, perc_test):
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
    file_ixs = np.array(range(len(ixs_randomize)))
    file_ixs_randomized = file_ixs[ixs_randomize]
    file_ixs_train = file_ixs_randomized[0: num_train]
    file_ixs_valid = file_ixs_randomized[num_train: num_train + num_valid]
    file_ixs_test = file_ixs_randomized[num_train + num_valid:]
    print("Dataset split")
    print("Percentages of split: training " + str(perc_train * 100) + "%, " +
          "validation " + str(perc_valid * 100) + "% and " +
          "test " + str(perc_test * 100) + "%")
    print("Number of files of split: training " + str(len(filenamebases_train)) + ", " +
          "validation " + str(len(filenamebases_valid)) + " and " +
          "test " + str(len(filenamebases_test)))
    print("Saving split into pickle file: " + splitfilename)
    data = {
        'dataset_root_folder': dataset_root_folder,
        'perc_train': perc_train,
        'perc_valid': perc_valid,
        'perc_test': perc_test,
        'filenamebases': filenamebases,
        'ixs_randomize': ixs_randomize,
        'filenamebases_train': filenamebases_train,
        'filenamebases_valid': filenamebases_valid,
        'filenamebases_test': filenamebases_test,
        'file_ixs_train': file_ixs_train,
        'file_ixs_valid': file_ixs_valid,
        'file_ixs_test': file_ixs_test,
    }
    if save_folder is None:
        with open(dataset_root_folder + splitfilename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_folder + splitfilename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_split(root_folder, splitfilename):
    return pickle.load(open(root_folder + splitfilename, "rb"))

def dataset_n_splits(dataset_root_folder, split_filename, filenamebases, save_folder, num_splits):
    print("Randomising file names...")
    ixs_randomize = np.random.choice(len(filenamebases), len(filenamebases), replace=False)
    filenamebases = np.array(filenamebases)
    filenamebases_randomized = filenamebases[ixs_randomize]
    file_ixs_randomized = np.array(range(len(ixs_randomize)))
    file_ixs_randomized = file_ixs_randomized[ixs_randomize]
    print("Splitting into {} splits...".format(num_splits))
    perc_split = 1 / num_splits
    num = int(np.floor(len(filenamebases) * perc_split))
    print('Number of images per split:\t{}'.format(num))
    filename_bases_list = []
    file_ixs_list = []
    end_ix = -1
    for i in range(num_splits - 1):
        beg_ix = end_ix + 1
        end_ix = (beg_ix + num) - 1
        filename_bases_list.append(filenamebases_randomized[beg_ix:end_ix])
        file_ixs_list.append(file_ixs_randomized[beg_ix:end_ix])
    filename_bases_list.append(filenamebases_randomized[end_ix+1:])
    print("Saving split into pickle file: " + split_filename)
    data = {
        'dataset_root_folder': dataset_root_folder,
        'filenamebases': filenamebases,
        'ixs_randomize': ixs_randomize,
        'filename_bases_list': filename_bases_list,
        'file_ixs_list': file_ixs_list,
    }
    if save_folder is None:
        with open(dataset_root_folder + split_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_folder + split_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_dataset_split(dataset_root_folder, split_filename, dataset_handler, num_splits=0, save_folder=None, perc_train=0.7, perc_valid=0.15, perc_test=0.15):
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
                filenamebases[ix] = os.path.join(root, filename[0:dataset_handler.SPLIT_PREFIX_LENGTH])[len_root_folder:]
                ix += 1
        tabs = '  ' * (len(root.split('/')) - orig_num_tabs)
        print(str(ix) + '/' + str(num_files_to_process) + ' files processed : ' + tabs + root)
    print("Done traversing files")
    if num_splits > 0:
        dataset_n_splits(dataset_root_folder, split_filename, filenamebases, save_folder, num_splits)
    else:
        dataset_save_split(dataset_root_folder, split_filename, filenamebases, save_folder, perc_train, perc_valid, perc_test)