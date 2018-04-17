import time
import numpy as np
import pickle
from debugger import print_verbose
from magic import display_est_time_loop
from tsne import tsne

HANDS2017_ROOTPATH = '/home/paulo/Downloads/'
HANDS2017_TRAIN_FOLDER = ''
HANDS2017_TRAIN_FILENAME = 'Training_Annotation.txt'
HANDS_2017_TRAIN_OUTPUT_FILEPATH = 'Training_Annotation.pkl'
HANDS_2017_TRAIN_FILEPATH = HANDS2017_ROOTPATH + HANDS2017_TRAIN_FOLDER + HANDS2017_TRAIN_FILENAME
HANDS_2017_TRAIN_OUTPUT_FILEPATH = '/home/paulo/handschallenge/' + HANDS_2017_TRAIN_OUTPUT_FILEPATH

def training_file_line_to_numpy_array(line, num_joints):
    joint_gt = np.zeros((num_joints, 3))
    joint_gt[:, :] = np.array(list(map(float, line.split()[1:]))).reshape(num_joints, 3)
    return joint_gt

def training_file_line_to_image_name(line):
    image_name = line.split()[0]
    return image_name

def get_joints_rel_ranges(joints):
    joints = joints - joints[:, 0, :].reshape((joints.shape[0], 1, joints.shape[2]))
    min_x = np.min(joints[:, :, 0])
    max_x = np.max(joints[:, :, 0])
    range_x = max_x - min_x
    min_y = np.min(joints[:, :, 1])
    max_y = np.max(joints[:, :, 1])
    range_y = max_y - min_y
    min_z = np.min(joints[:, :, 2])
    max_z = np.max(joints[:, :, 2])
    range_z = max_z - min_z
    return range_x, range_y, range_z

def load_training_annotation(filepath, verbose=False):
    output_filepath = filepath[:-4] + '_embed.pkl'
    image_names, joints = pickle.load(open(filepath, "rb"))
    joints = np.array(joints)
    joints_embed = tsne(joints.reshape((joints.shape[0], joints.shape[1] * joints.shape[2]))[0:10000, :])
    with open(output_filepath, 'wb') as pf:
        pickle.dump(joints_embed, pf)

def get_training_annotation(training_filepath, output_filepath, verbose=False):
    guess_num_lines = 1e6
    read_interval = 10000000
    num_joints = 21
    print_verbose("Training input file path: " + training_filepath, verbose)
    print_verbose("Testing if program can write to output: " + output_filepath, verbose)
    with open(output_filepath, 'wb') as f:
        pickle.dump([], f)
    joints = []
    image_names = []
    with open(training_filepath, 'r') as f:
        line = f.readline()
        curr_line_ix = 1
        tot_toc = 0
        while line:
            start = time.time()
            image_names.append(training_file_line_to_image_name(line))
            joints.append(training_file_line_to_numpy_array(line, num_joints))
            if curr_line_ix % read_interval == 0:
                with open(output_filepath + '.pkl', 'wb') as pf:
                    pickle.dump([image_names, joints], pf)
            line = f.readline()
            curr_line_ix += 1
            tot_toc = display_est_time_loop(tot_toc + (time.time() - start), curr_line_ix,
                                            guess_num_lines,
                                            prefix='Line: ' + str(curr_line_ix) + ' ')
    with open(output_filepath, 'wb') as pf:
        pickle.dump([image_names, joints], pf)

#get_training_annotation(HANDS_2017_TRAIN_FILEPATH, HANDS_2017_TRAIN_OUTPUT_FILEPATH, verbose=True)
load_training_annotation(HANDS_2017_TRAIN_OUTPUT_FILEPATH, verbose=True)
