import argparse
import dataset_handler
import egodexter_handler
import synthhands_handler

parser = argparse.ArgumentParser(description='Split a dataset into train, valid and test')
parser.add_argument('-r', dest='dataset_folder', default='', required=True, help='Root folder for dataset')
parser.add_argument('-f', dest='split_filename', default='', required=True, help='Filename for split file')
parser.add_argument('-s', '--splits', type=int, dest='num_splits', default=0,
                        help='Number of splits to perform. If not defined, will split into train, test and valid.')
args = parser.parse_args()

dataset_handler.save_dataset_split(args.dataset_folder,
                                   args.split_filename,
                                   num_splits=args.num_splits,
                                   prefix_length=11,
                                   save_folder='')