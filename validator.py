from debugger import print_verbose
import argparse
import os
import trainer

def initialize_train_vars(args):
    train_vars = {}
    train_vars['losses'] = []
    train_vars['losses_main'] = []
    train_vars['losses_joints'] = []
    train_vars['best_loss_joints'] = 1e10
    train_vars['total_joints_loss'] = 0
    train_vars['losses_heatmaps'] = []
    train_vars['best_loss_heatmaps'] = 1e10
    train_vars['total_heatmaps_loss'] = 0
    train_vars['pixel_losses'] = []
    train_vars['pixel_losses_sample'] = []
    train_vars['best_loss'] = 1e10
    train_vars['best_pixel_loss'] = 1e10
    train_vars['best_pixel_loss_sample'] = 1e10
    train_vars['best_model_dict'] = {}
    train_vars['joint_ixs'] = range(21)
    train_vars['use_cuda'] = False
    train_vars['cross_entropy'] = False
    train_vars['root_folder'] = os.path.dirname(os.path.abspath(__file__)) + '/'
    train_vars['checkpoint_filenamebase'] = 'trained_net_log_'
    return train_vars

# initialize control variables
def initialize_control_vars(args):
    control_vars = {}
    control_vars['start_epoch'] = 1
    control_vars['start_iter'] = 1
    control_vars['num_iter'] = args.num_iter
    control_vars['best_model_dict'] = 0
    control_vars['log_interval'] = args.log_interval
    control_vars['batch_size'] = args.batch_size
    control_vars['max_mem_batch'] = args.max_mem_batch
    control_vars['iter_size'] = int(args.batch_size / args.max_mem_batch)
    control_vars['n_iter_per_epoch'] = 0
    control_vars['done_training'] = False
    control_vars['tot_toc'] = 0
    control_vars['verbose'] = args.verbose
    return control_vars

def initialize_vars(args):
    control_vars = initialize_control_vars(args)
    train_vars = initialize_train_vars(args)
    return control_vars, train_vars


def parse_args(model_class):
    parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
    parser.add_argument('--num_iter', dest='num_iter', type=int,
                        help='Total number of iterations to train')
    parser.add_argument('-c', dest='checkpoint_filepath', default='', required=True,
                        help='Checkpoint file from which to begin training')
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=10,
                        help='Number of iterations interval on which to log'
                             ' a model checkpoint (default 10)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=True,
                        help='Verbose mode')
    parser.add_argument('--max_mem_batch', type=int, dest='max_mem_batch', default=8,
                        help='Max size of batch given GPU memory (default 8)')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=16,
                        help='Batch size for training (if larger than max memory batch, training will take '
                             'the required amount of iterations to complete a batch')
    parser.add_argument('-r', dest='root_folder', default='', required=True, help='Root folder for dataset')
    parser.add_argument('--visual', dest='visual_debugging', action='store_true', default=False,
                        help='Whether to visually inspect results')
    parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=False,
                        help='Whether to use cuda for training')
    parser.add_argument('--split_filename', default='', required=False,
                        help='Split filename for the file with dataset splits')
    args = parser.parse_args()

    control_vars, valid_vars = initialize_vars(args)
    control_vars['visual_debugging'] = args.visual_debugging


    print_verbose("Loading model and optimizer from file: " + args.checkpoint_filepath, args.verbose)

    model, optimizer, valid_vars, train_control_vars = \
        trainer.load_checkpoint(filename=args.checkpoint_filepath, model_class=model_class, use_cuda=args.use_cuda)

    valid_vars['root_folder'] = args.root_folder
    valid_vars['use_cuda'] = args.use_cuda

    random_int_str = args.checkpoint_filepath.split('_')[-2]
    valid_vars['checkpoint_filenamebase'] = 'valid_halnet_log_' + str(random_int_str) + '_'
    control_vars['output_filepath'] = 'validated_halnet_log_' + random_int_str + '.txt'
    msg = print_verbose("Printing also to output filepath: " + control_vars['output_filepath'], args.verbose)
    with open(control_vars['output_filepath'], 'w+') as f:
        f.write(msg + '\n')


    if valid_vars['use_cuda']:
        print_verbose("Using CUDA", args.verbose)
    else:
        print_verbose("Not using CUDA", args.verbose)

    control_vars['num_epochs'] = 100
    control_vars['verbose'] = True

    if valid_vars['cross_entropy']:
        print_verbose("Using cross entropy loss", args.verbose)

    control_vars['num_iter'] = 0

    valid_vars['split_filename'] = args.split_filename

    return model, optimizer, control_vars, valid_vars, train_control_vars

