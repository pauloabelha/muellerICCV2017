from debugger import print_verbose
import resnet
import numpy as np
import torch
import argparse
import os
import optimizers as my_optimizers
import io_data

def initialize_train_vars(args):
    train_vars = {}
    train_vars['losses'] = []
    train_vars['losses_main'] = []
    train_vars['losses_joints'] = []
    train_vars['pixel_losses'] = []
    train_vars['pixel_losses_sample'] = []
    train_vars['best_loss'] = 1e10
    train_vars['best_pixel_loss'] = 1e10
    train_vars['best_pixel_loss_sample'] = 1e10
    train_vars['best_model_dict'] = {}
    train_vars['joint_ixs'] = args.joint_ixs
    train_vars['use_cuda'] = args.use_cuda
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
    control_vars['num_epochs'] = args.num_epochs
    control_vars['best_model_dict'] = 0
    control_vars['log_interval'] = args.log_interval
    control_vars['log_interval_valid'] = args.log_interval_valid
    control_vars['batch_size'] = args.batch_size
    control_vars['max_mem_batch'] = args.max_mem_batch
    control_vars['iter_size'] = int(args.batch_size / args.max_mem_batch)
    control_vars['n_iter_per_epoch'] = 0
    control_vars['done_training'] = False
    control_vars['tot_toc'] = 0
    control_vars['output_filepath'] = args.output_filepath
    control_vars['verbose'] = args.verbose
    return control_vars

def initialize_vars(args):
    control_vars = initialize_control_vars(args)
    train_vars = initialize_train_vars(args)
    return control_vars, train_vars

def parse_args(model_class, random_id=-1):
    parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
    parser.add_argument('--num_iter', dest='num_iter', type=int, required=True,
                        help='Total number of iterations to train')
    parser.add_argument('-c', dest='checkpoint_filepath', default='',
                        help='Checkpoint file from which to begin training')
    parser.add_argument('--log_interval', type=int, dest='log_interval', default=10,
                        help='Number of iterations interval on which to log'
                             ' a model checkpoint (default 10)')
    parser.add_argument('--log_interval_valid', type=int, dest='log_interval_valid', default=1000,
                        help='Number of iterations interval on which to log'
                             ' a model checkpoint for validation (default 1000)')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100,
                        help='Total number of epochs to train')
    parser.add_argument('--cuda', dest='use_cuda', action='store_true', default=False,
                        help='Whether to use cuda for training')
    parser.add_argument('-o', dest='output_filepath', default='',
                        help='Output file for logging')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=True,
                        help='Verbose mode')
    parser.add_argument('-j', '--joint_ixs', dest='joint_ixs', nargs='+', help='', default=list(range(21)))
    parser.add_argument('--resnet', dest='load_resnet', action='store_true', default=False,
                        help='Whether to load RESNet weights onto the network when creating it')
    parser.add_argument('--max_mem_batch', type=int, dest='max_mem_batch', default=8,
                        help='Max size of batch given GPU memory (default 8)')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=16,
                        help='Batch size for training (if larger than max memory batch, training will take '
                             'the required amount of iterations to complete a batch')
    parser.add_argument('--cross_entropy', dest='cross_entropy', action='store_true', default=False,
                        help='Whether to use cross entropy loss on HALNet')
    parser.add_argument('-r', dest='root_folder', default='', required=True, help='Root folder for dataset')
    args = parser.parse_args()
    args.joint_ixs = list(map(int, args.joint_ixs))

    control_vars, train_vars = initialize_vars(args)
    train_vars['root_folder'] = args.root_folder

    if random_id < 0:
        train_vars['checkpoint_filenamebase'] = 'trained_halnet_log_'
    else:
        train_vars['checkpoint_filenamebase'] = 'trained_halnet_log_' + str(random_id) + '_'
    if control_vars['output_filepath'] == '':
        print_verbose("No output filepath specified", args.verbose)
    else:
        print_verbose("Printing also to output filepath: " + control_vars['output_filepath'], args.verbose)

    if args.checkpoint_filepath == '':
        print_verbose("Creating network from scratch", args.verbose)
        print_verbose("Building network...", args.verbose)
        train_vars['use_cuda'] = args.use_cuda
        train_vars['cross_entropy'] = args.cross_entropy
        params_dict = {}
        params_dict['joint_ixs'] = args.joint_ixs
        params_dict['use_cuda'] = args.use_cuda
        params_dict['cross_entropy'] = args.cross_entropy
        model = model_class(params_dict)
        if args.load_resnet:
            model = load_resnet_weights_into_HALNet(model, args.verbose)
        print_verbose("Done building network", args.verbose)
        optimizer = my_optimizers.get_adadelta_halnet(model)

    else:
        print_verbose("Loading model and optimizer from file: " + args.checkpoint_filepath, args.verbose)
        model, optimizer, train_vars, control_vars = \
            io_data.load_checkpoint(filename=args.checkpoint_filepath, model_class=model_class,
                                    num_iter=100000, log_interval=10,
                                    log_interval_valid=1000, batch_size=16, max_mem_batch=args.max_mem_batch)
    if train_vars['use_cuda']:
        print_verbose("Using CUDA", args.verbose)
    else:
        print_verbose("Not using CUDA", args.verbose)

    control_vars['num_epochs'] = 100
    control_vars['verbose'] = True


    if train_vars['cross_entropy']:
        print_verbose("Using cross entropy loss", args.verbose)

    return model, optimizer, control_vars, train_vars



def load_resnet_weights_into_HALNet(halnet, verbose, n_tabs=1):
    print_verbose("Loading RESNet50...", verbose, n_tabs)
    resnet50 = resnet.resnet50(pretrained=True)
    print_verbose("Done loading RESNet50", verbose, n_tabs)
    # initialize HALNet with RESNet50
    print_verbose("Initializaing network with RESNet50...", verbose, n_tabs)
    # initialize level 1
    # initialize conv1
    resnet_weight = resnet50.conv1.weight.data.cpu()
    float_tensor = np.random.normal(np.mean(resnet_weight.numpy()),
                                    np.std(resnet_weight.numpy()),
                                    (resnet_weight.shape[0],
                                     1, resnet_weight.shape[2],
                                     resnet_weight.shape[2]))
    resnet_weight_numpy = resnet_weight.numpy()
    resnet_weight = np.concatenate((resnet_weight_numpy, float_tensor), axis=1)
    resnet_weight = torch.FloatTensor(resnet_weight)
    halnet.conv1[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize level 2
    # initialize res2a
    resnet_weight = resnet50.layer1[0].conv1.weight.data
    halnet.res2a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv2.weight.data
    halnet.res2a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].conv3.weight.data
    halnet.res2a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[0].downsample[0].weight.data
    halnet.res2a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2b
    resnet_weight = resnet50.layer1[1].conv1.weight.data
    halnet.res2b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv2.weight.data
    halnet.res2b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[1].conv3.weight.data
    halnet.res2b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res2c
    resnet_weight = resnet50.layer1[2].conv1.weight.data
    halnet.res2c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv2.weight.data
    halnet.res2c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer1[2].conv3.weight.data
    halnet.res2c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3a
    resnet_weight = resnet50.layer2[0].conv1.weight.data
    halnet.res3a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv2.weight.data
    halnet.res3a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].conv3.weight.data
    halnet.res3a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[0].downsample[0].weight.data
    halnet.res3a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3b
    resnet_weight = resnet50.layer2[1].conv1.weight.data
    halnet.res3b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv2.weight.data
    halnet.res3b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[1].conv3.weight.data
    halnet.res3b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    # initialize res3c
    resnet_weight = resnet50.layer2[2].conv1.weight.data
    halnet.res3c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv2.weight.data
    halnet.res3c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
    resnet_weight = resnet50.layer2[2].conv3.weight.data
    halnet.res3c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
    print_verbose("Done initializaing network with RESNet50", verbose, n_tabs)
    print_verbose("Deleting resnet from memory", verbose, n_tabs)
    del resnet50
    return halnet

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("\tSaving a checkpoint...")
    torch.save(state, filename)

def pixel_stdev(norm_heatmap):
    mean_norm_heatmap = np.mean(norm_heatmap)
    stdev_norm_heatmap = np.std(norm_heatmap)
    lower_bound = mean_norm_heatmap - stdev_norm_heatmap
    upper_bound = mean_norm_heatmap + stdev_norm_heatmap
    pixel_count_lower = np.where(norm_heatmap >= lower_bound)
    pixel_count_upper = np.where(norm_heatmap <= upper_bound)
    pixel_count_mask = pixel_count_lower and pixel_count_upper
    return np.sqrt(norm_heatmap[pixel_count_mask].size)

def print_target_info(target):
    if len(target.shape) == 4:
        target = target[0, :, :, :]
    target = io_data.convert_torch_dataoutput_to_canonical(target.data.numpy()[0])
    norm_target = io_data.normalize_output(target)
    # get joint inference from max of heatmap
    max_heatmap = np.unravel_index(np.argmax(norm_target, axis=None), norm_target.shape)
    print("Heamap max: " + str(max_heatmap))
    # data_image = visualize.add_squares_for_joint_in_color_space(data_image, max_heatmap, color=[0, 50, 0])
    # sample from heatmap
    heatmap_sample_flat_ix = np.random.choice(range(len(norm_target.flatten())), 1, p=norm_target.flatten())
    heatmap_sample_uv = np.unravel_index(heatmap_sample_flat_ix, norm_target.shape)
    heatmap_mean = np.mean(norm_target)
    heatmap_stdev = np.std(norm_target)
    print("Heatmap mean: " + str(heatmap_mean))
    print("Heatmap stdev: " + str(heatmap_stdev))
    print("Heatmap pixel standard deviation: " + str(pixel_stdev(norm_target)))
    heatmap_sample_uv = (int(heatmap_sample_uv[0]), int(heatmap_sample_uv[1]))
    print("Heatmap sample: " + str(heatmap_sample_uv))

def print_header_info(model, dataset_loader, control_vars):
    msg = ''
    msg += print_verbose("-----------------------------------------------------------", control_vars['verbose']) + "\n"
    msg += print_verbose("Output filenamebase: " + control_vars['output_filepath'], control_vars['verbose']) + "\n"
    msg += print_verbose("Model info", control_vars['verbose']) + "\n"
    msg += print_verbose("Number of joints: " + str(len(model.joint_ixs)), control_vars['verbose']) + "\n"
    msg += print_verbose("Joints indexes: " + str(model.joint_ixs), control_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", control_vars['verbose']) + "\n"
    msg += print_verbose("Max memory batch size: " + str(control_vars['max_mem_batch']), control_vars['verbose']) + "\n"
    msg += print_verbose("Length of dataset (in max mem batch size): " + str(len(dataset_loader)),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("Training batch size: " + str(control_vars['batch_size']), control_vars['verbose']) + "\n"
    msg += print_verbose("Starting epoch: " + str(control_vars['start_epoch']), control_vars['verbose']) + "\n"
    msg += print_verbose("Starting epoch iteration: " + str(control_vars['start_iter_mod']),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("Starting overall iteration: " + str(control_vars['start_iter']),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", control_vars['verbose']) + "\n"
    msg += print_verbose("Number of iterations per epoch: " + str(control_vars['n_iter_per_epoch']),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("Number of iterations to train: " + str(control_vars['num_iter']),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("Approximate number of epochs to train: " +
                         str(round(control_vars['num_iter'] / control_vars['n_iter_per_epoch'], 1)),
                         control_vars['verbose']) + "\n"
    msg += print_verbose("-----------------------------------------------------------", control_vars['verbose']) + "\n"

    if not control_vars['output_filepath'] == '':
        with open(control_vars['output_filepath'], 'w+') as f:
            f.write(msg + '\n')

def print_log_info(model, optimizer, epoch, total_loss, vars, control_vars, save_best=True):
    verbose = control_vars['verbose']
    print_verbose("", verbose)
    print_verbose("-------------------------------------------------------------------------------------------",
                  verbose)
    print_verbose("Saving checkpoints:", verbose)
    print_verbose("-------------------------------------------------------------------------------------------",
                  verbose)
    if save_best:
        save_checkpoint(vars['best_model_dict'],
                                filename=vars['checkpoint_filenamebase'] + 'best.pth.tar')
    checkpoint_model_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'control_vars': control_vars,
        'train_vars': vars,
    }
    save_checkpoint(checkpoint_model_dict, filename=vars['checkpoint_filenamebase'] + '.pth.tar')
    msg = ''
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose('Training (Epoch #' + str(epoch) + ' ' + str(control_vars['curr_epoch_iter']) + '/' + \
                         str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx'] + 1) + \
                         '(' + str(control_vars['iter_size']) + ')' + '/' + \
                         str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) + \
                         '(' + str(control_vars['batch_size']) + ')' + \
                         ' - log every ' + str(control_vars['log_interval']) + ' iter): ', verbose) + '\n'
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose("Current loss: " + str(total_loss), verbose) + "\n"
    msg += print_verbose("Best loss: " + str(vars['best_loss']), verbose) + "\n"
    msg += print_verbose("Mean total loss: " + str(np.mean(vars['losses'])), verbose) + "\n"
    msg += print_verbose("Mean loss for last " + str(control_vars['log_interval']) +
                         " iterations (average total loss): " + str(
        np.mean(vars['losses'][-control_vars['log_interval']:])), verbose) + "\n"
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose("Joint pixel losses:", verbose) + "\n"
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    joint_loss_avg = 0
    for joint_ix in model.joint_ixs:
        msg += print_verbose("\tJoint index: " + str(joint_ix), verbose) + "\n"
        mean_joint_pixel_loss = np.mean(
                                 np.array(vars['pixel_losses'])
                                 [-control_vars['log_interval']:, joint_ix])
        joint_loss_avg += mean_joint_pixel_loss
        msg += print_verbose("\tTraining set mean error for last " + str(control_vars['log_interval']) +
                             " iterations (average pixel loss): " +
                             str(mean_joint_pixel_loss),
                             verbose) + "\n"
        msg += print_verbose("\tTraining set stddev error for last " + str(control_vars['log_interval']) +
                             " iterations (average pixel loss): " +
                             str(np.std(
                                 np.array(vars['pixel_losses'])[-control_vars['log_interval']:, joint_ix])),
                             verbose) + "\n"
        msg += print_verbose("\tThis is the last pixel dist loss: " + str(vars['pixel_losses'][-1][joint_ix]),
                             verbose) + "\n"
        msg += print_verbose("\tTraining set mean error for last " + str(control_vars['log_interval']) +
                             " iterations (average pixel loss of sample): " +
                             str(np.mean(np.array(vars['pixel_losses_sample'])[-control_vars['log_interval']:,
                                         joint_ix])), verbose) + "\n"
        msg += print_verbose("\tTraining set stddev error for last " + str(control_vars['log_interval']) +
                             " iterations (average pixel loss of sample): " +
                             str(np.std(np.array(vars['pixel_losses_sample'])[-control_vars['log_interval']:,
                                        joint_ix])), verbose) + "\n"
        msg += print_verbose(
            "\tThis is the last pixel dist loss of sample: " + str(vars['pixel_losses_sample'][-1][joint_ix]),
            verbose) + "\n"
        msg += print_verbose(
            "\t-------------------------------------------------------------------------------------------",
            verbose) + "\n"
        msg += print_verbose(
            "-------------------------------------------------------------------------------------------",
            verbose) + "\n"
    joint_loss_avg /= len(model.joint_ixs)
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    msg += print_verbose("Mean joint loss (pixel): " + str(joint_loss_avg), verbose) + '\n'
    msg += print_verbose("-------------------------------------------------------------------------------------------",
                         verbose) + "\n"
    if not control_vars['output_filepath'] == '':
        with open(control_vars['output_filepath'], 'a') as f:
            f.write(msg + '\n')