import sys
import JORNet
import optimizers as my_optimizers
import torch
from torch.autograd import Variable
import io_data
import numpy as np
import trainer
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
import argparse

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('--num_iter', dest='num_iter', type=int, required=True,
                    help='Total number of iterations to train')
parser.add_argument('-c', dest='checkpoint_filepath', default='',
                    help='Checkpoint file from which to begin training')
parser.add_argument('--log_interval', dest='log_interval', default=10,
                    help='Number of iterations interval on which to log'
                         ' a model checkpoint (default 10)')
parser.add_argument('--log_interval_valid', dest='log_interval_valid', default=1000,
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

args = parser.parse_args()

args.joint_ixs = list(map(int, args.joint_ixs))

if args.use_cuda:
    print_verbose("Using CUDA", args.verbose)
else:
    print_verbose("Not using CUDA", args.verbose)

if args.output_filepath == '':
    print_verbose("No output filepath specified", args.verbose)
else:
    f = open(args.output_filepath, 'w')
    sys.stdout = f

control_vars = trainer.initialize_control_vars(args.num_iter, args.max_mem_batch, args.batch_size,
                                               args.log_interval, args.log_interval_valid)
train_vars = trainer.initialize_train_vars(args.joint_ixs)
if args.checkpoint_filepath == '':
    print_verbose("Creating network from scratch", args.verbose)
    print_verbose("Building network...", args.verbose)
    jornet = JORNet.JORNet(joint_ixs=args.joint_ixs, use_cuda=args.use_cuda)
    if args.load_resnet:
        jornet = trainer.load_resnet_weights_into_HALNet(jornet, args.verbose)
    print_verbose("Done building network", args.verbose)
    optimizer = my_optimizers.get_adadelta_halnet(jornet)
else:
    print_verbose("Loading model and optimizer from file: " + args.checkpoint_filepath, args.verbose)
    jornet, optimizer, train_vars, control_vars =\
        io_data.load_checkpoint(filename=args.checkpoint_filepath, model_class=JORNet.JORNet,
                                num_iter=100000, log_interval=10,
                                log_interval_valid=1000, batch_size=16, max_mem_batch=args.max_mem_batch)

def train(model, optimizer, train_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        control_vars['batch_idx'] = batch_idx
        if batch_idx < control_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(control_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # check if arrived at iter to start
        if control_vars['curr_epoch_iter'] < control_vars['start_iter_mod']:
            if batch_idx % control_vars['iter_size'] == 0:
                print_verbose("\rGoing through iterations to arrive at last one saved... " +
                      str(int(control_vars['curr_epoch_iter']*100.0/control_vars['start_iter_mod'])) + "% of " +
                      str(control_vars['start_iter_mod']) + " iterations (" +
                      str(control_vars['curr_epoch_iter']) + "/" + str(control_vars['start_iter_mod']) + ")",
                              verbose, n_tabs=0, erase_line=True)
                control_vars['curr_epoch_iter'] += 1
                control_vars['curr_iter'] += 1
                curr_epoch_iter += 1
            continue
        # save checkpoint after final iteration
        if control_vars['curr_iter'] == control_vars['num_iter']:
            print_verbose("\nReached final number of iterations: " + str(control_vars['num_iter']), verbose)
            print_verbose("\tSaving final model checkpoint...", verbose)
            final_model_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'control_vars': control_vars,
                'train_vars': train_vars,
            }
            trainer.save_checkpoint(final_model_dict,
                            filename='final_model_iter_' + str(control_vars['num_iter']) + '.pth.tar')
            control_vars['done_training'] = True
            break
        # start time counter
        start = time.time()
        # get data and targetas cuda variables
        target_heatmaps, target_joints = target
        data, target_heatmaps, target_joints = \
            Variable(data).cuda(), Variable(target_heatmaps).cuda(), Variable(target_joints).cuda()
        # visualize if debugging
        # get model output
        output = model(data)
        # accumulate loss for sub-mini-batch
        loss = my_losses.calculate_loss_main_with_joints(output, target_heatmaps, target_joints,
                                                         control_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        train_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            train_vars['total_pixel_loss'], output[0], target_heatmaps, args.batch_size)
        train_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            train_vars['total_pixel_loss_sample'], output[0], target_heatmaps, args.batch_size)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
            # optimise for mini-batch
            optimizer.step()
            # clear optimiser
            optimizer.zero_grad()
            # append loss
            train_vars['losses'].append(train_vars['total_loss'].data[0])
            # erase loss
            train_vars['total_loss'] = 0
            # append dist loss
            train_vars['pixel_losses'].append(train_vars['total_pixel_loss'])
            # erase pixel dist loss
            train_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
            # append dist loss of sample from output
            train_vars['pixel_losses_sample'].append(train_vars['total_pixel_loss_sample'])
            # erase dist loss of sample from output
            train_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
            # check if loss is better
            if train_vars['losses'][-1] < train_vars['best_loss']:
                train_vars['best_loss'] = train_vars['losses'][-1]
                print_verbose("  This is a best loss found so far: " + str(train_vars['losses'][-1]), verbose)
                train_vars['best_model_dict'] = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': train_vars,
                }
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                print_verbose("", verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Saving checkpoints:", verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                trainer.save_checkpoint(train_vars['best_model_dict'], filename='best_model_log.pth.tar')
                checkpoint_model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': train_vars,
                }
                trainer.save_checkpoint(checkpoint_model_dict, filename='checkpoint_model_log.pth.tar')
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Main loss:", verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Training set mean error for last " + str(control_vars['log_interval']) +
                      " iterations (average loss): " + str(np.mean(train_vars['losses'][-control_vars['log_interval']:])), verbose)
                print_verbose("This is the last loss: " + str(train_vars['losses'][-1]), verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Joint pixel losses:", verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                for joint_ix in model.joint_ixs:
                    print_verbose("\tJoint index: " + str(joint_ix), verbose)
                    print_verbose("\tTraining set mean error for last " + str(control_vars['log_interval']) +
                          " iterations (average pixel loss): " +
                          str(np.mean(np.array(train_vars['pixel_losses'])[-control_vars['log_interval']:, joint_ix])), verbose)
                    print_verbose("\tTraining set stddev error for last " + str(control_vars['log_interval']) +
                          " iterations (average pixel loss): " +
                          str(np.std(np.array(train_vars['pixel_losses'])[-control_vars['log_interval']:, joint_ix])), verbose)
                    print_verbose("\tThis is the last pixel dist loss: " + str(train_vars['pixel_losses'][-1][joint_ix]), verbose)
                    print_verbose("\tTraining set mean error for last " + str(control_vars['log_interval']) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(train_vars['pixel_losses_sample'])[-control_vars['log_interval']:, joint_ix])), verbose)
                    print_verbose("\tTraining set stddev error for last " + str(control_vars['log_interval']) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(train_vars['pixel_losses_sample'])[-control_vars['log_interval']:, joint_ix])), verbose)
                    print_verbose("\tThis is the last pixel dist loss of sample: " + str(train_vars['pixel_losses_sample'][-1][joint_ix]), verbose)
                    print_verbose("\t-------------------------------------------------------------------------------------------", verbose)
                    print_verbose("-------------------------------------------------------------------------------------------", verbose)
            if control_vars['curr_iter'] % control_vars['log_interval_valid'] == 0:
                print_verbose("\nSaving model and checkpoint model for validation", verbose)
                checkpoint_model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': train_vars,
                }
                trainer.save_checkpoint(checkpoint_model_dict, filename='checkpoint_model_log_for_valid_' +
                                                                        str(control_vars['curr_iter']) + '.pth.tar')
            # print time lapse
            prefix = 'Training (Epoch #' + str(epoch) + ' ' + str(curr_epoch_iter) + '/' +\
                     str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx']+1) + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)
            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            curr_epoch_iter += 1
    return train_vars, control_vars

torch.set_default_tensor_type('torch.cuda.FloatTensor')
train_loader = io_data.get_SynthHands_trainloader(joint_ixs=jornet.joint_ixs,
                                              batch_size=args.max_mem_batch,
                                              verbose=args.verbose)
control_vars['num_batches'] = len(train_loader)
control_vars['n_iter_per_epoch'] = int(len(train_loader) / control_vars['iter_size'])

control_vars['tot_iter'] = int(len(train_loader) / control_vars['iter_size'])
control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

print_verbose("-----------------------------------------------------------", args.verbose)
print_verbose("Model info", args.verbose)
print_verbose("Number of joints: " + str(len(jornet.joint_ixs)), args.verbose)
print_verbose("Joints indexes: " + str(jornet.joint_ixs), args.verbose)
print_verbose("-----------------------------------------------------------", args.verbose)
print_verbose("Max memory batch size: " + str(args.max_mem_batch), args.verbose)
print_verbose("Length of dataset (in max mem batch size): " + str(len(train_loader)), args.verbose)
print_verbose("Training batch size: " + str(args.batch_size), args.verbose)
print_verbose("Starting epoch: " + str(control_vars['start_epoch']), args.verbose)
print_verbose("Starting epoch iteration: " + str(control_vars['start_iter_mod']), args.verbose)
print_verbose("Starting overall iteration: " + str(control_vars['start_iter']), args.verbose)
print_verbose("-----------------------------------------------------------", args.verbose)
print_verbose("Number of iterations per epoch: " + str(control_vars['n_iter_per_epoch']), args.verbose)
print_verbose("Number of iterations to train: " + str(control_vars['num_iter']), args.verbose)
print_verbose("Approximate number of epochs to train: " +
              str(round(control_vars['num_iter']/control_vars['n_iter_per_epoch'], 1)), args.verbose)
print_verbose("-----------------------------------------------------------", args.verbose)

jornet.train()
control_vars['curr_epoch_iter'] = control_vars['start_iter_mod'] - 1
control_vars['curr_iter'] = control_vars['start_iter_mod'] - 1
#control_vars['log_interval'] = 1

print(train_vars['losses'][-10:])

for epoch in range(args.num_epochs):
    if epoch + 1 < control_vars['start_epoch']:
        print_verbose("Advancing through epochs: " + str(epoch + 1), args.verbose, erase_line=True)
        control_vars['curr_iter'] += control_vars['n_iter_per_epoch']
        continue
    train_vars['total_loss'] = 0
    train_vars['total_pixel_loss'] = [0] * len(jornet.joint_ixs)
    train_vars['total_pixel_loss_sample'] = [0] * len(jornet.joint_ixs)
    optimizer.zero_grad()
    train_vars, control_vars = train(jornet, optimizer, train_vars, control_vars, args.verbose)
    if control_vars['done_training']:
        break
    if epoch + 1 >= control_vars['start_epoch']:
        for param_group in optimizer.param_groups:
           param_group['lr'] = param_group['lr']*0.5
