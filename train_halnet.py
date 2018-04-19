import torch
from torch.autograd import Variable
import io_data
import numpy as np
import trainer
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from HALNet import HALNet

CHECKPOINT_FILENAMEBASE ='trained_halnet_log_'

model, optimizer, control_vars, train_vars = trainer.parse_args(model_class=HALNet)

def train(train_loader, model, optimizer, train_vars, control_vars, verbose=True):
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
        data, target_heatmaps = Variable(data).cuda(), Variable(target_heatmaps).cuda()
        # visualize if debugging
        # get model output
        output = model(data)
        # accumulate loss for sub-mini-batch
        if train_vars['cross_entropy']:
            loss_func = my_losses.cross_entropy_loss_p_logq
        else:
            loss_func = my_losses.euclidean_loss
        loss = my_losses.calculate_loss_HALNet(loss_func,
            output, target_heatmaps, model.joint_ixs, model.WEIGHT_LOSS_INTERMED1,
            model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
            model.WEIGHT_LOSS_MAIN, control_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        train_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            train_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        train_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            train_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
            # optimise for mini-batch
            optimizer.step()
            # clear optimiser
            optimizer.zero_grad()
            # append total loss
            train_vars['losses'].append(train_vars['total_loss'].data[0])
            # erase total loss
            total_loss = train_vars['total_loss'].data[0]
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
                trainer.save_checkpoint(train_vars['best_model_dict'],
                                        filename=CHECKPOINT_FILENAMEBASE + 'best.pth.tar')
                checkpoint_model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': train_vars,
                }
                trainer.save_checkpoint(checkpoint_model_dict,
                                        filename=CHECKPOINT_FILENAMEBASE + '.pth.tar')
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Total loss: " + str(total_loss), verbose)
                print_verbose("-------------------------------------------------------------------------------------------", verbose)
                print_verbose("Training set mean error for last " + str(control_vars['log_interval']) +
                      " iterations (average total loss): " + str(
                    np.mean(train_vars['losses'][-control_vars['log_interval']:])), verbose)
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
                          str(np.std(np.array(train_vars['pixel_losses_sample'])[-control_vars['log_interval']:, joint_ix])), verbose)
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
                trainer.save_checkpoint(checkpoint_model_dict,
                                        filename=CHECKPOINT_FILENAMEBASE + 'for_valid_' +
                                                 str(control_vars['curr_iter']) + '.pth.tar')
            # print time lapse
            prefix = 'Training (Epoch #' + str(epoch) + ' ' + str(control_vars['curr_epoch_iter']) + '/' +\
                     str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx']+1) + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)
            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            control_vars['curr_epoch_iter'] += 1
    return train_vars, control_vars

torch.set_default_tensor_type('torch.cuda.FloatTensor')
train_loader = io_data.get_SynthHands_trainloader(joint_ixs=model.joint_ixs,
                                              batch_size=control_vars['max_mem_batch'],
                                              verbose=control_vars['verbose'])
control_vars['num_batches'] = len(train_loader)
control_vars['n_iter_per_epoch'] = int(len(train_loader) / control_vars['iter_size'])

control_vars['tot_iter'] = int(len(train_loader) / control_vars['iter_size'])
control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

print_verbose("-----------------------------------------------------------", control_vars['verbose'])
print_verbose("Model info", control_vars['verbose'])
print_verbose("Number of joints: " + str(len(model.joint_ixs)), control_vars['verbose'])
print_verbose("Joints indexes: " + str(model.joint_ixs), control_vars['verbose'])
print_verbose("-----------------------------------------------------------", control_vars['verbose'])
print_verbose("Max memory batch size: " + str(control_vars['max_mem_batch']), control_vars['verbose'])
print_verbose("Length of dataset (in max mem batch size): " + str(len(train_loader)), control_vars['verbose'])
print_verbose("Training batch size: " + str(control_vars['batch_size']), control_vars['verbose'])
print_verbose("Starting epoch: " + str(control_vars['start_epoch']), control_vars['verbose'])
print_verbose("Starting epoch iteration: " + str(control_vars['start_iter_mod']), control_vars['verbose'])
print_verbose("Starting overall iteration: " + str(control_vars['start_iter']), control_vars['verbose'])
print_verbose("-----------------------------------------------------------", control_vars['verbose'])
print_verbose("Number of iterations per epoch: " + str(control_vars['n_iter_per_epoch']), control_vars['verbose'])
print_verbose("Number of iterations to train: " + str(control_vars['num_iter']), control_vars['verbose'])
print_verbose("Approximate number of epochs to train: " +
              str(round(control_vars['num_iter']/control_vars['n_iter_per_epoch'], 1)), control_vars['verbose'])
print_verbose("-----------------------------------------------------------", control_vars['verbose'])

model.train()
control_vars['curr_iter'] = 1

for epoch in range(control_vars['num_epochs']):
    control_vars['curr_epoch_iter'] = 1
    if epoch + 1 < control_vars['start_epoch']:
        print_verbose("Advancing through epochs: " + str(epoch + 1), control_vars['verbose'], erase_line=True)
        control_vars['curr_iter'] += control_vars['n_iter_per_epoch']
        continue
    train_vars['total_loss'] = 0
    train_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
    train_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
    optimizer.zero_grad()
    # train model
    train_vars, control_vars = train(train_loader, model, optimizer, train_vars, control_vars, control_vars['verbose'])
    if control_vars['done_training']:
        break
