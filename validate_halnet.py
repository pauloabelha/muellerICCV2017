import io_data
import torch
import time
import trainer
import validator
import losses
import numpy as np
from HALNet import HALNet
from debugger import print_verbose, show_target_and_output_to_image_info
from torch.autograd import Variable
from magic import display_est_time_loop

args, model, optimizer, control_vars, train_vars = trainer.parse_args(model_class=HALNet)

torch.set_default_tensor_type('torch.cuda.FloatTensor')
valid_loader = io_data.get_SynthHands_validloader(joint_ixs=model.joint_ixs,
                                              batch_size=args.max_mem_batch,
                                              verbose=args.verbose)

print("Validating model that was trained for " + str(control_vars['curr_iter']) + " iterations")

validator.plot_losses(train_vars)

def validate(valid_loader, model, train_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
    for batch_idx, (data, target) in enumerate(valid_loader):
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
        if control_vars['curr_iter'] == control_vars['num_iter']:
            print_verbose("\nReached final number of iterations: " + str(control_vars['num_iter']), verbose)
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
        loss = losses.calculate_loss_HALNet_euclidean(
            output, target_heatmaps, model.joint_ixs, model.WEIGHT_LOSS_INTERMED1,
            model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
            model.WEIGHT_LOSS_MAIN, control_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        train_vars['total_pixel_loss'] = losses.accumulate_pixel_dist_loss_multiple(
            train_vars['total_pixel_loss'], output[3], target_heatmaps, args.batch_size)
        train_vars['total_pixel_loss_sample'] = [-1] * len(model.joint_ixs)
        #train_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
        #    train_vars['total_pixel_loss_sample'], output[3], target_heatmaps, args.batch_size)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
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
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                print("")
                show_target_and_output_to_image_info(data, target_heatmaps, output[3])
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
                          str(np.mean(np.array(train_vars['pixel_losses_sample'])[-control_vars['log_interval']:, joint_ix])), verbose)
                    print_verbose("\tThis is the last pixel dist loss of sample: " + str(train_vars['pixel_losses_sample'][-1][joint_ix]), verbose)
                    print_verbose("\t-------------------------------------------------------------------------------------------", verbose)
                    print_verbose("-------------------------------------------------------------------------------------------", verbose)
            # print time lapse
            prefix = 'Validating (Batch ' + str(control_vars['batch_idx']+1) + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)
            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            curr_epoch_iter += 1
    return train_vars, control_vars

validate(valid_loader, model, train_vars, control_vars, verbose=True)