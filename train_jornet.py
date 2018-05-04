import torch
from torch.autograd import Variable
import io_data
import trainer
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from JORNet import JORNet
from halnet_crop import crop_batch_input_images
from matplotlib import pyplot as plt
import numpy as np
import visualize

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

def train(train_loader, model, optimizer, train_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        control_vars['batch_idx'] = batch_idx
        if batch_idx < control_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(control_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # check if arrived at iter to start
        if control_vars['curr_epoch_iter'] < control_vars['start_iter_mod']:
            msg = ''
            if batch_idx % control_vars['iter_size'] == 0:
                msg += print_verbose("\rGoing through iterations to arrive at last one saved... " +
                      str(int(control_vars['curr_epoch_iter']*100.0/control_vars['start_iter_mod'])) + "% of " +
                      str(control_vars['start_iter_mod']) + " iterations (" +
                      str(control_vars['curr_epoch_iter']) + "/" + str(control_vars['start_iter_mod']) + ")",
                              verbose, n_tabs=0, erase_line=True)
                control_vars['curr_epoch_iter'] += 1
                control_vars['curr_iter'] += 1
                curr_epoch_iter += 1
            if not control_vars['output_filepath'] == '':
                with open(control_vars['output_filepath'], 'a') as f:
                    f.write(msg + '\n')
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
                            filename=train_vars['checkpoint_filenamebase'] +
                                     'final' + str(control_vars['num_iter']) + '.pth.tar')
            control_vars['done_training'] = True
            break
        # start time counter
        start = time.time()
        # get data and targetas cuda variables
        target_heatmaps, target_joints, target_roothand = target
        data, target_heatmaps, target_joints, target_roothand = Variable(data), Variable(target_heatmaps),\
                                               Variable(target_joints), Variable(target_roothand)
        if train_vars['use_cuda']:
            data = data.cuda()
            target_heatmaps = target_heatmaps.cuda()
            target_joints = target_joints.cuda()
        # get model output

        #visualize.plot_joints_from_heatmaps(target_heatmaps[0, :, :, :].data.numpy(),
        #                                    title='', data=data[0].data.numpy())
        #visualize.show()
        #visualize.plot_image_and_heatmap(target_heatmaps[0][20].data.numpy(),
        #                                 data=data[0].data.numpy(),
        #                                 title='')
        #visualize.show()
        output = model(data)
        # accumulate loss for sub-mini-batch
        if train_vars['cross_entropy']:
            loss_func = my_losses.cross_entropy_loss_p_logq
        else:
            loss_func = my_losses.euclidean_loss
        weights_heatmaps_loss, weights_joints_loss = get_loss_weights(control_vars['curr_iter'])
        loss, loss_heatmaps, loss_joints = my_losses.calculate_loss_JORNet(
            loss_func, output, target_heatmaps, target_joints, train_vars['joint_ixs'],
            weights_heatmaps_loss, weights_joints_loss, control_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss.data[0]
        train_vars['total_joints_loss'] += loss_joints.data[0]
        train_vars['total_heatmaps_loss'] += loss_heatmaps.data[0]
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
            train_vars['losses'].append(train_vars['total_loss'])
            # erase total loss
            total_loss = train_vars['total_loss']
            train_vars['total_loss'] = 0
            # append total joints loss
            train_vars['losses_joints'].append(train_vars['total_joints_loss'])
            # erase total joints loss
            train_vars['total_joints_loss'] = 0
            # append total joints loss
            train_vars['losses_heatmaps'].append(train_vars['total_heatmaps_loss'])
            # erase total joints loss
            train_vars['total_heatmaps_loss'] = 0
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
            if train_vars['losses_joints'][-1] < train_vars['best_loss_joints']:
                train_vars['best_loss_joints'] = train_vars['losses_joints'][-1]
            if train_vars['losses_heatmaps'][-1] < train_vars['best_loss_heatmaps']:
                train_vars['best_loss_heatmaps'] = train_vars['losses_heatmaps'][-1]
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                trainer.print_log_info(model, optimizer, epoch, total_loss, train_vars, control_vars)
                aa1 = target_joints[0].data.cpu().numpy().reshape((21, 3))
                aa2 = output[7][0].data.cpu().numpy().reshape((21, 3))
                output_joint_loss = np.sum(np.abs(aa1 - aa2)) / 63
                msg = ''
                msg += print_verbose(
                    "-------------------------------------------------------------------------------------------",
                    verbose) + "\n"
                msg += print_verbose('\tJoint Coord Avg Loss: ' +
                                     str(output_joint_loss) + '\n', control_vars['verbose'])
                msg += print_verbose(
                    "-------------------------------------------------------------------------------------------",
                    verbose) + "\n"
                if not control_vars['output_filepath'] == '':
                    with open(control_vars['output_filepath'], 'a') as f:
                        f.write(msg + '\n')
            if control_vars['curr_iter'] % control_vars['log_interval_valid'] == 0:
                print_verbose("\nSaving model and checkpoint model for validation", verbose)
                checkpoint_model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': train_vars,
                }
                trainer.save_checkpoint(checkpoint_model_dict,
                                        filename=train_vars['checkpoint_filenamebase'] + 'for_valid_' +
                                                 str(control_vars['curr_iter']) + '.pth.tar')



            # print time lapse
            prefix = 'Training (Epoch #' + str(epoch) + ' ' + str(control_vars['curr_epoch_iter']) + '/' +\
                     str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx']+1) +\
                     '(' + str(control_vars['iter_size']) + ')' + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     '(' + str(control_vars['batch_size']) + ')' +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)

            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            control_vars['curr_epoch_iter'] += 1


    return train_vars, control_vars

model, optimizer, control_vars, train_vars = trainer.get_vars(model_class=JORNet)
if train_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_loader = io_data.get_SynthHands_trainloader(root_folder=train_vars['root_folder'],
                                                  joint_ixs=model.joint_ixs,
                                                  heatmap_res=(128, 128),
                                              batch_size=control_vars['max_mem_batch'],
                                              verbose=control_vars['verbose'],
                                                  crop_hand=True)
control_vars['num_batches'] = len(train_loader)
control_vars['n_iter_per_epoch'] = int(len(train_loader) / control_vars['iter_size'])

control_vars['tot_iter'] = int(len(train_loader) / control_vars['iter_size'])
control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

trainer.print_header_info(model, train_loader, control_vars)

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
        print_verbose("Done training.", control_vars['verbose'])
        break