import torch
from torch.autograd import Variable
import synthhands_handler
import egodexter_handler
import trainer
import validator
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from JORNet import JORNet
import visualize
import numpy as np

DEBUG_VISUALLY = False

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

def validate(valid_loader, model, optimizer, valid_vars, control_vars, verbose=True):
    losses_main = []
    for batch_idx, (data, target) in enumerate(valid_loader):
        control_vars['batch_idx'] = batch_idx
        if batch_idx < control_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                          str(batch_idx + 1) + "/" + str(control_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # start time counter
        start = time.time()
        # get data and targetas cuda variables
        target_heatmaps, target_joints, target_handroot = target
        # make target joints be relative
        target_joints = target_joints[:, 3:]
        data, target_heatmaps = Variable(data), Variable(target_heatmaps)
        if valid_vars['use_cuda']:
            data = data.cuda()
            target_joints = target_joints.cuda()
            target_heatmaps = target_heatmaps.cuda()
            target_handroot = target_handroot.cuda()
        # visualize if debugging
        # get model output
        output = model(data)
        # accumulate loss for sub-mini-batch
        if model.cross_entropy:
            loss_func = my_losses.cross_entropy_loss_p_logq
        else:
            loss_func = my_losses.euclidean_loss
        weights_heatmaps_loss, weights_joints_loss = get_loss_weights(control_vars['curr_iter'])
        loss, loss_heatmaps, loss_joints, loss_main = my_losses.calculate_loss_JORNet_for_valid(
            loss_func, output, target_heatmaps, target_joints, valid_vars['joint_ixs'],
            weights_heatmaps_loss, weights_joints_loss, control_vars['iter_size'])
        losses_main.append(loss_main.item() / 63.0)
        valid_vars['total_loss'] += loss
        valid_vars['total_joints_loss'] += loss_joints
        valid_vars['total_heatmaps_loss'] += loss_heatmaps
        # accumulate pixel dist loss for sub-mini-batch
        valid_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            valid_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        valid_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            valid_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        valid_vars['total_loss'] += loss
        valid_vars['total_joints_loss'] += loss_joints
        valid_vars['total_heatmaps_loss'] += loss_heatmaps
        # accumulate pixel dist loss for sub-mini-batch
        valid_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            valid_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        valid_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            valid_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
            # append total loss
            valid_vars['losses'].append(valid_vars['total_loss'].item())
            # erase total loss
            total_loss = valid_vars['total_loss'].item()
            valid_vars['total_loss'] = 0
            # append total joints loss
            valid_vars['losses_joints'].append(valid_vars['total_joints_loss'].item())
            # erase total joints loss
            valid_vars['total_joints_loss'] = 0
            # append total joints loss
            valid_vars['losses_heatmaps'].append(valid_vars['total_heatmaps_loss'].item())
            # erase total joints loss
            valid_vars['total_heatmaps_loss'] = 0
            # append dist loss
            valid_vars['pixel_losses'].append(valid_vars['total_pixel_loss'])
            # erase pixel dist loss
            valid_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
            # append dist loss of sample from output
            valid_vars['pixel_losses_sample'].append(valid_vars['total_pixel_loss_sample'])
            # erase dist loss of sample from output
            valid_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
            # check if loss is better
            #if valid_vars['losses'][-1] < valid_vars['best_loss']:
            #    valid_vars['best_loss'] = valid_vars['losses'][-1]
            #    print_verbose("  This is a best loss found so far: " + str(valid_vars['losses'][-1]), verbose)
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                trainer.print_log_info(model, optimizer, 1, total_loss, valid_vars, control_vars,
                                       save_best=False, save_a_checkpoint=False)
                model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': valid_vars,
                }
                #trainer.save_checkpoint(model_dict,
                #                        filename=valid_vars['checkpoint_filenamebase'] +
                #                                 str(control_vars['num_iter']) + '.pth.tar')
            # print time lapse
            prefix = 'Validating (Epoch #' + str(1) + ' ' + str(control_vars['curr_epoch_iter']) + '/' +\
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

    total_avg_loss = np.mean(losses_main)
    return valid_vars, control_vars, total_avg_loss


model, optimizer, control_vars, valid_vars, train_control_vars = validator.parse_args(model_class=JORNet)
if valid_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if "EgoDexter" in valid_vars['root_folder']:
    dataset_func = egodexter_handler.EgoDexterDataset
else:
    dataset_func = synthhands_handler.SynthHandsDataset

dataset = dataset_func(root_folder=valid_vars['root_folder'],
                                               type_='split',
                                               joint_ixs=model.joint_ixs,
                                               heatmap_res=(320, 240),
                                               splitfilename=valid_vars['split_filename'])



num_splits = dataset.num_splits
valid_errors = []
for split_ix in range(dataset.num_splits):
    print('Performing validation for {} splits at split #{}'.format(num_splits, split_ix + 1))
    dataset = dataset_func(root_folder=valid_vars['root_folder'],
                                                   type_='split',
                                                   joint_ixs=model.joint_ixs,
                                                   heatmap_res=(128, 128),
                                                   splitfilename=valid_vars['split_filename'],
                                                   split_ix=split_ix,
                                                   crop_hand=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False)
    control_vars['log_interval'] = len(valid_loader)

    control_vars['num_batches'] = len(valid_loader)
    control_vars['n_iter_per_epoch'] = int(len(valid_loader) / control_vars['iter_size'])
    control_vars['num_iter'] = len(valid_loader)

    control_vars['tot_iter'] = int(len(valid_loader) / control_vars['iter_size'])
    control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

    trainer.print_header_info(model, valid_loader, control_vars)

    control_vars['curr_iter'] = 1
    control_vars['curr_epoch_iter'] = 1

    valid_vars['total_loss'] = 0
    valid_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
    valid_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)

    valid_vars, control_vars, tot_joint_loss_avg = validate(valid_loader, model, optimizer, valid_vars, control_vars, control_vars['verbose'])

    valid_errors.append(tot_joint_loss_avg)
    print('Mean valid error: {}'.format(np.mean(valid_errors)))
    print('Stddev valid error: {}'.format(np.std(valid_errors)))





