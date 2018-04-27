import torch
from torch.autograd import Variable
import io_data
import validator
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from HALNet import HALNet
import trainer
from debugger import show_target_and_output_to_image_info

model, optimizer, control_vars, valid_vars, train_control_vars = validator.parse_args(model_class=HALNet)

def validate(valid_loader, model, optimizer, valid_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
    for batch_idx, (data, target) in enumerate(valid_loader):
        control_vars['batch_idx'] = batch_idx
        if batch_idx < control_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(control_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # save checkpoint after final iteration
        if control_vars['curr_iter'] == control_vars['num_iter']:
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
        if valid_vars['cross_entropy']:
            loss_func = my_losses.cross_entropy_loss_p_logq
        else:
            loss_func = my_losses.euclidean_loss
        loss = my_losses.calculate_loss_HALNet(loss_func,
            output, target_heatmaps, model.joint_ixs, model.WEIGHT_LOSS_INTERMED1,
            model.WEIGHT_LOSS_INTERMED2, model.WEIGHT_LOSS_INTERMED3,
            model.WEIGHT_LOSS_MAIN, control_vars['iter_size'])
        loss.backward()
        valid_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        valid_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            valid_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        valid_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            valid_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % control_vars['iter_size'] == 0
        if minibatch_completed:
            # append total loss
            valid_vars['losses'].append(valid_vars['total_loss'].data[0])
            # erase total loss
            total_loss = valid_vars['total_loss'].data[0]
            valid_vars['total_loss'] = 0
            # append dist loss
            valid_vars['pixel_losses'].append(valid_vars['total_pixel_loss'])
            # erase pixel dist loss
            valid_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
            # append dist loss of sample from output
            valid_vars['pixel_losses_sample'].append(valid_vars['total_pixel_loss_sample'])
            # erase dist loss of sample from output
            valid_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
            # check if loss is better
            if valid_vars['losses'][-1] < valid_vars['best_loss']:
                valid_vars['best_loss'] = valid_vars['losses'][-1]
                print_verbose("  This is a best loss found so far: " + str(valid_vars['losses'][-1]), verbose)
            # log checkpoint
            if control_vars['curr_iter'] % control_vars['log_interval'] == 0:
                trainer.print_log_info(model, optimizer, 0, total_loss,
                                       valid_vars, control_vars, save_best=False)
                if control_vars['visual_debugging']:
                    show_target_and_output_to_image_info(data, target_heatmaps, output[3])

            # print time lapse
            prefix = 'Validating (' + str(control_vars['curr_epoch_iter']) + '/' +\
                     str(control_vars['tot_iter']) + ')' + ', (Batch ' + str(control_vars['batch_idx']+1) +\
                     '(' + str(control_vars['max_mem_batch']) + ')' + '/' +\
                     str(control_vars['num_batches']) + ')' + ', (Iter #' + str(control_vars['curr_iter']) +\
                     '(' + str(control_vars['batch_size']) + ')' +\
                     ' - log every ' + str(control_vars['log_interval']) + ' iter): '
            control_vars['tot_toc'] = display_est_time_loop(control_vars['tot_toc'] + time.time() - start,
                                                            control_vars['curr_iter'], control_vars['num_iter'],
                                                            prefix=prefix)

            control_vars['curr_iter'] += 1
            control_vars['start_iter'] = control_vars['curr_iter'] + 1
            control_vars['curr_epoch_iter'] += 1


    return valid_vars, control_vars

torch.set_default_tensor_type('torch.cuda.FloatTensor')
valid_loader = io_data.get_SynthHands_validloader(root_folder=valid_vars['root_folder'],
                                                  joint_ixs=model.joint_ixs,
                                              batch_size=control_vars['max_mem_batch'],
                                              verbose=control_vars['verbose'])
control_vars['num_batches'] = len(valid_loader)
control_vars['n_iter_per_epoch'] = int(len(valid_loader) / control_vars['iter_size'])

control_vars['tot_iter'] = int(len(valid_loader) / control_vars['iter_size'])
control_vars['start_iter_mod'] = control_vars['start_iter'] % control_vars['tot_iter']

trainer.print_header_info(model, valid_loader, control_vars)

msg = print_verbose("Number of iterations trained: " +
                    str(train_control_vars['curr_iter']), control_vars['verbose']) + '\n'
if not control_vars['output_filepath'] == '':
    with open(control_vars['output_filepath'], 'a') as f:
        f.write(msg + '\n')

control_vars['curr_iter'] = 1
control_vars['curr_epoch_iter'] = 1
valid_vars['total_loss'] = 0
valid_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
valid_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
optimizer.zero_grad()
# validate model
valid_vars, control_vars = validate(valid_loader, model, optimizer, valid_vars, control_vars, control_vars['verbose'])
if control_vars['done_training']:
    print_verbose("Done validating.", control_vars['verbose'])