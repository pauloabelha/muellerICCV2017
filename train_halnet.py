import torch
from torch.autograd import Variable
import synthhands_handler
import trainer
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from HALNet import HALNet
from trainer import run_until_curr_iter, save_final_checkpoint

def train(train_loader, model, optimizer, train_vars):
    verbose = train_vars['verbose']
    for batch_idx, (data, target) in enumerate(train_loader):
        train_vars['batch_idx'] = batch_idx
        # print info about performing first iter
        if batch_idx < train_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(train_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # check if arrived at iter to start
        arrived_curr_iter, train_vars = run_until_curr_iter(train_vars)
        if not arrived_curr_iter:
            continue
        # save checkpoint after final iteration
        if train_vars['curr_iter'] - 1 == train_vars['num_iter']:
            train_vars = save_final_checkpoint(train_vars)
            break
        # start time counter
        start = time.time()
        # get data and target as torch Variables
        target_heatmaps, target_joints, target_joints_z = target
        data, target_heatmaps = Variable(data), Variable(target_heatmaps)
        if train_vars['use_cuda']:
            data = data.cuda()
            target_heatmaps = target_heatmaps.cuda()
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
            model.WEIGHT_LOSS_MAIN, train_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss
        # accumulate pixel dist loss for sub-mini-batch
        train_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            train_vars['total_pixel_loss'], output[3], target_heatmaps, train_vars['batch_size'])
        if train_vars['cross_entropy']:
            train_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
                train_vars['total_pixel_loss_sample'], output[3], target_heatmaps, train_vars['batch_size'])
        else:
            train_vars['total_pixel_loss_sample'] = [-1] * len(model.joint_ixs)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % train_vars['iter_size'] == 0
        if minibatch_completed:
            # optimise for mini-batch
            optimizer.step()
            # clear optimiser
            optimizer.zero_grad()
            # append total loss
            train_vars['losses'].append(train_vars['total_loss'].item())
            # erase total loss
            total_loss = train_vars['total_loss'].item()
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
                    'train_vars': train_vars
                }
            # log checkpoint
            if train_vars['curr_iter'] % train_vars['log_interval'] == 0:
                trainer.print_log_info(model, optimizer, epoch, total_loss, train_vars, train_vars)

            if train_vars['curr_iter'] % train_vars['log_interval_valid'] == 0:
                print_verbose("\nSaving model and checkpoint model for validation", verbose)
                checkpoint_model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_vars': train_vars,
                }
                trainer.save_checkpoint(checkpoint_model_dict,
                                        filename=train_vars['checkpoint_filenamebase'] + 'for_valid_' +
                                                 str(train_vars['curr_iter']) + '.pth.tar')

            # print time lapse
            prefix = 'Training (Epoch #' + str(epoch) + ' ' + str(train_vars['curr_epoch_iter']) + '/' +\
                     str(train_vars['tot_iter']) + ')' + ', (Batch ' + str(train_vars['batch_idx']+1) +\
                     '(' + str(train_vars['iter_size']) + ')' + '/' +\
                     str(train_vars['num_batches']) + ')' + ', (Iter #' + str(train_vars['curr_iter']) +\
                     '(' + str(train_vars['batch_size']) + ')' +\
                     ' - log every ' + str(train_vars['log_interval']) + ' iter): '
            train_vars['tot_toc'] = display_est_time_loop(train_vars['tot_toc'] + time.time() - start,
                                                            train_vars['curr_iter'], train_vars['num_iter'],
                                                            prefix=prefix)

            train_vars['curr_iter'] += 1
            train_vars['start_iter'] = train_vars['curr_iter'] + 1
            train_vars['curr_epoch_iter'] += 1
    return train_vars


model, optimizer, train_vars = trainer.get_vars(model_class=HALNet)
if train_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_loader = synthhands_handler.get_SynthHands_trainloader(root_folder=train_vars['root_folder'],
                                                             joint_ixs=model.joint_ixs,
                                                             heatmap_res=(320, 240),
                                                             batch_size=train_vars['max_mem_batch'],
                                                             verbose=train_vars['verbose'])
train_vars['num_batches'] = len(train_loader)
train_vars['n_iter_per_epoch'] = int(len(train_loader) / train_vars['iter_size'])

train_vars['tot_iter'] = int(len(train_loader) / train_vars['iter_size'])
train_vars['start_iter_mod'] = train_vars['start_iter'] % train_vars['tot_iter']

trainer.print_header_info(model, train_loader, train_vars)

model.train()
train_vars['curr_iter'] = 1

for epoch in range(train_vars['num_epochs']):
    train_vars['curr_epoch_iter'] = 1
    if epoch + 1 < train_vars['start_epoch']:
        print_verbose("Advancing through epochs: " + str(epoch + 1), train_vars['verbose'], erase_line=True)
        train_vars['curr_iter'] += train_vars['n_iter_per_epoch']
        continue
    train_vars['total_loss'] = 0
    train_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
    train_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
    optimizer.zero_grad()
    # train model
    train_vars = train(train_loader, model, optimizer, train_vars)
    if train_vars['done_training']:
        print_verbose("Done training.", train_vars['verbose'])
        break