import torch
from torch.autograd import Variable
import synthhands_handler
import trainer
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from JORNet import JORNet
from trainer import run_until_curr_iter, save_final_checkpoint
import numpy as np
import visualize

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

def change_learning_rate(optimizer, lr, curr_iter):
    if curr_iter > 45000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer

def get_batch(loader, batch_idx, batch_size):
    data = []
    targets0 = []
    targets1 = []
    targets2 = []
    for i in range(batch_size):
        datum, target = loader.dataset[batch_idx + i]
        target0, target1, target2 = target
        data.append(datum)
        targets0.append(target0)
        targets1.append(target1)
        targets2.append(torch.from_numpy(target2))
    data = torch.stack(data)
    targets0 = torch.stack(targets0)
    targets1 = torch.stack(targets1)
    targets2 = torch.stack(targets2)
    targets = (targets0, targets1, targets2)
    return data, targets

def train(train_loader, model, optimizer, train_vars):
    verbose = train_vars['verbose']
    for batch_idx, (data, target) in enumerate(train_loader):
        train_vars['batch_idx'] = batch_idx
        # print info about performing first iter
        if batch_idx < train_vars['iter_size']:
            print_verbose("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(train_vars['iter_size']), verbose, n_tabs=0, erase_line=True)
        # check if arrived at iter to start
        arrived_curr_iter, train_vars = run_until_curr_iter(batch_idx, train_vars)
        if not arrived_curr_iter:
            continue
        # save checkpoint after final iteration
        if train_vars['curr_iter'] - 1 == train_vars['num_iter']:
            train_vars = trainer.save_final_checkpoint(train_vars, model, optimizer)
            break
        # start time counter
        start = time.time()
        # get data and target as torch Variables
        _, target_joints, target_heatmaps, target_joints_z = target
        # make target joints be relative
        target_joints = target_joints[:, 3:]
        data, target_heatmaps = Variable(data), Variable(target_heatmaps)
        if train_vars['use_cuda']:
            data = data.cuda()
            target_heatmaps = target_heatmaps.cuda()
            target_joints = target_joints.cuda()
            target_joints_z = target_joints_z.cuda()
        # get model output
        output = model(data)

        # accumulate loss for sub-mini-batch
        if train_vars['cross_entropy']:
            loss_func = my_losses.cross_entropy_loss_p_logq
        else:
            loss_func = my_losses.euclidean_loss
        weights_heatmaps_loss, weights_joints_loss = get_loss_weights(train_vars['curr_iter'])
        loss, loss_heatmaps, loss_joints = my_losses.calculate_loss_JORNet(
            loss_func, output, target_heatmaps, target_joints, train_vars['joint_ixs'],
            weights_heatmaps_loss, weights_joints_loss, train_vars['iter_size'])
        loss.backward()
        train_vars['total_loss'] += loss.item()
        train_vars['total_joints_loss'] += loss_joints.item()
        train_vars['total_heatmaps_loss'] += loss_heatmaps.item()
        # accumulate pixel dist loss for sub-mini-batch
        train_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            train_vars['total_pixel_loss'], output[3], target_heatmaps, train_vars['batch_size'])
        if train_vars['cross_entropy']:
            train_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
                train_vars['total_pixel_loss_sample'], output[3], target_heatmaps, train_vars['batch_size'])
        else:
            train_vars['total_pixel_loss_sample'] = [-1] * len(model.joint_ixs)

        '''
        For debugging training
        for i in range(train_vars['max_mem_batch']):
            filenamebase_idx = (batch_idx * train_vars['max_mem_batch']) + i
            filenamebase = train_loader.dataset.get_filenamebase(filenamebase_idx)
            visualize.plot_joints_from_heatmaps(target_heatmaps[i].data.cpu().numpy(),
                                                title='GT joints: ' + filenamebase, data=data[i].data.cpu().numpy())
            visualize.plot_joints_from_heatmaps(output[3][i].data.cpu().numpy(),
                                                title='Pred joints: ' + filenamebase, data=data[i].data.cpu().numpy())
            visualize.plot_image_and_heatmap(output[3][i][4].data.numpy(),
                                             data=data[i].data.numpy(),
                                             title='Thumb tib heatmap: ' + filenamebase)
            visualize.show()
        '''

        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % train_vars['iter_size'] == 0
        if minibatch_completed:
            # visualize
            # ax, fig = visualize.plot_3D_joints(target_joints[0])
            # visualize.plot_3D_joints(target_joints[1], ax=ax, fig=fig)
            if train_vars['curr_iter'] % train_vars['log_interval'] == 0:
                fig, ax = visualize.plot_3D_joints(target_joints[0])
                visualize.savefig('joints_GT_' + str(train_vars['curr_iter']) + '.png')
                #visualize.plot_3D_joints(target_joints[1], fig=fig, ax=ax, color_root='C7')
                #visualize.plot_3D_joints(output[7].data.cpu().numpy()[0], fig=fig, ax=ax, color_root='C7')
                visualize.plot_3D_joints(output[7].data.cpu().numpy()[0])
                visualize.savefig('joints_model_' + str(train_vars['curr_iter']) + '.png')
                #visualize.show()
                #visualize.savefig('joints_' + str(train_vars['curr_iter']) + '.png')
            # change learning rate to 0.01 after 45000 iterations
            optimizer = change_learning_rate(optimizer, 0.01, train_vars['curr_iter'])
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
                    'train_vars': train_vars
                }
            # log checkpoint
            if train_vars['curr_iter'] % train_vars['log_interval'] == 0:
                trainer.print_log_info(model, optimizer, epoch, total_loss, train_vars, train_vars)
                aa1 = target_joints[0].data.cpu().numpy()
                aa2 = output[7][0].data.cpu().numpy()
                output_joint_loss = np.sum(np.abs(aa1 - aa2)) / 63
                msg = ''
                msg += print_verbose(
                    "-------------------------------------------------------------------------------------------",
                    verbose) + "\n"
                msg += print_verbose('\tJoint Coord Avg Loss for first image of current mini-batch: ' +
                                     str(output_joint_loss) + '\n', train_vars['verbose'])
                msg += print_verbose(
                    "-------------------------------------------------------------------------------------------",
                    verbose) + "\n"
                if not train_vars['output_filepath'] == '':
                    with open(train_vars['output_filepath'], 'a') as f:
                        f.write(msg + '\n')
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


model, optimizer, train_vars = trainer.get_vars(model_class=JORNet)
if train_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_loader = synthhands_handler.get_SynthHands_trainloader(root_folder=train_vars['root_folder'],
                                                             joint_ixs=model.joint_ixs,
                                                             heatmap_res=(128, 128),
                                                             batch_size=train_vars['max_mem_batch'],
                                                             verbose=train_vars['verbose'],
                                                             crop_hand=train_vars['crop_hand'])

train_vars['num_batches'] = len(train_loader)
train_vars['n_iter_per_epoch'] = int(len(train_loader) / train_vars['iter_size'])

train_vars['tot_iter'] = int(len(train_loader) / train_vars['iter_size'])
train_vars['start_iter_mod'] = train_vars['start_iter'] % train_vars['tot_iter']

train_vars['start_epoch'] = int(train_vars['start_iter'] / train_vars['n_iter_per_epoch'])

trainer.print_header_info(model, train_loader, train_vars)

model.train()
train_vars['curr_iter'] = 1

msg = ''
for epoch in range(train_vars['num_epochs']):
    train_vars['curr_epoch_iter'] = 1
    if epoch + 1 < train_vars['start_epoch']:
        msg += print_verbose("\nAdvancing through epochs: " + str(epoch + 1), train_vars['verbose'], erase_line=True)
        train_vars['curr_iter'] += train_vars['n_iter_per_epoch']
        if not train_vars['output_filepath'] == '':
            with open(train_vars['output_filepath'], 'a') as f:
                f.write(msg + '\n')
        continue
    else:
        msg = ''
    train_vars['total_loss'] = 0
    train_vars['total_pixel_loss'] = [0] * len(model.joint_ixs)
    train_vars['total_pixel_loss_sample'] = [0] * len(model.joint_ixs)
    optimizer.zero_grad()
    # train model
    train_vars = train(train_loader, model, optimizer, train_vars)
    if train_vars['done_training']:
        msg += print_verbose("Done training.", train_vars['verbose'])
        if not train_vars['output_filepath'] == '':
            with open(train_vars['output_filepath'], 'a') as f:
                f.write(msg + '\n')
        break