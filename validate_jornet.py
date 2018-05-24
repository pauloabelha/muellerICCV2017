import torch
from torch.autograd import Variable
import synthhands_handler
import trainer
import validator
import time
from magic import display_est_time_loop
import losses as my_losses
from debugger import print_verbose
from JORNet import JORNet
import visualize
import converter as conv
import numpy as np
import camera

def get_loss_weights(curr_iter):
    weights_heatmaps_loss = [0.5, 0.5, 0.5, 1.0]
    weights_joints_loss = [1250, 1250, 1250, 2500]
    if curr_iter > 45000:
        weights_heatmaps_loss = [0.1, 0.1, 0.1, 1.0]
        weights_joints_loss = [250, 250, 250, 2500]
    return weights_heatmaps_loss, weights_joints_loss

def validate(valid_loader, model, optimizer, valid_vars, control_vars, verbose=True):
    curr_epoch_iter = 1
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
        loss, loss_heatmaps, loss_joints = my_losses.calculate_loss_JORNet(
            loss_func, output, target_heatmaps, target_joints, valid_vars['joint_ixs'],
            weights_heatmaps_loss, weights_joints_loss, control_vars['iter_size'])
        valid_vars['total_loss'] += loss
        valid_vars['total_joints_loss'] += loss_joints
        valid_vars['total_heatmaps_loss'] += loss_heatmaps
        # accumulate pixel dist loss for sub-mini-batch
        valid_vars['total_pixel_loss'] = my_losses.accumulate_pixel_dist_loss_multiple(
            valid_vars['total_pixel_loss'], output[3], target_heatmaps, control_vars['batch_size'])
        valid_vars['total_pixel_loss_sample'] = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            valid_vars['total_pixel_loss_sample'], output[3], target_heatmaps, control_vars['batch_size'])
        # get boolean variable stating whether a mini-batch has been completed

        for i in range(control_vars['max_mem_batch']):
            filenamebase_idx = (batch_idx * control_vars['max_mem_batch']) + i
            filenamebase = valid_loader.dataset.get_filenamebase(filenamebase_idx)

            print('')
            print(filenamebase)

            visualize.plot_image(data[i].data.numpy())
            visualize.show()

            output_batch_numpy = output[7][i].data.cpu().numpy()
            fig, ax = visualize.plot_3D_joints(target_joints[i])
            visualize.plot_3D_joints(output_batch_numpy, fig=fig, ax=ax, color='C6')

            visualize.title(filenamebase)
            visualize.show()

            temp = np.zeros((21, 3))
            output_batch_numpy_abs = output_batch_numpy.reshape((20, 3))
            temp[1:, :] = output_batch_numpy_abs
            output_batch_numpy_abs = temp
            output_joints_colorspace = camera.joints_depth2color(
                output_batch_numpy_abs,
                depth_intr_matrix=synthhands_handler.DEPTH_INTR_MTX,
                handroot=target_handroot[i].data.cpu().numpy())
            visualize.plot_3D_joints(output_joints_colorspace)
            visualize.show()

            #image_rgbd = conv.numpy_to_plottable_rgb(data[i].data.numpy())
            #visualize.plot_joints_from_colorspace(output_joints_colorspace,
            #                                      title=filenamebase,
            #                                      fig=fig,
            #                                      data=image_rgbd)
            #visualize.show()


            #filenamebase_idx = (batch_idx * control_vars['max_mem_batch']) + i
            #filenamebase = valid_loader.dataset.get_filenamebase(filenamebase_idx)
            #visualize.plot_image(data[i].data.cpu().numpy(), title=filenamebase)
            #target_joints_colorspace = camera.joints_depth2color(output[7][i].data.cpu().numpy().reshape((21, 3)),
            #                                                     target_handroot.data.cpu().numpy())
            #visualize.plot_3D_joints(target_joints_colorspace)
            #fig = visualize.create_fig()
            #visualize.plot_joints_from_heatmaps(target_heatmaps[i].data.cpu().numpy(),
            #                                    title='GT joints: ' + filenamebase, data=data[i].data.cpu().numpy())
            #visualize.plot_joints_from_heatmaps(output[3][i].data.cpu().numpy(),
            #                                    title='Pred joints: ' + filenamebase, data=data[i].data.cpu().numpy())
            #visualize.plot_image_and_heatmap(output[3][i][4].data.numpy(),
            #                                 data=data[i].data.numpy(),
            #                                 title='Thumb tib heatmap: ' + filenamebase)
            #print('\n-----------------------------------------')
            #aa = output[3][i][8].data.numpy()
            #aa = np.exp(aa)
            #print(np.min(aa))
            #bb = np.unravel_index(np.argmax(aa), aa.shape)
            #print(bb)
            #print(np.max(aa))
            #print(model.cross_entropy)
            #print('-----------------------------------------')
            #visualize.savefig('/home/paulo/' + filenamebase.replace('/', '_') + '_heatmap')
            #visualize.plot_joints_from_colorspace(labels_colorspace, title=filenamebase, fig=fig, data=data_crop_img)
            #labels_colorspace = conv.heatmaps_to_joints_colorspace(output[3][i].data.numpy())
            #data_crop, crop_coords, labels_heatmaps, labels_colorspace = \
            #    io_data.crop_image_get_labels(data[i].data.numpy(), labels_colorspace, range(21))
            #data_crop_img = conv.numpy_to_plottable_rgb(data_crop)
            #visualize.plot_image(data_crop_img, title=filenamebase, fig=fig)
            #visualize.plot_joints_from_colorspace(labels_colorspace, title=filenamebase, fig=fig, data=data_crop_img)
            #visualize.savefig('/home/paulo/' + filenamebase.replace('/', '_') + '_crop')
            #visualize.show()
            aa1 = target_joints[i].data.cpu().numpy().reshape((20, 3))
            aa2 = output[7][i].data.cpu().numpy().reshape((20, 3))
            print('\n----------------------------------')
            print(np.sum(np.abs(aa1 - aa2)) / 60)
            print('----------------------------------')

        #loss.backward()
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
            valid_vars['losses'].append(valid_vars['total_loss'].data[0])
            # erase total loss
            total_loss = valid_vars['total_loss'].data[0]
            valid_vars['total_loss'] = 0
            # append total joints loss
            valid_vars['losses_joints'].append(valid_vars['total_joints_loss'].data[0])
            # erase total joints loss
            valid_vars['total_joints_loss'] = 0
            # append total joints loss
            valid_vars['losses_heatmaps'].append(valid_vars['total_heatmaps_loss'].data[0])
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
                trainer.print_log_info(model, optimizer, 1, total_loss, valid_vars, control_vars)
                model_dict = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'control_vars': control_vars,
                    'train_vars': valid_vars,
                }
                trainer.save_checkpoint(model_dict,
                                        filename=valid_vars['checkpoint_filenamebase'] +
                                                 str(control_vars['num_iter']) + '.pth.tar')
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


    return valid_vars, control_vars

model, optimizer, control_vars, valid_vars, train_control_vars = validator.parse_args(model_class=JORNet)

if valid_vars['use_cuda']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
visualize.plot_line(valid_vars['losses'], 'Main loss')
visualize.show()

visualize.plot_line(valid_vars['losses_heatmaps'], 'batch_halnetHeatmap loss')
visualize.show()

visualize.plot_line(valid_vars['losses_joints'], 'Joint loss')
visualize.show()
'''

valid_loader = synthhands_handler.get_SynthHands_validloader(root_folder=valid_vars['root_folder'],
                                                             joint_ixs=model.joint_ixs,
                                                             heatmap_res=(128, 128),
                                                             batch_size=control_vars['max_mem_batch'],
                                                             verbose=control_vars['verbose'],
                                                             crop_hand=True)
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

valid_vars, control_vars = validate(valid_loader, model, optimizer, valid_vars, control_vars, control_vars['verbose'])