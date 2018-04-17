from torch.autograd import Variable
import io_data
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from magic import display_est_time_loop
import debugger
import losses as my_losses
import JORNet
import math

VERBOSE = True
DEBUGGING_VISUALLY = True
# max batch size that GPU can handle
MAX_MEM_BATCH_SIZE = 4
# actual batch size wanted
BATCH_SIZE = 16
LOG_INTERVAL = 1

def get_quant_results(model, valid_loader, results_filename='test_quant_results.p'):
    losses = []
    pixel_losses = []
    pixel_losses_sample = []
    curr_iter = 1
    iter_size = int(BATCH_SIZE / MAX_MEM_BATCH_SIZE)
    total_loss = 0
    curr_train_ix = 0
    tot_iter = int(len(valid_loader) / int(BATCH_SIZE / MAX_MEM_BATCH_SIZE))
    tot_toc = 0
    total_pixel_losses = [0] * len(model.joint_ixs)
    total_pixel_losses_sample = [0] * len(model.joint_ixs)
    results_dict = {}
    MAX_N_VALID_EXAMPLES = 1000
    for batch_idx, (data, target) in enumerate(valid_loader):
        if batch_idx < int(BATCH_SIZE / MAX_MEM_BATCH_SIZE):
            print("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(int(BATCH_SIZE / MAX_MEM_BATCH_SIZE)), end='')
        if curr_iter > MAX_N_VALID_EXAMPLES:
            break
        # start time counter
        start = time.time()
        curr_train_ix += 1
        # get data and targetas cuda variables
        target_heatmaps, target_joints = target
        data, target_heatmaps, target_joints = \
            Variable(data).cuda(), Variable(target_heatmaps).cuda(), Variable(target_joints).cuda()
        # visualize if debugging
        # get model output
        output = model(data)
        # accumulate loss for sub-mini-batch
        # accumulate loss for sub-mini-batch
        # accumulate loss for sub-mini-batch
        loss, loss_main, loss_joints = my_losses.calculate_loss_main_with_joints(
            output, target_heatmaps, target_joints, control_vars['iter_size'])
        loss.backward()
        total_loss += loss
        # accumulate pixel dist loss for sub-mini-batch
        total_pixel_losses = my_losses.accumulate_pixel_dist_loss_multiple(
            total_pixel_losses, output[0], target_heatmaps, BATCH_SIZE)
        total_pixel_losses_sample = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            total_pixel_losses_sample, output[0], target_heatmaps, BATCH_SIZE)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx + 1) % int(BATCH_SIZE / MAX_MEM_BATCH_SIZE) == 0
        if minibatch_completed:
            # optimise for mini-batch
            optimizer.step()
            # clear optimiser
            optimizer.zero_grad()
            # append loss
            losses.append(total_loss.data[0])
            # erase loss
            total_loss = 0
            # append dist loss
            pixel_losses.append(total_pixel_losses)
            # erase pixel dist loss
            total_pixel_losses = [0] * len(model.joint_ixs)
            # append dist loss of sample from output
            pixel_losses_sample.append(total_pixel_losses_sample)
            # erase dist loss of sample from output
            total_pixel_losses_sample = [0] * len(model.joint_ixs)
            # check if dist loss is better
            if curr_iter % LOG_INTERVAL == 0:
                if DEBUGGING_VISUALLY:
                    print("")
                    debugger.show_target_and_output_to_image_info(data, target_heatmaps, output[0])
                print("-------------------------------------------------------------------------------------------")
                print("Main loss:")
                print("-------------------------------------------------------------------------------------------")
                print("Training set mean error for last " + str(LOG_INTERVAL) +
                      " iterations (average loss): " + str(np.mean(losses[-LOG_INTERVAL:])))
                print("This is the last loss: " + str(losses[-1]))
                print("-------------------------------------------------------------------------------------------")
                print("Joint pixel losses:")
                print("-------------------------------------------------------------------------------------------")
                for joint_ix in model.joint_ixs:
                    print("\tJoint index: " + str(joint_ix))
                    print("\tTraining set mean error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss): " +
                          str(np.mean(np.array(pixel_losses)[-LOG_INTERVAL:, joint_ix])))
                    print("\tTraining set stddev error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss): " +
                          str(np.std(np.array(pixel_losses)[-LOG_INTERVAL:, joint_ix])))
                    print("\tThis is the last pixel dist loss: " + str(dist_losses[-1][joint_ix]))
                    print("\tTraining set mean error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(pixel_losses_sample)[-LOG_INTERVAL:, joint_ix])))
                    print("\tTraining set stddev error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(pixel_losses_sample)[-LOG_INTERVAL:, joint_ix])))
                    print("\tThis is the last pixel dist loss of sample: " + str(pixel_losses_sample[-1][joint_ix]))
                    print(
                        "\t-------------------------------------------------------------------------------------------")
                print("-------------------------------------------------------------------------------------------")
                results_dict = {
                    'losses': losses,
                    'pixel_losses': pixel_losses,
                    'pixel_losses_sample': pixel_losses_sample,
                    'JOINT_IXS': model.joint_ixs
                }
                with open(results_filename, 'wb') as handle:
                    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tot_toc = display_est_time_loop(tot_toc + time.time() - start,
                                            curr_iter, tot_iter,
                                            prefix='Validation: ' +
                                                   'Iter #' + str(curr_iter) + "/" + str(tot_iter) +
                                                   ' - show info every ' + str(LOG_INTERVAL) + ' iter): ')
            curr_iter += 1
    with open(results_filename, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return losses, pixel_losses, pixel_losses_sample

def load_quant_results(results_filename='test_quant_results.p'):
    return pickle.load(open(results_filename, "rb"))

def show_hist_quant_results(results, xlabel='', ylabel='', title=''):
    mu = np.round(np.mean(results), 1)
    sigma = np.round(np.std(results), 1)
    # the histogram of the data
    n, bins, patches = plt.hist(results,
                                50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(bins.min(), n.max(), r'$\mu='+str(mu) + ',\ \sigma=' + str(sigma))
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

model, optimizer, train_vars, control_vars = io_data.load_checkpoint(filename='trained_halnet_log_.pth.tar',
                                                                      model_class=JORNet.JORNet)
START_ITER = control_vars['curr_iter'] + 1

print("Validating model that was trained for " + str(control_vars['curr_iter']) + " iterations")

# plot train losses
if DEBUGGING_VISUALLY:
    main_losses = train_vars['losses']
    main_losses = np.divide(main_losses, math.log(2))
    main_loss_handle, = plt.plot(main_losses, label='Total loss (bits)')
    dist_losses = train_vars['pixel_losses']
    dist_losses = np.array(dist_losses)
    plot_handles = [main_loss_handle]
    for j in range(dist_losses.shape[1]):
        plot_handle, = plt.plot([x / 1.0 for x in dist_losses[:, j]], label='Joint ' + str(j) + ' loss (pixels)')
        plot_handles.append(plot_handle)
    plt.ylabel('Losses (total loss and joint losses are in different units)')
    plt.legend(handles=plot_handles)
    plt.show()

valid_loader = io_data.get_SynthHands_validloader(joint_ixs=model.joint_ixs,
                                              batch_size=MAX_MEM_BATCH_SIZE,
                                              verbose=VERBOSE)
model.eval()
losses, pixel_losses, pixel_losses_sample = get_quant_results(
    model, valid_loader, results_filename='valid_results_' + str(START_ITER) + '.p')

# plot valid losses
if DEBUGGING_VISUALLY:
    plt.plot(losses)
    plt.plot([x / 10.0 for x in pixel_losses])
    plt.ylabel('Valid loss and pixel loss (div by 10)')
    plt.show()

    quant_results = load_quant_results(results_filename='valid_results.p')

    show_hist_quant_results(results=quant_results['losses'],
                            xlabel='Cross-entropy loss',
                            ylabel='Count',
                            title='Histogram of cross-entropy')

    show_hist_quant_results(results=quant_results['pixel_losses'],
                            xlabel='Pixel distance from max of output',
                            ylabel='Count',
                            title='Histogram of pixel dist loss (max of heatmap)')

    show_hist_quant_results(results=quant_results['pixel_losses_sample'],
                            xlabel='Pixel distance from sample of output',
                            ylabel='Count',
                            title='Histogram of pixel dist loss (sample of heatmap)')


