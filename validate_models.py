from torch.autograd import Variable
import io_data
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time
from magic import display_est_time_loop
import debugger
import losses as my_losses

VERBOSE = True
DEBUGGING_VISUALLY = False
MAX_N_VALID_BATCHES = 10
# max batch size that GPU can handle
MAX_MEM_BATCH_SIZE = 8
# actual batch size wanted
BATCH_SIZE = 16
LOG_INTERVAL = 100

def get_validation_models_filenames(root_folder='/home/paulo/muellerICCV2017/',
                          valid_file_prefix='checkpoint_model_log_for_valid_'):
    filenames = []
    for file in os.listdir(root_folder):
        if os.path.isfile(os.path.join(root_folder, file)) \
                and valid_file_prefix in file:
            filenames.append(file)
            print("Found model to validate: " + file)
    return filenames

def get_quant_results(model, valid_loader, results_filename='test_quant_results.p'):
    losses = []
    pixel_losses = []
    pixel_losses_sample = []
    curr_iter = 1
    iter_size = int(BATCH_SIZE / MAX_MEM_BATCH_SIZE)
    total_loss = 0
    curr_train_ix = 0
    tot_iter = min(MAX_N_VALID_BATCHES, int(len(valid_loader) / int(BATCH_SIZE / MAX_MEM_BATCH_SIZE)))
    tot_toc = 0
    total_pixel_loss = 0
    total_pixel_loss_sample = 0
    results_dict = {}
    for batch_idx, (data, target) in enumerate(valid_loader):
        if curr_iter > MAX_N_VALID_BATCHES:
            break
        # start time counter
        start = time.time()
        curr_train_ix += 1
        # get data and targetas cuda variables
        data, target = Variable(data).cuda(), Variable(target).cuda()
        # get model output
        output = model(data.cuda())
        # accumulate loss for sub-mini-batch
        loss = my_losses.calculate_loss(output, target, iter_size,
                              model.WEIGHT_LOSS_INTERMED1,
                              model.WEIGHT_LOSS_INTERMED2,
                              model.WEIGHT_LOSS_INTERMED3,
                              model.WEIGHT_LOSS_MAIN)
        loss.backward()
        total_loss += loss
        # accumulate pixel dist loss for sub-mini-batch
        total_pixel_loss = my_losses.accumulate_pixel_dist_loss(
            total_pixel_loss, output[-1], target, BATCH_SIZE)
        total_pixel_loss_sample = my_losses.accumulate_pixel_dist_loss_from_sample(
            total_pixel_loss_sample, output[-1], target, BATCH_SIZE)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = curr_train_ix % int(BATCH_SIZE / MAX_MEM_BATCH_SIZE) == 0
        if minibatch_completed:
            # append loss
            losses.append(total_loss.data[0])
            # erase loss
            total_loss = 0
            # append dist loss
            pixel_losses.append(total_pixel_loss)
            # erase pixel dist loss
            total_pixel_loss = 0
            # append dist loss from sample
            pixel_losses_sample.append(total_pixel_loss_sample)
            # erase pixel dist loss from sample
            total_pixel_loss_sample = 0
            if curr_iter % LOG_INTERVAL == 0:
                if DEBUGGING_VISUALLY:
                    print("\nPixel loss: " + str(pixel_losses[-1]))
                    for idx in range(target.data.cpu().numpy().shape[0]):
                        debugger.show_target_and_output_to_image_info(data, target, output, idx)
                # check if dist loss is better
                print("\nValidation set mean error (loss): " + str(np.mean(losses)))
                print("Validation set stddev error (loss): " + str(np.std(losses)))
                print("Validation set mean error (pixel loss): " + str(np.mean(pixel_losses)))
                print("Validation set stddev error (pixel loss): " + str(np.std(pixel_losses)))
                print("Validation set mean error (pixel loss from sample of output): " + str(np.mean(pixel_losses_sample)))
                print("Validation set stddev error (pixel loss from sample of output): " + str(np.std(pixel_losses_sample)))
                print("Saving validation results in file: " + results_filename)
                results_dict = {
                    'losses': losses,
                    'pixel_losses': pixel_losses,
                    'pixel_losses_sample': pixel_losses_sample,
                }
                if not results_filename == '':
                    with open(results_filename, 'wb') as handle:
                        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tot_toc = display_est_time_loop(tot_toc + time.time() - start,
                                            curr_iter, tot_iter,
                                            prefix='Validation: ' +
                                                   'Iter #' + str(curr_iter) + "/" + str(tot_iter) +
                                                   ' - show info every ' + str(LOG_INTERVAL) + ' iter): ')
            curr_iter += 1
    if not results_filename == '':
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

valid_filenames = get_validation_models_filenames()
valid_dict = {}
valid_loader = io_data.get_HALNet_validloader(batch_size=MAX_MEM_BATCH_SIZE, verbose=VERBOSE)
halnet_dataset = io_data.SynthHandsHALNetValidDataset()
print("Max number of validation batches: " + str(MAX_N_VALID_BATCHES))

sorted_n_iters = []
n_iter_dict = {}
for valid_filename in valid_filenames:
    n_iter = int(valid_filename.split('_')[-1].split('.')[0])
    sorted_n_iters.append(n_iter)
    n_iter_dict[n_iter] = valid_filename
sorted_n_iters = sorted(sorted_n_iters)

idx = 0
for n_iter in sorted_n_iters:
    idx += 1
    valid_filename = n_iter_dict[n_iter]
    valid_iter_dict = {}
    pixel_losses_dict = {}
    pixel_losses_sample_dict = {}
    n_valid_iter = int(valid_filename.split('_')[-1].split('.')[0])
    model, optimizer, trained_dict = \
        io_data.load_checkpoint(filename=valid_filename)
    valid_model = {}
    valid_model['model'] = model
    valid_model['optimizer'] = optimizer
    valid_model['trained_dict'] = trained_dict
    trained_dict = valid_model['trained_dict']
    print("\nValidating model (" + str(idx) + "/" + str(len(sorted_n_iters)) +
          ") that was trained for " + str(trained_dict['curr_iter']) + " iterations")
    halnet = valid_model['model']
    halnet.eval()
    losses, pixel_losses, pixel_losses_sample = get_quant_results(
        halnet, valid_loader, results_filename='')
    print(" Mean loss: " + str(np.mean(losses)))
    valid_iter_dict['losses'] = losses
    valid_iter_dict['pixel_losses'] = pixel_losses
    valid_iter_dict['pixel_losses_sample'] = pixel_losses_sample
    valid_dict[n_valid_iter] = valid_iter_dict
    with open('valid_dict.p', 'wb') as handle:
        pickle.dump(valid_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)