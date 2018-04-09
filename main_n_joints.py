import sys
import HALNet2_torch
import optimizers as my_optimizers
import torch
from torch.autograd import Variable
import io_data
import torch.nn.functional as F
import numpy as np
import resnet
import time
from magic import display_est_time_loop
import visualize
import losses as my_losses
import debugger

f = open('output.txt', 'w')
sys.stdout = f

LOAD_MODEL_FILENAME = 'checkpoint_model_log.pth.tar'
LOAD_RESNET = False
DEBUGGING_VISUALLY = False
USE_CUDA = True

# max batch size that GPU can handle
MAX_MEM_BATCH_SIZE = 8
# actual batch size wanted
BATCH_SIZE = 16
# for adadelta optimizer

NUM_ITER_TO_TRAIN = 100000

VERBOSE = True

LOG_INTERVAL = 10
LOG_FOR_VALID_INTERVAL = 1000
NUM_EPOCHS = 100

JOINT_IXS = list(range(21))


if LOAD_MODEL_FILENAME == '':
    START_EPOCH = 0
    START_ITER = 1
    losses = []
    dist_losses = []
    dist_losses_sample = []
    best_loss = 1e10
    best_dist_loss = 1e10
    best_dist_loss_sample = 1e10
    best_model_dict = {}

    if VERBOSE:
        print("Building HALNet network...")
    halnet = HALNet2_torch.HALNet(joint_ixs=JOINT_IXS, use_cuda=USE_CUDA)
    if VERBOSE:
        print("Done building HALNet network")
    if not USE_CUDA:
        visualize.save_graph_pytorch_model(halnet,
                                           model_input_shape=(1, 4, 320, 240),
                                           folder='', modelname='halnet')

    if LOAD_RESNET:
        print("Loading RESNet50...")
        resnet50 = resnet.resnet50(pretrained=True)
        #visualize.save_graph_pytorch_model(resnet50,
         #                                  model_input_shape=(1, 3, 227, 227),
          #                                 folder='', modelname='resnet50')
        print("Done loading RESNet50")

        # initialize HALNet with RESNet50
        print("Initializaing HALNet with RESNet50...")
        # initialize level 1
        # initialize conv1
        resnet_weight = resnet50.conv1.weight.data
        float_tensor = np.random.normal(np.mean(resnet_weight.numpy()),
                                        np.std(resnet_weight.numpy()),
                                        (resnet_weight.shape[0],
                                         1, resnet_weight.shape[2],
                                         resnet_weight.shape[2]))
        resnet_weight_numpy = resnet_weight.numpy()
        resnet_weight = np.concatenate((resnet_weight_numpy, float_tensor), axis=1)
        resnet_weight = torch.FloatTensor(resnet_weight)
        halnet.conv1[0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize level 2
        # initialize res2a
        resnet_weight = resnet50.layer1[0].conv1.weight.data
        halnet.res2a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[0].conv2.weight.data
        halnet.res2a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[0].conv3.weight.data
        halnet.res2a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[0].downsample[0].weight.data
        halnet.res2a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize res2b
        resnet_weight = resnet50.layer1[1].conv1.weight.data
        halnet.res2b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[1].conv2.weight.data
        halnet.res2b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[1].conv3.weight.data
        halnet.res2b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize res2c
        resnet_weight = resnet50.layer1[2].conv1.weight.data
        halnet.res2c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[2].conv2.weight.data
        halnet.res2c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer1[2].conv3.weight.data
        halnet.res2c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize res3a
        resnet_weight = resnet50.layer2[0].conv1.weight.data
        halnet.res3a.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[0].conv2.weight.data
        halnet.res3a.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[0].conv3.weight.data
        halnet.res3a.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[0].downsample[0].weight.data
        halnet.res3a.left_res[0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize res3b
        resnet_weight = resnet50.layer2[1].conv1.weight.data
        halnet.res3b.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[1].conv2.weight.data
        halnet.res3b.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[1].conv3.weight.data
        halnet.res3b.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        # initialize res3c
        resnet_weight = resnet50.layer2[2].conv1.weight.data
        halnet.res3c.right_res[0][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[2].conv2.weight.data
        halnet.res3c.right_res[2][0]._parameters['weight'].data.copy_(resnet_weight)
        resnet_weight = resnet50.layer2[2].conv3.weight.data
        halnet.res3c.right_res[4][0]._parameters['weight'].data.copy_(resnet_weight)
        print("Done initializaing HALNet with RESNet50")
        del resnet50

    optimizer = my_optimizers.get_adadelta_halnet(halnet)
else:
    print("Loading model and optimizer from file: " + LOAD_MODEL_FILENAME)
    halnet, optimizer, train_dict = io_data.load_checkpoint(filename=LOAD_MODEL_FILENAME,
                                                            model_class=HALNet2_torch.HALNet)
    START_EPOCH = train_dict['epoch']
    START_ITER = train_dict['curr_iter'] + 2
    losses = train_dict['losses']
    dist_losses = train_dict['dist_losses']
    dist_losses_sample = train_dict['dist_losses_sample']
    best_loss = train_dict['best_loss']
    best_dist_loss = train_dict['best_dist_loss']
    # old model trained dicts don't have best_dist_loss_sample
    try:
        best_dist_loss_sample = train_dict['best_dist_loss_sample']
    except:
        best_dist_loss_sample = 1e10
    best_model_dict = train_dict

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("\tSaving a checkpoint...")
    torch.save(state, filename)

def validate(model, valid_loader):
    # switch to evaluate mode
    model.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, target)

def pixel_stdev(norm_heatmap):
    num_pixels = norm_heatmap.size
    mean_norm_heatmap = np.mean(norm_heatmap)
    stdev_norm_heatmap = np.std(norm_heatmap)
    lower_bound = mean_norm_heatmap - stdev_norm_heatmap
    upper_bound = mean_norm_heatmap + stdev_norm_heatmap
    pixel_count_lower = np.where(norm_heatmap >= lower_bound)
    pixel_count_upper = np.where(norm_heatmap <= upper_bound)
    pixel_count_mask = pixel_count_lower and pixel_count_upper
    return np.sqrt(norm_heatmap[pixel_count_mask].size)

def print_target_info(target):
    if len(target.shape) == 4:
        target = target[0, :, :, :]
    target = io_data.convert_torch_dataoutput_to_canonical(target.data.numpy()[0])
    norm_target = io_data.normalize_output(target)
    # get joint inference from max of heatmap
    max_heatmap = np.unravel_index(np.argmax(norm_target, axis=None), norm_target.shape)
    print("Heamap max: " + str(max_heatmap))
    # data_image = visualize.add_squares_for_joint_in_color_space(data_image, max_heatmap, color=[0, 50, 0])
    # sample from heatmap
    heatmap_sample_flat_ix = np.random.choice(range(len(norm_target.flatten())), 1, p=norm_target.flatten())
    heatmap_sample_uv = np.unravel_index(heatmap_sample_flat_ix, norm_target.shape)
    heatmap_mean = np.mean(norm_target)
    heatmap_stdev = np.std(norm_target)
    print("Heatmap mean: " + str(heatmap_mean))
    print("Heatmap stdev: " + str(heatmap_stdev))
    print("Heatmap pixel standard deviation: " + str(pixel_stdev(norm_target)))
    heatmap_sample_uv = (int(heatmap_sample_uv[0]), int(heatmap_sample_uv[1]))
    print("Heatmap sample: " + str(heatmap_sample_uv))

def train(START_ITER_MOD, NUM_ITER_TO_TRAIN, model, optimizer, train_loader, epoch, total_loss, total_dist_losses,
          total_dist_losses_sample, curr_iter, losses, dist_losses, dist_loss_sample, best_loss, best_dist_loss,
          best_dist_loss_sample, best_model_dict, tot_toc):
    tot_iter = int(len(train_loader) / int(BATCH_SIZE/MAX_MEM_BATCH_SIZE))
    curr_epoch_iter = 1
    done = False
    iter_size = int(BATCH_SIZE/MAX_MEM_BATCH_SIZE)
    num_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < int(BATCH_SIZE / MAX_MEM_BATCH_SIZE):
            print("\rPerforming first iteration; current mini-batch: " +
                  str(batch_idx+1) + "/" + str(int(BATCH_SIZE / MAX_MEM_BATCH_SIZE)), end='')
        # check if arrived at iter to start
        if curr_iter < START_ITER_MOD:
            if batch_idx % int(BATCH_SIZE / MAX_MEM_BATCH_SIZE) == 0:
                print("\rGoing through iterations to arrive at last one saved... " +
                      str(int(curr_iter*100.0/START_ITER_MOD)) + "% of " +
                      str(START_ITER_MOD) + " iterations (" +
                      str(curr_iter) + "/" + str(START_ITER_MOD) + ")", end='')
                curr_iter += 1
                curr_epoch_iter += 1
            continue
        # save checkpoint after final iteration
        if curr_iter == NUM_ITER_TO_TRAIN:
            print("\nReached final number of iterations: " + str(NUM_ITER_TO_TRAIN))
            print("\tSaving final checkpoint...")
            save_checkpoint(best_model_dict,
                            filename='final_model_iter_' + str(NUM_ITER_TO_TRAIN) + '.pth.tar')
            done = True
            break
        # start time counter
        start = time.time()
        # get data and targetas cuda variables
        data, target = Variable(data).cuda(), Variable(target).cuda()
        # visualize if debugging
        # get model output
        output = model(data.cuda())
        if DEBUGGING_VISUALLY and curr_iter % 10 == 0:
            debugger.show_target_and_output_to_image_info(data, target, output)
        # accumulate loss for sub-mini-batch
        loss = my_losses.calculate_loss_main(output, target, iter_size)
        loss.backward()
        total_loss += loss
        # accumulate pixel dist loss for sub-mini-batch
        total_dist_losses = my_losses.accumulate_pixel_dist_loss_multiple(
            total_dist_losses, output, target, BATCH_SIZE)
        total_dist_losses_sample = my_losses.accumulate_pixel_dist_loss_from_sample_multiple(
            total_dist_losses_sample, output, target, BATCH_SIZE)
        # get boolean variable stating whether a mini-batch has been completed
        minibatch_completed = (batch_idx+1) % int(BATCH_SIZE / MAX_MEM_BATCH_SIZE) == 0
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
            dist_losses.append(total_dist_losses)
            # erase pixel dist loss
            total_dist_losses = [0] * len(model.joint_ixs)
            # append dist loss of sample from output
            dist_losses_sample.append(total_dist_losses_sample)
            # erase dist loss of sample from output
            total_dist_losses_sample = [0] * len(model.joint_ixs)
            # check if dist loss is better
            '''
            if dist_losses[-1] < best_dist_loss:
                best_dist_loss = dist_losses[-1]
                print("  This is a best pixel dist loss found so far: " + str(dist_losses[-1]))
            if dist_losses_sample[-1] < best_dist_loss_sample:
                best_dist_loss_sample = dist_losses_sample[-1]
                print("  This is a best pixel dist loss (from sample) found so far: " + str(dist_losses_sample[-1]))
                '''
            # check if loss is better
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                print("  This is a best loss found so far: " + str(losses[-1]))
                best_model_dict = {
                    'batch_size': BATCH_SIZE,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'dist_losses': dist_losses,
                    'dist_losses_sample': dist_losses_sample,
                    'best_loss': best_loss,
                    'best_dist_loss': best_dist_loss,
                    'best_dist_loss_sample': best_dist_loss_sample,
                    'epoch': epoch,
                    'curr_epoch_iter': curr_epoch_iter,
                    'curr_iter': curr_iter,
                    'batch_idx': batch_idx,
                    'joint_ixs': model.joint_ixs
                }
            # log checkpoint
            if curr_iter % LOG_INTERVAL == 0:
                print("")
                print("-------------------------------------------------------------------------------------------")
                print("Saving checkpoints:")
                print("-------------------------------------------------------------------------------------------")
                save_checkpoint(best_model_dict, filename='best_model_log.pth.tar')
                checkpoint_model_dict = {
                    'batch_size': BATCH_SIZE,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'dist_losses': dist_losses,
                    'dist_losses_sample': dist_losses_sample,
                    'best_loss': best_loss,
                    'best_dist_loss': best_dist_loss,
                    'best_dist_loss_sample': best_dist_loss_sample,
                    'epoch': epoch,
                    'curr_epoch_iter': curr_epoch_iter,
                    'curr_iter': curr_iter,
                    'batch_idx': batch_idx,
                    'joint_ixs': model.joint_ixs
                }
                save_checkpoint(checkpoint_model_dict, filename='checkpoint_model_log.pth.tar')
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
                          str(np.mean(np.array(dist_losses)[-LOG_INTERVAL:, joint_ix])))
                    print("\tTraining set stddev error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss): " +
                          str(np.std(np.array(dist_losses)[-LOG_INTERVAL:, joint_ix])))
                    print("\tThis is the last pixel dist loss: " + str(dist_losses[-1][joint_ix]))
                    print("\tTraining set mean error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(dist_losses_sample)[-LOG_INTERVAL:, joint_ix])))
                    print("\tTraining set stddev error for last " + str(LOG_INTERVAL) +
                          " iterations (average pixel loss of sample): " +
                          str(np.mean(np.array(dist_losses_sample)[-LOG_INTERVAL:, joint_ix])))
                    print("\tThis is the last pixel dist loss of sample: " + str(dist_losses_sample[-1][joint_ix]))
                    print("\t-------------------------------------------------------------------------------------------")
                print("-------------------------------------------------------------------------------------------")
            if curr_iter % LOG_FOR_VALID_INTERVAL == 0:
                print("\nSaving model and checkpoint model for validation")
                checkpoint_model_dict = {
                    'batch_size': BATCH_SIZE,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                    'dist_losses': dist_losses,
                    'dist_losses_sample': dist_losses_sample,
                    'best_loss': best_loss,
                    'best_dist_loss': best_dist_loss,
                    'best_dist_loss_sample': best_dist_loss_sample,
                    'epoch': epoch,
                    'curr_epoch_iter': curr_epoch_iter,
                    'curr_iter': curr_iter,
                    'batch_idx': batch_idx,
                    'joint_ixs': model.joint_ixs
                }
                save_checkpoint(checkpoint_model_dict, filename='checkpoint_model_log_for_valid_'
                                                                + str(curr_iter) + '.pth.tar')
            # print time lapse
            tot_toc = display_est_time_loop(tot_toc + time.time() - start,
                                                curr_iter, NUM_ITER_TO_TRAIN,
                                                prefix='Training '
                                                       '(Epoch #' + str(epoch) +
                                                       ' ' + str(curr_epoch_iter) + '/' + str(tot_iter) + ')' +
                                                       ', (Batch ' + str(batch_idx) + '/' + str(num_batches) + ')' +
                                                       ', (Iter #' + str(curr_iter) +
                                                       ' - log every ' + str(LOG_INTERVAL) + ' iter): ')
            curr_iter += 1
            curr_epoch_iter += 1
    return curr_iter, done, losses, dist_losses, dist_loss_sample,\
           best_loss, best_dist_loss, best_dist_loss_sample, best_model_dict, tot_toc

torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_loader = io_data.get_HALNet_trainloader(joint_ixs=halnet.joint_ixs,
                                              batch_size=MAX_MEM_BATCH_SIZE,
                                              verbose=VERBOSE)

tot_iter = int(len(train_loader) / int(BATCH_SIZE/MAX_MEM_BATCH_SIZE))
START_ITER_MOD = START_ITER % tot_iter
curr_iter = 1
start_batch_idx = START_ITER * int(BATCH_SIZE/MAX_MEM_BATCH_SIZE)

halnet.train()
tot_toc = 0
print("-----------------------------------------------------------")
print("Model info")
print("Number of joints: " + str(len(halnet.joint_ixs)))
print("Joints indexes: " + str(halnet.joint_ixs))
print("-----------------------------------------------------------")
print("Max memory batch size: " + str(MAX_MEM_BATCH_SIZE))
print("Length of dataset (in max mem batch size): " + str(len(train_loader)))
print("Starting batch idx: " + str(start_batch_idx))
print("-----------------------------------------------------------")
print("Training batch size: " + str(BATCH_SIZE))
n_iter_per_epoch = int(len(train_loader) / int(BATCH_SIZE/MAX_MEM_BATCH_SIZE))
print("Starting epoch: " + str(START_EPOCH))
print("Starting epoch iteration: " + str(START_ITER_MOD))
print("Starting overall iteration: " + str(START_ITER))
print("-----------------------------------------------------------")
print("Number of iterations per epoch: " + str(n_iter_per_epoch))
print("Number of iterations to train: " + str(NUM_ITER_TO_TRAIN))
print("Approximate number of epochs to train: " + str(round(NUM_ITER_TO_TRAIN/n_iter_per_epoch, 1)))
print("-----------------------------------------------------------")

for epoch in range(NUM_EPOCHS):
    if epoch + 1 < START_EPOCH:
        curr_iter += n_iter_per_epoch
        continue
    total_loss = 0
    total_dist_losses = [0] * len(halnet.joint_ixs)
    total_dist_losses_sample = [0] * len(halnet.joint_ixs)
    optimizer.zero_grad()
    curr_iter, done, losses, dist_losses, dist_losses_sample, best_loss, best_dist_loss, best_dist_loss_sample,\
    best_model_dict, tot_toc =\
        train(START_ITER_MOD, NUM_ITER_TO_TRAIN, halnet, optimizer, train_loader, epoch + 1, total_loss,
              total_dist_losses, total_dist_losses_sample, curr_iter, losses, dist_losses, dist_losses_sample,
              best_loss, best_dist_loss, best_dist_loss_sample, best_model_dict, tot_toc)
    if done:
        break
    if epoch + 1 >= START_EPOCH:
        for param_group in optimizer.param_groups:
           param_group['lr'] = param_group['lr']*0.5
