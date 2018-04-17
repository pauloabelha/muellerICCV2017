import math
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_vars):
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