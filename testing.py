
root_folder = '/home/paulo/'
filepath_prior = root_folder + 'output_HALNet_prior_1804790995.txt'
filepath = root_folder + 'output_HALNet_1345115866.txt'
import matplotlib.pyplot as plt

mean_loss_prefix = 'Mean loss for last 10 iterations (average total loss): '
prior_loss_prefix = 'Mean loss (prior) for last 10 iterations (average total loss): '

mean_losses_hm = []
with open(filepath) as fp:
    for line in fp:
        if mean_loss_prefix in line:
            mean_loss_str = float(line[len(mean_loss_prefix):])
            mean_losses_hm.append(mean_loss_str / 21.0)

mean_losses_hm_prior = []
mean_losses_joint_dist_prior = []
with open(filepath_prior) as fp:
    for line in fp:
        if mean_loss_prefix in line:
            mean_loss = float(line[len(mean_loss_prefix):])
        if prior_loss_prefix in line:
            prior_loss = float(line[len(prior_loss_prefix):])
            heatmap_loss = mean_loss - prior_loss
            mean_losses_hm_prior.append(heatmap_loss / 21.0)
            mean_losses_joint_dist_prior.append(prior_loss / 210.0)

min_len = min(len(mean_losses_hm_prior), len(mean_losses_hm))

plt.plot(mean_losses_hm[:min_len])
plt.plot(mean_losses_hm_prior[:min_len])
plt.ylabel('Mean losses (heatmaps)')
plt.show()

plt.plot(mean_losses_joint_dist_prior)
plt.ylabel('Mean losses (joint pair dist)')
plt.show()