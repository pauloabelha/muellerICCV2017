import synthhands_handler
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pickle
from matplotlib import pyplot as plt

root_folder = '/home/paulo/rds_muri/paulo/synthhands/SynthHands_Release/'

load = True

if load:
    joint_prior = pickle.load(open("joint_prior.p", "rb"))
    plt.imshow(joint_prior['pair_dist_prob'].astype(int), cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(0, 210, 10.0))
    plt.xticks(np.arange(0, 300, 10.0))
    plt.show()



full_loader = synthhands_handler.get_SynthHands_fullloader(root_folder=root_folder,
                                                           joint_ixs=range(21),
                                                           heatmap_res=(640, 480),
                                                           batch_size=1,
                                                           verbose=True)

min_dist = 1e10
max_dist = -1
len_dataset = len(full_loader)
max_dist_span = 300  # in mm

pair_dist_prob = np.zeros((210, max_dist_span))
for batch_idx, (data, target) in enumerate(full_loader):
    filenamebase = full_loader.dataset.get_filenamebase(batch_idx)
    _, target_joints, target_roothand = target
    target_joints, target_roothand = target_joints.data.numpy(), target_roothand.data.numpy()
    D = squareform(pdist(target_joints.reshape((21, 3))))
    ix_pair = 0
    for i in range(D.shape[0]):
        j = i + 1
        while j < D.shape[1]:
            # print('(' + str(i) + ', ' + str(j) + '): ' + str(D[i, j]))
            dist = D[i, j]
            pair_dist_prob[ix_pair, int(dist)] += 1
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
            j += 1
            ix_pair += 1
    print(filenamebase + ': ' + str(batch_idx) + '/' + str(len_dataset))
    print(filenamebase + ': ' + 'Max dist: ' + str(max_dist))
    print(filenamebase + ': ' + 'Min dist: ' + str(min_dist))
    if batch_idx % 500 == 0:
        print("Saving checkpoint...")
        data = {
            'min_dist': min_dist,
            'max_dist': max_dist,
            'pair_dist_prob': pair_dist_prob,
            'batch_idx': batch_idx
        }
        with open('joint_prior.p', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Saving final checkpoint...")
data = {
    'min_dist': min_dist,
    'max_dist': max_dist,
    'pair_dist_prob': pair_dist_prob,
    'batch_idx': batch_idx
}
with open('joint_prior.p', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

