import synthhands_handler
import visualize
import argparse

NUM_EXAMPLES = 5

parser = argparse.ArgumentParser(description='Train a hand-tracking deep neural network')
parser.add_argument('-n', dest='num_examples', type=int, default=5,
                    help='Total number of examples to check')
parser.add_argument('-r', dest='root_folder', default='', required=True, help='Root folder for dataset')
args = parser.parse_args()

train_loader = synthhands_handler.get_SynthHands_trainloader(root_folder=args.root_folder,
                                                             joint_ixs=range(21),
                                                             heatmap_res=(320, 240),
                                                             batch_size=1,
                                                             verbose=True)

print("Checking " + str(NUM_EXAMPLES) + " examples from the training set")
for batch_idx, (data, target) in enumerate(train_loader):
    target_heatmaps, target_joints, target_roothand = target
    visualize.plot_image_and_heatmap(target_heatmaps[0][4].cpu().data.numpy(),
                                     data=data[0].cpu().data.numpy(),
                                     title='Training set\n' + train_loader.dataset.get_filenamebase(batch_idx) + '\nImage + Heatmap(thumb tip)')
    visualize.show()
    if (batch_idx + 1) == args.num_examples:
        break





