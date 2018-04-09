import io_data
import visualize

NUM_EXAMPLES = 5

halnet_train_set = io_data.SynthHandsHALNetTrainDataset()
halnet_valid_set = io_data.SynthHandsHALNetValidDataset()
halnet_test_set = io_data.SynthHandsHALNetTestDataset()

print("Checking " + str(NUM_EXAMPLES) + " examples from the training set")
for i in range(NUM_EXAMPLES):
    example_data, example_label = halnet_train_set[i]
    visualize.show_dataset_example_with_joints(halnet_train_set, i)

print("Checking " + str(NUM_EXAMPLES) + " examples from the validation set")
for i in range(NUM_EXAMPLES):
    example_data, example_label = halnet_valid_set[i]
    visualize.show_dataset_example_with_joints(halnet_valid_set, i)

print("Checking " + str(NUM_EXAMPLES) + " examples from the test set")
for i in range(NUM_EXAMPLES):
    example_data, example_label = halnet_test_set[i]
    visualize.show_dataset_example_with_joints(halnet_test_set, i)





