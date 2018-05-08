import synthhands_handler
import visualize

NUM_EXAMPLES = 5

halnet_train_set = synthhands_handler.SynthHandsHALNetTrainDataset()
halnet_valid_set = synthhands_handler.SynthHandsHALNetValidDataset()
halnet_test_set = synthhands_handler.SynthHandsHALNetTestDataset()

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





