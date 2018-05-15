import dataset_handler
import egodexter_handler

dataset_handler.save_dataset_split('/home/paulo/rds_muri/paulo/EgoDexter/',
                                   egodexter_handler.DATASET_SPLIT_FILENAME,
                                   prefix_length=11,
                                   save_folder='')