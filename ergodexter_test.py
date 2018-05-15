import egodexter_handler


ROOT_FOLDER = '/home/paulo/rds_muri/paulo/EgoDexter/'
BATCH_SIZE = 1

train_loader = egodexter_handler.get_loader(type='train',
                                            root_folder=ROOT_FOLDER,
                                            img_res=(640, 480),
                                            batch_size=BATCH_SIZE,
                                            verbose=True)