import synthhands_handler

valid_loader = synthhands_handler.\
    get_SynthHands_boundbox_loader(root_folder='/home/paulo/SynthHands_Release/',
                                                             heatmap_res=(320, 240),
                                                             batch_size=2,
                                                             verbose=True,
                                                                 type='train')


