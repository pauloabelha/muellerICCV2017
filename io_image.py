from scipy import misc


def change_res_image(image, new_res):
    image = misc.imresize(image, new_res)
    return image


def _read_RGB_image(image_filepath, new_res=None):
    image = misc.imread(image_filepath)
    image = image.swapaxes(0, 1)
    if new_res:
        image = change_res_image(image, new_res)
    return image