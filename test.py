import sys
import io_data
import numpy as np


try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Ignoring matplotlib import error")
    pass

#image_filepath = "/home/paulo/SynthHands_Release/male_object/seq06/cam04/01/00000311_color_on_depth.png"
image_filepath = sys.argv[1]
#image_filepath = "/home/paulo/beach.jpg"

module_name = sys.argv[2]

image = io_data._read_RGB_image(image_filepath, new_res=(240, 320), module_name=module_name)

print(image.shape)

print(image)
plt.imshow(image)
plt.title("Image " + module_name)
plt.show()




