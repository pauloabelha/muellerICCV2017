import io_data
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Ignoring matplotlib import error")
    pass

image_filepath = "/home/paulo/SynthHands_Release/male_object/seq06/cam04/01/00000311_color_on_depth.png"
image_filepath = "/home/paulo/beach.jpg"

image_scipy = io_data._read_RGB_image(image_filepath, new_res=(240, 320), read_image_func=io_data._read_RGB_image_scipy)

image_opencv = io_data._read_RGB_image(image_filepath, new_res=(240, 320), read_image_func=io_data._read_RGB_image_opencv)

print(np.sum(np.abs(image_scipy) - np.abs(image_opencv)))
print(image_scipy.shape)

plt.imshow(image_scipy)
plt.title("Image SciPy")
plt.show()
plt.imshow(image_opencv)
plt.title("Image OpenCV")
plt.show()




