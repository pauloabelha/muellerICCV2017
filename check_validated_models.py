import pickle
import matplotlib.pyplot as plt
import numpy as np

valid_dict = pickle.load(open('valid_dict.p', "rb"))

sorted_losses = []
sorted_pixel_losses = []
sorted_pixel_losses_sample = []
for valid_dict_key in sorted(valid_dict.keys()):
    valid_dict_iter = valid_dict[valid_dict_key]
    print("Number of trained iterations: " + str(valid_dict_key))
    sorted_losses.append(np.mean(valid_dict_iter['losses']))
    sorted_pixel_losses.append(np.mean(valid_dict_iter['pixel_losses']))
    sorted_pixel_losses_sample.append(np.mean(valid_dict_iter['pixel_losses_sample']))

plt.plot(sorted_losses)
#plt.ylabel('Validation losses')
#plt.show()

plt.plot(sorted_pixel_losses)
#plt.ylabel('Validation pixel losses (max of prob dist)')
#plt.show()

plt.plot(sorted_pixel_losses_sample)
#plt.ylabel('Validation pixel losses (sample from prob dist)')
plt.show()


