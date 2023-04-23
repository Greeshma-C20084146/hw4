from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)
# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=2, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        # convert from uint8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299,299,3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
        print(mean(scores))
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

from PIL import Image
import numpy as np
import os

# Set the directory path
dir_path = "/content/generated_images"

# Initialize an empty list to store the images
images = []

# Loop through the files in the directory
for filename in os.listdir(dir_path):
    # Load the image and convert it to a NumPy array
    img = Image.open(os.path.join(dir_path, filename))
    img_array = np.array(img)
    # Append the array to the list of images
    images.append(img_array)

# Convert the list of images to a NumPy array
images_array = np.array(images * 255)
shuffle(images_array)

print('loaded', images_array.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images_array)
print('score', is_avg, is_std)

