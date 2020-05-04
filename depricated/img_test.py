import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
# we create two instances with the same arguments
data_gen_args = dict(rescale=1./255, rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
target_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#image_datagen.fit(images, augment=True, seed=seed)
#target_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory('img_container', class_mode=None, seed=seed)
target_generator = target_datagen.flow_from_directory('lbl_sw_container', class_mode=None, seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, target_generator)
for img, tar in train_generator:
    #print(len(np.unique(np.reshape(tar, (-1,3)), axis=0)))
    plt.imshow(img[0,:,:,:], interpolation='nearest')
    plt.show()
    plt.imshow(tar[0,:,:,:], interpolation='nearest')
    plt.show()
