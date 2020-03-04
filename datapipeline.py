import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

class Datapipeline:
    def __init__(self, data_dir, target_dir, batch_size):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        #TODO: add to data augmentation, e.g. noise, flip
        self.data_gen_args = dict(rescale=1./255)  #vertical_flip=True, horizontal_flip=True, zoom_range=[0.7,1.0])#rotation_range=90)
        self.image_datagen = ImageDataGenerator(**self.data_gen_args)#, brightness_range=[0.5,1.0])
        self.target_datagen = ImageDataGenerator(**self.data_gen_args)
        # Provide the same seed and keyword arguments to the fit and flow methods
        self.seed = np.random.randint(0,10000)
        #image_datagen.fit(images, augment=True, seed=seed)
        #target_datagen.fit(masks, augment=True, seed=seed)

    def get_generator(self):
        rand_seed = np.random.rand()
        tf.random.set_seed(rand_seed)
        image_generator = self.image_datagen.flow_from_directory(self.data_dir, class_mode=None, seed=self.seed, batch_size=self.batch_size)
        tf.random.set_seed(rand_seed)
        target_generator = self.target_datagen.flow_from_directory(self.target_dir, class_mode=None, seed=self.seed, batch_size=self.batch_size)
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, target_generator)
        self.seed = np.random.randint(0,10000)
        return train_generator
