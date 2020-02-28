import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

class Datapipeline:
    def __init__(self, data_dir, target_dir):
        self.data_dir = data_dir
        self.target_dir = target_dir
        #TODO: add to data augmentation, e.g. noise, flip
        self.data_gen_args = dict(rescale=1./255, rotation_range=90)
        self.image_datagen = ImageDataGenerator(**self.data_gen_args)
        self.target_datagen = ImageDataGenerator(**self.data_gen_args)
        # Provide the same seed and keyword arguments to the fit and flow methods
        self.seed = np.random.randint(0,10000)
        #image_datagen.fit(images, augment=True, seed=seed)
        #target_datagen.fit(masks, augment=True, seed=seed)

    def get_generator(self):
        image_generator = self.image_datagen.flow_from_directory(self.data_dir, class_mode=None, seed=self.seed, batch_size=3)
        target_generator = self.target_datagen.flow_from_directory(self.target_dir, class_mode=None, seed=self.seed, batch_size=3)
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, target_generator)
        self.seed = np.random.randint(0,10000)
        return train_generator
