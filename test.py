import proto_stem_and_seg
from os import listdir
import PIL
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np

import sys
from hyperparametrs import *

rand_crop = True
if 'rand_crop' in sys.argv:
    rand_crop = bool(sys.argv[sys.argv.index('rand_crop')+1])
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


model = proto.ProtoDense()
model.load_weights(MODEL_TEST_DIR)
origin_target_directory = DATA_TEST_LBL
origin_data_directory =  DATA_TEST
pic_names = listdir(origin_data_directory)

for i in range(NUM_TESTS):
    rand_seed = np.random.randint(0,100000)
    rand_img_name = np.random.choice(pic_names)
    data_file = origin_data_directory + '/' + rand_img_name
    target_file = origin_target_directory + '/' + rand_img_name
    data_img = np.asarray(PIL.Image.open(data_file)).astype(np.float32)/255
    print(np.max(data_img))
    print(np.min(data_img))
    target_img = np.asarray(PIL.Image.open(target_file))
    #data_img=tf.image.random_crop(data_img, (256,256,3), seed=rand_seed)
    #target_img=tf.image.random_crop(target_img, (256,256,3), seed=rand_seed)
    data_img = np.expand_dims(data_img, 0)
    print(data_img.shape)
    t_1 = time.time()
    out_img_stem, out_img_seg = model(data_img)
    t_2 = time.time()
    print('Forward pass took: ' + str(t_2-t_1))
    out_img_stem = np.squeeze(out_img_stem)#
    out_img_seg = np.squeeze(out_img_seg)
    data_img = np.squeeze(data_img)

    plt.subplot(241)
    plt.imshow(data_img)
    plt.title('target')
    plt.subplot(242)
    pos = plt.imshow(np.squeeze(out_img_stem[:,:,0]), cmap='viridis')
    plt.colorbar(pos, cmap='viridis')
    plt.title('carrot')
    plt.subplot(243)
    pos = plt.imshow(np.squeeze(out_img_stem[:,:,1]), cmap='plasma')
    plt.colorbar(pos, cmap='plasma')
    plt.title('bad plant')
    plt.subplot((244))
    pos = plt.imshow(np.squeeze(out_img_stem[:,:,2]), cmap='inferno')
    plt.colorbar(pos, cmap='inferno')
    plt.title('ground')
    plt.subplot(245)
    plt.imshow(data_img)
    plt.title('target')
    plt.subplot(246)
    pos = plt.imshow(np.squeeze(out_img_seg[:,:,0]), cmap='viridis')
    plt.colorbar(pos, cmap='viridis')
    plt.title('carrot')
    plt.subplot(247)
    pos = plt.imshow(np.squeeze(out_img_seg[:,:,1]), cmap='plasma')
    plt.colorbar(pos, cmap='plasma')
    plt.title('bad plant')
    plt.subplot((248))
    pos = plt.imshow(np.squeeze(out_img_seg[:,:,2]), cmap='inferno')
    plt.colorbar(pos, cmap='inferno')
    plt.title('ground')
    plt.show()
