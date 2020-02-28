import proto
from os import listdir
import PIL
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
rand_crop = True
import sys
BATCH_SIZE = 4

if 'rand_crop' in sys.argv:
    rand_crop = bool(sys.argv[sys.argv.index('rand_crop')+1])
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

num_tests =10
model = proto.ProtoDense()
model.load_weights("models/proto_dense_1/model2020-02-27_15:05:53.878598")
origin_target_directory = 'stem_lbl_human'
origin_data_directory = 'stem_data'
pic_names = listdir(origin_data_directory)

for i in range(num_tests):
    rand_seed = np.random.randint(0,100000)
    rand_img_name = np.random.choice(pic_names)
    data_file = origin_data_directory + '/' + rand_img_name
    target_file = origin_target_directory + '/' + rand_img_name
    data_img = np.asarray(PIL.Image.open(data_file)).astype(np.float32)/255
    print(np.max(data_img))
    print(np.min(data_img))
    target_img = np.asarray(PIL.Image.open(target_file))
    data_img=tf.image.random_crop(data_img, (256,256,3), seed=rand_seed)
    target_img=tf.image.random_crop(target_img, (256,256,3), seed=rand_seed)
    data_img = np.expand_dims(data_img, 0)
    print(data_img.shape)
    t_1 = time.time()
    out_img = model(data_img)
    t_2 = time.time()
    print('Forward pass took: ' + str(t_2-t_1))
    out_img = np.squeeze(out_img)
    plt.subplot(221)
    plt.imshow(target_img)
    plt.title('target')
    plt.subplot(222)
    pos = plt.imshow(np.squeeze(out_img[:,:,0]), cmap='viridis')
    plt.colorbar(pos, cmap='viridis')
    plt.title('carrot')
    plt.subplot(223)
    pos = plt.imshow(np.squeeze(out_img[:,:,1]), cmap='plasma')
    plt.colorbar(pos, cmap='plasma')
    plt.title('bad plant')
    plt.subplot((224))
    pos = plt.imshow(np.squeeze(out_img[:,:,2]), cmap='inferno')
    plt.colorbar(pos, cmap='inferno')
    plt.title('ground')
    plt.show()
