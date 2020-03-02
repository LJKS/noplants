from os import listdir
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
THE_GOOD = [0,1,1] #cyan
THE_BAD = [1,0,1] #magenta
THE_UGLY = None #everything else
subpics = 1
crop_size = (256, 256, 3)
origin_target_directory = 'stem_lbl_human'
origin_data_directory = 'stem_data'
assert listdir(origin_data_directory) == listdir(origin_target_directory)

for img_name in listdir(origin_data_directory):
    target_source_path = origin_target_directory + '/' + img_name
    data_source_path = origin_data_directory + '/' + img_name
    img_target_human = image.imread(target_source_path)
    good = np.all(img_target_human == THE_GOOD, -1).astype(float)
    bad = np.all(img_target_human == THE_BAD, -1).astype(float)
    ugly = np.ones(bad.shape) - good - bad
    img_target = np.stack((good, bad, ugly), -1)*255
    img_target = img_target.astype(np.uint8)
    img_data = image.imread(data_source_path)
    img_data = img_data*255
    img_data = img_data.astype(np.uint8)
    img_target = tf.convert_to_tensor(img_target)
    img_data = tf.convert_to_tensor(img_data)

    for i in range(subpics):
        save_path_target = 'stem_lbl_cropped_container/stem_lbl_cropped/' + img_name[0:-4] + '_crop_' + str(i) + '.png'
        save_path_data = 'stem_data_cropped_container/stem_data_cropped/' + img_name[0:-4] + '_crop_' + str(i) + '.png'
        rand_seed = np.random.rand()
        tf.random.set_seed(rand_seed)
        sub_img_target = tf.image.random_crop(img_target, crop_size)
        tf.random.set_seed(rand_seed)
        sub_img_data = tf.image.random_crop(img_data, crop_size)
        # convert back from tensor to np array
        sub_img_target = sub_img_target.numpy()
        sub_img_data = sub_img_data.numpy()
        save_img_target = Image.fromarray(sub_img_target).convert('RGB')
        save_img_target.save(save_path_target)
        save_img_data = Image.fromarray(sub_img_data).convert('RGB')
        save_img_data.save(save_path_data)
        plt.imshow(sub_img_data)
        plt.show()
        plt.imshow(sub_img_target)
        plt.show()

"""
for img_str in listdir('lbl'):
    img_str_old = 'lbl/'+img_str
    img = image.imread(img_str_old)
    good = np.all(img == THE_GOOD, -1).astype(float)
    bad = np.all(img == THE_BAD, -1).astype(float)
    ugly = np.ones(bad.shape) - good - bad
    img_new = np.stack((good, bad, ugly), -1)*255
    img_new = img_new.astype(np.uint8)
    img_string_new = 'target_container/targets/' + img_str
    img_new = Image.fromarray(img_new).convert('RGB')
    img_new.save(img_string_new)
"""
