import proto
from os import listdir
import PIL
import numpy as np
import matplotlib.pyplot as plt
import time
num_tests =10
model = proto.ProtoDense()
model.load_weights("models/proto_dense_2/model_0_46400")
origin_target_directory = 'stem_lbl_human'
origin_data_directory = 'stem_data'
pic_names = listdir(origin_data_directory)

for i in range(num_tests):
    rand_img_name = np.random.choice(pic_names)
    data_file = origin_data_directory + '/' + rand_img_name
    target_file = origin_target_directory + '/' + rand_img_name
    data_img = np.asarray(PIL.Image.open(data_file)).astype(np.float32)/255
    print(np.max(data_img))
    print(np.min(data_img))
    target_img = np.asarray(PIL.Image.open(target_file))
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
