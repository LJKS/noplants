import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import  Model
import numpy as np
from datapipeline import Datapipeline
from datetime import datetime

class ProtoDense(Model):
    def __init__(self):
        super(ProtoDense, self).__init__()
        self.block_1 = DenseBlock(12, 3, (5,5), (1,1))
        self.block_2 = DenseBlock(12, 3, (5,5), (1,1))
        self.read_outs = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='SAME', use_bias=False)
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, x):
        print('model_call')
        x = self.block_1(x)
        x = tf.nn.max_pool(x, (2,2), (1,1), 'SAME')
        x = self.block_2(x)
        x = self.read_outs(x)
        x = tf.nn.softmax(x,-1)
        return x

class DenseBlock(Layer):
    def __init__(self, layers, filters, kernel_size, strides):
        super(DenseBlock, self).__init__()
        self.layers = [DenseLayer(filters, kernel_size, strides) for _ in range(layers)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseLayer(Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DenseLayer, self).__init__()
        self.convolutions = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        new_feature_maps = self.convolutions(x)
        feature_maps = tf.concat([new_feature_maps, x], -1)
        return feature_maps



def train(model, pipeline, iters, model_dir):
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    train_generator = pipeline.get_generator()
    optimizer = tf.keras.optimizers.Adam()
    losses = []
    print('starting training')
    for epoch in range(iters):
        for input, target in train_generator:
            print('before tape')
            print('input shape', tf.shape(input))
            print('input_max', tf.reduce_max(input), 'out_max', tf.reduce_max(target))
            with tf.GradientTape() as tape:
                predictions = model(input, training=True)
                print('made predictions')
                loss = cce(target, predictions, compute_update_weights(target))
                #print('loss_shape', loss)
                #print('got loss')
                gradients = tape.gradient(loss, model.trainable_variables)
                #print('got gradients')
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                #print('applied optimizer')
            losses.append(np.mean(loss))
            print(losses, "losses")
            model.save_weights(model_dir+'model'+str(datetime.now()).replace(' ', '_'))

def compute_update_weights(target_batch):
    batch_size = target_batch.shape[0]
    img_size = target_batch.shape[1]*target_batch.shape[2]
    num_good = np.sum(target_batch[:,:,:,0], (1,2)) #should have shape [batch_size]
    num_bad =  np.sum(target_batch[:,:,:,1], (1,2)) #shoud have shape [batch_size]
    num_ugly = np.sum(target_batch[:,:,:,2], (1,2)) #should have shape [batch_size]
    weights_good = np.reshape(num_good*img_size/3, (batch_size, 1,1))
    weights_bad = np.reshape(num_bad * img_size/3, (batch_size,1,1))
    weights_ugly = np.reshape(num_ugly*img_size/3, (batch_size,1,1))
    weights = np.stack([weights_good, weights_bad, weights_ugly], axis=-1)
    weights_mapped = weights*target_batch
    weights_total = np.sum(weights_mapped, axis=-1)
    print(weights_total.shape, 'weights shape')
    return weights_total


if __name__ == "__main__":
    #load_data
    model = ProtoDense()
    pipeline = Datapipeline('stem_data_cropped_container', 'stem_lbl_cropped_container')
    train(model, pipeline, 3, 'models/proto_dense_1/')
