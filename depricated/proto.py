import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import  Model
import numpy as np
from datapipeline import Datapipeline
import time
import agg
import sys
from matplotlib import pyplot as plt
from hyperparametrs import *


if 'batch_size' in sys.argv:
    BATCH_SIZE = int(sys.argv[sys.argv.index('batch_size')+1])
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
if 'clock' in sys.argv:
    CLOCK = True

print(BATCH_SIZE, 'BATCH_SIZE')

class ProtoDense(Model):
    def __init__(self):
        super(ProtoDense, self).__init__()
        self.block_1 = DenseBlock(6, 3, (5,5), (1,1))
        self.block_2 = DenseBlock(12, 6, (5,5), (1,1), bottleneck=24)
        self.block_3_s = DenseBlock(12,9,(5,5),(1,1),bottleneck=36)
        self.block_3_b = DenseBlock(12,6, (5,5), (1,1), bottleneck=24)
        self.block_4 = DenseBlock(12,6,(5,5),(1,1), bottleneck=24)
        self.read_outs = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='SAME', use_bias=False)
        self.optimizer = tf.keras.optimizers.Adam()




    @tf.function

    def call(self, x):
        x = self.block_1(x)
        x = tf.nn.avg_pool(x, (2,2), (2,2), padding='SAME')
        x = self.block_2(x)
        x_3_s = tf.nn.avg_pool(x, (2,2), (2,2), padding='SAME')
        x_3_s = self.block_3_s(x_3_s)
        x_3_s = repeat_2D(x_3_s)
        x = self.block_3_b(x)
        x = tf.concat([x,x_3_s],-1)
        x = self.block_4(x)
        x = self.read_outs(x)
        x = tf.nn.softmax(x,-1)
        return x

class DenseBlock(Layer):
    def __init__(self, layers, filters, kernel_size, strides, bottleneck=False):
        super(DenseBlock, self).__init__()
        self.has_bottleneck = bottleneck!=False

        if bottleneck!=False:
            self.layers = [BNDenseLayer(filters, kernel_size, strides, bottleneck) for _ in range(layers)]
        else:
            self.layers = [DenseLayer(filters, kernel_size, strides) for _ in range(layers)]

    #@tf.function
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


class BNDenseLayer(Layer):
    def __init__(self, filters, kernel_size, strides, bottleneck_size):
        super(BNDenseLayer, self).__init__()
        self.bottleneck = tf.keras.layers.Conv2D(bottleneck_size, (1,1), (1,1), padding='SAME', activation=tf.nn.relu)
        self.convolutions = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        bottlenecked = self.bottleneck(x)
        new_feature_maps = self.convolutions(bottlenecked)
        feature_maps = tf.concat([new_feature_maps, x], -1)
        return feature_maps

def repeat_2D(tensor):
    tensor = tf.repeat(tensor, 2, 1)
    tensor = tf.repeat(tensor, 2, 2)
    return tensor

def train(model, pipeline, iters, model_dir):
    cce = tf.keras.losses.CategoricalCrossentropy()
    train_generator = pipeline.get_generator()
    optimizer = tf.keras.optimizers.Adam()
    print('starting training')
    aggregator = agg.Smoothing_aggregator(model_dir)
    clock = Clock() if CLOCK else None
    for epoch in range(iters):
        for step, intar in enumerate(train_generator):
            if CLOCK:
                print('total step')
                clock.clock()
            input, target = intar
            target = tf.convert_to_tensor(target)
            target = tf.nn.avg_pool(target, (2,2), (2,2), 'SAME')
            target = target.numpy()
            weights = compute_update_weights(target)
            loss = train_step(model, input, target, weights, optimizer, cce)
            if aggregator.update(loss):
                model.save_weights(model_dir + '/'+ '_step_' + str(step))
            if step%SAVE_STEPS==0:
                print(step)
            #print('if_clock')
            #if_c.clock()
@tf.function
def train_step(model, input, target, weights, optimizer, cce):
    with tf.GradientTape() as tape:
        predictions = model(input, training=True)
        loss = cce(target, predictions, weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def compute_update_weights(target_batch):
    epsilon = 0.0001
    target_batch = target_batch.astype(np.float64)
    batch_size = target_batch.shape[0]
    img_size = target_batch.shape[1]*target_batch.shape[2]
    num_good = np.sum(target_batch[:,:,:,0], (1,2))  + epsilon#should have shape [batch_size] +
    num_bad =  np.sum(target_batch[:,:,:,1], (1,2)) + epsilon#shoud have shape [batch_size]
    num_ugly = np.sum(target_batch[:,:,:,2], (1,2)) + epsilon#should have shape [batch_size]
    # weights = for each pixel state weight (small for ground, high for carrot)
    weights_good = np.reshape((1/num_good) *img_size/3, (batch_size, 1,1))
    weights_bad = np.reshape((1/num_bad) * img_size/3, (batch_size,1,1))
    weights_ugly = np.reshape((1/num_ugly)*img_size/3, (batch_size,1,1))
    # get (batch_size, 1,1,3)
    weights = np.stack([weights_good, weights_bad, weights_ugly], axis=-1)
    weights_mapped = weights*target_batch
    weights_total = np.sum(weights_mapped, axis=-1)
    weights_total = np.clip(weights_total, 1/target_batch.shape[-1], 100)
    print(weights_total.shape, "weights shape")
    return weights_total

def plot_progress(losses, epoch, step, model_dir):
    plt.plot(losses)
    plt.savefig(model_dir + '_loss_' + str(epoch) + '_' + str(step), format='png')

class Clock:
    def __init__(self):
        self.time = time.time()
    def clock(self):
        time_new = time.time()
        tdif = time_new-self.time
        self.time = time.time()
        print(tdif)

if __name__ == "__main__":
    #load_data
    model = ProtoDense()
    pipeline = Datapipeline( DATA_TRAIN, DATA_TRAIN_LBL, BATCH_SIZE)
    train(model, pipeline, EPOCHS, MODEL_SAVE_DIR)
