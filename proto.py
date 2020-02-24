import tensorflow as tf
from tf.keras.layers import Layer
from tf.keras import  Model

class DenseLayer(Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DenseLayer).__init__()
        self.convolutions = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        new_feature_maps = self.convolutions(x)
        feature_maps = tf.concat([new_feature_maps, x], -1)
        return feature_maps

class DenseBlock(Layer):
    def __init__(self, layers, filters, kernel_size, strides):
        super(DenseBlock, self).__init__()
        self.layers = [DenseLayer(filters, kernel_size, strides) for _ in range(layers)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ProtoDense(Model):
    def __init__(self):
        super(ProtoDense, self).__init__()
        self.block_1 = DenseBlock(12, 3, (5,5), (1,1))
        self.block_2 = DenseBlock(12, 3, (5,5), (1,1))
        self.read_outs = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), padding='SAME' use_bias=False)
        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, x):
        x = self.block_1(x)
        x = tf.nn.max_pool(x, (2,2), (1,1), 'SAME')
        x = block_2(x)
        x = self.read_outs(x)
        x = tf.nn.softmax(x,-1)
        return x

def train(model, training_samples, training_targets, iters):
    cce = tf.keras.losses.CategoricalCrossentropy()
    for _ in range(iters):
        for input, target in zip(training_samples, training_targets):
            with tf.GradientTape() as tape:
                predictions = model(input, training=True)
                loss = cce(targets, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))


if __name__ == "__main__":
