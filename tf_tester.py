import tensorflow as tf
from PIL import Image
from checkpoint_manager import CheckpointManager
import numpy as np
import os

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 170
IMAGE_CHANNEL = 1

EPOCH_LENGTH = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001

RANDOM_SEED = 2

WEIGHT_COUNTER = 0
BIAS_COUNTER = 0
CONVOLUTION_COUNTER = 0
POOLING_COUNTER = 0


def main():
    tf.reset_default_graph()

    NETWORK_NUMBER = 4

    input_placeholder = tf.placeholder(
        tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='input_placeholder')

    output_placeholder = tf.placeholder(
        tf.float32, shape=[None, 1], name='output_placeholder')

    layer_conv_1, weights_conv_1 = new_conv_layer(
        input=input_placeholder,
        num_input_channels=IMAGE_CHANNEL,
        filter_size=5,
        num_filters=64,
        pooling=2
    )

    layer_conv_2, weights_conv_2 = new_conv_layer(
        input=layer_conv_1,
        num_input_channels=64,
        filter_size=3,
        num_filters=128,
        pooling=2
    )

    layer_conv_3, weights_conv_3 = new_conv_layer(
        input=layer_conv_2,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_4, weights_conv_4 = new_conv_layer(
        input=layer_conv_3,
        num_input_channels=128,
        filter_size=3,
        num_filters=128,
        pooling=None
    )

    layer_conv_5, weights_conv_5 = new_conv_layer(
        input=layer_conv_4,
        num_input_channels=128,
        filter_size=3,
        num_filters=256,
        pooling=3
    )

    layer_flat, num_features = flatten_layer(layer_conv_5)

    layer_fc_1 = new_fc_layer(
        input=layer_flat, num_inputs=num_features, num_outputs=4096)

    layer_fc_1 = tf.nn.sigmoid(layer_fc_1)

    layer_fc_2 = new_fc_layer(
        input=layer_fc_1, num_inputs=4096, num_outputs=4096)

    layer_fc_2 = tf.nn.sigmoid(layer_fc_2)

    layer_output = new_fc_layer(
        input=layer_fc_2, num_inputs=4096, num_outputs=1)

    layer_output = tf.nn.sigmoid(layer_output)

    checkpoint_manager = CheckpointManager(NETWORK_NUMBER)


    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        checkpoint_manager.restore_model(sess)

        imgs = []
        for file in os.listdir("./"):
            if file.endswith(".png"):
                path = os.path.join("./", file)
                im = Image.open(path)
                imgs.append(np.array(im).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL))

        print(layer_output.eval(feed_dict={input_placeholder: imgs},
                                session=sess))


def new_weights(shape):
    global WEIGHT_COUNTER
    weight = tf.Variable(tf.random_normal(
        shape=shape, dtype=tf.float32, seed=RANDOM_SEED), name='w_' + str(WEIGHT_COUNTER))
    WEIGHT_COUNTER += 1
    return weight


def new_biases(length):
    global BIAS_COUNTER
    bias = tf.Variable(
        tf.random_normal(shape=[length], dtype=tf.float32, seed=RANDOM_SEED + 1), name='b_' + str(BIAS_COUNTER))
    BIAS_COUNTER += 1
    return bias


def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling=2):
    global CONVOLUTION_COUNTER
    global POOLING_COUNTER
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME',
                         name='conv_' + str(CONVOLUTION_COUNTER))
    CONVOLUTION_COUNTER += 1

    layer = tf.add(layer, biases)

    layer = tf.nn.relu(layer)

    if pooling is not None and pooling > 1:
        layer = tf.nn.max_pool(value=layer, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME',
                               name='pool_' + str(POOLING_COUNTER))
    POOLING_COUNTER += 1

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.add(tf.matmul(input, weights), biases)
    # layer = tf.nn.relu(layer)
    return layer


if __name__ == '__main__':
    main()
