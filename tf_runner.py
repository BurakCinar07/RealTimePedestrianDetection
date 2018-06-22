import sys
import time
import tensorflow as tf

import nn_datasource as ds
from checkpoint_manager import CheckpointManager

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 170
IMAGE_CHANNEL = 1

EPOCH_LENGTH = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.001

RANDOM_SEED = 2

WEIGHT_COUNTER = 0
BIAS_COUNTER = 0
CONVOLUTION_COUNTER = 0
POOLING_COUNTER = 0

sess = None

def new_weights(shape):
    global WEIGHT_COUNTER
    weight = tf.Variable(tf.random_normal(
        shape=shape, seed=RANDOM_SEED), name='w_' + str(WEIGHT_COUNTER))
    WEIGHT_COUNTER += 1
    return weight


def new_biases(length):
    global BIAS_COUNTER
    bias = tf.Variable(
        tf.zeros(shape=[length]), name='b_' + str(BIAS_COUNTER))
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




tf.reset_default_graph()

TEST = True
NETWORK_NUMBER = 4

print(NETWORK_NUMBER)

input_placeholder = tf.placeholder(
    tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='input_placeholder')

output_placeholder = tf.placeholder(tf.float32, shape=[None, 2], name='output_placeholder')

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

layer_fc_1 = tf.nn.softmax(layer_fc_1)

if TEST is not True:
    layer_fc_1 = tf.nn.dropout(layer_fc_1, 0.5)

layer_fc_2 = new_fc_layer(
    input=layer_fc_1, num_inputs=4096, num_outputs=4096)

layer_fc_2 = tf.nn.softmax(layer_fc_2)

if TEST is not True:
    layer_fc_2 = tf.nn.dropout(layer_fc_2, 0.5)

layer_output = new_fc_layer(
    input=layer_fc_2, num_inputs=4096, num_outputs=2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=output_placeholder,
    logits=layer_output)

cost = tf.reduce_mean(cross_entropy)
# cost = tf.losses.mean_squared_error(output_placeholder, layer_output)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

predictions = tf.argmax(tf.nn.softmax(layer_output), dimension=1)

prediction_equalities = tf.equal(predictions, tf.argmax(output_placeholder, dimension=1))

accuracy = tf.reduce_mean(tf.cast(prediction_equalities, tf.float32))


def train_nn(number, input_placeholder, output_placeholder, accuracy, cost, optimizer):
    global TEST

    checkpoint_manager = CheckpointManager(number)
    
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        checkpoint_manager.on_training_start(
            ds.DATASET_FOLDER, EPOCH_LENGTH, BATCH_SIZE,
            LEARNING_RATE, "AdamOptimizer", True)

        for batch_index, batch_images, batch_labels in ds.training_batch_generator(BATCH_SIZE, grayscale=True):

            print("Starting batch {:3}".format(batch_index + 1))

            for current_epoch in range(EPOCH_LENGTH):

                feed = {
                    input_placeholder: batch_images,
                    output_placeholder: batch_labels
                }

                epoch_accuracy, epoch_cost, _ = sess.run(
                    [accuracy, cost, optimizer], feed_dict=feed)
                print("Batch {:3}, Epoch {:3} -> Accuracy: {:3.1%}, Cost: {}".format(
                    batch_index + 1, current_epoch + 1, epoch_accuracy, epoch_cost))

                checkpoint_manager.on_epoch_completed()
            
            TEST = True

            batch_accuracy_training, batch_cost_training = sess.run(
                [accuracy, cost], feed_dict=feed)

            TEST = False

            print("Batch {} has been finished. Accuracy: {:3.1%}, Cost: {}".format(
                batch_index + 1, batch_accuracy_training, batch_cost_training))


            checkpoint_manager.on_batch_completed(
                batch_cost_training, batch_accuracy_training)

            checkpoint_manager.save_model(sess)

        print("\nTraining finished at {}!".format(time.asctime()))

        # overall_accuracy, overall_cost = \
        #     test_nn(number, input_placeholder, output_placeholder, accuracy, cost)

        checkpoint_manager.on_training_completed(None)
        

def test_frame(frame):
    prediction = tf.argmax(tf.nn.softmax(layer_output), 1)
    print(prediction.eval(feed_dict={input_placeholder:[frame]}, session=sess))


def test_nn(number, input_placeholder, output_placeholder, accuracy, cost):
    checkpoint_manager = CheckpointManager(number)


    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        checkpoint_manager.restore_model(sess)

        total_accuracy = 0
        total_cost = 0
        batches = None
        for batch_index, test_images, test_labels in ds.test_batch_generator(100, grayscale=True):

            feed = {
                input_placeholder: test_images,
                output_placeholder: test_labels
            }

            test_accuracy, test_cost = sess.run(
                [accuracy, cost], feed_dict=feed)
            print("Batch {:3}, Accuracy: {:3.1%}, Cost: {}" \
                  .format(batch_index, test_accuracy, test_cost))

            total_accuracy += test_accuracy
            total_cost += test_cost
            batches = batch_index

        overall_accuracy = total_accuracy / (batches + 1)
        overall_cost = total_cost / (batches + 1)

        print("Total test accuracy: {:5.1%}".format(overall_accuracy))

        return overall_accuracy, overall_cost



def main():
    pass


def init():
    checkpoint_manager = CheckpointManager(NETWORK_NUMBER)

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as ses:
        ses.run(init_g)
        ses.run(init_l)
        checkpoint_manager.restore_model(ses)
        sess = ses


if __name__ == '__main__':
    main()
