import tensorflow as tf
import scipy.io
import numpy as np


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
    return dims


def prelu(x, name, data, pre_name):
    with tf.variable_scope(name):
        alphas = tf.constant(data[pre_name + "/" + name + "/alpha"])
    pos = tf.nn.relu(x)
    neg = tf.multiply(alphas, (x - abs(x)) * 0.5)
    return pos + neg


def first_conv(input, name, data, pre_name):
    with tf.variable_scope(name):
        pre_name += "/%s" % name
        network = tf.nn.conv2d(input, tf.constant(data[pre_name + "/conv2d/kernel"]), strides=(1, 2, 2, 1), padding='SAME')
        bias = data[pre_name + "/conv2d/bias"]
        network = tf.nn.bias_add(network, tf.constant(bias, shape=(bias.shape[1],)))
        network = prelu(network, name, data, pre_name)
        return network


def block(input, name, data, pre_name):
    with tf.variable_scope(name):
        tf.get_variable_scope()
        pre_name += "/%s" % name
        network = tf.nn.conv2d(input, tf.constant(data[pre_name + "/conv2d/kernel"]), strides=(1, 1, 1, 1), padding='SAME')
        network = prelu(network, 'name1', data, pre_name)
        network = tf.nn.conv2d(network, tf.constant(data[pre_name + "/conv2d_1/kernel"]), strides=(1, 1, 1, 1), padding='SAME')
        network = prelu(network, 'name2', data, pre_name)
        network = tf.add(input, network)
        return network


def net(path_to_fie_net, input_image):
    data = scipy.io.loadmat(path_to_fie_net)

    with tf.variable_scope("sphere"):
        # input = B*3*112*96
        with tf.variable_scope('conv1_'):
            pre_name = "sphere/conv1_"
            network = first_conv(input_image, 'conv1', data, pre_name)  # =>B*64*56*48
            network = block(network, 'conv1_23', data, pre_name)

        with tf.variable_scope('conv2_'):
            pre_name = "sphere/conv2_"
            network = first_conv(network, 'conv2', data, pre_name)  # =>B*128*28*24
            network = block(network, 'conv2_23', data, pre_name)
            network = block(network, 'conv2_45', data, pre_name)

        with tf.variable_scope('conv3_'):
            pre_name = "sphere/conv3_"
            network = first_conv(network, 'conv3', data, pre_name)  # =>B*256*14*12
            network = block(network, 'conv3_23', data, pre_name)
            network = block(network, 'conv3_45', data, pre_name)
            network = block(network, 'conv3_67', data, pre_name)
            network = block(network, 'conv3_89', data, pre_name)

        with tf.variable_scope('conv4_'):
            pre_name = "sphere/conv4_"
            network = first_conv(network, 'conv4', data, pre_name)
            network = block(network, 'conv4_23', data, pre_name)
        with tf.variable_scope('feature'):
            pre_name = "sphere/feature"
            w = data[pre_name + "/dense/kernel"]
            b = data[pre_name + "/dense/bias"]
            dims = get_shape(network)
            feature = tf.matmul(tf.reshape(network, [dims[0], np.prod(dims[1:])]),
                                tf.constant(w)) + tf.constant(b, shape=(b.shape[1],))

            return tf.nn.l2_normalize(feature, axis=1)
