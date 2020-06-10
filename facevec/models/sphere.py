import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import math

l2_regularizer = tf.contrib.layers.l2_regularizer(1.0)
xavier = tf.contrib.layers.xavier_initializer_conv2d()


def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.25),
                                 regularizer=l2_regularizer, dtype=
                                 tf.float32)
    pos = tf.nn.relu(x)
    neg = tf.multiply(alphas, (x - abs(x)) * 0.5)
    return pos + neg


def first_conv(input, num_output, name):
    with tf.variable_scope(name):
        zero_init = tf.zeros_initializer()
        network = tf.layers.conv2d(input, num_output, kernel_size=[3, 3], strides=(2, 2), padding='SAME',
                                   kernel_initializer=xavier, bias_initializer=zero_init,
                                   kernel_regularizer=l2_regularizer,
                                   bias_regularizer=l2_regularizer)
        network = prelu(network, name=name)
        return network


def block(input, name, num_output):
    with tf.variable_scope(name):
        network = tf.layers.conv2d(input, num_output, kernel_size=[3, 3], strides=[1, 1], padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False,
                                   kernel_regularizer=l2_regularizer)
        network = prelu(network, name='name' + '1')
        network = tf.layers.conv2d(network, num_output, kernel_size=[3, 3], strides=[1, 1], padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False,
                                   kernel_regularizer=l2_regularizer)
        network = prelu(network, name='name' + '2')
        network = tf.add(input, network)
        return network


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
    return dims


def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc': rand_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def coco_forward(xw, y, m, name=None):
    # pdb.set_trace()
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind, y] -= m
    return xw_copy


def coco_help(grad, y):
    grad_copy = grad.copy()
    return grad_copy


def coco_backward(op, grad):
    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help, [grad, y], tf.float32)
    return grad_copy, y, m


def coco_func(xw, y, m, name=None):
    with tf.op_scope([xw, y, m], name, "Coco_func") as name:
        coco_out = py_func(coco_forward, [xw, y, m], tf.float32, name=name, grad_func=coco_backward)
        return coco_out


class Sphere(object):

    @classmethod
    def original_softmax_loss(cls, embeddings, labels, num_cls, bs=128):
        """
        This is the orginal softmax loss, nothing to say
        """
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], num_cls],
                                      initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)  # output probability
            # define cross entropy
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss

    '''
    @classmethod
    def modified_softmax_loss(cls, embeddings, labels, num_cls, bs=128):
        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], num_cls],
                                      initializer=tf.contrib.layers.xavier_initializer())
            # normalize weights 得到一个batch中每个维度的l2范数[[0.1, 0.2,...,0.94]]
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            # 对每个维度做归一化
            weights = tf.div(weights, weights_norm, name="normalize_weights")
            logits = tf.matmul(embeddings, weights)
            pred_prob = tf.nn.softmax(logits=logits)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            return pred_prob, loss
    '''

    '''
    @classmethod
    def angular_softmax_loss(cls, embeddings, labels, num_cls, bs, l=0):
        embeddings_norm = tf.norm(embeddings, axis=1)

        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], num_cls],
                                      initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.nn.l2_normalize(weights, axis=0)
            # cacualting the cos value of angles between embeddings and weights
            orgina_logits = tf.matmul(embeddings, weights)
            single_sample_label_index = tf.stack([tf.constant(list(range(bs)), tf.int32), labels], axis=1)
            # N = 128, labels = [1,0,...,9]
            # single_sample_label_index:
            # [ [0,1],
            #   [1,0],
            #   ....
            #   [128,9]]
            selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index)
            cos_theta = tf.div(selected_logits, embeddings_norm)
            cos_theta_power = tf.square(cos_theta)
            cos_theta_biq = tf.pow(cos_theta, 4)
            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2 * cos_theta_power - 1), sign0)
            sign4 = 2 * sign0 + sign3 - 3
            result = sign3 * (8 * cos_theta_biq - 8 * cos_theta_power + 1) + sign4

            margin_logits = tf.multiply(result, embeddings_norm)
            f = 1.0 / (1.0 + l)
            ff = 1.0 - f
            combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                                  tf.subtract(margin_logits, selected_logits),
                                                                  tf.constant(
                                                                      [bs, orgina_logits.get_shape().as_list()[1]])))
            updated_logits = ff * orgina_logits + f * combined_logits
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
            pred_prob = tf.nn.softmax(logits=updated_logits)
            return pred_prob, loss
    '''

    '''
    @classmethod
    def a_softmax_loss(cls, embeddings, labels, num_cls, bs, l=1, m=4, name='asoftmax'):
        xs = embeddings.get_shape().as_list()
        w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        eps = 1e-8

        xw = tf.matmul(embeddings, w)

        if m == 0:
            return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=xw))

        w_norm = tf.norm(w, axis=0) + eps
        logits = xw / w_norm

        if labels is None:
            return logits, None

        ordinal = tf.constant(list(range(0, bs)), tf.int32)
        ordinal_y = tf.stack([ordinal, labels], axis=1)

        x_norm = tf.norm(embeddings, axis=1) + eps

        sel_logits = tf.gather_nd(logits, ordinal_y)

        cos_th = tf.div(sel_logits, x_norm)

        if m == 1:

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        else:

            if m == 2:

                cos_sign = tf.sign(cos_th)
                res = 2 * tf.multiply(tf.sign(cos_th), tf.square(cos_th)) - 1

            elif m == 4:

                cos_th2 = tf.square(cos_th)
                cos_th4 = tf.pow(cos_th, 4)
                sign0 = tf.sign(cos_th)
                sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
                sign4 = 2 * sign0 + sign3 - 3
                res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
            else:
                raise ValueError('unsupported value of m')

            scaled_logits = tf.multiply(res, x_norm)

            f = 1.0 / (1.0 + l)
            ff = 1.0 - f
            comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits),
                                                            tf.constant([bs, logits.get_shape().as_list()[1]])))
            updated_logits = ff * logits + f * comb_logits_diff

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))

        return logits, loss
    '''

    '''
    @classmethod
    def sphere_loss(cls, embeddings, labels, num_cls, bs, fraction=1, scope='Logits', eplion=1e-8):
        inputs_shape = embeddings.get_shape().as_list()
        with tf.variable_scope(name_or_scope=scope):
            weight = tf.Variable(
                initial_value=tf.random_normal((num_cls, inputs_shape[1])) * tf.sqrt(2 / inputs_shape[1]),
                dtype=tf.float32, name='weights')  # shaep =classes, features,

        weight_unit = tf.nn.l2_normalize(weight, dim=1)

        inputs_mo = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1) + eplion)  # shape=[batch

        inputs_unit = tf.nn.l2_normalize(embeddings, dim=1)  # shape = [batch,features_num]

        logits = tf.matmul(embeddings, tf.transpose(weight_unit))  # shape = [batch,classes] x * w_unit

        weight_unit_batch = tf.gather(weight_unit, labels)  # shaep =batch,features_num,

        logits_inputs = tf.reduce_sum(tf.multiply(embeddings, weight_unit_batch), axis=1)  # shaep =batch,

        cos_theta = tf.reduce_sum(tf.multiply(inputs_unit, weight_unit_batch), axis=1)  # shaep =batch,

        cos_theta_square = tf.square(cos_theta)
        cos_theta_biq = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign2 = tf.sign(2 * cos_theta_square - 1)
        sign3 = tf.multiply(sign2, sign0)
        sign4 = 2 * sign0 + sign3 - 3
        cos_far_theta = sign3 * (8 * cos_theta_biq - 8 * cos_theta_square + 1) + sign4
        # print("cos_far_theta  = ", cos_far_theta.get_shape().as_list())

        logit_ii = tf.multiply(cos_far_theta, inputs_mo)  # shape = batch

        index_range = tf.range(start=0, limit=tf.shape(embeddings, out_type=tf.int32)[0], delta=1, dtype=tf.int32)
        index_labels = tf.stack([index_range, labels], axis=1)
        index_logits = tf.scatter_nd(index_labels, tf.subtract(logit_ii, logits_inputs),
                                     tf.shape(logits, out_type=tf.int32))

        logits_final = tf.add(logits, index_logits)
        logits_final = fraction * logits_final + (1 - fraction) * logits

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_final))

        return logits_final, loss
    '''

    @classmethod
    def cos_loss(cls, embeddings, labels, num_cls, bs, reuse=False, alpha=0.25, scale=64, name='cos_loss'):
        '''
        embeddings: B x D - features
        y: B x 1 - labels
        num_cls: 1 - total class number
        alpah: 1 - margin
        scale: 1 - scaling paramter
        '''
        # define the classifier weights
        xs = embeddings.get_shape().as_list()
        with tf.variable_scope('centers_var', reuse=reuse) as center_scope:
            w = tf.get_variable("centers", [xs[1], num_cls], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

        # normalize the feature and weight
        # (N,D)
        x_feat_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10)
        # (D,C)
        w_feat_norm = tf.nn.l2_normalize(w, 0, 1e-10)

        # get the scores after normalization
        # (N,C)
        xw_norm = tf.matmul(x_feat_norm, w_feat_norm)
        # implemented by py_func
        # value = tf.identity(xw)
        # substract the marigin and scale it
        value = coco_func(xw_norm, labels, alpha) * scale

        # implemented by tf api
        # margin_xw_norm = xw_norm - alpha
        # label_onehot = tf.one_hot(y,num_cls)
        # value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)

        # compute the loss as softmax loss
        cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=value))

        return value, cos_loss


    @classmethod
    def arc_loss(cls, embeddings, labels, num_cls, bs, logits_scale=64.0, logits_margin=0.5, scope="Logits"):
        inputs_shape = embeddings.get_shape().as_list()
        with tf.variable_scope(name_or_scope=scope):
            weight = tf.Variable(
                initial_value=tf.random_normal((inputs_shape[1], num_cls)) * tf.sqrt(2 / inputs_shape[1]),
                dtype=tf.float32, name='weights')  # shaep =classes, features,
        embds = tf.nn.l2_normalize(embeddings, axis=1, name='normed_embd')
        weights = tf.nn.l2_normalize(weight, axis=0)
        s = logits_scale
        m = logits_margin

        cos_m = math.cos(m)
        sin_m = math.sin(m)

        mm = sin_m * m

        threshold = math.cos(math.pi - m)

        cos_t = tf.matmul(embds, weights, name='cos_t')

        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(labels, depth=num_cls, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))
        return output, loss

    @classmethod
    def inference(cls, input, embedding_size=512, name="sphere"):
        with tf.variable_scope(name):
            # input = B*3*112*96
            with tf.variable_scope('conv1_'):
                network = first_conv(input, 64, name='conv1')  # =>B*64*56*48
                network = block(network, 'conv1_23', 64)

            with tf.variable_scope('conv2_'):
                network = first_conv(network, 128, name='conv2')  # =>B*128*28*24
                network = block(network, 'conv2_23', 128)
                network = block(network, 'conv2_45', 128)

            with tf.variable_scope('conv3_'):
                network = first_conv(network, 256, name='conv3')  # =>B*256*14*12
                network = block(network, 'conv3_23', 256)
                network = block(network, 'conv3_45', 256)
                network = block(network, 'conv3_67', 256)
                network = block(network, 'conv3_89', 256)

            with tf.variable_scope('conv4_'):
                network = first_conv(network, 512, name='conv4')
                network = block(network, 'conv4_23', 512)

            with tf.variable_scope('feature'):
                dims = get_shape(network)
                print(dims)
                feature = tf.layers.dense(tf.reshape(network, [dims[0], np.prod(dims[1:])]), embedding_size,
                                          kernel_regularizer=l2_regularizer, kernel_initializer=xavier)
            return feature
