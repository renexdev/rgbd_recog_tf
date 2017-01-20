import os, ipdb
import tensorflow as tf
import numpy as np
import configure as cfg
from tfcommon import *

FLAGS = tf.app.flags.FLAGS

def _extract_feature(images, net_data):
    # conv-1 layer
    with tf.name_scope('conv1') as scope:
        conv1W = tf.Variable(net_data['conv1'][0], trainable=False, name='weight')
        conv1b = tf.Variable(net_data['conv1'][1], trainable=False, name='biases')
        conv1_in = conv(images, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in, name=scope)
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
    lrn1 = tf.nn.local_response_normalization(maxpool1, depth_radius=5, alpha=1e-4, beta=0.75, name='norm1')


    # conv-2 layer
    with tf.name_scope('conv2') as scope:
        conv2W = tf.Variable(net_data['conv2'][0], trainable=False, name='weight')
        conv2b = tf.Variable(net_data['conv2'][1], trainable=False, name='biases')
        conv2_in = conv(lrn1, conv2W, conv2b, 5, 5, 256, 1, 1, padding='SAME', group=2)
        conv2 = tf.nn.relu(conv2_in, name=scope)
    maxpool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
    lrn2 = tf.nn.local_response_normalization(maxpool2, depth_radius=5, alpha=1e-4, beta=0.75, name='norm2')


    # conv-3 layer
    with tf.name_scope('conv3') as scope:
        conv3W = tf.Variable(net_data['conv3'][0], trainable=False, name='weight')
        conv3b = tf.Variable(net_data['conv3'][1], trainable=False, name='biases')
        conv3_in = conv(lrn2, conv3W, conv3b, 3, 3, 384, 1, 1, padding='SAME', group=1)
        conv3 = tf.nn.relu(conv3_in, name=scope)


    # conv-4 layer
    with tf.name_scope('conv4') as scope:
        conv4W = tf.Variable(net_data['conv4'][0], trainable=False, name='weight')
        conv4b = tf.Variable(net_data['conv4'][1], trainable=False, name='biases')
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding='SAME', group=2)
        conv4 = tf.nn.relu(conv4_in, name=scope)


    # conv-5 layer
    with tf.name_scope('conv5') as scope:
        conv5W = tf.Variable(net_data['conv5'][0], trainable=False, name='weight')
        conv5b = tf.Variable(net_data['conv5'][1], trainable=False, name='biases')
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding='SAME', group=2)
        conv5 = tf.nn.relu(conv5_in, name=scope)
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')


    # fc6 layer
    with tf.name_scope('fc6') as scope:
        fc6W = tf.Variable(net_data['fc6'][0], trainable=False, name='weight')
        fc6b = tf.Variable(net_data['fc6'][1], trainable=False, name='biases')
        fc6_in = tf.reshape(maxpool5, [FLAGS.batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6b, name=scope)


    # fc7 layer
    with tf.name_scope('fc7') as scope:
        fc7W = tf.Variable(net_data['fc7'][0], trainable=False, name='weight')
        fc7b = tf.Variable(net_data['fc7'][1], trainable=False, name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name=scope)


    # fc8 layer - classifier
    with tf.name_scope('fc8') as scope:
        fc8W = tf.Variable(net_data['fc8'][0], trainable=False, name='weight')
        fc8b = tf.Variable(net_data['fc8'][1], trainable=False, name='biases')
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b, name=scope)

    return fc7 #fc8


def inference(rgb_data, dep_data, rgb_model, dep_model, keep_prob, tag='fus'):
    tag += '_'
    batch_size = FLAGS.batch_size
    n_classes = FLAGS.n_classes
    feat_len = FLAGS.feat_len
    
    rgb_feat = _extract_feature(rgb_data, rgb_model)
    dep_feat = _extract_feature(dep_data, dep_model)
    concat_feat = tf.concat(1, [rgb_feat, dep_feat], name='concat_feat')

    # fc1-fus
    with tf.name_scope(tag+'fc1_fus') as scope:
        fc1_fusW = tf.Variable(tf.random_normal([feat_len*2,feat_len], stddev=0.01), name='weight')
        fc1_fusb = tf.Variable(tf.zeros([feat_len]), name='biases')
        fc1_fus  = tf.nn.relu_layer(concat_feat, fc1_fusW, fc1_fusb, name=scope)

    # classifier
    with tf.name_scope(tag+'class') as scope:
        classW = tf.Variable(tf.random_normal([feat_len,n_classes], stddev=0.01), name='weight')
        classb = tf.Variable(tf.zeros([n_classes]), name='biases')
        classifier = tf.nn.xw_plus_b(fc1_fus, classW, classb, name=scope)

    # prob
    #prob = tf.nn.softmax(classifier, name='prob')
    return classifier #prob


def loss(score, labels, tag='fus'):
    def _get_partial_regularizer(scope_name, w_shape):
        with tf.variable_scope(scope_name):
            w = tf.get_variable('weight', w_shape, dtype=tf.float32)
        # compute l2 loss
        w_l2 = tf.nn.l2_loss(w)

        return w_l2

    tag += '_'
    #prob = tf.nn.softmax(score, name='prob')
    #logits = tf.log(tf.clip_by_value(prob, 1e-10, 1.0), name='logits')
    #L = -tf.reduce_sum(labels * logits, reduction_indices=1)
    #loss = tf.reduce_sum(L, reduction_indices=0, name='loss')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(score, labels), name='loss')

    # regularize weights
    regularizers = _get_partial_regularizer('fus_fc1_fus',[4096*2,4096]) + \
            _get_partial_regularizer('fus_class', [4096,FLAGS.n_classes])
    loss += 1e-4 * regularizers
    return loss

