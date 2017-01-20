import os, ipdb
import tensorflow as tf
import numpy as np
from tfcommon import *

FLAGS = tf.app.flags.FLAGS

def inference(images, net_data, keep_prob, tag=''):
    """Build the inference for one single channel.

    Args:
        images: 4D tensor: [batch_size, image_size, image_size, 3]
        net_data: pretrained weights from AlexNet
        keep_prob: Tensor, 0.5 if training, 1.0 otherwise

    Returns:
        prob: softmax result
    """
    tag += '_'

    # conv-1 layer
    ## conv(11,11,96,4,4,padding='VALID',name='conv1')
    with tf.name_scope(tag+'conv1') as scope:
        conv1W = tf.Variable(net_data['conv1'][0], name='weight')
        conv1b = tf.Variable(net_data['conv1'][1], name='biases')
        conv1_in = conv(images, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in, name=scope)
    ## max_pool(3,3,2,2,padding='VALID',name='pool1')
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
    ## lrn(2,2e-05,0.75,name='norm1')
    lrn1 = tf.nn.local_response_normalization(maxpool1, depth_radius=5, alpha=1e-4, beta=0.75, name='norm1')


    # conv-2 layer
    ## conv(5,5,256,1,1,group=2,name='conv2')
    with tf.name_scope(tag+'conv2') as scope:
        conv2W = tf.Variable(net_data['conv2'][0], name='weight')
        conv2b = tf.Variable(net_data['conv2'][1], name='biases')
        conv2_in = conv(lrn1, conv2W, conv2b, 5, 5, 256, 1, 1, padding='SAME', group=2)
        conv2 = tf.nn.relu(conv2_in, name=scope)
    ## max_pool(3,3,2,2,padding='VALID',name='pool2')
    maxpool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
    ## lrn(2,2e-05,0.75,name='norm2')
    lrn2 = tf.nn.local_response_normalization(maxpool2, depth_radius=5, alpha=1e-4, beta=0.75, name='norm2')


    # conv-3 layer
    ## conv(3,3,384,1,1,name='conv3')
    with tf.name_scope(tag+'conv3') as scope:
        conv3W = tf.Variable(net_data['conv3'][0], name='weight')
        conv3b = tf.Variable(net_data['conv3'][1], name='biases')
        conv3_in = conv(lrn2, conv3W, conv3b, 3, 3, 384, 1, 1, padding='SAME', group=1)
        conv3 = tf.nn.relu(conv3_in, name=scope)


    # conv-4 layer
    ## conv(3,3,384,1,1,group=2,name='conv4')
    with tf.name_scope(tag+'conv4') as scope:
        conv4W = tf.Variable(net_data['conv4'][0], name='weight')
        conv4b = tf.Variable(net_data['conv4'][1], name='biases')
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding='SAME', group=2)
        conv4 = tf.nn.relu(conv4_in, name=scope)


    # conv-5 layer
    ## conv(3,3,256,1,1,group=2,name='conv5')
    with tf.name_scope(tag+'conv5') as scope:
        conv5W = tf.Variable(net_data['conv5'][0], name='weight')
        conv5b = tf.Variable(net_data['conv5'][1], name='biases')
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding='SAME', group=2)
        conv5 = tf.nn.relu(conv5_in, name=scope)
    ## max_pool(3,3,2,2,padding='VALID',name='pool5')
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')


    # fc6 layer
    ## fc(4096, name='fc6')
    with tf.name_scope(tag+'fc6') as scope:
        fc6W = tf.Variable(net_data['fc6'][0], name='weight')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        fc6_in = tf.reshape(maxpool5, [FLAGS.batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6b, name=scope)
        fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='drop')


    # fc7 layer
    ## fc(4096, name='fc7')
    with tf.name_scope(tag+'fc7') as scope:
        fc7W = tf.Variable(net_data['fc7'][0], name='weight')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b, name=scope)
        fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='drop')


    # fc8 layer - classifier
    ## fc(1000, relu=False, name='fc8')
    with tf.name_scope(tag+'fc8') as scope:
        # do not use net_data as we have differenet number of classes here
        #fc8W = tf.Variable(net_data['fc8'][0], name='weight')
        #fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        '''
        fc8W_mean = np.mean(net_data['fc8'][0])
        fc8W_std  = np.std(net_data['fc8'][0])
        fc8b_mean = np.mean(net_data['fc8'][1])
        fc8b_std  = np.std(net_data['fc8'][1])
        fc8W = tf.Variable(tf.random_normal([4096,FLAGS.n_classes], 
            mean=fc8W_mean, stddev=fc8W_std), name='weight')
        fc8b = tf.Variable(tf.random_normal([FLAGS.n_classes],
            mean=fc8b_mean, stddev=fc8b_std), name='biases')
        '''
        fc8W = tf.Variable(tf.random_normal([4096,FLAGS.n_classes], stddev=0.01), name='weight')
        fc8b = tf.Variable(tf.zeros([FLAGS.n_classes]), name='biases')
        fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b, name=scope)


    # prob
    ## softmax(name='prob')
    #prob = tf.nn.softmax(fc8, name='prob')
    return fc8


def loss(score, labels, tag):
    """Return the loss as categorical cross-entropy

    Args:
        prob: results from inference
        labels: 2D Tensor [batch_size, n_classes], 1 if object in that class, 0 otherwise

    Returns:
        loss: categorical crossentropy loss
    """
    tag += '_'

    #prob = tf.nn.softmax(score, name='prob')
    #logits = tf.log(tf.clip_by_value(prob, 1e-10, 1.0), name='logits')
    #L = -tf.reduce_sum(labels * logits, reduction_indices=1)
    #loss = tf.reduce_sum(L, reduction_indices=0, name='loss')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(score, labels), name='loss')

    # regularize weights
    layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
    shapes = [[11,11,3,96],[5,5,48,256],[3,3,256,384],[3,3,192,384],[3,3,192,256],[9216,4096],[4096,4096],[4096,FLAGS.n_classes]]

    regularizers = 0
    for i in range(len(layers)):
        with tf.variable_scope(tag+layers[i]):
            weight = tf.get_variable('weight', shapes[i], dtype=tf.float32)
            regularizers += tf.nn.l2_loss(weight)

    loss += 1e-4*regularizers
    return loss

