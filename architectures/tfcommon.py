import os, ipdb
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding='VALID', group=1):
    """Wrapper for TensorFlow's 2D convolution

    Args:
        input: input data
        kernel: kernel's weight
        biases: kernel's bias
        k_h: kernel's height
        k_s: kernel's width
        c_o: number of kernels
        s_h: stride's height
        s_w: stride's width

    Returns:
        Result of convolution
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1,s_h,s_w,1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i,k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    #result = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    result = tf.nn.bias_add(conv, biases)
    return result


def training(loss, learning_rate=None):
    """Sets up the training ops.

    Args:
        loss: loss tensor, from loss()
        learning_rate: if not specified -> use FLAGS.learning_rate

    Returns:
        train_op: the op for training
    """
    if learning_rate is None:
        learning_rate = FLAGS.learning_rate

    # Add a scalar summary for the snapshot loss
    tf.scalar_summary('loss', loss)

    # Create the optimizer with given learning rate
    #optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimzer to minimize the loss
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(score, labels):
    """Find the number of correct classification (top prob among classes), based on labels

    Args:
        prob: results of model's inference
        labels: 2D Tensor [batch_size, n_classes], 1 if object in that class, 0 otherwise

    Returns:
        Number of correct classification
    """
    id_labels = tf.argmax(labels, dimension=1) # convert from binary sequences to class id
    correct = tf.nn.in_top_k(score, id_labels, 1)

    # add to summary
    #tf.histogram_summary('score', score)
    #tf.histogram_summary('ground truth', id_labels)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
