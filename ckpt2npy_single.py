import numpy as np
import tensorflow as tf
import configure as cfg
import os, ipdb

'''
def load_tf_model(tag):
    tag = tag + '_'

    var_lst = []
    # conv-1 layer
    with tf.name_scope(tag+'conv1') as scope:
        conv1W = tf.Variable(tf.zeros(shape=[11,11,3,96]), name='weight')
        conv1b = tf.Variable(tf.zeros(shape=[96]), name='biases')

    # conv-2 layer
    with tf.name_scope(tag+'conv2') as scope:
        conv2W = tf.Variable(tf.zeros(shape=[5,5,48,256]), name='weight')
        conv2b = tf.Variable(tf.zeros(shape=[256]), name='biases')

    # conv-3 layer
    with tf.name_scope(tag+'conv2') as scope:
        conv3W = tf.Variable(tf.zeros(shape=[3,3,256,384]), name='weight')
        conv3b = tf.Variable(tf.zeros(shape=[384]), name='biases')

    # conv-4 layer
    with tf.name_scope(tag+'conv4') as scope:
        conv4W = tf.Variable(tf.zeros(shape=[3,3,192,384]), name='weight')
        conv4b = tf.Variable(tf.zeros(shape=[384]), name='biases')

    # conv-5 layer
    with tf.name_scope(tag+'conv5') as scope:
        conv5W = tf.Variable(tf.zeros(shape=[3,3,192,256]), name='weight')
        conv5b = tf.Variable(tf.zeros(shape=[256]), name='biases')

    # fc6 layer
    with tf.name_scope(tag+'fc6') as scope:
        fc6W = tf.Variable(tf.zeros(shape=[9216,4096]), name='weight')
        fc6b = tf.Variable(tf.zeros(shape=[4096]), name='biases')

    # fc7 layer
    with tf.name_scope(tag+'fc7') as scope:
        fc7W = tf.Variable(tf.zeros(shape=[4096,4096]), name='weight')
        fc7b = tf.Variable(tf.zeros(shape=[4096]), name='biases')

    # fc8 layer - classifier
    with tf.name_scope(tag+'fc8') as scope:
        fc8W = tf.Variable(tf.zeros([4096,51]), name='weight')
        fc8b = tf.Variable(tf.zeros([51]), name='biases')

    #var_lst = [conv1W, conv1b, conv2W, conv2b, conv3W, conv3b, conv4W, conv4b, conv5W, conv5b, fc6W, fc6b, fc7W, fc7b, fc8W, fc8b]

    var_lst = [conv1W]
    return var_lst
'''


def convert(path, filename):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path+'.meta')
    saver.restore(sess, path)
    var_lst = tf.trainable_variables()
    for v in var_lst:
        print v.name


    model = dict()
    model['conv1'] = [sess.run(var_lst[0]), sess.run(var_lst[1])]
    model['conv2'] = [sess.run(var_lst[2]), sess.run(var_lst[3])]
    model['conv3'] = [sess.run(var_lst[4]), sess.run(var_lst[5])]
    model['conv4'] = [sess.run(var_lst[6]), sess.run(var_lst[7])]
    model['conv5'] = [sess.run(var_lst[8]), sess.run(var_lst[9])]
    model['fc6'] = [sess.run(var_lst[10]), sess.run(var_lst[11])]
    model['fc7'] = [sess.run(var_lst[12]), sess.run(var_lst[13])]
    model['fc8'] = [sess.run(var_lst[14]), sess.run(var_lst[15])]

    np.save(filename, model)
    return



if __name__ == '__main__':
    print 'Converting RGB model...'
    with tf.Graph().as_default():
        convert(os.path.join(cfg.DIR_BESTCKPT, 'rgb-best'), cfg.PTH_RGB_MODEL)

    print 'Converting Dep model...'
    with tf.Graph().as_default():
        convert(os.path.join(cfg.DIR_BESTCKPT, 'dep-best'), cfg.PTH_DEP_MODEL)

