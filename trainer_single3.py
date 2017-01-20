import tensorflow as tf
import numpy as np
import os, sys, time, shutil, ipdb
from utils import common
import configure as cfg 
from architectures import model_single_channel as model


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 50, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('img_s_raw', cfg.IMG_RAW_S, """"Size of a raw square image.""")
tf.app.flags.DEFINE_integer('n_classes', len(cfg.CLASSES), """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 3, """How often to evaluate and write checkpoint.""")


#=========================================================================================
def placeholder_inputs(batch_size):
    """Create placeholders for tensorflow with some specific batch_size

    Args:
        batch_size: size of each batch

    Returns:
        images_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
    """
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='images_placeholder') 
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

    return images_ph, labels_ph, keep_prob_ph


def fill_feed_dict(img_batch, lbl_batch, images_ph, labels_ph, keep_prob_ph, is_training, crop):
    """Fills the feed_dict. If the batch has fewer samples than the placeholder, it is padded
    with zeros.

    Args:
        img_batch: 4D numpy array of shape [batch_size, image_size, image_size, 3]
        lbl_batch: 2D numpy array of shape [batch_size, num_classes]
        images_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
        is_training: True or False. If True, keep_prob = 0.5, 1.0 otherwise

    Returns:
        feed_dict: feed dictionary
    """
    if img_batch.shape[0] < FLAGS.batch_size: # pad the remainder with zeros
        if is_training: # only pad for evaluation
            return None
        else:
            M = FLAGS.batch_size - img_batch.shape[0]
            img_batch = np.pad(img_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
            lbl_batch = np.pad(lbl_batch, ((0,M),(0,0)), 'constant', constant_values=0)

    if crop == 'random':
        img_batch = common.random_crop(img_batch)
    elif crop == 'central':
        img_batch = common.central_crop(img_batch)

    kp = 0.5 if is_training else 1.0

    feed_dict = {images_ph: img_batch, labels_ph: lbl_batch, keep_prob_ph: kp}
    return feed_dict


def do_eval(sess, score, eval_correct, images_ph, labels_ph, keep_prob_ph, 
        all_data, all_labels, logfile, tag, step, crop):
    ''' Run one evaluation against the full epoch of data

    Args:
        sess: the session in which the model has been trained
        score: model inference operator
        eval_correct: the tensor that returns the number of correct predictions
        images_ph: tensor place holder for images
        labels_ph: tensor place holder for labels
        keep_prob_ph: tensor place holder for keep_prob
        all_data: all loaded images
        all_labels: all labels corresponding to all_images
        logfile: opened logfile

    Return
        precision: percentage of correct recognition
    '''
    true_count, start_idx = 0, 0
    num_samples = all_data.shape[0]
    indices = np.arange(num_samples)
    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx: stop_idx] # no need to shuffle for evaluation
        is_training = False

        fd = fill_feed_dict(
            all_data[batch_idx], all_labels[batch_idx], 
            images_ph, labels_ph, keep_prob_ph, 
            is_training, crop)
        score_val = sess.run(score, feed_dict=fd)

        # throw out zero padding results
        used_score = score_val[:stop_idx-start_idx]
        used_lbl   = all_labels[start_idx:stop_idx]

        common.write_score(used_score, used_lbl, tag, step)
        #true_count += sess.run(eval_correct, feed_dict=fd)

        # count correct predictions
        for i in range(stop_idx - start_idx):
            if used_score[i].argmax() == used_lbl[i].argmax():
                true_count += 1

        start_idx = stop_idx

    precision = true_count*1.0 / num_samples
    common.writer('    Num-samples:%d   Num-correct:%d   Precision:%0.04f', (num_samples, true_count, precision), logfile)

    return precision


#=========================================================================================
def run_training(pth_train_lst, pth_eval_lst, train_dir, eval_dir, tag):
    logfile = open(os.path.join(cfg.DIR_LOG, 'training_'+tag+'.log'), 'w', 0)

    # load data
    print 'Loading AlexNet...'
    net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()

    print 'Loading lists...'
    with open(pth_train_lst, 'r') as f: train_lst = f.read().splitlines()
    with open(pth_eval_lst, 'r') as  f: eval_lst  = f.read().splitlines()
    if tag == 'rgb': ext = cfg.EXT_RGB
    elif tag == 'dep': ext = cfg.EXT_D
    #train_lst = train_lst[:500]; eval_lst=eval_lst[:500] #TODO

    print 'Loading training data...'
    train_data, train_labels = common.load_images(train_lst, train_dir, ext, ccrop=False)
    num_train = len(train_data)

    print 'Loading validation data...'
    eval_data, eval_labels = common.load_images(eval_lst, eval_dir, ext, ccrop=True)

    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    images_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    score = model.inference(images_ph, net_data, keep_prob_ph, tag)
    loss = model.loss(score, labels_ph, tag)
    train_op = model.training(loss)
    eval_correct = model.evaluation(score, labels_ph)
    init_op = tf.initialize_all_variables()

    # tensorflow monitor
    summary = tf.merge_all_summaries()
    #saver = tf.train.Saver(max_to_keep=1000)
    saver = tf.train.Saver()
   
    # initialize graph
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(cfg.DIR_SUMMARY, sess.graph)
    sess.run(init_op)


    # start the training loop
    old_loss, best_loss, old_precision, best_precision = sys.maxsize, sys.maxsize, 0, 0
    patience_count = 0
    print 'Start the training loop...'
    for step in range(FLAGS.max_iter):
        # training phase----------------------------------------------
        start_time = time.time()

        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx = 0, 0
        lim = 10
        while start_idx != num_train:
            if start_idx*100.0 / num_train > lim:
                print 'Trained %d/%d' % (start_idx, num_train)
                lim += 10

            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            # filling feed dict
            fd = fill_feed_dict(
                train_data[batch_idx], train_labels[batch_idx], 
                images_ph, labels_ph, keep_prob_ph,
                is_training=True, crop='random')
            if fd is None: break # throw the incomplete batch

            # train
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)

            # write summary
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            # update loss
            total_loss += loss_value
            start_idx = stop_idx

        duration = time.time() - start_time
        common.writer('Step %d: loss = %.3f (%.3f sec)', (step,total_loss,duration), logfile)


        # write summary------------------------------------------------
        #if step % FLAGS.summary_frequency == 0:
            #summary_str = sess.run(summary, feed_dict=fd)
            #summary_writer.add_summary(summary_str, step)
            #summary_writer.flush()


        # write checkpoint---------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:# or total_loss<best_loss:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)
            
            common.writer('  Training data eval:', (), logfile)
            do_eval(
                sess, score, eval_correct, 
                images_ph, labels_ph, keep_prob_ph, 
                train_data, train_labels, 
                logfile, tag+'train', step, crop='central')

            common.writer('  Validation data eval:', (), logfile)
            precision = do_eval(
                sess, score, eval_correct, 
                images_ph, labels_ph, keep_prob_ph, 
                eval_data, eval_labels, 
                logfile, tag+'eval', step, crop=None)
            common.writer('Precision: %.4f', precision, logfile)

            if precision > best_precision:
                import shutil
                src = os.path.join(cfg.DIR_CKPT, tag+'-'+str(step))
                dst = os.path.join(cfg.DIR_BESTCKPT, tag+'-best')
                shutil.copy2(src, dst)
                shutil.copy2(src+'.meta', dst+'.meta')
                best_precision = precision


        common.writer('', (), logfile)


        # early stopping-----------------------------------------------
        '''
        to_stop, patience_count = common.early_stopping(
                old_loss, total_loss, patience_count, expect_greater=False)
        old_loss = total_loss
        if to_stop: 
            common.writer('Early stopping...', (), logfile)
            break
        '''
    common.writer('Best precision: %.4f', best_precision, logfile)
    logfile.close()
    return


#=========================================================================================
def main(argv=None):
    trial = 1
    print 'Trial: %d' % trial

    #pth_train_lst = cfg.PTH_TRAIN_LST[trial-1]
    pth_train_lst = cfg.PTH_TRAIN_SHORT_LST[trial-1]
    pth_eval_lst = cfg.PTH_EVAL_LST[trial-1]
    train_dir = cfg.DIR_DATA_FRINGE
    eval_dir = cfg.DIR_DATA_FRINGE
    #train_dir = cfg.DIR_DATA_MASKED
    #eval_dir = cfg.DIR_DATA_EVAL

    with tf.Graph().as_default():
        run_training(pth_train_lst, pth_eval_lst, train_dir, eval_dir, tag='rgb')

    with tf.Graph().as_default():
        run_training(pth_train_lst, pth_eval_lst, train_dir, eval_dir, tag='dep')


if __name__ == '__main__':
    tf.app.run(main)
