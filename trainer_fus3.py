import tensorflow as tf
import numpy as np
import os, sys, time, shutil, ipdb
import configure as cfg
from utils import common
from architectures import model_fusion3 as model


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 100, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('img_s_raw', cfg.IMG_RAW_S, """Size of a raw square image.""")
tf.app.flags.DEFINE_integer('n_classes', len(cfg.CLASSES), """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 3, """How often to evaluate and write checkpoint.""")
tf.app.flags.DEFINE_integer('feat_len', 4096, """Length of a feature (rgb or dep).""")


#=================================================================================================
def placeholder_inputs(batch_size):
    rgb_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='rgb_placeholder')
    dep_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='dep_placeholder')
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')
    return rgb_ph, dep_ph, labels_ph, keep_prob_ph


def fill_feed_dict(rgb_batch, dep_batch, lbl_batch, rgb_ph, dep_ph, labels_ph, keep_prob_ph, is_training, crop):
    if lbl_batch.shape[0] < FLAGS.batch_size: # zero-padding
        if is_training:
            return None
        else:
            M = FLAGS.batch_size - lbl_batch.shape[0]
            rgb_batch = np.pad(rgb_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
            dep_batch = np.pad(dep_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
            lbl_batch = np.pad(lbl_batch, ((0,M),(0,0)), 'constant', constant_values=0)

    if crop == 'random':
        rgb_batch, dep_batch = common.random_crop_pair(rgb_batch, dep_batch)
    elif crop == 'central':
        rgb_batch = common.central_crop(rgb_batch)
        dep_batch = common.central_crop(dep_batch)

    kp = 0.5 if is_training else 1.0
    feed_dict = {rgb_ph:rgb_batch, dep_ph:dep_batch, labels_ph:lbl_batch, keep_prob_ph: kp}
    return feed_dict


def do_eval(sess, score, eval_correct, rgb_ph, dep_ph, labels_ph, keep_prob_ph, 
        all_rgb, all_dep, all_labels, logfile, tag, step, crop):
    true_count, start_idx = 0, 0
    num_samples = all_labels.shape[0]
    indices = np.arange(num_samples)
    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx: stop_idx]
        is_training = False
        
        fd = fill_feed_dict(
            all_rgb[batch_idx], all_dep[batch_idx], all_labels[batch_idx],
            rgb_ph, dep_ph, labels_ph, keep_prob_ph,
            is_training, crop)
        score_val = sess.run(score, feed_dict=fd)

        # throw out zero padding results
        used_score = score_val[:stop_idx-start_idx]
        used_lbl   = all_labels[start_idx:stop_idx]
        common.write_score(used_score, used_lbl, tag, step)

        # count correct predictions
        for i in range(stop_idx - start_idx):
            if used_score[i].argmax() == used_lbl[i].argmax():
                true_count += 1
        #true_count += sess.run(eval_correct, feed_dict=fd)
        start_idx = stop_idx
        
    precision = true_count*1.0 / num_samples
    common.writer('    Num-samples:%d   Num-correct:%d   Precision:%0.04f', 
            (num_samples, true_count, precision), logfile)
    return precision


#=================================================================================================
def run_training(pth_train_lst, pth_eval_lst, train_dir, eval_dir, tag='fus'):
    logfile = open(os.path.join(cfg.DIR_LOG, 'training_'+tag+'.log'),'w',0)

    # load data-----------------------------------------------------------------
    print 'Loading color and depth models...'
    rgb_model = np.load(cfg.PTH_RGB_MODEL).item()
    dep_model = np.load(cfg.PTH_DEP_MODEL).item()

    print 'Loading lists...'
    train_lst = open(pth_train_lst, 'r').read().splitlines()
    eval_lst = open(pth_eval_lst, 'r').read().splitlines()
    #train_lst = train_lst[:100]; eval_lst = eval_lst[:100] #TODO

    print 'Loading training data...'
    rgb_train_data, train_labels = common.load_images(
            train_lst, train_dir, cfg.EXT_RGB, ccrop=False)
    dep_train_data, _ = common.load_images(
            train_lst, train_dir, cfg.EXT_D, ccrop=False)
    num_train = len(rgb_train_data)

    print 'Loading validation data...'
    rgb_eval_data, eval_labels = common.load_images(
            eval_lst, eval_dir, cfg.EXT_RGB, ccrop=True)
    dep_eval_data, _= common.load_images(
            eval_lst, eval_dir, cfg.EXT_D, ccrop=True)

    
    # tensorflow variables and operations---------------------------------------
    print 'Preparing tensorflow...'
    rgb_ph, dep_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    score = model.inference(rgb_ph, dep_ph, rgb_model, dep_model, keep_prob_ph)
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
        # training phase-------------------------------------------------------
        start_time = time.time()

        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx = 0, 0
        lim = 10
        while start_idx != num_train:
            if start_idx*100.0 / num_train > lim:
                print 'Trained %d / %d' % (start_idx, num_train)
                lim += 10

            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            # fill feed dict
            fd = fill_feed_dict(
                rgb_train_data[batch_idx], dep_train_data[batch_idx], train_labels[batch_idx],
                rgb_ph, dep_ph, labels_ph, keep_prob_ph, 
                is_training=True, crop='random')
            if fd is None: break # throw the incompete batch

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


        # write summary--------------------------------------------------------
        #if step % FLAGS.summary_frequency == 0:
        #    common.writer('Step %d: loss = %.3f (%.3f sec)', (step,total_loss,duration), logfile)
        #    summary_str = sess.run(summary, feed_dict=fd)
        #    summary_writer.add_summary(summary_str, step)
        #    summary_writer.flush()
        #else:
        #    common.writer('Step', step, logfile)


        # write checkpoint------------------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter: #or best_loss<total_loss:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)

            common.writer('  Training data eval:', (), logfile)
            do_eval(
                sess, score, eval_correct, 
                rgb_ph, dep_ph, labels_ph, keep_prob_ph, 
                rgb_train_data, dep_train_data, train_labels, 
                logfile, 'fustrain', step, crop='central')

            common.writer('  Validation data eval:', (), logfile)
            precision = do_eval(
                sess, score, eval_correct, 
                rgb_ph, dep_ph, labels_ph, keep_prob_ph,
                rgb_eval_data, dep_eval_data, eval_labels, 
                logfile, 'fuseval', step, crop=None)
            common.writer('Precision: %.4f', precision, logfile)

            if precision > best_precision: 
                import shutil
                src = os.path.join(cfg.DIR_CKPT, tag+'-'+str(step))
                dst = os.path.join(cfg.DIR_BESTCKPT, tag+'-best')
                shutil.copy2(src, dst)
                shutil.copy2(src+'.meta', dst+'.meta')
                best_precision = precision
        
        
        # early stopping-------------------------------------------------------
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


#=================================================================================================
def main(argv=None):
    trial = 1
    print 'Trial: %d' % trial

    #pth_train_lst = cfg.PTH_TRAIN_LST[trial]
    pth_train_lst = cfg.PTH_TRAIN_SHORT_LST[trial-1]
    pth_eval_lst  = cfg.PTH_EVAL_LST[trial-1]
    train_dir = cfg.DIR_DATA_FRINGE
    eval_dir  = cfg.DIR_DATA_FRINGE

    with tf.Graph().as_default():
        run_training(pth_train_lst, pth_eval_lst, train_dir, eval_dir, tag='fus')
    return


if __name__ == '__main__':
    tf.app.run(main)
