"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

import sys
import logging

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("opt", "sgd_anneal", "Optimizer, 'sgd_anneal' or 'adam' [sgd_anneal]")
tf.flags.DEFINE_integer("anneal_period", 25, "Anneal period [25]")
tf.flags.DEFINE_float("anneal_ratio", 0.5, "Anneal ratio [0.5]")
tf.flags.DEFINE_float("random_time", 0.1, "Random time [0.1]")
tf.flags.DEFINE_boolean("linear_start", False, "Linear start [False]")
tf.flags.DEFINE_integer("ls_epoch", 20, "Linear start ending epoch [20]")
tf.flags.DEFINE_string("weight_tying", "adj", "Weight tying, adjacent 'adj' or layer-wise 'lw' [adj]")
FLAGS = tf.flags.FLAGS

def get_temporal_encoding(d, random_time=0.):
    te = []
    for i in range(len(d)):
        l = int(np.sign(d[i].sum(axis=1)).sum())
        temporal_encoding = np.zeros(d.shape[1])
        if random_time > 0.:
            nblank = np.random.randint(0, np.ceil(l * random_time) + 1)
            rt = np.random.permutation(l + nblank) + 1 # +1: permutation starts from 0
            rt = np.vectorize(lambda x: d.shape[1] if x > d.shape[1] else x)(rt)
            temporal_encoding[:l] = np.sort(rt[:l])[::-1]
        else:
            temporal_encoding[:l] = np.arange(l, 0, -1)
        te.append(temporal_encoding)
    return te

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(" ".join(sys.argv))
    logger.info("Started Task: %d" % FLAGS.task_id)
    
    # task data
    train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
    data = train + test
    
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(FLAGS.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position
    
    logger.info("Longest sentence length %d" % sentence_size)
    logger.info("Longest story length %d" % max_story_size)
    logger.info("Average story length %d" % mean_story_size)
    
    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)
    
    logger.info("Training set shape " + str(trainS.shape))
    
    # params
    n_train = trainS.shape[0]
    n_test = testS.shape[0]
    n_val = valS.shape[0]
    
    logger.info("Training Size %d" % n_train)
    logger.info("Validation Size %d" % n_val)
    logger.info("Testing Size %d" % n_test)
    
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    val_labels = np.argmax(valA, axis=1)
    
    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    
    global_step = None
    optimizer = None
    
    if FLAGS.opt == 'sgd_anneal':
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
                           FLAGS.learning_rate, 
                           global_step,
                           FLAGS.anneal_period * (n_train / batch_size), 
                           FLAGS.anneal_ratio, 
                           staircase=True
                        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    
    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]
    
    linear_start = FLAGS.linear_start
    last_train_acc, last_val_acc = None, None
    
    with tf.Session() as sess:
        model = MemN2N(
                    batch_size, 
                    vocab_size, 
                    sentence_size, 
                    memory_size, 
                    FLAGS.embedding_size, 
                    session=sess,
                    weight_tying=FLAGS.weight_tying,
                    hops=FLAGS.hops, 
                    max_grad_norm=FLAGS.max_grad_norm, 
                    optimizer=optimizer,
                    global_step=global_step
                )
        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            if t > FLAGS.ls_epoch:
                linear_start = False
            
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                temporal = get_temporal_encoding(s, random_time=FLAGS.random_time)
                cost_t = model.batch_fit(s, q, a, temporal, linear_start)
                total_cost += cost_t
    
            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    temporal = get_temporal_encoding(s, random_time=0.0)
                    pred = model.predict(s, q, temporal, linear_start)
                    train_preds += list(pred)
    
                val_preds = model.predict(valS, valQ, get_temporal_encoding(valS, random_time=0.0), linear_start)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                val_acc = metrics.accuracy_score(val_preds, val_labels)
                
                last_train_acc = train_acc
                last_val_acc = val_acc
                
                logger.info('-----------------------')
                logger.info('Epoch %d' % t)
                logger.info('Total Cost: %f' % total_cost)
                logger.info('Training Accuracy: %f' % train_acc)
                logger.info('Validation Accuracy: %f' % val_acc)
                logger.info('-----------------------')
    
        test_preds = model.predict(testS, testQ, get_temporal_encoding(testS, random_time=0.0), linear_start)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        logger.info(" ".join(sys.argv))
        logger.info("Last Training Accuracy: %f" % last_train_acc)
        logger.info("Last Validation Accuracy: %f" % last_val_acc)
        logger.info("Testing Accuracy: %f" % test_acc)
