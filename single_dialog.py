"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils_dialog import load_task, vectorize_data, parse_kb
from sklearn import cross_validation, metrics
from memn2n import MemN2N_Dialog
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
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks-converted/", "Directory containing bAbI-dialog tasks")
tf.flags.DEFINE_string("opt", "sgd_anneal", "Optimizer, 'sgd_anneal' or 'adam' [sgd_anneal]")
tf.flags.DEFINE_integer("anneal_period", 25, "Anneal period [25]")
tf.flags.DEFINE_float("anneal_ratio", 0.5, "Anneal ratio [0.5]")
tf.flags.DEFINE_float("random_time", 0.1, "Random time [0.1]")
tf.flags.DEFINE_boolean("linear_start", False, "Linear start [False]")
tf.flags.DEFINE_integer("ls_epoch", 20, "Linear start ending epoch [20]")
tf.flags.DEFINE_string("weight_tying", "adj", "Weight tying, adjacent 'adj' or layer-wise 'lw' [adj]")
tf.flags.DEFINE_boolean("oov", False, "Test on the OOV set [False]")
tf.flags.DEFINE_boolean("match", False, "Use the match features [False]")
tf.flags.DEFINE_string("kb_file", "data/dialog-bAbI-tasks-converted/dialog-babi-kb-all.txt", "KB file path")
FLAGS = tf.flags.FLAGS

def get_temporal_encoding(d, random_time=0.):
    te = []
    for i in range(len(d)):
        l = int(np.sign(d[i].sum(axis=1)).sum())
        temporal_encoding = np.zeros(d.shape[1])
        if l != 0:
            if random_time > 0.:
                nblank = np.random.randint(0, np.ceil(l * random_time) + 1)
                rt = np.random.permutation(l + nblank) + 1 # +1: permutation starts from 0
                rt = np.vectorize(lambda x: d.shape[1] if x > d.shape[1] else x)(rt)
                temporal_encoding[:l] = np.sort(rt[:l])[::-1]
            else:
                temporal_encoding[:l] = np.arange(l, 0, -1)
        te.append(temporal_encoding)
    return te

def get_answer_dict(data):
    ans_dict = {}
    for d in data:
        story, question, answer = d
        a = ' '.join(answer)
        if a not in ans_dict:
            ans_dict[a] = len(ans_dict)
    return ans_dict

kb_types = [
    'R_cuisine',
    'R_location',
    'R_price',
    'R_rating',
    'R_phone',
    'R_address',
    'R_number',
]

def get_kb_type_idx(t):
    assert t in kb_types
    return kb_types.index(t)

def get_kb_type(kb, word):
    for t, v in kb.iteritems():
        if word in v:
            return t
    return None

def find_match_in_story(word, story):
    for s in story:
        if word in s:
            return True
    return False

def create_match_features(data, idx2ans, kb):
    ret = np.zeros((len(data), 7, len(idx2ans)))
    for i, (story, _, _) in enumerate(data):
        for j in range(len(idx2ans)):
            a = idx2ans[j].split(' ')
            m = np.zeros(7)
            for w in a:
                kb_type = get_kb_type(kb, w)
                if kb_type and find_match_in_story(w, story):
                    m[get_kb_type_idx(kb_type)] = 1
            ret[i, :, j] = m
    return ret

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
    
    kb = parse_kb(FLAGS.kb_file)
    
    # task data
    if FLAGS.task_id != 6:
        train, dev, test, testOOV = load_task(FLAGS.data_dir, FLAGS.task_id)
        if FLAGS.oov:
            logger.info("Using the OOV set")
            test = testOOV
    else:
        train, dev, test = load_task(FLAGS.data_dir, FLAGS.task_id)
        if FLAGS.oov:
            logger.warning("No OOV set for dialog task 6, using the test set instead")
    data = train + dev + test
        
    ans2idx = get_answer_dict(data)
    idx2ans = dict(zip(ans2idx.values(), ans2idx.keys()))
    
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    idx_word = dict(zip(word_idx.values(), word_idx.keys()))
    
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(FLAGS.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position
    
    # create answer n-hot matrix (|V| x |C|)
    answer_n_hot = np.zeros((vocab_size, len(ans2idx)))
    for ans_it in range(len(idx2ans)):
        ans = idx2ans[ans_it]
        n_hot = np.zeros((vocab_size, ))
        for w in ans.split(' '):
            assert w in word_idx
            n_hot[word_idx[w]] = 1
        answer_n_hot[:, ans_it] = n_hot
        
    logger.info("Longest sentence length %d" % sentence_size)
    logger.info("Longest story length %d" % max_story_size)
    logger.info("Average story length %d" % mean_story_size)
    
    # train/validation/test sets
    trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size, ans2idx)
    valS, valQ, valA = vectorize_data(dev, word_idx, sentence_size, memory_size, ans2idx)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size, ans2idx)

    trainM, valM, testM = None, None, None
    
    if FLAGS.match:
        logger.info("Building match features for training set ...")
        trainM = create_match_features(train, idx2ans, kb)
        logger.info("Done")
        logger.info("Building match features for validation set ...")
        valM = create_match_features(dev, idx2ans, kb)
        logger.info("Done")
        logger.info("Building match features for test set ...")
        testM = create_match_features(test, idx2ans, kb)
        logger.info("Done")
    
    logger.info("Training story set shape " + str(trainS.shape))
    logger.info("Training question set shape " + str(trainQ.shape))
    logger.info("Training answer set shape " + str(trainA.shape))
    
    logger.info("Validation story set shape " + str(valS.shape))
    logger.info("Validation question set shape " + str(valQ.shape))
    logger.info("Validation answer set shape " + str(valA.shape))
    
    logger.info("Test story set shape " + str(testS.shape))
    logger.info("Test question set shape " + str(testQ.shape))
    logger.info("Test answer set shape " + str(testA.shape))
    
#     tsetOOVS, testOOVQ, testOOVA = None, None, None
#     if FLAGS.task_id != 6:
#         testOOVS, testOOVQ, testOOVA = vectorize_data(testOOV, word_idx, sentence_size, memory_size, ans2idx)
#         logger.info("Test OOV story set shape " + str(testOOVS.shape))
#         logger.info("Test OOV question set shape " + str(testOOVQ.shape))
#         logger.info("Test OOV answer set shape " + str(testOOVA.shape))
    
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
        model = MemN2N_Dialog(
                    batch_size, 
                    vocab_size, 
                    sentence_size, 
                    memory_size, 
                    FLAGS.embedding_size,
                    answer_n_hot,
                    match=FLAGS.match,
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
                m = trainM[start:end] if FLAGS.match else None
                temporal = get_temporal_encoding(s, random_time=FLAGS.random_time)
                cost_t = model.batch_fit(s, q, a, temporal, linear_start, m)
                total_cost += cost_t
    
            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    m = trainM[start:end] if FLAGS.match else None
                    temporal = get_temporal_encoding(s, random_time=0.0)
                    pred = model.predict(s, q, temporal, linear_start, m)
                    train_preds += list(pred)
    
                val_preds = model.predict(valS, valQ, get_temporal_encoding(valS, random_time=0.0), linear_start, valM)
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
    
        test_preds = model.predict(testS, testQ, get_temporal_encoding(testS, random_time=0.0), linear_start, testM)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        logger.info(" ".join(sys.argv))
        logger.info("Last Training Accuracy: %f" % last_train_acc)
        logger.info("Last Validation Accuracy: %f" % last_val_acc)
        logger.info("Testing Accuracy: %f" % test_acc)
