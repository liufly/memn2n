"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def get_tensorflow_version():
    ver = tf.__version__
    vers = ver.split(".")
    assert len(vers) == 3
    return int(vers[1])

if get_tensorflow_version() >= 11:
    from tensorflow import name_scope as op_scope
else:
    from tensorflow import op_scope

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
#     with tf.op_scope([t], name, "zero_nil_slot") as name:
#     with tf.name_scope(name, "zero_nil_slot", [t]) as name:
    with op_scope(values=[t], name=name, default_name="zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
#     with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
#     with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
    with op_scope(values=[t, stddev], name=name, default_name="add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
        weight_tying="adj",
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        global_step=None,
        encoding=position_encoding,
        session=tf.Session(),
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._weight_tying = weight_tying
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._global_step = global_step
        self._name = name

        self._build_inputs()
        
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        if self._weight_tying == "adj":
            self._build_vars_adj()
            logits = self._inference_adj(
                                self._stories, 
                                self._queries, 
                                self._temporal, 
                                self._linear_start
                    ) # (batch_size, vocab_size)
        elif self._weight_tying == "lw":
            self._build_vars_lw()
            logits = self._inference_lw(
                                self._stories, 
                                self._queries, 
                                self._temporal, 
                                self._linear_start
                    ) # (batch_size, vocab_size)
        else:
            raise # not implemented
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, global_step=self._global_step, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._temporal = tf.placeholder(tf.int32, [None, self._memory_size], name="temproal")
        self._linear_start = tf.placeholder(tf.bool, [], name="linear_start")

    def _build_vars_adj(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])

            self._weight_matrices = []
            self._T_weight_matrices = []
            
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            self._weight_matrices.append(tf.Variable(A, name="A"))
            TA = tf.concat(0, [ nil_word_slot, self._init([self._memory_size, self._embedding_size]) ])
            self._T_weight_matrices.append(tf.Variable(TA, name='TA'))
            
            for i in range(1, self._hops + 1):
                C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
                self._weight_matrices.append(tf.Variable(C, name="C" + str(i)))

                TC = tf.concat(0, [ nil_word_slot, self._init([self._memory_size, self._embedding_size]) ])
                self._T_weight_matrices.append(tf.Variable(TC, name='TC'))

        self._nil_vars = set([mat.name for mat in self._weight_matrices + self._T_weight_matrices])

    def _inference_adj(self, stories, queries, temporal, linear_start):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self._weight_matrices[0], queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            for it_hop in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self._weight_matrices[it_hop], stories)
                m = tf.reduce_sum(m_emb * self._encoding, 2)
                m = m + tf.nn.embedding_lookup(self._T_weight_matrices[it_hop], temporal)
                
                c_emb = tf.nn.embedding_lookup(self._weight_matrices[it_hop + 1], stories)
                c = tf.reduce_sum(c_emb * self._encoding, 2)
                c = c + tf.nn.embedding_lookup(self._T_weight_matrices[it_hop + 1], temporal)
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.cond(linear_start, lambda: dotted, lambda: tf.nn.softmax(dotted))

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(c, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = u[-1] + o_k
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            return tf.matmul(u_k, tf.transpose(self._weight_matrices[-1]))

    def _build_vars_lw(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            self.A = tf.Variable(A, name="A")
            self.B = tf.Variable(B, name="B")
            self.C = tf.Variable(C, name="C")

            TA = tf.concat(0, [ nil_word_slot, self._init([self._memory_size, self._embedding_size]) ])
            self.TA = tf.Variable(TA, name='TA')
            TC = tf.concat(0, [ nil_word_slot, self._init([self._memory_size, self._embedding_size]) ])
            self.TC = tf.Variable(TC, name='TC')

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")
        self._nil_vars = set([self.A.name, self.B.name, self.C.name, self.TA.name, self.TC.name])

    def _inference_lw(self, stories, queries, temporal, linear_start):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            for _ in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb * self._encoding, 2)
                m = m + tf.nn.embedding_lookup(self.TA, temporal)
                
                c_emb = tf.nn.embedding_lookup(self.C, stories)
                c = tf.reduce_sum(c_emb * self._encoding, 2)
                c = c + tf.nn.embedding_lookup(self.TC, temporal)
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.cond(linear_start, lambda: dotted, lambda: tf.nn.softmax(dotted))

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(c, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            return tf.matmul(u_k, self.W)

    def batch_fit(self, stories, queries, answers, temporal, linear_start):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {
            self._stories: stories, 
            self._queries: queries, 
            self._answers: answers, 
            self._temporal: temporal,
            self._linear_start: linear_start
        }
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries, temporal, linear_start):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {
            self._stories: stories, 
            self._queries: queries, 
            self._temporal: temporal,
            self._linear_start: linear_start
        }
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)
