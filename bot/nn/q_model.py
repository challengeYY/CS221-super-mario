from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

FLAGS = tf.app.flags.FLAGS


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class QModel(object):
    def __init__(self, state_size, num_actions, optimizer='adam', lr=0.01, decay_step=1000, decay_rate=0,
                 regularization=0):
        """
        Initializes your System
        :param stateVectorLength: Length of vector used to represent state and action.
        :param optimizer: Name of optimizer.
        """
        self.regularization = regularization

        # ==== set up placeholder tokens ========
        self.stateSize = state_size
        self.numActions = num_actions
        self.placeholders = {}
        self.placeholders['input_state'] = tf.placeholder(tf.float32, shape=(None, self.stateSize))
        self.placeholders['target_q'] = tf.placeholder(tf.float32, shape=(None))

        # ==== assemble pieces ====
        self.setup_model()
        self.setup_loss()
        self.setup_train(lr, decay_step, decay_rate, optimizer)

        # ==== set up training/updating procedure ====
        # implement learning rate annealing

        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess = tf.Session()
        self.setup_tensorboard()

    def setup_model(self):
        """
        Construct the tf graph.
        """
        with tf.variable_scope("QModel", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            h_1 = tf.layers.dense(self.placeholders['input_state'], 256, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_2 = tf.layers.dense(h_1, 256, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_3 = tf.layers.dense(h_2, 256, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_4 = tf.layers.dense(h_3, 256, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_5 = tf.layers.dense(h_4, 128, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_6 = tf.layers.dense(h_5, 128, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_7 = tf.layers.dense(h_6, 64, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_8 = tf.layers.dense(h_7, 64, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_9 = tf.layers.dense(h_8, 32, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.predicted_Q = tf.layers.dense(h_9, self.numActions, activation=None,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.soft_max_selection = tf.nn.softmax(self.predicted_Q)

    def setup_tensorboard(self):
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train',
                                                  self.sess.graph)

    def setup_loss(self):
        """
        Set up your loss computation here
        """
        with vs.variable_scope("loss"):
            self.losses = []
            reg_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('Regularization Loss', reg_loss)
            tf.summary.scalar('Total Loss', tf.losses.get_total_loss())
            for i in range(self.numActions):
                raw_loss = tf.losses.mean_squared_error(self.placeholders['target_q'], self.predicted_Q[:,i])
                self.losses.append(raw_loss + reg_loss)
                tf.summary.scalar('Q Loss of action {}'.format(i), raw_loss)

    def setup_train(self, lr, decay_step, decay_rate, optimizer):
        """
        Setup train ops.
        """
        with vs.variable_scope("train"):
            self.train_ops = []
            global_step = tf.Variable(0, trainable=False)
            self.lr = tf.train.exponential_decay(lr, global_step, decay_step, decay_rate,
                                                 staircase=True)

            self.global_step = global_step
            optimizer = get_optimizer(optimizer)(self.lr)
            for loss in self.losses:
                self.train_ops.append(optimizer.minimize(loss, global_step=global_step))

    def inference_Q(self, state):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state: A state which is represented by a list of float32.
        :return Predicted Q value for all actions of given state.
        """
        predicted_Q = self.sess.run(self.predicted_Q,
                                    feed_dict={self.placeholders['input_state']: state})
        return predicted_Q

    def inference_Prob(self, state):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state: A state which is represented by a list of float32.
        :return softmax of Q for all actions of given state.
        """
        prob = self.sess.run(self.soft_max_selection,
                                    feed_dict={self.placeholders['input_state']: state})
        return prob

    def update_weights(self, states, actions, target_Qs):
        """
        Update one step on the given state_and_actions batch.
        :param state: A list of states which is represented by a list of float32.
        :param actions: A list of indices of action which is represented by a int.
        :param target_Q: A list of r + /gamma V.
        """
        losses = []
        for state, action, target_Q in zip(states, actions, target_Qs):
            _, loss, summary, global_step = self.sess.run(
                [self.train_ops[action], self.losses[action], self.merged_summary, self.global_step],
                feed_dict={self.placeholders['input_state']: [state], self.placeholders['target_q']: [target_Q]})
            losses.append(loss)
            self.train_writer.add_summary(summary, global_step)
            if not global_step % 10000:
                self.save_model('./model')
        return sum(losses) / len(losses)

    def save_model(self, output_path):
        """
        Save the current graph and weights to the output_path.
        :return:
        """
        # save model weights
        model_path = output_path + "/{:%Y%m%d_%H%M%S}".format(
            datetime.now()) + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logging.info("Saving model parameters...")
        self.saver.save(self.sess, model_path + "model.weights")

    def initialize_model(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        init = tf.global_variables_initializer()
        self.sess.run(init)
