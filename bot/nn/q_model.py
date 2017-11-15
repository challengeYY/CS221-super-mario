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
    def __init__(self, stateVectorLength, optimizer='adam', lr=0.01, decay_step=1000, decay_rate=0, regularization=0):
        """
        Initializes your System
        :param stateVectorLength: Length of vector used to represent state and action.
        :param optimizer: Name of optimizer.
        """
        self.regularization = regularization

        # ==== set up placeholder tokens ========
        self.stateVectorLength = stateVectorLength
        self.placeholders = {}
        self.placeholders['input_state_action'] = tf.placeholder(tf.float32, shape=(None, stateVectorLength))
        self.placeholders['target_q'] = tf.placeholder(tf.float32, shape=(None, 1))

        # ==== assemble pieces ====
        self.setup_model()

        # ==== set up training/updating procedure ====
        # implement learning rate annealing
        global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(lr, global_step, decay_step, decay_rate,
                                             staircase=True)

        self.global_step = global_step
        optimizer = get_optimizer(optimizer)(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess = tf.Session()
        self.setup_tensorboard()

    def setup_model(self):
        """
        Construct the tf graph.
        """
        with tf.variable_scope("QModel", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            h_1 = tf.layers.dense(self.placeholders['input_state_action'], 512, activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_2 = tf.layers.dense(h_1, 256, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_3 = tf.layers.dense(h_2, 256, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_4 = tf.layers.dense(h_3, 256, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_5 = tf.layers.dense(h_4, 128, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_6 = tf.layers.dense(h_5, 128, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_7 = tf.layers.dense(h_6, 64, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_8 = tf.layers.dense(h_7, 64, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            h_9 = tf.layers.dense(h_8, 32, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.predicted_Q = tf.layers.dense(h_9, 1, activation=tf.nn.relu,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.setup_loss()

    def setup_tensorboard(self):
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train',
                                                  self.sess.graph)

    def setup_loss(self):
        """
        Set up your loss computation here
        """
        with vs.variable_scope("loss"):
            raw_loss = tf.losses.mean_squared_error(self.placeholders['target_q'], self.predicted_Q)
            reg_loss = tf.losses.get_regularization_loss()
            self.loss = raw_loss + reg_loss
            tf.summary.scalar('Total Loss', self.loss)
            tf.summary.scalar('Q Loss', raw_loss)
            tf.summary.scalar('Regularization Loss', reg_loss)

    def inference(self, state_and_actions):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state_and_actions: A list of state and actions. Each state action pair is represented by a list of float32.
        :return Predicted Q value of given state and action.
        """
        predicted_Q = self.sess.run(self.predicted_Q,
                                    feed_dict={self.placeholders['input_state_action']: state_and_actions})
        return predicted_Q

    def update_weights(self, state_and_actions, target_Q):
        """
        Update one step on the given state_and_actions batch.
        :param state_and_actions: A list of state and actions. Each state action pair is represented by a list of float32.
        :param target_Q: A list of r + /gamma V.
        """
        predicted_Q, summary, global_step = self.sess.run([self.train_op, self.merged_summary, self.global_step],
                                                          feed_dict={
                                                              self.placeholders[
                                                                  'input_state_action']: state_and_actions,
                                                              self.placeholders['target_q']: target_Q})
        self.train_writer.add_summary(summary, global_step)
        if not global_step % 1000:
            self.save_model('./model')
        return predicted_Q

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
