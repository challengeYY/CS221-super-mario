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
    def __init__(self, info_size, num_actions, tile_row, tile_col, window_size, optimizer='adam', lr=0.01,
                 decay_step=1000, decay_rate=1, regularization=0, conv=True, save_period=500, gradient_clip=10):
        """
        Initializes your System
        :param stateVectorLength: Length of vector used to represent state and action.
        :param optimizer: Name of optimizer.
        """
        self.regularization = regularization
        self.conv = conv
        self.save_period = save_period
        self.gradient_clip = gradient_clip

        # ==== set up placeholder tokens ========
        self.info_size = info_size
        self.numActions = num_actions
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.window_size = window_size
        self.placeholders = {}
        self.placeholders['tile'] = tf.placeholder(tf.float32, shape=(None, self.tile_row, self.tile_col, 1))
        self.placeholders['info'] = tf.placeholder(tf.float32, shape=(None, self.info_size))
        self.placeholders['target_q'] = tf.placeholder(tf.float32, shape=(None,))
        self.placeholders['action'] = tf.placeholder(tf.int32, shape=(None,))

        # ==== assemble pieces ====
        self.predicted_Q = {}
        self.scope_vars = {}
        self.prediction_vs = 'prediction_network'
        self.target_vs = 'target_network'
        (self.predicted_Q[self.prediction_vs], self.predicted_prob,
         self.scope_vars[self.prediction_vs]) = self.create_model(self.prediction_vs)
        self.predicted_Q[self.target_vs], _, self.scope_vars[self.target_vs] = self.create_model(self.target_vs)
        self.setup_target_update(self.prediction_vs, self.target_vs)
        self.setup_loss()
        self.setup_train(lr, decay_step, decay_rate, optimizer)

        # ==== set up training/updating procedure ====
        # implement learning rate annealing

        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess = tf.Session()
        self.setup_tensorboard()

    def create_model(self, variable_scope):
        """
        Construct the tf graph.
        """
        with tf.variable_scope(variable_scope, initializer=tf.uniform_unit_scaling_initializer(1.0)):
            if self.conv:
                conv_in = tf.squeeze(tf.one_hot(tf.cast(self.placeholders['tile'] - 1, tf.uint8), 4, axis=-1), axis=3)
                conv_1 = tf.layers.conv2d(conv_in, 8, 5, activation=tf.nn.relu,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.constant_initializer(0))
                pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
                conv_2 = tf.contrib.layers.flatten(
                    tf.layers.conv2d(pool_1, 16, 3, activation=tf.nn.relu,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                         self.regularization),
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0)))
                conv_out = tf.layers.dense(conv_2, 128, activation=tf.nn.relu,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                h_0 = tf.layers.dense(self.placeholders['info'], 16, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
                h_1 = tf.layers.dense(tf.concat([conv_out, h_0], axis=1), 128, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                h_0 = tf.layers.dense(self.placeholders['info'], 32, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
                h_0 = tf.layers.dense(h_0, 32, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
                h_1 = tf.layers.dense(h_0, 32, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.layers.dense(h_1, self.numActions, activation=None,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            return (out, tf.nn.softmax(out),
                    tf.contrib.framework.get_variables(variable_scope))

    def setup_tensorboard(self):
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train',
                                                  self.sess.graph)

    def setup_loss(self):
        """
        Set up your loss computation here
        """
        with vs.variable_scope("loss"):
            indices = self.placeholders['action']
            depth = self.numActions
            action_masks = tf.cast(tf.one_hot(indices, depth), tf.bool)
            reg_loss = tf.losses.get_regularization_loss()
            tf.summary.scalar('Regularization Loss', reg_loss)
            raw_loss = tf.losses.huber_loss(self.placeholders['target_q'],
                                            tf.boolean_mask(self.predicted_Q[self.prediction_vs], action_masks))
            self.loss = raw_loss + reg_loss
            tf.summary.scalar('Q Loss', raw_loss)
            tf.summary.scalar('Total Loss', self.loss)

    def setup_train(self, lr, decay_step, decay_rate, optimizer_name):
        """
        Setup train ops.
        """
        with vs.variable_scope("train"):
            global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            self.lr = tf.train.exponential_decay(lr, global_step, decay_step, decay_rate,
                                                 staircase=True)

            self.global_step = global_step
            optimizer = get_optimizer(optimizer_name)(self.lr)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            grads = [grad_and_var[0] for grad_and_var in grad_and_vars]
            variables = [grad_and_var[1] for grad_and_var in grad_and_vars]
            grads_clipped, self.grad_norm = tf.clip_by_global_norm(grads, self.gradient_clip)
            self.clipped_grad_norm = tf.global_norm(grads_clipped)
            self.train_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=self.global_step)

    def setup_target_update(self, source_scope, target_scope):
        update_target_expr = []
        for var, var_target in zip(sorted(self.scope_vars[source_scope], key=lambda v: v.name),
                                   sorted(self.scope_vars[target_scope], key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        self.update_target_network_ops = tf.group(*update_target_expr)

    def inference_Q(self, variable_scope, info, tile=None):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state: A state which is represented by a list of float32.
        :return Predicted Q value for all actions of given state.
        """
        feed_dict = {self.placeholders['info']: info}
        if self.conv:
            feed_dict[self.placeholders['tile']] = tile
        predicted_Q = self.sess.run(self.predicted_Q[variable_scope], feed_dict=feed_dict)

        return predicted_Q

    def inference_Prob(self, info, tile=None):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state: A state which is represented by a list of float32.
        :return softmax of Q for all actions of given state.
        """
        feed_dict = {self.placeholders['info']: info}
        if self.conv:
            feed_dict[self.placeholders['tile']] = tile
        prob = self.sess.run(self.predicted_prob, feed_dict=feed_dict)
        return prob

    def update_weights(self, infos, actions, target_Qs, tiles=None):
        """
        Update one step on the given state_and_actions batch.
        :param tiles: A list of tiles which is represented by a 2-D array of float32 (width, height).
        :param infos: A list of states which is represented by a list of float32.
        :param actions: A list of indices of action which is represented by a int.
        :param target_Q: A list of r + /gamma V.
        """
        feed_dict = {self.placeholders['info']: infos,
                     self.placeholders['target_q']: target_Qs,
                     self.placeholders['action']: actions}
        if self.conv:
            feed_dict[self.placeholders['tile']] = tiles

        losses = []
        _, loss, summary, global_step = self.sess.run(
            [self.train_op, self.loss, self.merged_summary, self.global_step],
            feed_dict=feed_dict)
        losses.append(loss)
        self.train_writer.add_summary(summary, global_step)
        if not global_step % self.save_period:
            self.save_model('./model')
        return sum(losses) / len(losses)

    def update_target_network(self):
        self.sess.run([self.update_target_network_ops])

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
        self.saver.save(self.sess, model_path + "model.weights", global_step=self.global_step)

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
