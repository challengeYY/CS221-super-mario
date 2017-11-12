from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar, get_minibatch

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
    def __init__(self, stateVectorLength, optimizer='adam', lr=0.01, decay_step=1000, decay_rate=0):
        """
        Initializes your System
        :param stateVectorLength: Length of vector used to represent state and action.
        :param optimizer: Name of optimizer.
        """

        # ==== set up placeholder tokens ========

        # ==== assemble pieces ====
        with tf.variable_scope("QModel", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_model()

        # ==== set up training/updating procedure ====
        # implement learning rate annealing
        global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(lr, global_step, decay_step, decay_rate,
                                        staircase=True)

        self.stateVectorLength = stateVectorLength
        self.global_step = global_step
        optimizer = get_optimizer(optimizer)(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        self.saver = tf.train.Saver(max_to_keep=50)

    def setup_model(self):
        """
        Construct the tf graph.
        """

        rep = self.encoder.encode((qn_embeddings, con_embeddings),
                                  (self.qns_mask_placeholder, self.cons_mask_placeholder),
                                  dropout=self.dropout_placeholder)

        self.start_pred, self.end_pred = self.decoder.decode(rep, self.cons_mask_placeholder,
                                                             dropout=self.dropout_placeholder)
        self.yp = tf.nn.softmax(self.start_pred)
        self.yp2 = tf.nn.softmax(self.end_pred)
        self.setup_loss()

    def setup_loss(self):
        """
        Set up your loss computation here
        """
        with vs.variable_scope("loss"):

    def inference(self, state_and_actions):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        :param state_and_actions: A list of state and actions. Each state action pair is represented by a list of float32.
        :return Predicted Q value of given state and action.
        """
        # padded_train_qns, mask_train_qns, padded_train_cons, mask_train_cons, train_y, train_answers = train_examples
        # train_examples = [padded_train_qns, mask_train_qns, padded_train_cons, mask_train_cons, train_y]
        # prog = Progbar(target=1 + int(len(train_examples[0]) / FLAGS.batch_size))
        # for i, batch in enumerate(get_minibatches(train_examples, FLAGS.batch_size)):
        #     loss, clipped_grad_norm, grad_norm, lr, gs = self.optimize(sess, *batch)
        #     prog.update(i + 1, [("train loss", loss)],
        #                 [("clipped grad norm", clipped_grad_norm), ("grad norm", grad_norm), ("learning rate", lr),
        #                  ("global step", gs)])

        # logging.info("Calculating F1 and EM for 1000 training examples...")
        # train_examples_for_eval = zip(padded_train_qns, mask_train_qns, padded_train_cons, mask_train_cons,
        #                               train_answers)
        # self.evaluate_answer(sess, train_examples_for_eval, rev_vocab, sample=1000)

        # logging.info("Evaluating on validation set...")
        # padded_val_qns, mask_val_qns, padded_val_cons, mask_val_cons, valid_y, val_answers = valid_examples
        # valid_examples = [padded_val_qns, mask_val_qns, padded_val_cons, mask_val_cons, valid_y]
        # prog = Progbar(target=1 + len(valid_examples[0]) / FLAGS.batch_size)
        # total_val_loss = 0
        # for i, batch in enumerate(get_minibatches(valid_examples, FLAGS.batch_size)):
        #     loss = self.validate(sess, *batch)
        #     prog.update(i + 1, [("validation loss", loss[0])])
        #     total_val_loss += loss[1]
        # total_val_loss = total_val_loss / len(valid_examples[0])

        # # evaluate in each epoch
        # logging.info("Calculating F1 and EM for 100 validation examples...")
        # valid_examples_for_eval = zip(padded_val_qns, mask_val_qns, padded_val_cons, mask_val_cons, val_answers)
        # self.evaluate_answer(sess, valid_examples_for_eval, rev_vocab)
        # return total_val_loss

    def update_weights(self, state_and_actions, target_Q):
        """
        Update one step on the given state_and_actions batch.
        :param state_and_actions: A list of state and actions. Each state action pair is represented by a list of float32.
        :param target_Q: A list of r + /gamma V.
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        # tic = time.time()
        # params = tf.trainable_variables()
        # num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        # toc = time.time()
        # logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # # extract the inputs
        # raw_train_examples, raw_val_set = dataset
        # train_qns, train_cons, train_y, train_answer = raw_train_examples
        # val_qns, val_cons, valid_y, val_answer = raw_val_set
        # # preprocess the inputs
        # padded_train_qns, mask_train_qns = preprocess_inputs(train_qns, FLAGS.qn_max_len)
        # padded_train_cons, mask_train_cons = preprocess_inputs(train_cons, FLAGS.con_max_len)
        # padded_val_qns, mask_val_qns = preprocess_inputs(val_qns, FLAGS.qn_max_len)
        # padded_val_cons, mask_val_cons = preprocess_inputs(val_cons, FLAGS.con_max_len)
        # logging.info("Finished preprocessing")
        # train_examples = [padded_train_qns, mask_train_qns, padded_train_cons, mask_train_cons, train_y, train_answer]
        # valid_examples = [padded_val_qns, mask_val_qns, padded_val_cons, mask_val_cons, valid_y, val_answer]
        # # training
        # for epoch in range(FLAGS.epochs):
        #     logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
        #     total_val_loss = self.run_epoch(session, train_examples, valid_examples, rev_vocab)
        #     # save model weights
        #     model_path = FLAGS.train_dir + "/{:%Y%m%d_%H%M%S}".format(
        #         datetime.now()) + "_val_loss_" + str(total_val_loss) + "/"
        #     if not os.path.exists(model_path):
        #         os.makedirs(model_path)
        #     logging.info("Saving model parameters...")
        #     self.saver.save(session, model_path + "model.weights")
