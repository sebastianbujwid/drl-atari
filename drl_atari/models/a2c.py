import tensorflow as tf
import numpy as np
import os

from drl_atari import utils


class A2C:

    def __init__(self, cfg, input_shape, output_shape):
        self.cfg = cfg
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.sess = None
        self.total_updates = 0

        dir = os.path.join(os.path.join(utils.output_dir(self.cfg), self.cfg['models_dir']))
        if not os.path.exists(dir):
            os.makedirs(dir)

        self._bias_initializer = tf.constant_initializer(value=0.)

    def _kernel_initializer(self, scale=np.sqrt(2)):
        return tf.orthogonal_initializer(gain=scale)

    def conv2d(self, x, filters, kernel_size, strides, name):
        return tf.layers.conv2d(x, filters, kernel_size, strides, padding='same',
                                activation=tf.nn.relu, name=name,
                                kernel_initializer=self._kernel_initializer(np.sqrt(2)),
                                bias_initializer=self._bias_initializer)

    def build(self, initial_learning_rate):
        num_actions = self.output_shape[0]
        batch = None
        height, width, channels = self.input_shape

        self.input = tf.placeholder(tf.float32, shape=(batch, height, width, channels), name='input')
        with tf.variable_scope("a2c_model"):

            x = (self.input / 255. - 0.5) * 2.
            if self.cfg['architecture'] == 'a3c_like':
                x = self.conv2d(x, filters=16, kernel_size=[8, 8], strides=[4, 4], name='conv1')
                x = self.conv2d(x, filters=32, kernel_size=[4, 4], strides=[2, 2], name='conv2')
                x_flat = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
                dense = tf.layers.dense(x_flat, units=256, activation=tf.nn.relu, name='dense')
            elif self.cfg['architecture'] == 'dqn_like':
                x = self.conv2d(x, filters=32, kernel_size=(8, 8), strides=[4, 4], name='conv1')
                x = self.conv2d(x, filters=64, kernel_size=(4, 4), strides=[2, 2], name='conv2')
                x = self.conv2d(x, filters=64, kernel_size=(3, 3), strides=[1, 1], name='conv3')
                x_flat = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
                dense = tf.layers.dense(x_flat, units=512, activation=tf.nn.relu, name='dense',
                                        kernel_initializer=self._kernel_initializer(np.sqrt(2)),
                                        bias_initializer=self._bias_initializer)

            self.value = tf.layers.dense(dense, units=1, name='value_func',
                                         kernel_initializer=self._kernel_initializer(scale=1.),
                                         bias_initializer=self._bias_initializer)
            self.value = tf.squeeze(self.value, axis=-1)
            self.action = tf.layers.dense(dense, units=num_actions, name='action',
                                          kernel_initializer=self._kernel_initializer(scale=1.),
                                          bias_initializer=self._bias_initializer)
            self.predict_action = tf.nn.softmax(self.action, name='action_softmax')

        self.cum_reward = tf.placeholder(tf.float32, shape=(batch, ), name='cum_reward')
        self.advantage = tf.placeholder(tf.float32, shape=(batch, ), name='advantage')
        self.action_labels = tf.placeholder(tf.int32, shape=(batch, ), name='action_label')
        self.masks_valid = tf.placeholder(tf.float32, shape=(batch, ), name='masks_valid')

        policy_loss = tf.reduce_mean(self.masks_valid * self.advantage *
                                     tf.nn.sparse_softmax_cross_entropy_with_logits(
                                         logits=self.action, labels=self.action_labels),
                                     name='policy_loss')

        entropy_reg = tf.reduce_mean(self.masks_valid * utils.cat_entropy(self.action),
                                     name='entropy_regularization')

        value_loss = tf.reduce_mean(self.masks_valid * tf.square(self.cum_reward - self.value),
                                    name='value_loss')

        total_loss = self.cfg['policy_loss_weight'] * policy_loss\
            - self.cfg['entropy_reg_value'] * entropy_reg\
            + self.cfg['value_loss_weight'] * value_loss

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step,
                                                   self.cfg['decay_steps'], self.cfg['decay_rate'],
                                                   staircase=True)
        if self.cfg['optimizer'] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=self.cfg['rmsprop_decay'],
                                                  momentum=self.cfg['momentum'], epsilon=1e-05)
        elif self.cfg['optimizer'] == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-05)
        else:
            raise ValueError("Optimizer %s not supported!" % self.cfg['optimizer'])

        trainable_vars = tf.trainable_variables()

        grads, vars = zip(*optimizer.compute_gradients(total_loss, trainable_vars))
        grads_clipped, _ = tf.clip_by_global_norm(grads, self.cfg['max_grad_norm'])

        self.update_op = optimizer.apply_gradients(zip(grads_clipped, vars),
                                                   global_step=self.global_step)

        self.summaries_scalar = tf.summary.merge([
            tf.summary.scalar("learning_rate", learning_rate),
            tf.summary.scalar("mean_policy_loss", tf.reduce_mean(policy_loss)),
            tf.summary.scalar("mean_value_loss", tf.reduce_mean(value_loss)),
            tf.summary.scalar("entropy_reg_loss", tf.reduce_mean(entropy_reg)),
            tf.summary.scalar("total_loss", tf.reduce_mean(total_loss)),
            tf.summary.scalar("global_step", self.global_step)
        ])

        self.summaries_histogram = tf.summary.merge([
            tf.summary.histogram("predicted_actions", tf.argmax(self.predict_action, axis=1)),
            tf.summary.histogram("predict_actions_probs", self.predict_action),
            tf.summary.histogram("value", self.value),
            tf.summary.histogram("cum_rewards", self.cum_reward),
            tf.summary.histogram("action_labels", self.action_labels),
            tf.summary.histogram("advantage", self.advantage)
        ])

        self.summaries_grad = tf.summary.merge([
            [tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in zip(grads, vars)],
            [tf.summary.histogram("%s-grad-clip" % g[1].name, g[0]) for g in zip(grads_clipped, vars)]
        ])

        tf_log_dir = os.path.join(utils.output_dir(self.cfg), self.cfg['logs_dir'])
        self.summary_writer = tf.summary.FileWriter(tf_log_dir, graph=tf.get_default_graph())

        self.saver = tf.train.Saver(
            max_to_keep=self.cfg['max_checkpoints_to_keep'],
            keep_checkpoint_every_n_hours=self.cfg['keep_checkpoint_every_n_hours']
        )

    def set_session(self, sess):
        self.sess = sess

    def update_params(self, states, actions, disc_cum_rewards, values, masks_valid):
        if self.total_updates % 4000 == 0:
            summaries_ops = [self.summaries_scalar, self.summaries_histogram, self.summaries_grad]
        else:
            summaries_ops = [self.summaries_scalar]

        advantages = disc_cum_rewards - values
        _, summary = self.sess.run([self.update_op, summaries_ops], feed_dict={
            self.input: states,
            self.cum_reward: disc_cum_rewards,
            self.action_labels: actions,
            self.advantage: advantages,
            self.masks_valid: masks_valid
        })
        [self.summary_writer.add_summary(x, self.total_updates) for x in summary]

        self.total_updates += 1
        if self.total_updates % self.cfg['save_model_frequency'] == 0 and self.total_updates > 1:
            dir = os.path.join(utils.output_dir(self.cfg), self.cfg['models_dir'])
            path = os.path.join(dir, 'a2c')
            self.saver.save(self.sess, path, global_step=self.global_step)

    def predict_action_probs(self, inputs):
        action_probs = self.sess.run(self.predict_action, feed_dict={self.input: inputs})
        return action_probs

    def predict_act(self, inputs):
        action_probs = self.sess.run(self.predict_action, feed_dict={self.input: inputs})
        action = np.argmax(action_probs, axis=-1)
        return action

    def predict_values(self, inputs):
        values = self.sess.run(self.value, feed_dict={
            self.input: inputs
        })

        return values

    def predict_action_probs_and_values(self, inputs):
        action_probs, values = self.sess.run(
            [self.predict_action, self.value],
            feed_dict={
                self.input: inputs
            }
        )
        return action_probs, values

    def load_weights(self, weights_path):
        self.saver.restore(self.sess, weights_path)

    def report_score(self, score):
        self.summary_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag="score", simple_value=score)]), self.total_updates)
