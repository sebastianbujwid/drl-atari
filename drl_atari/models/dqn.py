import os
import logging
import numpy as np
import tensorflow as tf

from drl_atari import utils

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DQN:

    def __init__(self, cfg, input_shape, output_shape):
        self.cfg = cfg
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.sess = None
        self.saver = None
        self.summary_writer = None

        self.total_updates = 0
        self.total_num_actions = 0
        self.new_updates = 0
        self.experiences = []

    def conv2d(self, x, filters, kernel_size, strides, name):
        return tf.layers.conv2d(x, filters, kernel_size, strides, padding='same',
                                activation=tf.nn.relu, name=name)

    def _add_experience(self, experience):
        self.experiences.append(experience)
        num_elements_to_remove = len(self.experiences) - self.cfg['replay_memory_size']
        if num_elements_to_remove > 0:
            del self.experiences[:num_elements_to_remove]

    def update_target_network(self):
        self.sess.run(self.update_target_op)
        log.info("Target network updated!")

    def report_score(self, score):
        self.summary_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag="score", simple_value=score)]), self.total_updates)

    def get_exploration_prob(self, total_num_actions):
        ratio_of_frames = total_num_actions / self.cfg['final_exploration_frame']
        explore_prob_diff = self.cfg['initial_exploration_prob'] - self.cfg['final_exploration_prob']
        exploration_prob = self.cfg['initial_exploration_prob'] - ratio_of_frames * explore_prob_diff
        exploration_prob = max(exploration_prob, self.cfg['final_exploration_prob'])
        return exploration_prob

    def qnetwork(self, scope):

        with tf.variable_scope(scope):
            input_op = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='input')

            x = (input_op / 255. - 0.5) * 2.
            x = self.conv2d(x, filters=32, kernel_size=(8, 8), strides=[4, 4], name='conv1')
            x = self.conv2d(x, filters=64, kernel_size=(4, 4), strides=[2, 2], name='conv2')
            x = self.conv2d(x, filters=64, kernel_size=(3, 3), strides=[1, 1], name='conv3')

            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])
            x = tf.layers.dense(x, units=512, activation=tf.nn.relu, name="dense")
            state_actions_values = tf.layers.dense(x, units=self.output_shape[0], name='state_actions_values')

        return input_op, state_actions_values

    def build(self):

        # Target Q-Network
        self.target_input_op, self.target_state_action_values = self.qnetwork(scope='target')

        # Q-Network
        self.input_op, self.state_action_values = self.qnetwork(scope='main')

        self.target_value = tf.placeholder(tf.float32, shape=(None,), name='target_value')
        self.action_ohe_op = tf.placeholder(tf.float32, shape=(None, self.output_shape[0]), name='action_ohe')
        q_value = tf.reduce_sum(self.state_action_values * self.action_ohe_op, axis=1)
        diff_op = self.target_value - q_value
        error_op = diff_op
        self.loss_op = tf.reduce_sum(tf.square(error_op))

        # optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.cfg['optimizer'] == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(self.cfg['learning_rate'],
                                                  decay=self.cfg['rmsprop_decay'],
                                                  momentum=self.cfg['momentum'],
                                                  centered=self.cfg['rmsprop_centered'],
                                                  epsilon=0.01)
        else:
            raise ValueError("Optimizer '%s' not supported!" % self.cfg['optimizer'])

        target_vars = tf.trainable_variables(scope='target')
        main_vars = tf.trainable_variables(scope='main')

        self.update_target_op = []
        for target, var in zip(target_vars, main_vars):
            self.update_target_op.append(target.assign(var))

        # grads, vars = zip(*optimizer.compute_gradients(self.loss_op, main_vars))
        # grads_clipped, _ = tf.clip_by_global_norm(grads, self.cfg['max_grad_norm'])
        # Clipping by value seems to give better results
        grads, vars = zip(*optimizer.compute_gradients(self.loss_op, main_vars))
        grads_clipped = [grad if grad is None
                         else tf.clip_by_value(grad, self.cfg['min_grad'], self.cfg['max_grad'])
                         for grad in grads]

        self.update_op = optimizer.apply_gradients(zip(grads_clipped, vars),
                                                   global_step=self.global_step)

        self.summaries_scalar = tf.summary.merge([
            tf.summary.scalar("loss", self.loss_op),
            tf.summary.scalar("global_step", self.global_step)
        ])

        self.summaries_histogram = tf.summary.merge([
            tf.summary.histogram("target_value", self.target_value),
            tf.summary.histogram("state_action_value", q_value),
            tf.summary.histogram("diff", diff_op)
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

    def load_weights(self, weights_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_path)

    def update_params(self, states, actions, rewards, dones, next_states):
        next_states_qs = self.sess.run(self.target_state_action_values, feed_dict={self.target_input_op: next_states})
        next_states_qs_max = next_states_qs.max(axis=-1)
        next_states_qs_max = next_states_qs_max * (dones == False)

        targets = rewards + self.cfg['gamma'] * next_states_qs_max

        if self.total_updates % 4000 == 0:
            summaries_ops = [self.summaries_scalar, self.summaries_histogram, self.summaries_grad]
        else:
            summaries_ops = [self.summaries_scalar]

        _, summary = self.sess.run(
            [self.update_op, summaries_ops],
            feed_dict={
                self.input_op: states,
                self.target_value: targets,
                self.action_ohe_op: utils.ohe(actions, self.output_shape[0])
            }
        )

        [self.summary_writer.add_summary(x, self.total_updates) for x in summary]

        self.total_updates += 1
        self.new_updates += 1

        if self.total_updates % self.cfg['save_model_frequency'] == 0 and self.total_updates > 1:
            dir = os.path.join(os.path.join(utils.output_dir(self.cfg), self.cfg['models_dir']))
            path = os.path.join(dir, 'dqn')
            self.saver.save(self.sess, path, global_step=self.global_step)

        if self.new_updates >= self.cfg['target_update_frequency']:
            self.update_target_network()
            self.new_updates = 0

    def predict_act(self, inputs):
        state_action_vals = self.sess.run(self.state_action_values, feed_dict={self.input_op: inputs})
        action = np.argmax(state_action_vals, axis=-1)
        return action
