#!/usr/bin/env python3

import argparse
import logging
import yaml
import gym
import tensorflow as tf
import numpy as np

from drl_atari.models import A2C, DQN
from drl_atari import utils


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("model_path")
    parser.add_argument("--record_dir", default=None)
    return parser.parse_args()


def main(args):
    config_file = args.config_file

    log.info("Using '" + str(config_file) + "' as a config file.")
    with open(config_file, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    env = gym.make(cfg['env_id'])

    if args.record_dir is not None:
        env = gym.wrappers.Monitor(env, args.record_dir, video_callable=lambda x: True)

    model_input_states_shape, model_output_action_shape = utils.in_out_shapes(cfg, env)
    num_state_imgs = cfg['num_input_images']

    if cfg['model_class'] == 'A2C':
        model = A2C(cfg, input_shape=model_input_states_shape, output_shape=model_output_action_shape)
        model.build(initial_learning_rate=0.0)

    elif cfg['model_class'] == 'DQN':
        model = DQN(cfg, input_shape=model_input_states_shape, output_shape=model_output_action_shape)
        model.build()
    else:
        raise ValueError("model_class: %s not supported!" % cfg['model_class'])

    with tf.Session() as sess:

        model.set_session(sess)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model.load_weights(args.model_path)

        done = True
        next_state = None
        total_reward = 0.0
        episode = 0

        while True:

            env.render()
            state = next_state

            if done:
                if episode > 0:
                    log.info('Episode: ' + str(episode))
                    log.info('Total rewards: ' + str(total_reward))

                episode += 1
                total_reward = 0.0
                ob = env.reset()
                observations = [ob]
                for _ in range(num_state_imgs - 1):
                    ob, r, done, _ = env.step(env.action_space.sample())
                    observations.append(ob)
                next_state = utils.convert_observation_list(model_input_states_shape, observations)
            else:
                action = model.predict_act(state)
                ob, r, done, _ = env.step(action)
                total_reward += r

                conv_ob = utils.convert_observation(model_input_states_shape, ob)
                next_state = np.concatenate([state, conv_ob], axis=-1)[:, :, :, -num_state_imgs:]


if __name__ == "__main__":
    main(parse_args())
