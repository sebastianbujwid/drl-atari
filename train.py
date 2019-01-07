#!/usr/bin/env python3

import argparse
import logging
import yaml
import gym
import os
import shutil
import tensorflow as tf
import numpy as np

from drl_atari.models import A2C, DQN
from drl_atari import utils
from drl_atari.multiple_sync_env import MultipleSyncEnv
from drl_atari.experience_replay import ExperienceReplay

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    return parser.parse_args()


def prepare_output_dir(cfg, config_file):
    output_dir = utils.output_dir(cfg)
    if os.path.exists(output_dir):
        raise ValueError("Output directory '%s' already exits!" % output_dir)
    os.makedirs(output_dir)
    shutil.copy2(config_file, output_dir)
    utils.save_code_info(output_dir)


def main(args):

    config_file = args.config_file

    log.info("Using '" + str(config_file) + "' as a config file.")
    with open(config_file, 'r') as yml_file:
        cfg = yaml.load(yml_file)
    prepare_output_dir(cfg, config_file)

    env = gym.make(cfg['env_id'])
    model_input_state_shape, model_output_action_shape = utils.in_out_shapes(cfg, env)

    if cfg['model_class'] == 'DQN':

        num_state_imgs = model_input_state_shape[-1]
        update_frequency = cfg['update_frequency']
        replay_start_size = cfg['replay_start_size']
        batch_size = cfg['batch_size']

        dqn = DQN(cfg, input_shape=model_input_state_shape, output_shape=model_output_action_shape)
        dqn.build()

        experience_replay = ExperienceReplay(max_size=cfg['max_replay_memory_size'])

        with tf.Session() as sess:

            dqn.set_session(sess)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            dqn.update_target_network()

            done = True
            episode = 0
            total_reward = 0.0
            total_actions = 0

            while True:

                if done:
                    if episode > 0:
                        log.info('Episode: ' + str(episode))
                        log.info('Total reward: ' + str(total_reward))
                        dqn.report_score(total_reward)

                    episode += 1
                    total_reward = 0.0
                    ob = env.reset()
                    observations = [ob]
                    for _ in range(num_state_imgs - 1):
                        ob, r, done, _ = env.step(env.action_space.sample())
                        observations.append(ob)
                    state = utils.convert_observation_list(model_input_state_shape, observations)

                exploration_prob = dqn.get_exploration_prob(total_actions)
                explore_randomly = np.random.rand() < exploration_prob or experience_replay.size() < replay_start_size
                if explore_randomly:
                    action = env.action_space.sample()
                else:
                    action = dqn.predict_act(state)[0]
                ob, r, done, _ = env.step(action)

                total_reward += r
                r = np.clip(r, -1., 1.)

                conv_ob = utils.convert_observation(model_input_state_shape, ob)
                next_state = np.concatenate([state, conv_ob], axis=-1)[:, :, :, -num_state_imgs:]

                experience_replay.add(state, action, r, done, next_state)

                if total_actions % update_frequency == 0 and experience_replay.size() >= replay_start_size:
                    rand_experiences = experience_replay.sample(batch_size)
                    states, actions, rewards, dones, next_states = zip(*rand_experiences)
                    states = np.array(states).reshape((batch_size,) + model_input_state_shape)
                    actions = np.array(actions).reshape((batch_size,))
                    rewards = np.array(rewards).reshape((batch_size,))
                    dones = np.array(dones).reshape((batch_size,))
                    next_states = np.array(next_states).reshape((batch_size,) + model_input_state_shape)

                    dqn.update_params(states, actions, rewards, dones, next_states)

                total_actions += 1
                state = next_state

    elif cfg['model_class'] == 'A2C':

        initial_lr = np.exp(np.random.uniform(np.log(cfg['min_initial_lr']), np.log(cfg['max_initial_lr'])))
        log.info('Random initial learning rate: %f' % initial_lr)

        num_agent_steps = cfg['num_agent_steps']
        num_agents = cfg['num_agents']
        gamma = cfg['gamma']
        batch_size = num_agents * num_agent_steps

        a2c = A2C(cfg, input_shape=model_input_state_shape, output_shape=model_output_action_shape)

        def pred_actions_and_values(states):
            inputs = np.array(states).reshape((num_agents, ) + model_input_state_shape)
            actions_probs, values = a2c.predict_action_probs_and_values(inputs)
            actions = [np.random.choice(prob.size, p=prob) for prob in actions_probs]
            return actions, values

        def discounted_cum_rewards(rewards, next_states_values, gamma):
            R = next_states_values
            disc_cum_rewards_inverted = []
            for r in rewards[::-1]:
                R = r + gamma * R
                disc_cum_rewards_inverted.append(R)
            dis_cum_rewards = disc_cum_rewards_inverted[::-1]
            return dis_cum_rewards

        def report_score(score):
            log.info('Score: %f' % score)
            a2c.report_score(score)

        env = MultipleSyncEnv(state_shape=model_input_state_shape, num_agents=num_agents)
        states = env.make(cfg['env_id'])

        a2c.build(initial_lr)

        with tf.Session() as sess:

            a2c.set_session(sess)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            while True:
                states, actions, rewards, values, masks_valid, not_terminated, next_states =\
                    env.step(states, num_steps=num_agent_steps,
                             predict_actions_func=pred_actions_and_values,
                             report_score_func=report_score)

                m_states = np.array(states).reshape((batch_size, ) + model_input_state_shape)
                m_actions = np.array(actions).reshape((batch_size, ))
                m_values = np.array(values).reshape((batch_size, ))
                m_masks_valid = np.array(masks_valid).reshape((batch_size, ))

                m_next_states = np.array(next_states).reshape((num_agents, ) + model_input_state_shape)
                next_states_values = a2c.predict_values(m_next_states)
                next_states_values = next_states_values * not_terminated
                disc_cum_rewards = discounted_cum_rewards(rewards, next_states_values, gamma)
                m_disc_cum_rewards = np.array(disc_cum_rewards).reshape((batch_size, ))

                a2c.update_params(m_states, m_actions, m_disc_cum_rewards, m_values, m_masks_valid)
                states = next_states

    else:
        raise ValueError("model_class: %s not supported" % cfg['model_class'])


if __name__ == "__main__":
    main(parse_args())
