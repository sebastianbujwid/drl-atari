import numpy as np
import gym

from drl_atari import utils


class MultipleSyncEnv:

    def __init__(self, state_shape, num_agents=1):
        self.state_shape = state_shape
        self.num_state_imgs = self.state_shape[-1]
        self.num_agents = num_agents
        self.env_id = None

        self.total_scores = np.zeros((self.num_agents, ))

    def convert_observations(self, observations):
        states = [utils.convert_observation(self.state_shape, img) for img in observations]
        return states

    def append_observations(self, states, new_observations):
        new_states = [np.concatenate([state, obs], axis=-1)[:, :, :, -self.num_state_imgs:]
                      for state, obs in zip(states, new_observations)]
        return new_states

    def make(self, env_id):
        self.env_id = env_id
        self.envs = [gym.make(self.env_id) for _ in range(self.num_agents)]

        observations = [env.reset() for env in self.envs]
        states = self.convert_observations(observations)
        for _ in range(self.num_state_imgs - 1):
            outcomes = map(lambda env: env.step(env.action_space.sample()), self.envs)
            obs, r, dones, _ = zip(*outcomes)
            obs = self.convert_observations(observations)
            next_states = self.append_observations(states, obs)
            states = next_states

        return states

    def step(self, states, num_steps, predict_actions_func, report_score_func):
        l_states = []
        l_actions = []
        l_rewards = []
        l_values = []
        num_agents = len(states)
        l_masks_valid = []

        agents_masks_valid = np.array([True] * num_agents)

        for _ in range(num_steps):
            actions, values = predict_actions_func(states)
            outcomes = map(lambda env_act: env_act[0].step(env_act[1]), zip(self.envs, actions))
            next_obs, rewards, dones, _ = zip(*outcomes)
            self.total_scores += rewards
            rewards = np.clip(rewards, -1., 1.)

            dones_np = np.array(dones)
            masks_valid = np.array(dones_np) == False

            if not np.alltrue(masks_valid):
                for i, valid in enumerate(masks_valid):
                    if valid:
                        continue
                    self.envs[i].reset()
                    report_score_func(self.total_scores[i])
                    self.total_scores[i] = 0.

            next_obs = self.convert_observations(next_obs)
            next_states = self.append_observations(states, next_obs)

            l_states.append(states)
            l_actions.append(actions)
            l_rewards.append(rewards)
            l_values.append(values)
            l_masks_valid.append(agents_masks_valid)

            states = next_states
            agents_masks_valid = agents_masks_valid * masks_valid

        not_terminated = agents_masks_valid
        return l_states, l_actions, l_rewards, l_values, l_masks_valid, not_terminated, next_states
