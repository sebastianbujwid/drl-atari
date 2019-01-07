import gym
import numpy as np
import os
import subprocess
from PIL import Image
import tensorflow as tf


def in_out_shapes(cfg, env):
    if isinstance(env.observation_space, gym.spaces.Box):
        model_input_state_shape = cfg['input_img_shape'] + (cfg['num_input_images'], )
    else:
        raise NotImplementedError("Only Box observation space is supported!")
    if isinstance(env.action_space, gym.spaces.Discrete):
        model_output_action_shape = (env.action_space.n, )
    else:
        raise NotImplementedError("Only Discrete action space is supported!")

    return model_input_state_shape, model_output_action_shape


# from OpenAI implementation
def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def convert_observation_list(input_shape, l_images):
    # TODO - max of current and previous - Atari paper
    _input = np.zeros((1,) + input_shape)
    for i, image in enumerate(l_images):
        img_rescaled = Image.fromarray(image).resize(input_shape[:2])
        img_luminance = np.array(img_rescaled.convert('L'))
        _input[0, :, :, i] = img_luminance
    return _input.astype(np.uint8)


def convert_observation(input_shape, image):
    height, width, channels = input_shape
    _input = np.zeros((1, height, width, 1))
    img_rescaled = Image.fromarray(image).resize(input_shape[:2])
    img_luminance = np.array(img_rescaled.convert('L'))
    _input[0, :, :, 0] = img_luminance
    return _input.astype(np.uint8)


def ohe(x, num_classes):
    assert(x.ndim == 1)
    num_objects = x.shape[0]
    x_ohe = np.zeros((num_objects, num_classes))
    x_ohe[np.arange(num_objects), x] = 1
    return x_ohe


def output_dir(cfg):
    return os.path.join(cfg['output_dir'], cfg['run_name'])


def git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def git_history():
    return subprocess.check_output(['git', 'log'])


def git_diff():
    return subprocess.check_output(['git', 'diff'])


def save_code_info(dir):
    revision_hash = git_revision_hash().decode('utf-8')
    rev_file = os.path.join(dir, 'code_version.txt')
    with open(rev_file, 'w') as f:
        f.write(revision_hash)

    local_changes = git_diff().decode('utf-8')
    diff_file = os.path.join(dir, 'git_diff.txt')
    with open(diff_file, 'w') as f:
        f.write(local_changes)

    log = git_history().decode('utf-8')
    log_file = os.path.join(dir, 'git_log.txt')
    with open(log_file, 'w') as f:
        f.write(log)
