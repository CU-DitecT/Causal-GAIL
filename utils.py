
import h5py
import numpy as np
import os
import tensorflow as tf

from garage.misc import logger
from garage.tf.policies import GaussianMLPPolicy

import transgail
from transgail.critic.critic import WassersteinCritic
from transgail.misc.datasets import CriticDataset
from transgail.core.models import CriticNetwork
from transgail.core.models import ObservationActionMLP, ObservationActionMLPBound
from transgail.baselines.gaussian_mlp_baseline import GaussianMLPBaseline


def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def str2bool(v):
    if v.lower() == 'true':
        return True
    return False


def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std

"""
Some code is borrowed from:
1. https://github.com/sisl/ngsim_env
"""

def build_critic(args, data, env, writer=None):

    if args.use_critic_replay_memory:
        critic_replay_memory = transgail.misc.utils.KeyValueReplayMemory(maxsize=3 * args.batch_size)
    else:
        critic_replay_memory = None

    critic_dataset = CriticDataset(
        data,
        replay_memory=critic_replay_memory,
        flat_recurrent=args.policy_recurrent,
        batch_size=args.critic_batch_size,
    )

    if args.bound_critic:
        critic_network = ObservationActionMLPBound(
            name='critic',
            hidden_layer_dims=args.critic_hidden_layer_dims,
            dropout_keep_prob=args.critic_dropout_keep_prob,
            score_bound=args.score_bound,
        )
    else:
        critic_network = ObservationActionMLP(
            name='critic',
            hidden_layer_dims=args.critic_hidden_layer_dims,
            dropout_keep_prob=args.critic_dropout_keep_prob,
        )
    
    if args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps':
        critic = WassersteinCritic(
            obs_dim=7,  # fixed value for the specific env, LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps
            act_dim=1,  # fixed value for the specific env, LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps
            dataset=critic_dataset,
            network=critic_network,
            gradient_penalty=args.gradient_penalty,
            optimizer=tf.train.AdamOptimizer(args.critic_learning_rate, beta1=.5, beta2=.9),
            n_train_epochs=args.n_critic_train_epochs,
            summary_writer=writer,
            grad_norm_rescale=args.critic_grad_rescale,
            verbose=2,
            debug_nan=True,
        )
    elif args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps':
        critic = WassersteinCritic(
            obs_dim=11,  # fixed value for the specific env, LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps
            act_dim=1,  # fixed value for the specific env, LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps
            dataset=critic_dataset,
            network=critic_network,
            gradient_penalty=args.gradient_penalty,
            optimizer=tf.train.AdamOptimizer(args.critic_learning_rate, beta1=.5, beta2=.9),
            n_train_epochs=args.n_critic_train_epochs,
            summary_writer=writer,
            grad_norm_rescale=args.critic_grad_rescale,
            verbose=2,
            debug_nan=True,
        )
    else:
        critic = WassersteinCritic(
            obs_dim=3,  # fixed value for the specific env, LeaderFollowerEnv
            act_dim=1,  # fixed value for the specific env, LeaderFollowerEnv
            dataset=critic_dataset,
            network=critic_network,
            gradient_penalty=args.gradient_penalty,
            optimizer=tf.train.AdamOptimizer(args.critic_learning_rate, beta1=.5, beta2=.9),
            n_train_epochs=args.n_critic_train_epochs,
            summary_writer=writer,
            grad_norm_rescale=args.critic_grad_rescale,
            verbose=2,
            debug_nan=True,
        )
    return critic


def build_policy(args, env, latent_sampler=None):
    if args.policy_hidden_nonlinearity == 'tanh':
        policy_hidden_nonlinearity = tf.nn.tanh
    elif args.policy_hidden_nonlinearity == 'relu':
        policy_hidden_nonlinearity = tf.nn.relu
    elif args.policy_hidden_nonlinearity == 'leaky_relu':
        policy_hidden_nonlinearity = tf.nn.leaky_relu
    else:
        raise TypeError("Please choose the correct policy_hidden_nonlinearity.")

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=args.policy_mean_hidden_layer_dims,
        std_hidden_sizes=args.policy_std_hidden_layer_dims,
        std_hidden_nonlinearity=policy_hidden_nonlinearity,
        hidden_nonlinearity=policy_hidden_nonlinearity,
        adaptive_std=True,
        output_nonlinearity=None,
        learn_std=True
    )
    return policy


def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)


def set_up_experiment(
        exp_name,
        phase,
        exp_home='../data/experiments/',
        snapshot_gap=5):
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    log_dir = os.path.join(phase_dir, 'log')
    maybe_mkdir(log_dir)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('gap')
    logger.set_snapshot_gap(snapshot_gap)
    log_filepath = os.path.join(log_dir, 'log.txt')
    logger.add_text_output(log_filepath)
    return exp_dir
