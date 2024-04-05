import sys, os
import time
import math
import random
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.97):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        # print("delta:", delta)
        # print("masks[step]:", masks[step])
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns



"""
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]
"""


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :] 


def ppo_update(actor, critic, optimizer_actor, optimizer_critic, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    
    actor_loss_avg = 0.0
    critic_loss_avg = 0.0
    entropy_avg = 0.0
    kl_penalty_avg = 0.0

    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = actor(state)
            value = critic(state)
            
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            # critic_loss = (return_ - value).pow(2).mean()
            critic_loss = F.smooth_l1_loss(value, return_)

            log_ratio = new_log_probs - old_log_probs
            kl_penalty = torch.mean(-log_ratio) + torch.mean(torch.exp(log_ratio) * log_ratio)

            actor_loss_all = actor_loss - 0.001 * entropy + 3.0 * kl_penalty

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            
            actor_loss_all.backward()
            critic_loss.backward()

            optimizer_actor.step()
            optimizer_critic.step()

            actor_loss_avg += actor_loss.item()
            critic_loss_avg += critic_loss.item()
            entropy_avg += entropy.item()
            kl_penalty_avg += kl_penalty.item()
    
    actor_loss_avg = actor_loss_avg/ppo_epochs
    critic_loss_avg = critic_loss_avg/ppo_epochs
    entropy_avg = entropy_avg/ppo_epochs
    kl_penalty_avg = kl_penalty_avg/ppo_epochs

    return actor_loss_avg, critic_loss_avg, entropy_avg, kl_penalty_avg

