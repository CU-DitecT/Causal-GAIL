"""
GymEnv is no longer available in the garage package

"""

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


from core_net import Actor, Critic, Discriminator
from ppo import ppo_iter, ppo_update, compute_gae

import torch_utils
from utils import str2bool, normalize

from transgail.envs.leader_follower_env import LeaderFollowerRewardEngEnvUC_Imitator, \
                                                LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps, \
                                                LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps


# use_cuda = torch.cuda.is_available()
use_cuda = False
device   = torch.device("cuda" if use_cuda else "cpu")




def expert_reward_concatenated(args, discriminator, state_action, output_mode="np"):
    # 'linear', 'bce', 'bce_neg', 'ace'
    if args.loss_type == 'linear':
        if output_mode == "np":
            return discriminator(state_action).cpu().data.numpy()
        elif output_mode == "torch":
            return discriminator(state_action)
        else:
            return None
    elif args.loss_type == 'bce':
        if output_mode == "np":
            return -F.logsigmoid(-discriminator(state_action)).cpu().data.numpy()
        elif output_mode == "torch":
            return -F.logsigmoid(-discriminator(state_action))
        else:
            return None
    elif args.loss_type == 'bce_neg':
        if output_mode == "np":
            return F.logsigmoid(-discriminator(state_action)).cpu().data.numpy()
        elif output_mode == "torch":
            return F.logsigmoid(-discriminator(state_action))
        else:
            return None
    else:
        raise TypeError('Please choose a correct loss_type!')


def expert_reward(args, discriminator, state, action, output_mode="np"):
    state = state.cpu().numpy()
    state_action = torch.from_numpy(np.concatenate([state, action], -1)).float()
    
    return expert_reward_concatenated(args, discriminator, state_action, output_mode=output_mode)


def compute_grad_pen(expert_s_a, policy_s_a, discriminator, lambda_=5):
    alpha = torch.rand(expert_s_a.size(0), 1)
    alpha = alpha.expand_as(expert_s_a).to(expert_s_a.device)
    mixup_data = alpha * expert_s_a + (1 - alpha) * policy_s_a
    mixup_data.requires_grad = True

    disc = discriminator(mixup_data)
    ones = torch.ones(disc.size()).to(disc.device)

    grad = autograd.grad(
        outputs=disc,
        inputs=mixup_data,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen


def obtain_one_traj(args, actor: Actor, critic: Critic, env, discriminator: Discriminator, max_steps=200):
    
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    state = env.reset()
    done = False
    idx_step = 0

    while not done and idx_step < max_steps:
        # print("idx_step:", idx_step)
        idx_step += 1
        state = torch.from_numpy(state).float()

        dist = actor(state)
        value = critic(state)
        
        action = dist.sample()
        action_np = action.cpu().numpy()
        action_np = action_np.astype(dtype=np.float64)
        
        next_state, _, done, _ = env.step(action_np)
        
        reward = expert_reward(args, discriminator, state, action_np)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(torch.unsqueeze(log_prob, dim=0))
        values.append(torch.unsqueeze(value, dim=0))
        rewards.append(torch.unsqueeze(torch.from_numpy(reward).float().to(device), dim=0))
        masks.append(torch.unsqueeze(torch.from_numpy(np.array(1 - int(done))).unsqueeze(-1).to(device), dim=0))

        states.append(torch.unsqueeze(state, dim=0))
        actions.append(torch.unsqueeze(action, dim=0))

        state = next_state


    next_state = torch.from_numpy(next_state).float().to(device)
    next_value = critic(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns    = torch.cat(returns).detach()
    values     = torch.cat(values).detach()
    advantages = returns - values
    advantages = advantages - advantages.mean()
    
    log_probs  = torch.cat(log_probs).detach()
    states     = torch.cat(states)
    actions    = torch.cat(actions)

    return returns, log_probs, values, states, actions, advantages


def obtain_mul_trajs(args, actor: Actor, critic: Critic, env, discriminator: Discriminator, max_steps=200):

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    state = env.reset()
    n_agents = state.shape[0]
    done = [False] * n_agents
    idx_step = 0

    while idx_step < max_steps:
        # print("idx_step:", idx_step)
        idx_step += 1
        state = torch.from_numpy(state).float()
        
        dist = actor(state)
        value = critic(state)
        
        action = dist.sample()
        action_np = action.cpu().numpy()
        action_np = action_np.astype(dtype=np.float64)
        
        next_state, _, done, _ = env.step(action_np)
        
        reward = expert_reward(args, discriminator, state, action_np)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.from_numpy(reward).float().to(device))
        masks.append(torch.from_numpy(1 - np.array(done).astype(int)).unsqueeze(-1).to(device))

        states.append(state)
        actions.append(action)

        state = next_state
    
    next_state = torch.from_numpy(next_state).float().to(device)
    next_value = critic(next_state)
    returns = compute_gae(next_value, rewards, masks, values)
    
    returns    = torch.cat(returns, dim=0).detach()
    values     = torch.cat(values, dim=0).detach()

    advantages = returns - values
    advantages = advantages - advantages.mean()
    
    log_probs  = torch.cat(log_probs, dim=0).detach()
    states     = torch.cat(states, dim=0)
    actions    = torch.cat(actions, dim=0)

    return returns, log_probs, values, states, actions, advantages



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dims", type=int, default=3)
    parser.add_argument("--act_dims", type=int, default=1)

    # logistics
    parser.add_argument("--exp_name", type=str, default='single_gail_1_step_1')
    parser.add_argument("--exp_home", type=str, default='../data/experiments')
    parser.add_argument('--n_envs', type=int, default=50)
    parser.add_argument('--normalize_env_data', type=str2bool, default=False)
    parser.add_argument('--mask_temporal', type=str2bool, default=False)

    # policy
    parser.add_argument('--ppo_epochs', type=int, default=30)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    parser.add_argument('--policy_activation', type=str, default='relu', choices=['relu', 'leakyrelu', 'tanh', 'sigmoid'])

    # critic
    parser.add_argument('--critic_hidden_size', type=int, default=128)
    parser.add_argument('--critic_activation', type=str, default='relu', choices=['relu', 'leakyrelu', 'tanh', 'sigmoid'])

    # discriminator
    parser.add_argument('--loss_type', type=str, default='linear', choices=['linear', 'bce', 'bce_neg', 'ace'])
    parser.add_argument('--discrim_hidden_size', type=int, default=64)
    parser.add_argument('--discrim_learning_rate', type=float, default=.0004)
    parser.add_argument('--discrim_activation', type=str, default='relu', choices=['relu', 'leakyrelu', 'tanh', 'sigmoid'])
    parser.add_argument('--instance_noise', type=str2bool, default=False)

    # env
    parser.add_argument('--idm_data_flag', type=str2bool, default=True)
    parser.add_argument("--random_init", type=str2bool, default=True)
    parser.add_argument('--env_uc', type=str2bool, default=False)
    parser.add_argument('--env_multiagent', type=str2bool, default=False)
    parser.add_argument('--env_vectorize', type=str2bool, default=False)
    parser.add_argument('--env_reward', type=int, default=0)
    parser.add_argument('--env_type', type=str, default='LeaderFollowerRewardEngEnvUC_Imitator', 
                        choices=['LeaderFollowerRewardEngEnvUC_Imitator', 
                                 'LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps',
                                 'LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps',
                                 ])

    # gail
    parser.add_argument('--num_epochs', type=int, default=1200)

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_home, 'pytorch', args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    
    print("exp_dir:", exp_dir)

    saver_dir = os.path.join(exp_dir, 'imitate', 'log')
    if not os.path.exists(os.path.join(exp_dir, 'imitate')):
        os.mkdir(os.path.join(exp_dir, 'imitate'))
    if not os.path.exists(saver_dir):
        os.mkdir(saver_dir)

    print("saver_dir:", saver_dir)

    np.savez(os.path.join(saver_dir, 'args'), args=args)

    # load Experts data
    if args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps':
        with open('../data/rx_ra_policy_randomEnv_uc_trajs.npz', 'rb') as f:
            print("loading rx_ra_policy_randomEnv_uc_trajs.npz")
            states_action_dict = np.load(f)
            batch_size, T, orig_feat_size = states_action_dict['rx'].shape
            ra = states_action_dict['ra']
            rx = np.zeros((batch_size, T, orig_feat_size*2+1))
            # s_{t}, a_{t-1}, s_{t-1} -> a_{t}
            rx[:, :, :3] = states_action_dict['rx']
            rx[:, 1:, 3] = states_action_dict['ra'][:, :-1]
            rx[:, 1:, 4:] = states_action_dict['rx'][:, :-1, :]
            if ra.shape[-1] != 1:
                ra = np.expand_dims(ra, axis=-1)
            rx_flat = np.reshape(rx, (-1, rx.shape[-1]))
            ra_flat = np.reshape(ra, (-1, ra.shape[-1]))
            assert (rx_flat.shape[-1] == 7)
            assert (ra_flat.shape[-1] == 1)
    elif args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps':
        with open('../data/rx_ra_policy_randomEnv_uc_trajs.npz', 'rb') as f:
            print("loading rx_ra_policy_randomEnv_uc_trajs.npz")
            states_action_dict = np.load(f)
            batch_size, T, orig_feat_size = states_action_dict['rx'].shape
            ra = states_action_dict['ra']
            rx = np.zeros((batch_size, T, orig_feat_size*3+2))
            rx[:, :, :3] = states_action_dict['rx'] # s_{t}, 0:3
            rx[:, 1:, 3] = states_action_dict['ra'][:, :-1] # a_{t-1}, 3
            rx[:, 1:, 4:7] = states_action_dict['rx'][:, :-1, :] # s_{t-1}, 4:7
            rx[:, 2:, 7] = states_action_dict['ra'][:, :-2] # a_{t-2}, 7
            rx[:, 2:, 8:] = states_action_dict['rx'][:, :-2, :] # s_{t-2}, 8:
            if ra.shape[-1] != 1:
                ra = np.expand_dims(ra, axis=-1)
            rx_flat = np.reshape(rx, (-1, rx.shape[-1]))
            ra_flat = np.reshape(ra, (-1, ra.shape[-1]))
            assert (rx_flat.shape[-1] == 11)
            assert (ra_flat.shape[-1] == 1)
    else:
        if args.env_type == 'LeaderFollowerRewardEngEnvUC_Imitator':
            with open('../data/rx_ra_policy_randomEnv_uc_trajs.npz', 'rb') as f:
                print("loading rx_ra_policy_randomEnv_uc_trajs.npz")
                states_action_dict = np.load(f)
                rx = states_action_dict['rx']
                ra = states_action_dict['ra']
        
        if ra.shape[-1] != 1:
            ra = np.expand_dims(ra, axis=-1)

        rx_flat = np.reshape(rx, (-1, rx.shape[-1]))
        ra_flat = np.reshape(ra, (-1, ra.shape[-1]))

        assert (rx_flat.shape[-1] == 3)
        assert (ra_flat.shape[-1] == 1)
    
    print("rx_flat:", rx_flat.shape)
    print("ra_flat:", ra_flat.shape)

    obs = rx_flat
    act = ra_flat

    # build the env
    if args.env_type == 'LeaderFollowerRewardEngEnvUC_Imitator':
        env = LeaderFollowerRewardEngEnvUC_Imitator(env_reward=args.env_reward, random_init=args.random_init)
    elif args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps':
        env = LeaderFollowerRewardEngEnvUC_ImitatorTwoSteps(env_reward=args.env_reward, random_init=args.random_init)
    elif args.env_type == 'LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps':
        env = LeaderFollowerRewardEngEnvUC_ImitatorThreeSteps(env_reward=args.env_reward, random_init=args.random_init)
    else:
        raise TypeError("Please choose the correct env_type.")


    # hyperparameters
    num_epochs = args.num_epochs
    mini_batch_size = 10000
    ppo_epochs = args.ppo_epochs
    discrim_hidden_size = args.discrim_hidden_size

    writer = SummaryWriter('./tensorboard')

    actor = Actor(
        args.state_dims, 
        args.act_dims, 
        hidden_size=args.policy_hidden_size, 
        activation=args.policy_activation, 
        std=1.
        ).to(device)
    critic = Critic(
        num_inputs=args.state_dims, 
        num_outputs=1, 
        hidden_size=args.critic_hidden_size, 
        activation=args.critic_activation,
        ).to(device)

    optimizer_actor = optim.Adam(actor.parameters(),)
    optimizer_critic = optim.Adam(critic.parameters(),)


    discriminator = Discriminator(
        args.state_dims + args.act_dims, 
        hidden_size=discrim_hidden_size, 
        activation=args.discrim_activation, 
        ).to(device)
    optimizer_discrim = optim.RMSprop(discriminator.parameters(), lr=args.discrim_learning_rate)

    print("Using instance noise:", args.instance_noise)

    for epoch in range(num_epochs):
        
        print()
        print("epoch:", epoch)

        # if epoch <= (num_epochs // 3):
        #     n_discriminator_train_epochs = 1
        # elif (num_epochs // 3) < epoch <= (num_epochs*2 // 3):
        #     n_discriminator_train_epochs = 3
        # else:
        #     n_discriminator_train_epochs = 5
        n_discriminator_train_epochs = 10

        actor.train()
        critic.train()
        discriminator.train()
        
        if args.env_multiagent or args.env_vectorize:
            returns_trajs, log_probs_trajs, values_trajs, states_trajs, actions_trajs, advantages_trajs = obtain_mul_trajs(args=args, actor=actor, critic=critic, env=env, discriminator=discriminator, max_steps=100)
        else:
            returns_trajs = []
            log_probs_trajs = []
            values_trajs = []
            states_trajs = []
            actions_trajs = []
            advantages_trajs = []
            for _idx_traj in range(256):
                returns, log_probs, values, states, actions, advantages = obtain_one_traj(args=args, actor=actor, critic=critic, env=env, discriminator=discriminator, max_steps=100)
                
                returns_trajs.append(returns)
                log_probs_trajs.append(log_probs)
                values_trajs.append(values)
                states_trajs.append(states)
                actions_trajs.append(actions)
                advantages_trajs.append(advantages)
        
            returns_trajs = torch.cat(returns_trajs, dim=0)
            log_probs_trajs = torch.cat(log_probs_trajs, dim=0)
            values_trajs = torch.cat(values_trajs, dim=0)
            states_trajs = torch.cat(states_trajs, dim=0)
            actions_trajs = torch.cat(actions_trajs, dim=0)
            advantages_trajs = torch.cat(advantages_trajs, dim=0)


        # update the policy
        actor.train()
        critic.train()
        discriminator.eval()
        if epoch % 1 == 0:
            # ppo_update(actor, critic, optimizer_actor, optimizer_critic, 
            # critic_loss_fn, ppo_epochs, mini_batch_size, states, actions, log_probs, 
            # returns, advantages, clip_param=0.2)
            actor_loss_avg, critic_loss_avg, entropy_avg, kl_penalty_avg = ppo_update(actor=actor, critic=critic, optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic,
                                                                                    ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size, states=states_trajs, actions=actions_trajs, 
                                                                                    log_probs=log_probs_trajs, returns=returns_trajs, advantages=advantages_trajs)
            print("actor_loss_avg:", actor_loss_avg)
            print("critic_loss_avg:", critic_loss_avg)
            print("entropy_avg:", entropy_avg)
            print("kl_penalty_avg:", kl_penalty_avg)
            print()


        # update the discriminator
        actor.eval()
        critic.eval()
        discriminator.train()
        for _idx_disc in range(n_discriminator_train_epochs):
            
            sample_idx = np.random.randint(0, obs.shape[0], size=(states_trajs.size(0)))
            
            expert_state_action = np.concatenate((obs[sample_idx, :], act[sample_idx, :]), axis=-1)
            expert_state_action = torch.from_numpy(expert_state_action).float().to(device)
            state_action        = torch.cat([states_trajs, actions_trajs], -1)


            if args.instance_noise:
                instance_noise_sigma  = (1. - (epoch+1)/num_epochs) * 1.0
                
                instance_noise_expert = torch.normal(mean=0., std=instance_noise_sigma, size=expert_state_action.size())
                instance_noise_actor  = torch.normal(mean=0., std=instance_noise_sigma, size=state_action.size())
                
                expert_state_action   = expert_state_action + instance_noise_expert
                state_action          = state_action + instance_noise_actor


            grad_pen = compute_grad_pen(expert_s_a=expert_state_action, 
                                        policy_s_a=state_action, 
                                        discriminator=discriminator,
                                        lambda_=5.,
                                        )

            real = expert_reward_concatenated(args, discriminator, expert_state_action, output_mode="torch")
            fake = expert_reward_concatenated(args, discriminator, state_action, output_mode="torch")

            
            # 'linear', 'bce', 'bce_neg', 'ace'
            if args.loss_type == 'linear':
                
                real_loss = - torch.mean(real)
                gen_loss = torch.mean(fake)
            
            elif args.loss_type == 'bce':
                adversarial_loss = torch.nn.BCEWithLogitsLoss()

                # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                label_real = 1
                label_fake = 0
                
                label_real = Variable(torch.FloatTensor(expert_state_action.size(0), 1).fill_(label_real), requires_grad=False).to(device)
                label_fake = Variable(torch.FloatTensor(state_action.size(0), 1).fill_(label_fake), requires_grad=False).to(device)
                
                real_loss = adversarial_loss(real, label_real)
                gen_loss = adversarial_loss(fake, label_fake)

            elif args.loss_type == 'bce_neg':
                adversarial_loss = torch.nn.BCEWithLogitsLoss()
                
                # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
                label_real = 0
                label_fake = 1
                
                label_real = Variable(torch.FloatTensor(expert_state_action.size(0), 1).fill_(label_real), requires_grad=False).to(device)
                label_fake = Variable(torch.FloatTensor(state_action.size(0), 1).fill_(label_fake), requires_grad=False).to(device)
                
                real_loss = adversarial_loss(real, label_real)
                gen_loss = adversarial_loss(fake, label_fake)

            else:
                raise TypeError('Please choose a correct loss_type!')
            
            discrim_loss = real_loss + gen_loss + grad_pen
            
            print("real_loss:", real_loss.item())
            print("gen_loss:", gen_loss.item())
            print("grad_pen:", grad_pen.item())
            print("discrim_loss:", discrim_loss.item())
            print()
            optimizer_discrim.zero_grad()
            discrim_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 40.)
            optimizer_discrim.step()


        # if (epoch+1) >= (num_epochs-50):
        #     # save the parameters
        #     torch.save(actor.state_dict(), os.path.join(saver_dir, 'actor_params_{}.pt'.format(epoch+1)))
        params = dict()
        params['actor'] = actor.state_dict()
        params['critic'] = critic.state_dict()
        params['discriminator'] = discriminator.state_dict()
        
        params['optimizer_actor'] = optimizer_actor.state_dict()
        params['optimizer_critic'] = optimizer_critic.state_dict()
        params['optimizer_discrim'] = optimizer_discrim.state_dict()
        
        torch_utils.save_params(saver_dir, params, epoch+1, max_to_keep=20)