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


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=1024, activation='relu', std=0.0):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = F.sigmoid

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear4 = nn.Linear(hidden_size, num_outputs)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

        self.log_std_actor = nn.Parameter(torch.ones(num_outputs,) * std)
    
    def forward(self, x):
        out1 = self.activation_fn(self.linear1(x))
        out2 = self.activation_fn(self.linear2(out1))
        out3 = self.activation_fn(self.linear3(out1 + out2))
        mu = self.linear4(out1 + out2 + out3)
        std   = self.log_std_actor.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs=1, hidden_size=1024, activation='relu',):
        super(Critic, self).__init__()

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = F.sigmoid

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        
        self.linear4 = nn.Linear(hidden_size, num_outputs)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, x):
        out1 = self.activation_fn(self.linear1(x))
        out2 = self.activation_fn(self.linear2(out1))
        out3 = self.activation_fn(self.linear3(out1 + out2))
        value = self.linear4(out1 + out2 + out3)
        return value



class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, activation='relu', dropout=0.2, clip=5.):
        super(Discriminator, self).__init__()
        
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = F.sigmoid

        self.linear1   = nn.Linear(num_inputs, hidden_size)
        self.linear2   = nn.Linear(hidden_size, hidden_size)
        self.linear3   = nn.Linear(hidden_size, hidden_size)
        self.linear4   = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

        self.dropout   = nn.Dropout(dropout)
        self.clip      = clip
    
    def forward(self, x):
        out1 = self.activation_fn(self.linear1(x))
        out1 = self.dropout(out1)
        out2 = self.activation_fn(self.linear2(out1))
        out2 = self.dropout(out2)
        out3 = self.activation_fn(self.linear3(out1 + out2))
        out3 = self.dropout(out3)
        out4 = self.linear4(out1 + out2 + out3)

        if self.clip > 0:   # if clip, we use sigmoid to bound in (0, clip) (positive argument)
            out4 = torch.sigmoid(out4) * self.clip

        return out4
