import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    """A neural net model for value function."""

    def __init__(self, params, q=True):
        super().__init__()
        self.params = params
        self.obser_n = params['obser_n']
        self.action_n = params['action_n']
        self.output_n = self.action_n if q else 1

        self.linear1 = nn.Linear(self.obser_n, 64)
        self.linear2 = nn.Linear(64, self.output_n)

    def forward(self, x):
        """
        x: batch of states with shape (batch_size, obser_n)
        return: Q-values of x with shape (batch_size, output_n)
        """
        x = F.relu_(self.linear1(x))
        val = self.linear2(x)

        return val
