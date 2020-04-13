import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    """A neural net model for value function."""

    def __init__(self, params, q=True):
        """
        params (dict):
            keys:
                obser_n: dimension of observation space
                action_n: dimension of action space
        q (bool): if q, Critic is Q-function, else V-function.
        """
        super().__init__()
        self.params = params
        self.obser_n = params['obser_n']
        self.action_n = params['action_n']
        # we use V-function if q is False
        self._q = q
        self.output_n = self.action_n if q else 1

        self.linear1 = nn.Linear(self.obser_n, 64)
        self.linear2 = nn.Linear(64, self.output_n)

    def forward(self, x):
        """
        x (torch.FloatTensor): batch of states with shape (batch_size, obser_n)
        return (torch.FloatTensor): the value function value of x with shape (batch_size, output_n)
        """
        x = F.relu_(self.linear1(x))
        val = self.linear2(x)

        return val

    def _initialize(self):
        """Initialize training parameters."""
        for m in self.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
