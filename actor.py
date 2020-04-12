import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """A simple neural net model for policy."""

    def __init__(self, env, params):
        super().__init__()
        self.env = env
        self.params = params
        self.obser_n = params['obser_n']
        self.action_n = params['action_n']

        self.linear1 = nn.Linear(self.obser_n, 64)
        self.linear2 = nn.Linear(64, self.action_n)

    def forward(self, x):
        '''
        x: batch of states with shape (batch_size, obser_n)

        return: "log(softmax(scores))" of shape (batch_size, action_n)
        '''
        x = F.relu(self.linear1(x))
        score = self.linear2(x)

        return F.log_softmax(score, dim=1)

    def get_action(self, state, strategy='greedy', epsilon=1e-1):
        '''
        Choose an action for given a state following the strategy

        strategy in ['greedy', 'epsilon-greedy']

        state: torch.Tensor of shape [1, obser_n]
        '''
        logit = self.forward(state)

        if strategy == 'greedy':
            return torch.argmax(logit, dim=1).cpu().numpy()[0]
        elif strategy == 'epsilon-greedy':
            if np.random.random_sample() < epsilon:
                return self.env.action_space.sample()
            else:
                return torch.argmax(logit, dim=1).cpu().numpy()[0]