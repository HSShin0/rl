"""Trainer for policy gradient agent."""
import time
from argparse import ArgumentParser

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from agent import Agent
from critic import Critic
from utils.replay_buffer import ReplayBuffer


class Trainer:
    """Trainer for policy gradient agent."""

    def __init__(self, agent, params):
        """Initialize the trainer."""
        self.agent = agent
        self._seed = params['seed']
        self.max_episode = params['max_episode']
        self.savepath = params['savepath']
        self.device = params['device']
        self._fix_random_seed(self._seed)
        # If we use target network, initialize the target network
        # via the critic.
        if self.agent.use_target:
            self.agent._update_target()

    def _fix_random_seed(self, seed):
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run_training_loop(self):
        """Run training loop."""
        assert self.agent.episode < self.max_episode,\
            'already done for maximal episode'
        while self.agent.episode < self.max_episode:
            # train one episode
            accumulated_rewards = self.agent.train_one_episode()
            self.agent.writer.add_scalar('Total Rewards',
                                         accumulated_rewards,
                                         self.agent.episode)
            self.agent.history['rewards'].append(accumulated_rewards)
            # Stop learning if critic "converges".
            if (self.agent.episode % 10 == 0 and
                    len(self.agent.history['critic_loss']) > 10):
                recent_crit_loss = np.array(
                        self.agent.history['critic_loss'][-10:])
                mean_loss = recent_crit_loss.mean()
                if mean_loss < self.agent.critic_threshold:
                    print(f"Early stopping with mean loss : {mean_loss}!!")
                    break
        # Save the parameters of models if ...
        self.agent.save(self.savepath)


def get_args():
    """Get arguments from the shell command."""
    parser = ArgumentParser()
    parser.add_argument('--render', help="show rendered image",
                        action='store_true')
    parser.add_argument('--capacity', help="capacity of replay buffer\
                        (default=100)", type=int, default=100)
    parser.add_argument('--max_episode', help="maximal number of episode\
                        for training (default=10)", type=int, default=10)
    parser.add_argument('--random_exploration', help="number of iterations to\
                        explore randomly without learning (default=50)",
                        type=int, default=50)
    parser.add_argument('--critic_update', help='frequency for updating critic\
                        (default=1)', type=int, default=1)
    parser.add_argument('--actor_update', help='frequency for updating actor\
                        (default=1)', type=int, default=1)
    parser.add_argument('--critic_threshold', help='early stopping criterion\
                        on convergence of critic (default=1e-1)', type=float,
                        default=1e-1)

    parser.add_argument('--actor_lr', help='learning rate of actor\
                        (default=1e-3)', type=float, default=1e-3)
    parser.add_argument('--critic_lr', help='learning rate of critic\
                        (default=1e-3)', type=float, default=1e-3)
    parser.add_argument('--batch_size', help='batch size for training actor &\
                        critic (default=16)', type=int, default=16)
    parser.add_argument('--env', help="environment (default='CartPole-v0')",
                        type=str, choices=['CartPole-v0', 'Pendulum-v0'],
                        default='CartPole-v0')
    parser.add_argument('--gamma', help='decreasing rate of future rewards\
                        (default=0.99)', type=float, default=0.99)
    parser.add_argument('--seed', help="random seed (default=2020)", type=int,
                        default=2021)
    parser.add_argument('--load', help="filename for loading the agent\
                        (default=None)", default=None)
    parser.add_argument('--savepath', help="Path to save outputs\
                        (default=backup.pth)", default='backup.pth')
    parser.add_argument('--cuda', help="cuda usage (default='auto')",
                        choices=['auto', 'cpu', '0', '1'], default='auto')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    LOAD = args.load
    CAPACITY = args.capacity

    env = gym.make(args.env)
    # set configuration parameters
    params = {
            'obser_n': env.observation_space.shape[0],
            'action_n': env.action_space.n,
            }
    actor_params = {
            'learning_rate': args.actor_lr,
            'batch_size': args.batch_size,
            }
    critic_params = {
            'learning_rate': args.critic_lr,
            'batch_size': args.batch_size,
            }
    agent_params = {
            'gamma': args.gamma,
            'render': args.render,
            'random_expl': args.random_exploration,
            'critic_update': args.critic_update,
            'actor_update': args.actor_update,
            'critic_threshold': args.critic_threshold,
            }
    actor_params.update(params)
    critic_params.update(params)

    # CPU/GPU device choice
    if args.cuda == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.cuda == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.cuda))
    print(f'Use {device}')

    actor = Actor(env, actor_params)
    critics = list()
    for _ in range(2):
        critics.append(Critic(critic_params, q=True))
    actor.to(device)
    for idx in range(2):
        critics[idx].to(device)
    assert len(critics) == 2

    actor_optim = torch.optim.Adam(actor.parameters(),
                                   lr=actor.params['learning_rate'])
    critic_optim = torch.optim.Adam(critics[0].parameters(),
                                    lr=critics[0].params['learning_rate'])
    # summary writer for tensorboard
    writer = SummaryWriter()

    # create the actor-critic agent
    agent_params.update(params)
    memory = ReplayBuffer(CAPACITY)
    agent = Agent(env, actor, critics, memory, actor_optim, critic_optim,
                  agent_params, device, writer)

    # Initialize or load the model
    if not LOAD:
        agent.actor._initialize()
        for critic in agent.critics:
            critic._initialize()
    else:
        agent.load(LOAD)

    # Parameters for trainer
    trainer_params = {
            'seed': args.seed,
            'max_episode': args.max_episode,
            'savepath': args.savepath,
            'device': device,
            }
    trainer = Trainer(agent, trainer_params)
    trainer.run_training_loop()
    print(f"Stop learning after {trainer.agent.episode} episodes and \
          {trainer.agent.n_iter} steps!")
